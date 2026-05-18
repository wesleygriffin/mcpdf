# mcpdf

Local indexer that extracts, chunks, and uploads PDFs to a **Cloudflare
Vectorize**-backed search index. Pairs with a Worker (`worker/`) that exposes
the resulting index as a remote MCP server.

Originally a self-contained local MCP server with `sqlite-vec`; rebuilt to push
the storage + serving side to Cloudflare so claude.ai and other remote clients
can reach it. The local CLI now does PDF extraction and bulk upload only.

A **corpus** is the triple (Vectorize index, D1 database, deployed Worker URL).
Each corpus is fully isolated — own URL, own data, own auth token, own Durable
Object state. Every corpus lives under its own `[env.X]` block in
`wrangler.toml`; there is no privileged default. See
[Adding a corpus](#adding-a-corpus).

## Layout

```
mcpdf/
├── src/mcpdf/              # Python — local extract+upload CLI
│   ├── chunking.py         # Token-budgeted chunking with page tracking
│   ├── tokenizer.py        # EmbeddingGemma tokenizer (gated HF download)
│   ├── pdf.py              # PyMuPDF text extraction
│   ├── indexer.py          # Orchestrates extract → ChunkRecord stream
│   ├── cloudflare.py       # Workers AI + Vectorize + D1 REST client
│   ├── cli.py              # `mcpdf-index extract` and `mcpdf-index upload`
│   └── config.py           # Loads creds from ~/Source/mm-env
└── worker/                 # TypeScript — Cloudflare Worker (MCP endpoint)
    ├── wrangler.toml       # Top-level + [env.X] blocks per corpus
    └── src/index.ts        # McpdfAgent (Durable Object) with 4 MCP tools
```

## Prerequisites

1. **Python toolchain**: `uv` for the local indexer.
2. **Node.js + npm** for the Worker (`worker/`).
3. **Cloudflare account** (paid plan recommended — Workers Free's 10k Neurons/day
   throttles the initial bulk re-embed).
4. **Hugging Face account** with the EmbeddingGemma license accepted at
   <https://huggingface.co/google/embeddinggemma-300m>.

## Credentials

The CLI loads credentials from `~/Source/mm-env` by default (override with
`MCPDF_ENV_FILE`):

```sh
chmod 600 ~/Source/mm-env
cat > ~/Source/mm-env <<'EOF'
CLOUDFLARE_ACCOUNT_ID=...
CLOUDFLARE_API_TOKEN=...
CLOUDFLARE_WORKERS_SUBDOMAIN=...workers.dev
HF_TOKEN=hf_...
EOF
```

`wrangler` also picks up `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` from
env, so `source`ing this file before Worker work is enough.

## Local indexer (`mcpdf-index`)

### Install

```sh
cd /path/to/mcpdf
uv sync
```

### Extract

```sh
uv run mcpdf-index extract /path/to/PDFs --out chunks.jsonl
```

This walks the directory (recursively by default), extracts text page-by-page
with PyMuPDF, and chunks against EmbeddingGemma's SentencePiece tokenizer.
Each line of `chunks.jsonl` is one chunk with denormalized document metadata
(title, sha, page range). No network calls are made beyond the one-time
tokenizer download from Hugging Face.

By default, each chunk's `document_path` is stored **relative to the input
root** you passed (e.g. `cubase/v15/manual.pdf` rather than the full
absolute path). Single-file inputs store just the basename. Pass
`--absolute-paths` to keep the full filesystem path instead. Don't mix
relative and absolute paths in the same corpus — they're treated as
different documents.

Defaults:

| Env var | Default | Purpose |
| --- | --- | --- |
| `CHUNK_TOKENS` | `1500` | Target tokens per chunk |
| `CHUNK_OVERLAP_TOKENS` | `150` | Overlap between consecutive chunks |
| `EMBED_BATCH_SIZE` | `32` | Texts per Workers AI request (upload-side) |
| `TOKENIZER_REPO` | `google/embeddinggemma-300m` | HF repo for `tokenizer.json` |
| `WORKERS_AI_MODEL` | `@cf/google/embeddinggemma-300m` | Embedding model on Cloudflare |
| `VECTORIZE_INDEX` | `mcpdf` | Vectorize index name |
| `D1_DATABASE` | `mcpdf` | D1 database name |

### Upload

```sh
# Dry-run prints per-document summaries without calling Cloudflare.
uv run mcpdf-index upload chunks.jsonl --dry-run

# Real run: embeds via Workers AI, upserts to Vectorize, writes documents row to D1.
uv run mcpdf-index upload chunks.jsonl

# Send to a non-default corpus (Vectorize index + D1 database must already exist):
uv run mcpdf-index upload chunks.jsonl --corpus mcpdf-music

# Stamp every uploaded vector + D1 row with a per-document version:
uv run mcpdf-index upload chunks.jsonl --corpus studio --version 15
```

`--corpus NAME` overrides both `VECTORIZE_INDEX` and `D1_DATABASE` — the
convention is that one name labels both. The default is `mcpdf`.

`--version VERSION` is **part of the document's identity**, not a tag —
uploading the same path with two different versions creates two coexisting
entries (e.g. `cubase_op_man.pdf --version 15` and then
`cubase_op_man.pdf --version 16`, both searchable). Re-uploading the same
`(path, version)` upserts in place. Omit `--version` to upload as the
'unversioned' entry, which is itself a distinct version. Independent of
this, every vector also carries `document_sha256` (the content hash from
extract time) — automatic, not overridable, useful for change detection
within a single version.

Vector IDs are derived as `sha256(path||\0||version)[:16]:<chunk_index>`, so
each `(path, version)` occupies its own ID space. The D1 `documents` row is
upserted by `(path, version)`.

## Worker (`worker/`)

A remote MCP server (Streamable HTTP transport) that exposes four tools to
clients like claude.ai:

| Tool | What it does |
| --- | --- |
| `search` | Embeds the query via Workers AI, runs `topK` Vectorize lookup, returns hits with text, page range, content sha, and version. Optional filters: `document_path`, `version`, `document_sha256`. |
| `list_documents` | Returns every `(path, version)` row from D1's `documents` table. Multiple versions of the same path appear as separate rows. |
| `get_document_info` | Returns every version of `path`, or just one if `version` is given. |
| `remove_document` | Without `version`, removes every version of `path`. With `version`, removes only that one. |

The MCP endpoint is `/mcp` (Streamable HTTP). `/healthz` returns `ok` for
uptime checks. The agent runs as a SQLite-backed Durable Object — one
instance per session — declared in `wrangler.toml` under `MCP_OBJECT`.

There is no privileged "default" corpus — every corpus is its own `[env.X]`
block in `wrangler.toml`. Every `wrangler` command needs `--env <corpus>`;
running it without `--env` deliberately errors.

### Adding a corpus

Walkthrough for a corpus named `studio` (the same recipe with `studio` →
`<name>` applies to any corpus). All `wrangler` commands run from `worker/`.

```sh
cd worker
```

1. **Provision Cloudflare resources**:

   ```sh
   npx wrangler vectorize create studio --dimensions=768 --metric=cosine
   npx wrangler vectorize create-metadata-index studio \
     --property-name=document_path --type=string
   npx wrangler vectorize create-metadata-index studio \
     --property-name=version --type=string
   npx wrangler d1 create studio          # note the printed UUID
   ```

   The metadata indexes are what make `search`'s `document_path` and
   `version` filters fast. Cheap to create up front, painful to backfill
   later. (You can add a `document_sha256` metadata index too if you plan to
   filter on raw content hash; Vectorize allows up to 10 per index.)

2. **Create the D1 tables** — one for per-document metadata, one for per-chunk text:

   ```sh
   npx wrangler d1 execute studio --remote --command "CREATE TABLE documents (
     path TEXT NOT NULL,
     version TEXT NOT NULL DEFAULT '',
     title TEXT NOT NULL,
     content_sha256 TEXT NOT NULL,
     total_pages INTEGER NOT NULL,
     chunk_count INTEGER NOT NULL,
     indexed_at TEXT NOT NULL,
     PRIMARY KEY (path, version)
   );
   CREATE TABLE chunks (
     vector_id TEXT PRIMARY KEY,
     text TEXT NOT NULL
   );"
   ```

   - **`documents`**: `(path, version)` is a composite PK — different versions
     of the same path coexist as separate rows. The `version` column is `''`
     for unversioned uploads (which is itself a distinct version, not "no row").
   - **`chunks`**: holds the full chunk text, keyed by `vector_id`. Vectorize's
     metadata cap (10KB per vector) is too small for big chunks; this table is
     where the full text actually lives. The worker JOINs to it at search time
     to populate the `text` field in hits.

3. **Provision the KV namespace for OAuth state**:

   ```sh
   npx wrangler kv namespace create OAUTH_KV --env studio
   ```

   Paste the printed `id` into `wrangler.toml` at the
   `[[env.studio.kv_namespaces]]` block, replacing
   `REPLACE_WITH_WRANGLER_KV_NAMESPACE_CREATE_OUTPUT`.

4. **Confirm `[env.studio]` block in `wrangler.toml`** (already present for
   `studio`; copy the commented template at the bottom for new corpora and
   rename throughout). Paste the D1 UUID from step 1 and the KV id from
   step 3. Bindings are not inherited from the top-level config — every
   binding must be declared in the env block.

5. **Deploy**:

   ```sh
   npx wrangler deploy --env studio
   # → https://studio.<your-subdomain>.workers.dev
   curl https://studio.<your-subdomain>.workers.dev/healthz   # → ok
   ```

6. **Gate `/authorize` with Cloudflare Access.** Auth is handled by Zero
   Trust at the edge — no shared password, no email loop. In the
   Cloudflare dashboard → Zero Trust → Access → Applications → Add an
   application → Self-hosted:

   - **Application Domain**: the worker host (e.g.
     `studio.fraktured.workers.dev`).
   - **Path**: `/authorize` — *only*. Do NOT gate the whole hostname or
     any of `/register`, `/token`, `/.well-known/oauth-*`, `/mcp`. Those
     are reached by claude.ai server-to-server and have no Access session.
   - **Identity providers**: whatever you have configured (Google or
     GitHub SSO recommended — one click and a persistent session).
   - **Policy**: Allow → Include → Emails → your owner address.

   Then set `OWNER_EMAIL` in `wrangler.toml` to the same address as a
   defense-in-depth check (the worker rejects sessions whose Access email
   doesn't match):

   ```toml
   [env.studio.vars]
   CORPUS_NAME = "studio"
   OWNER_EMAIL = "you@example.com"
   ```

   No secrets to set. The worker fails closed (403) if Access isn't in
   front when it should be.

7. **Upload PDFs**:

   ```sh
   cd ..   # back to repo root
   uv run mcpdf-index extract /path/to/studio-PDFs --out studio.jsonl
   uv run mcpdf-index upload studio.jsonl --corpus studio
   ```

8. **Add as a connector in claude.ai or Claude Desktop** — Settings →
   Connectors → Add custom connector. Just the URL:

   - **URL**: `https://studio.<your-subdomain>.workers.dev/mcp`

   The client will redirect you through the OAuth flow on first connect.
   If you're already logged in to Access (via the identity provider you
   configured), the entire `/authorize` step is invisible — the browser
   bounces through and back to claude.ai. If you're not logged in, you'll
   see the Access login page once. The MCP server name will read as
   `studio` (the `CORPUS_NAME` var from the env block).

### Local dev

```sh
cd worker
npx wrangler dev --env studio    # http://127.0.0.1:8787/healthz
```

`wrangler dev` uses remote bindings by default for AI / Vectorize / D1, so
local queries hit your real Cloudflare resources.

## Development

### Continuous deployment (Cloudflare Workers Builds)

Once a corpus's worker has been deployed once manually, you can wire it to
push-to-deploy via **Cloudflare Workers Builds** (CWB). CWB is configured in
the Cloudflare dashboard (no workflow YAML in the repo), so the CI config is
not tied to a specific git provider — switch between GitHub and GitLab by
reconnecting in the dashboard.

The pattern is **one Worker per environment, each with its own CWB
connection and branch filter**. For the production `studio` corpus:

1. Cloudflare dashboard → Workers & Pages → `studio` → Settings →
   **Builds** → **Connect**.
2. Authorize the git provider, pick the `mcpdf` repo.
3. Configure:
   - **Branch**: `main`
   - **Root directory**: `worker` (wrangler.toml lives here, not at the
     repo root)
   - **Build command**: `npm ci`
   - **Deploy command**: `npx wrangler deploy --env studio`
4. Save. Pushes to `main` will now build and deploy automatically.

Worker secrets persist across deploys — no need to re-set them. KV
namespace, D1, and `send_email` bindings are re-attached on each deploy
from `wrangler.toml`, so changes to those bindings (including the
`destination_address` allowlist) *do* require a redeploy to take effect.

### Dev/prod separation

To iterate on worker code without risking the live corpus, add a second
worker (e.g. `studio-dev`):

1. Add an `[env.studio-dev]` block to `wrangler.toml` (copy `[env.studio]`,
   rename throughout). Provision its own Vectorize index, D1 database, and
   `documents` table — see [Adding a corpus](#adding-a-corpus). Use a small
   set of test PDFs; the dev corpus doesn't need real data.
2. Deploy it once manually: `npx wrangler deploy --env studio-dev`. CWB
   attaches to existing deployed workers — the Builds tab only appears
   after the first deploy.
3. In the `studio-dev` worker's Builds settings, configure:
   - **Branch**: any pattern that excludes `main` (e.g. `!main` or a
     specific `dev` branch)
   - **Deploy command**: `npx wrangler deploy --env studio-dev`

Now main pushes deploy prod (`studio`); feature-branch pushes deploy dev
(`studio-dev`). The two workers have entirely separate URLs, Vectorize
indexes, D1 databases, and auth tokens.

**Don't try to use Workers' built-in preview deployments for this.**
Previews share bindings with the production worker, which means a dev query
would hit your real `studio` D1 and Vectorize. Separate `[env.X]` blocks
are the only way to get true isolation.

## How it works

1. **Extract** — PyMuPDF reads each PDF page-by-page, preserving page numbers.
2. **Chunk** — Pages are concatenated into a (token, page) stream tokenized
   with **EmbeddingGemma's SentencePiece** vocabulary. Sliding windows produce
   chunks that record the first and last page they touch. Using the embedder's
   own tokenizer means the chunk's token count is exactly what Workers AI will
   bill against.
3. **Embed (upload-side)** — Chunks are batched and sent to
   `@cf/google/embeddinggemma-300m` via Workers AI. Output is 768-dim float
   vectors.
4. **Store** — Vectors go into Vectorize keyed by
   `sha256(path||\0||version)[:16]:<idx>` with small per-chunk metadata (path,
   title, document_sha256, version, page_start, page_end). Full chunk text
   goes into D1's `chunks` table keyed by the same vector_id. One row per
   `(path, version)` goes into D1's `documents` table. Re-uploading a doc
   with fewer chunks than before automatically cleans up the now-ghost
   vectors and chunks rows at the high indices.
5. **Search** — Worker receives an MCP `search` call, embeds the query via
   Workers AI, queries Vectorize with `topK` (and any filters), then batches a
   single `SELECT text FROM chunks WHERE vector_id IN (…)` to populate the
   text on each hit before returning.
