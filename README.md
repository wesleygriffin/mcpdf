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
```

`--corpus NAME` overrides both `VECTORIZE_INDEX` and `D1_DATABASE` — the
convention is that one name labels both. The default is `mcpdf`.

Vector IDs are derived as `sha256(path)[:16]:<chunk_index>`, so re-uploading the
same file upserts in place. The D1 `documents` row is upserted by path.

## Worker (`worker/`)

A remote MCP server (Streamable HTTP transport) that exposes four tools to
clients like claude.ai:

| Tool | What it does |
| --- | --- |
| `search` | Embeds the query via Workers AI, runs `topK` Vectorize lookup, returns hits with text + page range. Optional `document_path` filter. |
| `list_documents` | Returns every row from D1's `documents` table. |
| `get_document_info` | Single-row lookup by `path`. |
| `remove_document` | Deletes the doc's row from D1 and all its chunk vectors from Vectorize. |

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
   npx wrangler d1 create studio          # note the printed UUID
   ```

   The metadata index is what makes `search`'s `document_path` filter fast.
   Cheap to create up front, painful to backfill later.

2. **Create the `documents` table**:

   ```sh
   npx wrangler d1 execute studio --remote --command "CREATE TABLE documents (
     path TEXT PRIMARY KEY,
     title TEXT NOT NULL,
     content_sha256 TEXT NOT NULL,
     total_pages INTEGER NOT NULL,
     chunk_count INTEGER NOT NULL,
     indexed_at TEXT NOT NULL
   );"
   ```

3. **Add an `[env.studio]` block to `wrangler.toml`** (already present for
   `studio`; copy the commented template at the bottom of the file for new
   corpora and rename throughout). Paste the D1 UUID from step 1 into the
   block's `database_id`. Bindings are not inherited from the top-level
   config — every binding must be declared in the env block.

4. **Deploy**:

   ```sh
   npx wrangler deploy --env studio
   # → https://studio.<your-subdomain>.workers.dev
   curl https://studio.<your-subdomain>.workers.dev/healthz   # → ok
   ```

5. **Set the auth secret** (must be after first deploy):

   ```sh
   openssl rand -hex 32 | tee /dev/tty | npx wrangler secret put MCP_AUTH_TOKEN --env studio
   ```

   `tee /dev/tty` prints the token so you can copy it for claude.ai before
   piping it into wrangler. Without `MCP_AUTH_TOKEN` set, `/mcp` is open.

6. **Upload PDFs**:

   ```sh
   cd ..   # back to repo root
   uv run mcpdf-index extract /path/to/studio-PDFs --out studio.jsonl
   uv run mcpdf-index upload studio.jsonl --corpus studio
   ```

7. **Add as a connector in claude.ai** — Settings → Connectors → Add custom
   connector:

   - **URL**: `https://studio.<your-subdomain>.workers.dev/mcp`
   - **Auth**: Bearer; paste the value from step 5.

   The connector's MCP server name will read as `studio` (the `CORPUS_NAME`
   var from the env block).

### Local dev

```sh
cd worker
npx wrangler dev --env studio    # http://127.0.0.1:8787/healthz
```

`wrangler dev` uses remote bindings by default for AI / Vectorize / D1, so
local queries hit your real Cloudflare resources.

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
4. **Store** — Vectors go into Vectorize keyed by `sha256(path)[:16]:<idx>` with
   per-chunk metadata (path, title, page_start, page_end, text). Per-document
   rows go into D1's `documents` table.
5. **Search (Worker-side, future)** — Worker receives an MCP `search` call,
   embeds the query via Workers AI, queries Vectorize with `topK`, returns the
   hits with metadata.
