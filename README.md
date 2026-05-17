# mcpdf

Local indexer that extracts, chunks, and uploads PDFs to a **Cloudflare
Vectorize**-backed search index. Pairs with a Worker (`worker/`) that exposes
the resulting index as a remote MCP server.

Originally a self-contained local MCP server with `sqlite-vec`; rebuilt to push
the storage + serving side to Cloudflare so claude.ai and other remote clients
can reach it. The local CLI now does PDF extraction and bulk upload only.

A **corpus** is the triple (Vectorize index, D1 database, deployed Worker URL).
Each corpus is fully isolated — own URL, own data, own auth token, own Durable
Object state. You can run a single corpus (defaults to `mcpdf`) or many — see
[Multiple corpora](#multiple-corpora).

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

### Provision Cloudflare resources

```sh
cd worker
npx wrangler vectorize create mcpdf --dimensions=768 --metric=cosine
npx wrangler d1 create mcpdf
# Paste the printed D1 UUID into wrangler.toml under database_id.

# Create the documents table:
npx wrangler d1 execute mcpdf --remote --command "CREATE TABLE documents (
  path TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  content_sha256 TEXT NOT NULL,
  total_pages INTEGER NOT NULL,
  chunk_count INTEGER NOT NULL,
  indexed_at TEXT NOT NULL
);"
```

Vectorize's `search` tool filters by `document_path`. To make that filter
fast, also create a metadata index (cheap, one-time):

```sh
npx wrangler vectorize create-metadata-index mcpdf \
  --property-name=document_path --type=string
```

### Local dev

```sh
cd worker
npm install
npm run dev      # http://127.0.0.1:8787/healthz
```

`wrangler dev` uses remote bindings by default for AI / Vectorize / D1, so
local queries hit your real Cloudflare resources.

### Deploy

```sh
cd worker
npm run deploy   # publishes to mcpdf.<your-subdomain>.workers.dev
```

### Auth

If `MCP_AUTH_TOKEN` is set as a secret, the Worker requires
`Authorization: Bearer <token>` on every `/mcp` request. If unset, `/mcp` is
open — fine for a quick demo, not fine for a publicly-deployed Worker.

```sh
cd worker
openssl rand -hex 32 | npx wrangler secret put MCP_AUTH_TOKEN
```

### Connect from claude.ai

In claude.ai → Settings → Connectors → Add custom connector:

- **URL**: `https://mcpdf.<your-subdomain>.workers.dev/mcp`
- **Auth**: Bearer token; paste the value of `MCP_AUTH_TOKEN`.

The four tools will appear in the connector's tool list.

## Multiple corpora

Each corpus is a fully independent (Vectorize index, D1 database, Worker URL)
triple. The worker code is shared — additional corpora are deployed via
`wrangler.toml` `[env.X]` blocks, with `wrangler deploy --env X`.

To add a corpus called `mcpdf-music`:

1. **Provision Cloudflare resources** for the new corpus:

   ```sh
   cd worker
   npx wrangler vectorize create mcpdf-music --dimensions=768 --metric=cosine
   npx wrangler vectorize create-metadata-index mcpdf-music \
     --property-name=document_path --type=string
   npx wrangler d1 create mcpdf-music   # note the printed UUID

   npx wrangler d1 execute mcpdf-music --remote --command "CREATE TABLE documents (
     path TEXT PRIMARY KEY,
     title TEXT NOT NULL,
     content_sha256 TEXT NOT NULL,
     total_pages INTEGER NOT NULL,
     chunk_count INTEGER NOT NULL,
     indexed_at TEXT NOT NULL
   );"
   ```

2. **Add an `[env.music]` block to `wrangler.toml`** (copy the commented
   `[env.example]` template at the bottom of the file and change `example` to
   `music`, including the `database_id`). Bindings are not inherited from the
   top-level config — every binding must be declared in the env block.

3. **Set the auth secret for this env**:

   ```sh
   openssl rand -hex 32 | npx wrangler secret put MCP_AUTH_TOKEN --env music
   ```

4. **Deploy**:

   ```sh
   npx wrangler deploy --env music
   # → https://mcpdf-music.<your-subdomain>.workers.dev/mcp
   ```

5. **Upload PDFs to the corpus**:

   ```sh
   uv run mcpdf-index extract /path/to/music-PDFs --out music.jsonl
   uv run mcpdf-index upload music.jsonl --corpus mcpdf-music
   ```

6. **Add as a separate connector in claude.ai** with the new URL and its own
   `MCP_AUTH_TOKEN`. The connector's name will read as `mcpdf-music` (the
   `CORPUS_NAME` var from the env block).

The default top-level config in `wrangler.toml` deploys to `mcpdf` when you run
`wrangler deploy` with no `--env`. Treat it as your primary corpus or remove it
if every corpus should be explicitly named.

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
