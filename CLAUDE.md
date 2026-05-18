# CLAUDE.md

Project-specific guidance for future Claude Code sessions. Read alongside
the global `~/.claude/CLAUDE.md` rules, not in place of them.

## What this is

A local PDF extractor + uploader paired with a Cloudflare Worker that
exposes the resulting index as a remote MCP server.

- `src/mcpdf/` — Python CLI (`mcpdf-index`). Extracts PDF text with
  PyMuPDF, chunks against EmbeddingGemma's tokenizer (token counts match
  Workers AI billing), pushes vectors + chunk text to Cloudflare.
- `worker/` — TypeScript Cloudflare Worker using Cloudflare's `agents`
  SDK (Streamable HTTP MCP transport). `McpdfAgent` runs as a SQLite-backed
  Durable Object; exposes `search`, `list_documents`, `get_document_info`,
  `remove_document`.

Storage is split deliberately: Vectorize holds embeddings + a small slice of
metadata for filtering only; D1 holds per-document rows (`documents`) and
the full per-chunk text (`chunks`).

Auth: OAuth 2.1 + PKCE via `@cloudflare/workers-oauth-provider`, with the
worker hosting `/authorize`, `/token`, `/register`, and the well-known
discovery docs. Login is a magic-link flow (in `defaultHandler` inside
`worker/src/index.ts`): `POST /authorize` stores the OAuth `AuthRequest`
under `magic:<uuid>` in `OAUTH_KV` (10 min TTL) and emails a single-use
sign-in link via Cloudflare's `send_email` binding; `GET /authorize/verify`
deletes the KV entry (burn-on-use) and hands control back to the OAuth
provider via `completeAuthorization`. Per-corpus state (client
registrations, auth codes, tokens, magic tokens) all live in a per-env KV
namespace bound as `OAUTH_KV` — internal OAuth keys and our `magic:` keys
don't collide.

## Core concepts

### Corpus

A **corpus** is the triple `(Vectorize index, D1 database, deployed Worker URL)`.
All three share the same name (e.g., `studio`, `papers`). Each corpus is an
independently deployed Worker — see "Multi-corpus model" below.

### Document identity is `(path, version)`

Not just `path`. v15 and v16 of the same path coexist as distinct documents.
The `version` column is `''` when `--version` isn't passed at upload time —
the unversioned entry is itself one distinct version, not "no row." SQLite
composite PKs require every column `NOT NULL` because `NULL ≠ NULL` breaks
`ON CONFLICT`.

### Vector ID format is a wire contract

`sha256(document_path || \x00 || version)[:16] : <chunk_index:05d>`

Generated in `src/mcpdf/cli.py::_vector_id` and reconstructed in
`worker/src/index.ts::chunkIds`. **Bit-identical across Python and Node** —
verified. Don't change one side without the other, and don't change the
format at all without re-uploading every existing corpus.

## Multi-corpus model (wrangler.toml)

- **No top-level corpus.** Every corpus is its own `[env.X]` block.
  `wrangler deploy` without `--env <name>` deliberately errors (top-level
  has no `name`) — this prevents accidental misconfigured deploys.
- Bindings (AI / Vectorize / D1 / DO / migrations / vars) are **not
  inherited** from top-level into env blocks. Each env block redeclares
  everything. This is a wrangler design choice, not a bug.
- `CORPUS_NAME` in each block becomes the `McpServer({ name: … })` value
  so claude.ai shows distinct connector names per corpus.

## D1 schema (per corpus)

```sql
CREATE TABLE documents (
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
);
```

`chunks.text` is the source of truth for chunk text. Worker `search` does a
batched `SELECT … WHERE vector_id IN (…)` after the Vectorize query. **Do
not put chunk text back into Vectorize metadata** — the 10KB metadata cap
won't hold 1500-token chunks (we burned a session learning this).

## Hard limits we design around

- **Vectorize metadata**: 10KB JSON per vector. Keep small — no text.
- **D1 REST API**: 100 bound params per query, ~100KB SQL per statement.
  Chunks insert batches at 40 rows (80 params); deletes at 90 IDs.
- **EmbeddingGemma via Workers AI**: 768-dim vectors, 2048 max tokens per
  input. HF-gated — license must be accepted at
  <https://huggingface.co/google/embeddinggemma-300m>.
- **Workers Free**: 10k Neurons/day will throttle a bulk re-embed; user is
  on a paid plan.

## Credentials

CLI loads from a typical `.env` file. Required:
`CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN`, `HF_TOKEN`.

Per-corpus Worker configuration lives in `[env.<name>.vars]` and
`[[env.<name>.send_email]]` blocks in `wrangler.toml` (not secrets — none
of these are sensitive on their own, and putting them in vars makes
`wrangler.toml` the single source of truth):

- `OWNER_EMAIL` — recipient of sign-in links. **Must equal the
  `destination_address`** on the `send_email` binding (Cloudflare enforces
  the allowlist at the runtime layer) and must be a verified Email Routing
  destination on a domain bound to the account.
- `MAIL_FROM` — sender address for sign-in emails. Must live on a domain
  this account controls. No per-address verification needed beyond owning
  the domain.
- `CORPUS_NAME` — display name used by `McpServer({ name })` and the
  login page header.

There is no shared password and no identity model beyond "anyone who can
read `OWNER_EMAIL`'s inbox is `owner`." Rotate access by changing
`OWNER_EMAIL` + the binding's `destination_address` together and
redeploying — existing OAuth tokens stay valid (they don't re-check the
recipient) but new sign-ins go to the new address.

`SHARED_PASSWORD` and `MCP_AUTH_TOKEN` are **both no longer used** —
shared-password gating was replaced by magic-link, and the earlier
bearer-auth model was replaced by OAuth when claude.ai/Desktop went
OAuth-only. Safe to remove from any corpus that still has them:
`wrangler secret delete SHARED_PASSWORD --env <name>` /
`wrangler secret delete MCP_AUTH_TOKEN --env <name>`.

## Common commands

Python (from repo root):
```sh
uv sync
uv run ruff check src/mcpdf/
uv run mcpdf-index extract <path> --out X.jsonl
uv run mcpdf-index upload X.jsonl --corpus <name> [--version V]
```

Worker (from `worker/`):
```sh
npm run typecheck                                   # tsc --noEmit
npx wrangler deploy --dry-run --env <name>          # validate config only
npx wrangler deploy --env <name>                    # real deploy
npx wrangler d1 execute <name> --remote --command   # ad-hoc SQL
```

There is no `npm run dev` or `npm run deploy` shortcut — they were footguns
without `--env`. Use the `wrangler` commands directly.

## Sharp edges

- **`wrangler dev` uses remote bindings by default.** Local queries hit
  real Vectorize/D1 unless you opt out. Don't extract+upload test PDFs
  during local dev without realizing it.
- **`path_relative_to`** in `extract_chunks` makes `document_path`
  relative to the input root by default; `--absolute-paths` opts out.
  **Don't mix relative and absolute paths in the same corpus** — composite
  PK treats them as different documents.
- **CWB attaches to existing workers** — must `wrangler deploy --env X`
  once manually before the Builds tab appears in the dashboard.
- **Worker preview deployments share bindings with prod** — never use
  them for dev. Always a separate `[env.X]` block.
- **Re-uploading a shrunk doc** automatically cleans up high-index ghost
  vectors and chunks rows (the old-count > new-count branch in
  `_upload_one_doc`). Don't remove that logic to "simplify" — it solves a
  real correctness issue.
- **Wrangler binding changes don't take effect until a redeploy.** Editing
  `wrangler.toml` (e.g. pasting a real KV id over a placeholder) updates
  the *next* deploy's bindings; the live worker keeps the bindings it had
  at its last deploy. Symptom is usually a runtime `TypeError: Cannot read
  properties of undefined (reading 'get')` from inside whatever package
  expected `env.X`.
- **Don't fill in `client_id` / `client_secret` in claude.ai's connector
  UI.** Our worker advertises `/register` (dynamic client registration),
  so claude.ai/Desktop registers itself silently on first contact. The
  fields are escape hatches for OAuth servers that pre-issue static
  credentials; filling them with arbitrary values breaks the lookup.
- **The OAuth `defaultHandler` is the only place a user sees branded UX**
  (the magic-link request page, the "check your inbox" page, and the
  "link expired" page). Keep it minimal; if you ever add a logo, copy, or
  per-corpus styling, do it there. The MCP traffic itself never renders.
- **Magic-link tokens share `OAUTH_KV` with the OAuth provider's own
  state**, under the `magic:<uuid>` prefix. The library's keys are
  namespaced (`grant:`, `token:`, `client:`), so no collision. If you ever
  want to wipe just our magic tokens without disturbing OAuth state, you
  can scan `magic:` keys; conversely, blowing away `OAUTH_KV` wholesale
  invalidates both magic tokens *and* every issued OAuth credential.

## What's intentionally NOT here

- **No tests yet** (Python or TS). Validation surface is `ruff check` +
  `tsc --noEmit` + `wrangler deploy --dry-run`. Adding a test framework
  would be welcome but not silently — discuss first.
- **No CI YAML in the repo.** User chose Cloudflare Workers Builds
  (dashboard-only) so the CI config isn't coupled to a specific git host.
- **No git remote yet.** First commit hasn't been made; everything in
  `git status` is uncommitted as of session handoff.
