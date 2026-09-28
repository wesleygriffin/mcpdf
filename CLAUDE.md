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
discovery docs. User authentication for `/authorize` is delegated to
**Cloudflare Access** — a self-hosted Access application gates the
`/authorize` path at the edge, and `defaultHandler` (in
`worker/src/index.ts`) trusts the `Cf-Access-Authenticated-User-Email`
header Access injects, optionally cross-checks it against `OWNER_EMAIL`,
and immediately calls `completeAuthorization`. No browser UI in the
worker; no email, no token. PKCE on the OAuth side keeps the
no-consent-button flow safe — codes go to the registered `redirect_uri`
and only redeem with the verifier the legitimate client generated.
Per-corpus OAuth state (registrations, auth codes, tokens) lives in a
per-env KV namespace bound as `OAUTH_KV`.

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
won't hold 1500-token chunks.

## Hard limits we design around

- **Vectorize metadata**: 10KB JSON per vector. Keep small — no text.
- **D1 REST API**: 100 bound params per query, ~100KB SQL per statement.
  Chunks insert batches at 40 rows (80 params); deletes at 90 IDs.
- **EmbeddingGemma via Workers AI**: 768-dim vectors, 2048 max tokens per
  input. HF-gated — license must be accepted at
  <https://huggingface.co/google/embeddinggemma-300m>.
- **Workers Free**: 10k Neurons/day will throttle a bulk re-embed; a paid
  plan is recommended for the initial bulk upload.

## Credentials

CLI loads from a typical `.env` file. Required:
`CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN`, `HF_TOKEN`.

Per-corpus Worker configuration lives in `[env.<name>.vars]` in
`wrangler.toml` (not secrets — none of these are sensitive):

- `CORPUS_NAME` — display name used by `McpServer({ name })`.
- `OWNER_EMAIL` — defense-in-depth check against the Access-injected
  email header. Access already enforces the policy at the edge; this
  guards against an over-broad Access policy ever being deployed. Leave
  unset to trust whatever identity Access lets through.

The Access app itself is configured **outside** this repo in the
Cloudflare Zero Trust dashboard:

1. Zero Trust → Access → Applications → Add application → Self-hosted.
2. Application Domain: the worker host (e.g.
   `studio.<your-subdomain>.workers.dev`). **Path: `/authorize` only** — do not
   gate the whole hostname. `/register`, `/token`, `/.well-known/oauth-*`,
   and `/mcp` are reached by claude.ai server-to-server and must remain
   reachable without an Access session.
3. Policy: Allow → Include → Emails → the owner address.
4. Identity provider: whatever's configured for the account (Google /
   GitHub SSO recommended; one-time PIN if you don't want federation).

The worker fails closed: if `Cf-Access-Authenticated-User-Email` is
missing, `/authorize` returns 403 with a message pointing at the Access
misconfiguration.

`SHARED_PASSWORD`, `MAIL_FROM`, the `send_email` binding, and the
magic-link `/authorize/verify` route are all gone in the Access world.
Stale secrets are safe to delete: `wrangler secret delete SHARED_PASSWORD
--env <name>` (and `wrangler secret delete MCP_AUTH_TOKEN --env <name>`
if it's still around from the pre-OAuth bearer-token era).

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
- **Upload skips unchanged docs.** If the `documents` row for `(path,
  version)` already has the same `content_sha256` and `chunk_count`, the
  doc is skipped with no embedding. Re-extracting and re-uploading a whole
  directory is the intended way to pick up new or changed PDFs. If you
  change chunking in a way that keeps the chunk count the same, delete the
  rows (or `remove_document`) to force a re-embed.
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
- **The worker never renders branded UX anymore.** `defaultHandler` is
  pure machinery: header check → `completeAuthorization` → 303. The only
  HTML/UX a user sees during sign-in is Cloudflare Access's login page;
  customize there if you want a logo or custom copy.
- **Access app path scoping is load-bearing.** The Access policy MUST be
  scoped to `/authorize` only. Gating the whole hostname (or any of
  `/register`, `/token`, `/.well-known/oauth-*`, `/mcp`) breaks dynamic
  client registration immediately — claude.ai is a server-side OAuth
  client and has no Access session to present.
- **Header trust is safe because workers have no separate origin.** All
  traffic reaches a worker via Cloudflare's edge, and the edge strips
  inbound `Cf-Access-*` headers and re-injects them only after Access
  validates. Spoofing requires bypassing the edge, which isn't possible
  for `*.workers.dev`. If we ever front a worker with a custom origin or
  multi-cloud routing, upgrade to validating the `Cf-Access-Jwt-Assertion`
  JWT instead of trusting the email header.
- **CWB deploy command MUST include `--env <name>`.** A bare
  `npx wrangler deploy` succeeds against our config (CWB passes the
  worker name implicitly) but deploys *with every env-scoped binding
  stripped* — the live worker ends up with code but no KV/D1/Vectorize/AI
  attached. Symptom: runtime `TypeError: Cannot read properties of
  undefined (reading 'put')` from inside the OAuth library at
  `handleClientRegistration` (it hardcodes `env.OAUTH_KV.put`). Fix is in
  the dashboard, not the code: Workers & Pages → worker → Settings →
  Builds → Deploy command.

## What's intentionally NOT here

- **No tests yet** (Python or TS). Validation surface is `ruff check` +
  `tsc --noEmit` + `wrangler deploy --dry-run`. Adding a test framework
  would be welcome but not silently — discuss first.
- **No CI YAML in the repo.** Cloudflare Workers Builds (dashboard-only)
  handles CI so the config isn't coupled to a specific git host.
