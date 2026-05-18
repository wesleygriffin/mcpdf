/**
 * mcpdf Worker — remote MCP server for PDF search.
 *
 * Exposes four MCP tools backed by Workers AI (query embedding), Vectorize
 * (chunk vectors + metadata), and D1 (per-document metadata + per-chunk text).
 * The local `mcpdf-index upload` CLI populates Vectorize and D1; this Worker
 * only reads/deletes from them.
 *
 * Auth: OAuth 2.1 + PKCE via @cloudflare/workers-oauth-provider. Login is a
 * magic-link flow served by `defaultHandler`: POST /authorize stores the
 * OAuth AuthRequest under `magic:<uuid>` in OAUTH_KV and emails a sign-in
 * link via Cloudflare's send_email binding; GET /authorize/verify consumes
 * the token (single-use, KV-delete-before-complete) and hands control back
 * to the OAuth provider via completeAuthorization.
 *
 * Vector ID format MUST stay in sync with `src/mcpdf/cli.py::_vector_id`:
 *   sha256(path || \x00 || version).slice(0,16) : <chunk_index zero-padded to 5>
 */

import OAuthProvider, { type AuthRequest, type OAuthHelpers } from "@cloudflare/workers-oauth-provider";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { McpAgent } from "agents/mcp";
import { EmailMessage } from "cloudflare:email";
import { z } from "zod";

export interface Env {
  AI: Ai;
  VECTORIZE: Vectorize;
  DB: D1Database;
  MCP_OBJECT: DurableObjectNamespace<McpdfAgent>;
  OAUTH_KV: KVNamespace;
  // Injected at runtime by OAuthProvider — not declared in wrangler.toml.
  OAUTH_PROVIDER: OAuthHelpers;
  // Cloudflare send_email binding. `destination_address` in wrangler.toml
  // restricts what `to` we can pass; must include OWNER_EMAIL.
  SEND_EMAIL: SendEmail;
  WORKERS_AI_EMBED_MODEL?: string;
  // Set per-corpus via `[vars]` in wrangler.toml. Shown by claude.ai as the
  // connector's MCP server name; useful for distinguishing corpora.
  CORPUS_NAME?: string;
  // Recipient for sign-in links. Must be a verified Email Routing destination
  // on a domain bound to this account, and must match the send_email
  // binding's destination_address.
  OWNER_EMAIL?: string;
  // Sender for sign-in emails. Must live on a domain this account controls.
  MAIL_FROM?: string;
}

const MAGIC_TTL_SEC = 600;

// Props are stashed in the OAuth grant and exposed on the DO as `this.props`.
// Single-user shared-password setup just stamps a fixed "owner" identity.
type Props = { userId: string };

interface DocumentRow {
  path: string;
  version: string;
  title: string;
  content_sha256: string;
  total_pages: number;
  chunk_count: number;
  indexed_at: string;
}

interface ChunkMetadata {
  document_path?: string;
  document_title?: string;
  document_sha256?: string;
  page_start?: number;
  page_end?: number;
  version?: string;
}

const DEFAULT_EMBED_MODEL = "@cf/google/embeddinggemma-300m";

export class McpdfAgent extends McpAgent<Env, Record<string, never>, Props> {
  server = new McpServer({
    name: this.env.CORPUS_NAME ?? "mcpdf",
    version: "0.1.0",
  });

  async init() {
    this.server.registerTool(
      "search",
      {
        description:
          "Semantic search over indexed PDF chunks. Returns the most relevant " +
          "chunks with their document title, page range, surrounding text, and " +
          "version metadata (content sha + optional user tag).",
        inputSchema: {
          query: z.string().min(1).describe("Natural-language query to embed and match."),
          top_k: z
            .number()
            .int()
            .min(1)
            .max(50)
            .default(8)
            .describe("Number of chunks to return (1-50)."),
          document_path: z
            .string()
            .optional()
            .describe("If set, restrict results to chunks from this document path."),
          version: z
            .string()
            .optional()
            .describe(
              "If set, restrict results to chunks stamped with this user-tag version " +
                "(the --version value at upload time).",
            ),
          document_sha256: z
            .string()
            .optional()
            .describe(
              "If set, restrict results to chunks from a document with this content " +
                "SHA-256 (auto-tracked at extract time, independent of --version).",
            ),
        },
      },
      async ({ query, top_k, document_path, version, document_sha256 }) => {
        const embedding = await embedQuery(this.env, query);
        const queryOpts: VectorizeQueryOptions = {
          topK: top_k,
          returnMetadata: "all",
        };
        const filter: Record<string, string> = {};
        if (document_path) filter.document_path = document_path;
        if (version) filter.version = version;
        if (document_sha256) filter.document_sha256 = document_sha256;
        if (Object.keys(filter).length > 0) {
          queryOpts.filter = filter as VectorizeVectorMetadataFilter;
        }
        const result = await this.env.VECTORIZE.query(embedding, queryOpts);
        const textById = await fetchChunkText(
          this.env,
          result.matches.map((m) => m.id),
        );
        const hits = result.matches.map((m) => {
          const meta = (m.metadata ?? {}) as ChunkMetadata;
          return {
            id: m.id,
            score: m.score,
            document_path: meta.document_path ?? null,
            document_title: meta.document_title ?? null,
            document_sha256: meta.document_sha256 ?? null,
            version: meta.version ?? null,
            page_start: meta.page_start ?? null,
            page_end: meta.page_end ?? null,
            text: textById.get(m.id) ?? "",
          };
        });
        return {
          content: [{ type: "text", text: JSON.stringify({ hits }, null, 2) }],
        };
      },
    );

    this.server.registerTool(
      "list_documents",
      {
        description:
          "List every indexed (path, version) entry with title, page count, chunk count, " +
          "and indexed timestamp. Multiple versions of the same path appear as separate rows.",
        inputSchema: {},
      },
      async () => {
        const { results } = await this.env.DB.prepare(
          "SELECT path, version, title, content_sha256, total_pages, chunk_count, indexed_at " +
            "FROM documents ORDER BY title COLLATE NOCASE, version",
        ).all<DocumentRow>();
        return {
          content: [
            { type: "text", text: JSON.stringify({ documents: results ?? [] }, null, 2) },
          ],
        };
      },
    );

    this.server.registerTool(
      "get_document_info",
      {
        description:
          "Look up document metadata by path. Returns every version of that path " +
          "unless `version` is given, in which case returns just that one.",
        inputSchema: {
          path: z.string().min(1).describe("Document path as stored at upload time."),
          version: z
            .string()
            .optional()
            .describe(
              "If set, return only the row for this version. Omit to return all versions.",
            ),
        },
      },
      async ({ path, version }) => {
        const stmt =
          version !== undefined
            ? this.env.DB.prepare(
                "SELECT path, version, title, content_sha256, total_pages, chunk_count, indexed_at " +
                  "FROM documents WHERE path = ? AND version = ?",
              ).bind(path, version)
            : this.env.DB.prepare(
                "SELECT path, version, title, content_sha256, total_pages, chunk_count, indexed_at " +
                  "FROM documents WHERE path = ? ORDER BY version",
              ).bind(path);
        const { results } = await stmt.all<DocumentRow>();
        const documents = results ?? [];
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(
                { found: documents.length > 0, path, documents },
                null,
                2,
              ),
            },
          ],
        };
      },
    );

    this.server.registerTool(
      "remove_document",
      {
        description:
          "Remove indexed content for a path. Without `version`, removes every version " +
          "of the path (all D1 rows and all chunk vectors). With `version`, removes only " +
          "that one. Idempotent (returns ok even if nothing was indexed).",
        inputSchema: {
          path: z.string().min(1).describe("Document path as stored at upload time."),
          version: z
            .string()
            .optional()
            .describe(
              "If set, remove only this version. Omit to remove every version of the path.",
            ),
        },
      },
      async ({ path, version }) => {
        const stmt =
          version !== undefined
            ? this.env.DB.prepare(
                "SELECT version, chunk_count FROM documents WHERE path = ? AND version = ?",
              ).bind(path, version)
            : this.env.DB.prepare(
                "SELECT version, chunk_count FROM documents WHERE path = ?",
              ).bind(path);
        const { results } = await stmt.all<{ version: string; chunk_count: number }>();
        const rows = results ?? [];
        if (rows.length === 0) {
          return {
            content: [
              {
                type: "text",
                text: JSON.stringify({ removed: false, path, version, reason: "not_indexed" }),
              },
            ],
          };
        }
        let totalIds = 0;
        const allIds: string[] = [];
        for (const row of rows) {
          const ids = await chunkIds(path, row.version, row.chunk_count);
          await this.env.VECTORIZE.deleteByIds(ids);
          allIds.push(...ids);
          totalIds += ids.length;
        }
        // Delete from chunks table in batches under D1's 100-param cap.
        for (let i = 0; i < allIds.length; i += 90) {
          const batch = allIds.slice(i, i + 90);
          const placeholders = batch.map(() => "?").join(",");
          await this.env.DB.prepare(
            `DELETE FROM chunks WHERE vector_id IN (${placeholders})`,
          )
            .bind(...batch)
            .run();
        }
        if (version !== undefined) {
          await this.env.DB.prepare("DELETE FROM documents WHERE path = ? AND version = ?")
            .bind(path, version)
            .run();
        } else {
          await this.env.DB.prepare("DELETE FROM documents WHERE path = ?").bind(path).run();
        }
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(
                {
                  removed: true,
                  path,
                  versions_removed: rows.map((r) => r.version),
                  vectors_deleted: totalIds,
                },
                null,
                2,
              ),
            },
          ],
        };
      },
    );
  }
}

async function fetchChunkText(env: Env, ids: string[]): Promise<Map<string, string>> {
  if (ids.length === 0) return new Map();
  const placeholders = ids.map(() => "?").join(",");
  const { results } = await env.DB.prepare(
    `SELECT vector_id, text FROM chunks WHERE vector_id IN (${placeholders})`,
  )
    .bind(...ids)
    .all<{ vector_id: string; text: string }>();
  return new Map((results ?? []).map((r) => [r.vector_id, r.text]));
}

async function embedQuery(env: Env, text: string): Promise<number[]> {
  const model = env.WORKERS_AI_EMBED_MODEL ?? DEFAULT_EMBED_MODEL;
  const result = (await env.AI.run(model as keyof AiModels, { text: [text] } as never)) as {
    data?: number[][];
  };
  const vec = result.data?.[0];
  if (!vec) {
    throw new Error(`Workers AI returned no embedding for model ${model}`);
  }
  return vec;
}

async function chunkIds(path: string, version: string, chunkCount: number): Promise<string[]> {
  // Must match src/mcpdf/cli.py::_vector_id — `path\x00version` keyed sha,
  // first 16 hex chars, then `:<chunk_index:05d>`.
  const key = `${path}\x00${version}`;
  const prefix = await sha256Hex(key).then((h) => h.slice(0, 16));
  const ids: string[] = [];
  for (let i = 0; i < chunkCount; i++) {
    ids.push(`${prefix}:${i.toString().padStart(5, "0")}`);
  }
  return ids;
}

async function sha256Hex(input: string): Promise<string> {
  const bytes = new TextEncoder().encode(input);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

// Handles everything that isn't /mcp: /healthz, the login UX at /authorize
// and /authorize/verify, and 404 for everything else. The OAuth provider
// routes /mcp to apiHandler (after auth validation) and /authorize, /token,
// /register, /.well-known/* to itself — but it delegates the *user-facing*
// part of /authorize (login) to this defaultHandler.
const defaultHandler = {
  async fetch(request: Request, env: Env, _ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);

    if (url.pathname === "/healthz") {
      return new Response("ok\n", {
        headers: { "content-type": "text/plain; charset=utf-8" },
      });
    }

    if (url.pathname === "/authorize/verify" && request.method === "GET") {
      return handleVerify(request, env, url);
    }

    if (url.pathname !== "/authorize") {
      return new Response("not found\n", {
        status: 404,
        headers: { "content-type": "text/plain; charset=utf-8" },
      });
    }

    const corpus = env.CORPUS_NAME ?? "mcpdf";

    if (request.method === "GET") {
      const oauthReqInfo = await env.OAUTH_PROVIDER.parseAuthRequest(request);
      if (!oauthReqInfo.clientId) {
        return new Response("invalid OAuth request: missing client_id\n", { status: 400 });
      }
      if (!env.OWNER_EMAIL) {
        return new Response(
          "server misconfigured: OWNER_EMAIL var is not set\n",
          { status: 500 },
        );
      }
      const state = encodeState(oauthReqInfo);
      return htmlResponse(requestLinkPage(state, corpus, maskEmail(env.OWNER_EMAIL)));
    }

    if (request.method === "POST") {
      if (!env.OWNER_EMAIL || !env.MAIL_FROM) {
        return new Response(
          "server misconfigured: OWNER_EMAIL or MAIL_FROM var is not set\n",
          { status: 500 },
        );
      }

      const form = await request.formData();
      const stateStr = String(form.get("state") ?? "");
      let oauthReqInfo: AuthRequest;
      try {
        oauthReqInfo = decodeState(stateStr);
      } catch {
        return new Response("invalid state\n", { status: 400 });
      }

      const token = crypto.randomUUID();
      await env.OAUTH_KV.put(magicKey(token), JSON.stringify(oauthReqInfo), {
        expirationTtl: MAGIC_TTL_SEC,
      });
      const verifyUrl = new URL("/authorize/verify", url);
      verifyUrl.searchParams.set("token", token);

      try {
        await sendMagicLink(env, verifyUrl.toString(), corpus);
      } catch (err) {
        // Best-effort cleanup so the unused token doesn't sit in KV.
        await env.OAUTH_KV.delete(magicKey(token));
        const msg = err instanceof Error ? err.message : String(err);
        return new Response(`failed to send email: ${msg}\n`, { status: 502 });
      }

      return htmlResponse(linkSentPage(corpus, maskEmail(env.OWNER_EMAIL)));
    }

    return new Response("method not allowed\n", { status: 405 });
  },
};

async function handleVerify(request: Request, env: Env, url: URL): Promise<Response> {
  const token = url.searchParams.get("token");
  if (!token) return new Response("missing token\n", { status: 400 });

  const key = magicKey(token);
  const stored = await env.OAUTH_KV.get(key);
  // Single-use: delete before completing so a refresh / re-click can't
  // re-grant. KV TTL is the freshness fence; this is the burn-on-use fence.
  await env.OAUTH_KV.delete(key);
  if (!stored) {
    return htmlResponse(
      messagePage(
        env.CORPUS_NAME ?? "mcpdf",
        "Link expired or already used",
        "Sign-in links expire after 10 minutes and can only be used once. Restart the connection from your client to get a new one.",
      ),
      410,
    );
  }

  let oauthReqInfo: AuthRequest;
  try {
    oauthReqInfo = JSON.parse(stored) as AuthRequest;
  } catch {
    return new Response("invalid stored state\n", { status: 500 });
  }

  const { redirectTo } = await env.OAUTH_PROVIDER.completeAuthorization({
    request: oauthReqInfo,
    userId: "owner",
    scope: oauthReqInfo.scope,
    props: { userId: "owner" } satisfies Props,
    metadata: {},
  });
  // Use a 303 so the client follows with GET (the redirect target is the
  // OAuth client's callback URL, which expects GET).
  return Response.redirect(redirectTo, 303);
}

async function sendMagicLink(env: Env, verifyUrl: string, corpus: string): Promise<void> {
  const from = env.MAIL_FROM!;
  const to = env.OWNER_EMAIL!;
  const subject = `Sign in to ${corpus}`;
  const body = [
    `Click the link below to sign in to the "${corpus}" mcpdf connector.`,
    ``,
    verifyUrl,
    ``,
    `This link is single-use and expires in ${Math.round(MAGIC_TTL_SEC / 60)} minutes.`,
    `If you didn't request it, you can ignore this email.`,
  ].join("\r\n");

  const raw = buildMimeMessage({ from, to, subject, body });
  await env.SEND_EMAIL.send(new EmailMessage(from, to, raw));
}

function buildMimeMessage(opts: {
  from: string;
  to: string;
  subject: string;
  body: string;
}): string {
  // RFC 5322 wants CRLF; Cloudflare's send_email validator is strict about it.
  const domain = opts.from.split("@")[1] ?? "localhost";
  const messageId = `<${crypto.randomUUID()}@${domain}>`;
  const headers = [
    `From: ${opts.from}`,
    `To: ${opts.to}`,
    `Subject: ${opts.subject}`,
    `Message-ID: ${messageId}`,
    `Date: ${new Date().toUTCString()}`,
    `MIME-Version: 1.0`,
    `Content-Type: text/plain; charset=utf-8`,
    `Content-Transfer-Encoding: 7bit`,
  ];
  return headers.join("\r\n") + "\r\n\r\n" + opts.body + "\r\n";
}

function magicKey(token: string): string {
  return `magic:${token}`;
}

function maskEmail(addr: string): string {
  const [local, domain] = addr.split("@");
  if (!local || !domain) return addr;
  const head = local.length > 1 ? local[0] : local;
  return `${head}***@${domain}`;
}

function encodeState(req: AuthRequest): string {
  return btoa(JSON.stringify(req));
}
function decodeState(s: string): AuthRequest {
  return JSON.parse(atob(s)) as AuthRequest;
}

function htmlResponse(body: string, status = 200): Response {
  return new Response(body, {
    status,
    headers: { "content-type": "text/html; charset=utf-8" },
  });
}

const pageStyles = `
    body { font-family: system-ui, -apple-system, sans-serif; max-width: 420px;
           margin: 6rem auto; padding: 0 1.5rem; color: #222; }
    h1 { margin: 0 0 0.25rem; font-size: 1.4rem; }
    p.sub { margin: 0 0 1.5rem; color: #666; font-size: 0.9rem; }
    p { line-height: 1.5; }
    button { padding: 0.6rem 1.4rem; font-size: 1rem; background: #222;
             color: white; border: none; border-radius: 4px; cursor: pointer; }
    code { background: #f3f3f3; padding: 0.1rem 0.3rem; border-radius: 3px; }
`;

function requestLinkPage(state: string, corpusName: string, maskedTo: string): string {
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Sign in to ${escapeHtml(corpusName)}</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>${pageStyles}</style>
</head>
<body>
  <h1>Sign in</h1>
  <p class="sub">mcpdf corpus: <code>${escapeHtml(corpusName)}</code></p>
  <p>We'll email a single-use sign-in link to <code>${escapeHtml(maskedTo)}</code>. The link expires in 10 minutes.</p>
  <form method="POST">
    <input type="hidden" name="state" value="${escapeHtml(state)}">
    <button type="submit">Email me a sign-in link</button>
  </form>
</body>
</html>`;
}

function linkSentPage(corpusName: string, maskedTo: string): string {
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Check your inbox — ${escapeHtml(corpusName)}</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>${pageStyles}</style>
</head>
<body>
  <h1>Check your inbox</h1>
  <p class="sub">mcpdf corpus: <code>${escapeHtml(corpusName)}</code></p>
  <p>We sent a sign-in link to <code>${escapeHtml(maskedTo)}</code>. Open it in this browser to finish connecting. You can close this tab.</p>
</body>
</html>`;
}

function messagePage(corpusName: string, title: string, body: string): string {
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>${escapeHtml(title)} — ${escapeHtml(corpusName)}</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>${pageStyles}</style>
</head>
<body>
  <h1>${escapeHtml(title)}</h1>
  <p class="sub">mcpdf corpus: <code>${escapeHtml(corpusName)}</code></p>
  <p>${escapeHtml(body)}</p>
</body>
</html>`;
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

export default new OAuthProvider({
  apiRoute: "/mcp",
  apiHandler: McpdfAgent.serve("/mcp") as never,
  defaultHandler: defaultHandler as never,
  authorizeEndpoint: "/authorize",
  tokenEndpoint: "/token",
  clientRegistrationEndpoint: "/register",
});
