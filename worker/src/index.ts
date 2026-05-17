/**
 * mcpdf Worker — remote MCP server for PDF search.
 *
 * Exposes four MCP tools backed by Workers AI (query embedding), Vectorize
 * (chunk vectors + metadata), and D1 (per-document metadata + per-chunk text).
 * The local `mcpdf-index upload` CLI populates Vectorize and D1; this Worker
 * only reads/deletes from them.
 *
 * Auth: OAuth 2.1 + PKCE via @cloudflare/workers-oauth-provider. Login is a
 * shared-password form served by `defaultHandler` at /authorize; on match,
 * we hand control back to the OAuth provider via completeAuthorization.
 *
 * Vector ID format MUST stay in sync with `src/mcpdf/cli.py::_vector_id`:
 *   sha256(path || \x00 || version).slice(0,16) : <chunk_index zero-padded to 5>
 */

import OAuthProvider, { type AuthRequest, type OAuthHelpers } from "@cloudflare/workers-oauth-provider";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { McpAgent } from "agents/mcp";
import { z } from "zod";

export interface Env {
  AI: Ai;
  VECTORIZE: Vectorize;
  DB: D1Database;
  MCP_OBJECT: DurableObjectNamespace<McpdfAgent>;
  OAUTH_KV: KVNamespace;
  // Injected at runtime by OAuthProvider — not declared in wrangler.toml.
  OAUTH_PROVIDER: OAuthHelpers;
  SHARED_PASSWORD?: string;
  WORKERS_AI_EMBED_MODEL?: string;
  // Set per-corpus via `[vars]` in wrangler.toml. Shown by claude.ai as the
  // connector's MCP server name; useful for distinguishing corpora.
  CORPUS_NAME?: string;
}

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

// Handles everything that isn't /mcp: /healthz, the login form at /authorize,
// and 404 for everything else. The OAuth provider routes /mcp to apiHandler
// (after auth validation) and /authorize, /token, /register, /.well-known/*
// to itself — but it delegates the *user-facing* part of /authorize (the
// login UX) to this defaultHandler.
const defaultHandler = {
  async fetch(request: Request, env: Env, _ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);

    if (url.pathname === "/healthz") {
      return new Response("ok\n", {
        headers: { "content-type": "text/plain; charset=utf-8" },
      });
    }

    if (url.pathname !== "/authorize") {
      return new Response("not found\n", {
        status: 404,
        headers: { "content-type": "text/plain; charset=utf-8" },
      });
    }

    if (request.method === "GET") {
      const oauthReqInfo = await env.OAUTH_PROVIDER.parseAuthRequest(request);
      if (!oauthReqInfo.clientId) {
        return new Response("invalid OAuth request: missing client_id\n", { status: 400 });
      }
      const state = encodeState(oauthReqInfo);
      return htmlResponse(loginPage(state, env.CORPUS_NAME ?? "mcpdf"));
    }

    if (request.method === "POST") {
      const form = await request.formData();
      const password = String(form.get("password") ?? "");
      const state = String(form.get("state") ?? "");

      if (!env.SHARED_PASSWORD) {
        return new Response(
          "server misconfigured: SHARED_PASSWORD secret is not set\n",
          { status: 500 },
        );
      }
      if (password !== env.SHARED_PASSWORD) {
        return htmlResponse(
          loginPage(state, env.CORPUS_NAME ?? "mcpdf", "Incorrect password."),
          401,
        );
      }

      let oauthReqInfo: AuthRequest;
      try {
        oauthReqInfo = decodeState(state);
      } catch {
        return new Response("invalid state\n", { status: 400 });
      }

      const { redirectTo } = await env.OAUTH_PROVIDER.completeAuthorization({
        request: oauthReqInfo,
        userId: "owner",
        scope: oauthReqInfo.scope,
        props: { userId: "owner" } satisfies Props,
        metadata: {},
      });
      return Response.redirect(redirectTo, 302);
    }

    return new Response("method not allowed\n", { status: 405 });
  },
};

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

function loginPage(state: string, corpusName: string, error?: string): string {
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Sign in to ${escapeHtml(corpusName)}</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    body { font-family: system-ui, -apple-system, sans-serif; max-width: 420px;
           margin: 6rem auto; padding: 0 1.5rem; color: #222; }
    h1 { margin: 0 0 0.25rem; font-size: 1.4rem; }
    p.sub { margin: 0 0 2rem; color: #666; font-size: 0.9rem; }
    label { display: block; font-size: 0.85rem; color: #555; margin-bottom: 0.3rem; }
    input[type=password] { width: 100%; padding: 0.6rem; font-size: 1rem;
           box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px;
           margin-bottom: 1rem; }
    button { padding: 0.6rem 1.4rem; font-size: 1rem; background: #222;
             color: white; border: none; border-radius: 4px; cursor: pointer; }
    .err { color: #c00; margin-bottom: 1rem; font-size: 0.9rem; }
  </style>
</head>
<body>
  <h1>Sign in</h1>
  <p class="sub">mcpdf corpus: <code>${escapeHtml(corpusName)}</code></p>
  ${error ? `<div class="err">${escapeHtml(error)}</div>` : ""}
  <form method="POST">
    <label for="p">Password</label>
    <input id="p" type="password" name="password" autofocus required autocomplete="current-password">
    <input type="hidden" name="state" value="${escapeHtml(state)}">
    <button type="submit">Sign in</button>
  </form>
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
