/**
 * mcpdf Worker — remote MCP server for PDF search.
 *
 * Exposes four MCP tools backed by Workers AI (query embedding), Vectorize
 * (chunk vectors + metadata), and D1 (per-document metadata + per-chunk text).
 * The local `mcpdf-index upload` CLI populates Vectorize and D1; this Worker
 * only reads/deletes from them.
 *
 * Auth: OAuth 2.1 + PKCE via @cloudflare/workers-oauth-provider, with the
 * user-authentication step delegated to Cloudflare Access. `defaultHandler`
 * trusts the `Cf-Access-Authenticated-User-Email` header injected by the
 * edge after Access validates, optionally cross-checks against OWNER_EMAIL,
 * and immediately calls completeAuthorization. The Access application MUST
 * be scoped to /authorize only — /register, /token, /.well-known/oauth-*,
 * and /mcp are reached by claude.ai server-to-server and cannot live behind
 * a browser-based gate.
 *
 * Vector ID format MUST stay in sync with `src/mcpdf/cli.py::_vector_id`:
 *   sha256(path || \x00 || version).slice(0,16) : <chunk_index zero-padded to 5>
 */

import OAuthProvider, { type OAuthHelpers } from "@cloudflare/workers-oauth-provider";
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
  WORKERS_AI_EMBED_MODEL?: string;
  // Set per-corpus via `[vars]` in wrangler.toml. Shown by claude.ai as the
  // connector's MCP server name; useful for distinguishing corpora.
  CORPUS_NAME?: string;
  // Defense-in-depth check against the Cf-Access-Authenticated-User-Email
  // header. Access already enforces the policy at the edge; this guards
  // against an over-broad policy ever being deployed. Optional — leave unset
  // to trust whatever identity Access lets through.
  OWNER_EMAIL?: string;
}

const ACCESS_EMAIL_HEADER = "cf-access-authenticated-user-email";

// Props are stashed in the OAuth grant and exposed on the DO as `this.props`.
// Stamps the Access-authenticated email so future tools can see which
// identity opened the session.
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

// Handles /healthz and /authorize (delegated to us by the OAuth provider).
// The /authorize handler trusts the Cf-Access-Authenticated-User-Email
// header injected by the Cloudflare edge after Access validates the user,
// then immediately completes the OAuth authorization. PKCE on the OAuth
// side keeps this safe even without an explicit consent click — the code
// the library issues goes to the client's registered redirect_uri and is
// only redeemable with the verifier the legitimate client generated.
const defaultHandler = {
  async fetch(request: Request, env: Env, _ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);

    if (url.pathname === "/healthz") {
      return new Response("ok\n", {
        headers: { "content-type": "text/plain; charset=utf-8" },
      });
    }

    if (url.pathname !== "/authorize" || request.method !== "GET") {
      return new Response("not found\n", {
        status: 404,
        headers: { "content-type": "text/plain; charset=utf-8" },
      });
    }

    const accessEmail = request.headers.get(ACCESS_EMAIL_HEADER);
    if (!accessEmail) {
      // Hard-fail closed: Access isn't in front of /authorize. Without this
      // guard, anyone on the public internet could complete the OAuth flow
      // and mint a token. The fix is a Cloudflare Access self-hosted app
      // scoped to <worker-host>/authorize, not a code change.
      return new Response(
        "forbidden: Cloudflare Access must be configured for /authorize\n",
        { status: 403, headers: { "content-type": "text/plain; charset=utf-8" } },
      );
    }
    if (env.OWNER_EMAIL && accessEmail !== env.OWNER_EMAIL) {
      return new Response("forbidden: not owner\n", { status: 403 });
    }

    const oauthReqInfo = await env.OAUTH_PROVIDER.parseAuthRequest(request);
    if (!oauthReqInfo.clientId) {
      return new Response("invalid OAuth request: missing client_id\n", { status: 400 });
    }

    const { redirectTo } = await env.OAUTH_PROVIDER.completeAuthorization({
      request: oauthReqInfo,
      userId: accessEmail,
      scope: oauthReqInfo.scope,
      props: { userId: accessEmail } satisfies Props,
      metadata: {},
    });
    // 303 so the OAuth client's callback (registered redirect_uri) is
    // followed with GET regardless of how this request arrived.
    return Response.redirect(redirectTo, 303);
  },
};

export default new OAuthProvider({
  apiRoute: "/mcp",
  apiHandler: McpdfAgent.serve("/mcp") as never,
  defaultHandler: defaultHandler as never,
  authorizeEndpoint: "/authorize",
  tokenEndpoint: "/token",
  clientRegistrationEndpoint: "/register",
});
