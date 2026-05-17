from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from . import db, indexer
from .config import Config

log = logging.getLogger("mcpdf")


def _format_search_results(query: str, hits: list[db.SearchHit]) -> str:
    if not hits:
        return f"No results for: {query!r}"
    lines = [f"Top {len(hits)} results for: {query!r}", ""]
    for i, h in enumerate(hits, start=1):
        page = (
            f"p. {h.page_start}"
            if h.page_start == h.page_end
            else f"pp. {h.page_start}-{h.page_end}"
        )
        snippet = h.text.replace("\n", " ").strip()
        if len(snippet) > 600:
            snippet = snippet[:597] + "..."
        lines.append(f"{i}. [{h.document_title}, {page}] (distance={h.distance:.3f})")
        lines.append(f"   path: {h.document_path}")
        lines.append(f"   {snippet}")
        lines.append("")
    return "\n".join(lines)


def build_server(cfg: Config) -> Server:
    server: Server = Server("mcpdf")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="index_directory",
                description=(
                    "Recursively scan a directory for PDF files and embed them into the local "
                    "vector store. Idempotent: unchanged files are skipped. Returns a summary."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Directory or PDF file path"},
                        "recursive": {"type": "boolean", "default": True},
                    },
                    "required": ["path"],
                },
            ),
            Tool(
                name="search",
                description=(
                    "Semantic search across indexed PDFs. Returns the most relevant chunks with "
                    "document title and page range."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "default": 8, "minimum": 1, "maximum": 50},
                        "document_ids": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Optional list of document IDs to restrict the search to.",
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="list_documents",
                description="List all indexed documents with their IDs, titles, and chunk counts.",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="get_document_info",
                description="Get metadata for a single indexed document.",
                inputSchema={
                    "type": "object",
                    "properties": {"document_id": {"type": "integer"}},
                    "required": ["document_id"],
                },
            ),
            Tool(
                name="remove_document",
                description="Remove a document and all its chunks/embeddings from the index.",
                inputSchema={
                    "type": "object",
                    "properties": {"document_id": {"type": "integer"}},
                    "required": ["document_id"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        if name == "index_directory":
            path = Path(arguments["path"]).expanduser()
            recursive = bool(arguments.get("recursive", True))
            result = await indexer.index_directory(cfg, path, recursive=recursive)
            summary = (
                f"Indexed: {len(result.indexed)}\n"
                f"Re-indexed (changed): {len(result.reindexed)}\n"
                f"Skipped (unchanged): {len(result.skipped_unchanged)}\n"
                f"Failed: {len(result.failed)}\n"
            )
            if result.failed:
                summary += "\nFailures:\n" + "\n".join(
                    f"  - {p}: {err}" for p, err in result.failed
                )
            return [TextContent(type="text", text=summary)]

        if name == "search":
            query = arguments["query"]
            top_k = int(arguments.get("top_k", 8))
            doc_ids = arguments.get("document_ids") or None
            hits = await indexer.search(cfg, query, top_k=top_k, document_ids=doc_ids)
            return [TextContent(type="text", text=_format_search_results(query, hits))]

        if name == "list_documents":
            conn = db.connect(cfg.db_path, embedding_dim=cfg.embedding_dim)
            try:
                rows = db.list_documents(conn)
            finally:
                conn.close()
            if not rows:
                return [TextContent(type="text", text="No documents indexed.")]
            lines = [
                f"{r['id']:>4}  {r['title']}  ({r['chunk_count']} chunks, "
                f"{r['total_pages']} pages, indexed {r['indexed_at']})\n"
                f"       {r['path']}"
                for r in rows
            ]
            return [TextContent(type="text", text="\n".join(lines))]

        if name == "get_document_info":
            doc_id = int(arguments["document_id"])
            conn = db.connect(cfg.db_path, embedding_dim=cfg.embedding_dim)
            try:
                row = db.get_document(conn, doc_id)
            finally:
                conn.close()
            if row is None:
                return [TextContent(type="text", text=f"No document with id {doc_id}.")]
            return [
                TextContent(
                    type="text",
                    text=(
                        f"id: {row['id']}\n"
                        f"title: {row['title']}\n"
                        f"path: {row['path']}\n"
                        f"total_pages: {row['total_pages']}\n"
                        f"indexed_at: {row['indexed_at']}\n"
                        f"sha256: {row['content_sha256']}"
                    ),
                )
            ]

        if name == "remove_document":
            doc_id = int(arguments["document_id"])
            conn = db.connect(cfg.db_path, embedding_dim=cfg.embedding_dim)
            try:
                row = db.get_document(conn, doc_id)
                if row is None:
                    return [TextContent(type="text", text=f"No document with id {doc_id}.")]
                db.delete_document(conn, doc_id)
            finally:
                conn.close()
            return [TextContent(type="text", text=f"Removed document {doc_id}: {row['title']}")]

        raise ValueError(f"Unknown tool: {name}")

    return server


async def _run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,  # stdout is the JSON-RPC channel; never log there
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = Config.from_env()
    log.info("mcpdf starting; db=%s model=%s", cfg.db_path, cfg.embedding_model)
    server = build_server(cfg)
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
