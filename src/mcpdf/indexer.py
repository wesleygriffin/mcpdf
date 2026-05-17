from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from . import db
from .chunking import chunk_pages
from .config import Config
from .embeddings import LMStudioEmbeddingClient
from .pdf import PdfDocument, extract_pdf, iter_pdfs

log = logging.getLogger(__name__)


@dataclass
class IndexResult:
    indexed: list[str]
    skipped_unchanged: list[str]
    reindexed: list[str]
    failed: list[tuple[str, str]]

    def as_dict(self) -> dict:
        return {
            "indexed": self.indexed,
            "skipped_unchanged": self.skipped_unchanged,
            "reindexed": self.reindexed,
            "failed": [{"path": p, "error": e} for p, e in self.failed],
        }


async def _embed_chunks_in_batches(
    client: LMStudioEmbeddingClient, texts: list[str], *, batch_size: int
) -> list[list[float]]:
    out: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        out.extend(await client.embed(batch, prefix="document"))
    return out


async def _index_one(
    conn: sqlite3.Connection,
    client: LMStudioEmbeddingClient,
    cfg: Config,
    pdf: PdfDocument,
) -> str:
    """Returns one of: 'indexed', 'skipped', 'reindexed'."""
    path_str = str(pdf.path)
    existing = db.get_document_by_path(conn, path_str)
    status: str
    if existing is not None:
        if existing["content_sha256"] == pdf.content_sha256:
            log.info("Skipping unchanged: %s", path_str)
            return "skipped"
        log.info("Re-indexing changed: %s", path_str)
        db.delete_document(conn, int(existing["id"]))
        status = "reindexed"
    else:
        status = "indexed"

    chunks = chunk_pages(
        pdf.pages,
        chunk_tokens=cfg.chunk_tokens,
        overlap_tokens=cfg.chunk_overlap_tokens,
    )
    if not chunks:
        log.warning("No extractable text: %s", path_str)
        return status  # nothing to embed; still record the document below? -- skip entirely

    embeddings = await _embed_chunks_in_batches(
        client, [c.text for c in chunks], batch_size=cfg.embed_batch_size
    )
    if len(embeddings) != len(chunks):
        raise RuntimeError(
            f"embedding/chunk count mismatch: {len(embeddings)} vs {len(chunks)} for {path_str}"
        )

    doc_id = db.insert_document(
        conn,
        path=path_str,
        title=pdf.title,
        content_sha256=pdf.content_sha256,
        total_pages=pdf.total_pages,
    )
    rows = [
        (i, c.page_start, c.page_end, c.token_count, c.text, emb)
        for i, (c, emb) in enumerate(zip(chunks, embeddings))
    ]
    db.insert_chunks(conn, document_id=doc_id, chunks_with_embeddings=rows)
    return status


async def index_directory(
    cfg: Config,
    root: Path,
    *,
    recursive: bool = True,
) -> IndexResult:
    pdfs = iter_pdfs(root, recursive=recursive)
    if not pdfs:
        return IndexResult(indexed=[], skipped_unchanged=[], reindexed=[], failed=[])

    result = IndexResult(indexed=[], skipped_unchanged=[], reindexed=[], failed=[])
    conn = db.connect(cfg.db_path, embedding_dim=cfg.embedding_dim)
    try:
        async with LMStudioEmbeddingClient(cfg.lm_studio_url, cfg.embedding_model) as client:
            for path in pdfs:
                try:
                    pdf = extract_pdf(path)
                    status = await _index_one(conn, client, cfg, pdf)
                except Exception as exc:  # noqa: BLE001 — surface as a per-file failure
                    log.exception("Failed to index %s", path)
                    result.failed.append((str(path), f"{type(exc).__name__}: {exc}"))
                    continue
                if status == "indexed":
                    result.indexed.append(str(path))
                elif status == "reindexed":
                    result.reindexed.append(str(path))
                else:
                    result.skipped_unchanged.append(str(path))
    finally:
        conn.close()
    return result


async def search(
    cfg: Config,
    query: str,
    *,
    top_k: int = 10,
    document_ids: list[int] | None = None,
) -> list[db.SearchHit]:
    conn = db.connect(cfg.db_path, embedding_dim=cfg.embedding_dim)
    try:
        async with LMStudioEmbeddingClient(cfg.lm_studio_url, cfg.embedding_model) as client:
            query_vec = (await client.embed([query], prefix="query"))[0]
        return db.search(conn, query_embedding=query_vec, top_k=top_k, document_ids=document_ids)
    finally:
        conn.close()
