from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .chunking import chunk_pages
from .config import Config
from .pdf import extract_pdf, iter_pdfs
from .tokenizer import GemmaTokenizer

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkRecord:
    """One emitted chunk, denormalized with its parent document's metadata.

    The Cloudflare upload step turns each record into a Vectorize vector
    (id derived from path+chunk_index) plus a D1 documents row (once per path).
    """

    document_path: str
    document_title: str
    document_sha256: str
    document_total_pages: int
    chunk_index: int
    page_start: int
    page_end: int
    token_count: int
    text: str


@dataclass(frozen=True)
class ExtractFailure:
    path: Path
    error: str


def extract_chunks(
    paths: list[Path],
    tokenizer: GemmaTokenizer,
    cfg: Config,
    *,
    path_relative_to: Path | None = None,
) -> Iterator[ChunkRecord | ExtractFailure]:
    """Emit ChunkRecords for each PDF.

    `path_relative_to`: if set, each chunk's `document_path` is stored relative
    to this directory (must be an absolute path). If None, absolute paths are
    stored (the pre-normalization default).
    """
    for path in paths:
        try:
            pdf = extract_pdf(path)
        except Exception as exc:  # noqa: BLE001 — surface as a per-file failure
            log.exception("Failed to extract %s", path)
            yield ExtractFailure(path=path, error=f"{type(exc).__name__}: {exc}")
            continue
        chunks = chunk_pages(
            pdf.pages,
            tokenizer,
            chunk_tokens=cfg.chunk_tokens,
            overlap_tokens=cfg.chunk_overlap_tokens,
        )
        if not chunks:
            log.warning("No extractable text: %s", path)
            continue
        document_path = (
            str(pdf.path.relative_to(path_relative_to))
            if path_relative_to is not None
            else str(pdf.path)
        )
        for idx, c in enumerate(chunks):
            yield ChunkRecord(
                document_path=document_path,
                document_title=pdf.title,
                document_sha256=pdf.content_sha256,
                document_total_pages=pdf.total_pages,
                chunk_index=idx,
                page_start=c.page_start,
                page_end=c.page_end,
                token_count=c.token_count,
                text=c.text,
            )


def collect_paths(root: Path, *, recursive: bool = True) -> list[Path]:
    return iter_pdfs(root, recursive=recursive)
