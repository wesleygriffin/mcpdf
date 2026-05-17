from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .pdf import PdfPage
from .tokenizer import GemmaTokenizer


@dataclass(frozen=True)
class Chunk:
    text: str
    page_start: int  # inclusive, 1-indexed
    page_end: int  # inclusive, 1-indexed
    token_count: int


def chunk_pages(
    pages: Iterable[PdfPage],
    tokenizer: GemmaTokenizer,
    *,
    chunk_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    """Build overlapping token-budgeted chunks while tracking source page ranges.

    Pages are concatenated into a single (token, page_number) stream; a sliding
    window walks the stream producing chunks that record the first and last
    page they touch. Token boundaries are the embedder's own — EmbeddingGemma's
    SentencePiece — so the resulting count is exactly what Workers AI will see.
    """
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens must be in [0, chunk_tokens)")

    flat: list[tuple[int, int]] = []
    newline_ids = tokenizer.encode_ids("\n")
    for page in pages:
        if not page.text:
            continue
        for tok in tokenizer.encode_ids(page.text):
            flat.append((tok, page.page_number))
        for tok in newline_ids:
            flat.append((tok, page.page_number))

    if not flat:
        return []

    step = chunk_tokens - overlap_tokens
    chunks: list[Chunk] = []
    i = 0
    while i < len(flat):
        window = flat[i : i + chunk_tokens]
        ids = [t for t, _ in window]
        text = tokenizer.decode(ids).strip()
        if text:
            chunks.append(
                Chunk(
                    text=text,
                    page_start=window[0][1],
                    page_end=window[-1][1],
                    token_count=len(ids),
                )
            )
        if i + chunk_tokens >= len(flat):
            break
        i += step
    return chunks
