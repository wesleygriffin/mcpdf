from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import tiktoken

from .pdf import PdfPage

# cl100k_base is the GPT-3.5/4 encoder. Used here only as a stable token *budget*
# approximation — the Nomic models have their own tokenizer, but we don't need
# exact alignment, just a reliable cap so we never exceed the model's input limit.
_ENCODING = tiktoken.get_encoding("cl100k_base")


@dataclass(frozen=True)
class Chunk:
    text: str
    page_start: int  # inclusive, 1-indexed
    page_end: int  # inclusive, 1-indexed
    token_count: int


def _tokenize(text: str) -> list[int]:
    return _ENCODING.encode(text, disallowed_special=())


def _detokenize(tokens: list[int]) -> str:
    return _ENCODING.decode(tokens)


def chunk_pages(
    pages: Iterable[PdfPage],
    *,
    chunk_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    """Build overlapping token-budgeted chunks while tracking source page ranges.

    Pages are concatenated into a single token stream tagged with their source
    page index, then a sliding window walks the stream producing chunks that
    record the first and last page they touch.
    """
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens must be in [0, chunk_tokens)")

    # Build a flat list of (token, page_number) pairs.
    flat: list[tuple[int, int]] = []
    for page in pages:
        if not page.text:
            continue
        for tok in _tokenize(page.text):
            flat.append((tok, page.page_number))
        # Treat the page break as a single newline token so chunks read naturally.
        for tok in _tokenize("\n"):
            flat.append((tok, page.page_number))

    if not flat:
        return []

    step = chunk_tokens - overlap_tokens
    chunks: list[Chunk] = []
    i = 0
    while i < len(flat):
        window = flat[i : i + chunk_tokens]
        tokens = [t for t, _ in window]
        page_start = window[0][1]
        page_end = window[-1][1]
        text = _detokenize(tokens).strip()
        if text:
            chunks.append(
                Chunk(
                    text=text,
                    page_start=page_start,
                    page_end=page_end,
                    token_count=len(tokens),
                )
            )
        if i + chunk_tokens >= len(flat):
            break
        i += step
    return chunks
