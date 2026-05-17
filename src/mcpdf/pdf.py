from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pymupdf  # PyMuPDF


@dataclass(frozen=True)
class PdfPage:
    page_number: int  # 1-indexed, matches what users see in viewers
    text: str


@dataclass(frozen=True)
class PdfDocument:
    path: Path
    title: str
    content_sha256: str
    pages: list[PdfPage]

    @property
    def total_pages(self) -> int:
        return len(self.pages)


def hash_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk_size):
            h.update(block)
    return h.hexdigest()


def extract_pdf(path: Path) -> PdfDocument:
    """Extract per-page text from a PDF. Empty pages are kept so page numbers stay aligned."""
    content_hash = hash_file(path)
    pages: list[PdfPage] = []
    with pymupdf.open(path) as doc:
        title = (doc.metadata or {}).get("title") or path.stem
        for idx, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            pages.append(PdfPage(page_number=idx, text=text.strip()))
    return PdfDocument(
        path=path.resolve(),
        title=title.strip(),
        content_sha256=content_hash,
        pages=pages,
    )


def iter_pdfs(root: Path, *, recursive: bool = True) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(root)
    if root.is_file():
        return [root] if root.suffix.lower() == ".pdf" else []
    pattern = "**/*.pdf" if recursive else "*.pdf"
    return sorted(p for p in root.glob(pattern) if p.is_file())
