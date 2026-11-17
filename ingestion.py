"""PDF text extraction and page-aware chunking."""
from __future__ import annotations

import re
from dataclasses import dataclass

import fitz  # PyMuPDF


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_name: str
    page: int
    text: str


def _clean(text: str) -> str:
    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(pdf_bytes: bytes) -> list[tuple[int, str]]:
    """Return (1-based page number, cleaned text) for every page of a PDF."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several exception types
        raise ValueError(f"Could not open PDF: {exc}") from exc
    with doc:
        return [(i, _clean(page.get_text("text"))) for i, page in enumerate(doc, start=1)]


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into roughly chunk_size-character pieces with overlap.

    Cuts move back to the nearest paragraph, sentence, or word boundary when one
    exists in the second half of the window, so chunks rarely end mid-word.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    text = text.strip()
    chunks: list[str] = []
    start, n = 0, len(text)

    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            window = text[start:end]
            boundary = max(
                window.rfind("\n\n"),
                window.rfind(". "),
                window.rfind("\n"),
                window.rfind(" "),
            )
            if boundary > chunk_size // 2:
                end = start + boundary + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - chunk_overlap, start + 1)  # always make progress

    return chunks


def build_chunks(
    doc_name: str,
    pages: list[tuple[int, str]],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Chunk each page separately so every chunk knows its exact page number."""
    chunks: list[Chunk] = []
    for page_no, text in pages:
        for idx, piece in enumerate(split_text(text, chunk_size, chunk_overlap)):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_name}|p{page_no}|c{idx}",
                    doc_name=doc_name,
                    page=page_no,
                    text=piece,
                )
            )
    return chunks
