"""Document parsing helpers for PDF / Markdown / plain text."""

from __future__ import annotations

import io
from pathlib import Path


def extract_text_from_upload(*, filename: str, data: bytes) -> tuple[str, str]:
    """
    Extract text and a normalized content type from an uploaded file.

    Returns:
        (text, content_type)
    """
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(data), "application/pdf"
    if suffix in {".md", ".markdown"}:
        return data.decode("utf-8", errors="ignore"), "text/markdown"
    if suffix in {".txt", ".text", ".log"}:
        return data.decode("utf-8", errors="ignore"), "text/plain"
    # Fallback: try UTF-8 text.
    return data.decode("utf-8", errors="ignore"), "application/octet-stream"


def _extract_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n\n".join(part.strip() for part in pages if part and part.strip())
