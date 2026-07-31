"""Semantic-ish text chunking with overlap and metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class TextChunk:
    """A single chunk ready for embedding and indexing."""

    index: int
    content: str
    token_estimate: int
    metadata: dict[str, Any]


_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars / token)."""
    return max(1, len(text) // 4)


def chunk_text(
    text: str,
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    base_metadata: dict[str, Any] | None = None,
) -> list[TextChunk]:
    """
    Split text into overlapping chunks preferring paragraph/sentence boundaries.

    `chunk_size` / `chunk_overlap` are measured in approximate tokens.
    """
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []

    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT.split(normalized) if p.strip()]
    units: list[str] = []
    for paragraph in paragraphs:
        if estimate_tokens(paragraph) <= chunk_size:
            units.append(paragraph)
            continue
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(paragraph) if s.strip()]
        if not sentences:
            units.append(paragraph)
            continue
        units.extend(sentences)

    chunks: list[TextChunk] = []
    buffer = ""
    meta = dict(base_metadata or {})

    def _flush(buf: str) -> None:
        content = buf.strip()
        if not content:
            return
        idx = len(chunks)
        chunks.append(
            TextChunk(
                index=idx,
                content=content,
                token_estimate=estimate_tokens(content),
                metadata={**meta, "chunk_id": idx},
            )
        )

    for unit in units:
        candidate = f"{buffer}\n\n{unit}".strip() if buffer else unit
        if estimate_tokens(candidate) <= chunk_size:
            buffer = candidate
            continue
        if buffer:
            _flush(buffer)
            # Overlap: keep the tail of the previous buffer.
            if chunk_overlap > 0:
                words = buffer.split()
                keep = max(1, chunk_overlap // 2)
                tail = " ".join(words[-keep:]) if words else ""
                buffer = f"{tail}\n\n{unit}".strip() if tail else unit
            else:
                buffer = unit
        else:
            # Single unit larger than chunk_size — hard split by characters.
            step = max(chunk_size * 4, 1)
            for start in range(0, len(unit), step):
                _flush(unit[start : start + step])
            buffer = ""

    if buffer:
        _flush(buffer)

    return chunks
