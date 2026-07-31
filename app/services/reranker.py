"""Cross-encoder reranking with graceful fallback."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RankedChunk:
    """Candidate chunk with fused and optional rerank scores."""

    chunk_id: str
    document_id: str
    source: str
    content: str
    metadata: dict[str, Any]
    fused_score: float
    vector_score: float
    fts_score: float
    rerank_score: float | None = None


class RerankerService:
    """
    Cross-encoder reranker.

    Uses sentence-transformers when available; otherwise falls back to fused RRF scores.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model: Any | None = None
        self._load_attempted = False

    def _ensure_model(self) -> bool:
        if not self.settings.reranker_enabled:
            return False
        if self._model is not None:
            return True
        if self._load_attempted:
            return False
        self._load_attempted = True
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.settings.reranker_model)
            logger.info("Loaded cross-encoder model %s", self.settings.reranker_model)
            return True
        except Exception:
            logger.warning(
                "Cross-encoder unavailable; continuing with RRF-only ranking",
                exc_info=True,
            )
            self._model = None
            return False

    def rerank(self, *, query: str, candidates: list[RankedChunk], top_k: int) -> tuple[list[RankedChunk], bool]:
        """
        Rerank candidates and return (top_k list, used_cross_encoder).
        """
        if not candidates:
            return [], False

        if not self._ensure_model():
            ordered = sorted(candidates, key=lambda c: c.fused_score, reverse=True)
            return ordered[:top_k], False

        pairs = [(query, c.content) for c in candidates]
        scores = self._model.predict(pairs)
        scored: list[RankedChunk] = []
        for candidate, score in zip(candidates, scores, strict=True):
            candidate.rerank_score = float(score)
            scored.append(candidate)
        scored.sort(key=lambda c: c.rerank_score or float("-inf"), reverse=True)
        return scored[:top_k], True
