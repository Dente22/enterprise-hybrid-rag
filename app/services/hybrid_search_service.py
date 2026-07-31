"""
Hybrid search: dense vector retrieval + lexical FTS fused with RRF, then reranked.
"""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.models.documents import Document, DocumentChunk
from app.services.embedding_service import EmbeddingService
from app.services.reranker import RankedChunk, RerankerService

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    *,
    k: int = 60,
    weights: list[float] | None = None,
) -> dict[str, float]:
    """
    Reciprocal Rank Fusion over multiple ranked id lists.

    score(d) = sum_i weight_i * 1 / (k + rank_i(d))
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights length must match ranked_lists")

    scores: dict[str, float] = defaultdict(float)
    for weight, ranked in zip(weights, ranked_lists, strict=True):
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] += weight * (1.0 / (k + rank))
    return dict(scores)


class HybridSearchService:
    """Enterprise hybrid retrieval over document chunks."""

    def __init__(
        self,
        session: AsyncSession,
        settings: Settings | None = None,
        reranker: RerankerService | None = None,
    ) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.reranker = reranker or RerankerService(self.settings)

    async def retrieve(self, *, query: str, top_k: int | None = None) -> tuple[list[RankedChunk], str, int, bool]:
        """
        Run hybrid retrieval + reranking.

        Returns:
            (chunks, retrieval_mode, candidates_considered, reranked)
        """
        top_k = top_k or self.settings.rerank_top_k
        limit = self.settings.hybrid_candidate_limit

        async with EmbeddingService(settings=self.settings) as embedder:
            query_vector, _provider = await embedder.embed(query)

        if self.settings.is_postgres:
            vector_hits = await self._vector_search_postgres(query_vector, limit=limit)
            fts_hits = await self._fts_search_postgres(query, limit=limit)
            mode = "pgvector+fts+rrf"
        else:
            vector_hits = await self._vector_search_sqlite(query_vector, limit=limit)
            fts_hits = await self._fts_search_sqlite(query, limit=limit)
            mode = "sqlite-cosine+fts+rrf"

        vector_ids = [chunk_id for chunk_id, _score in vector_hits]
        fts_ids = [chunk_id for chunk_id, _score in fts_hits]
        fused = reciprocal_rank_fusion(
            [vector_ids, fts_ids],
            weights=[self.settings.hybrid_vector_weight, self.settings.hybrid_fts_weight],
        )
        if not fused:
            return [], mode, 0, False

        vector_score_map = dict(vector_hits)
        fts_score_map = dict(fts_hits)
        chunk_ids = list(fused.keys())

        stmt = (
            select(DocumentChunk, Document)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(DocumentChunk.id.in_(chunk_ids))
        )
        rows = (await self.session.execute(stmt)).all()
        by_id = {chunk.id: (chunk, document) for chunk, document in rows}

        candidates: list[RankedChunk] = []
        for chunk_id, fused_score in fused.items():
            pair = by_id.get(chunk_id)
            if pair is None:
                continue
            chunk, document = pair
            candidates.append(
                RankedChunk(
                    chunk_id=chunk.id,
                    document_id=document.id,
                    source=document.source,
                    content=chunk.content,
                    metadata=dict(chunk.metadata_json or {}),
                    fused_score=fused_score,
                    vector_score=float(vector_score_map.get(chunk_id, 0.0)),
                    fts_score=float(fts_score_map.get(chunk_id, 0.0)),
                )
            )

        ranked, used_reranker = self.reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=top_k,
        )
        return ranked, mode, len(candidates), used_reranker

    async def _vector_search_postgres(
        self,
        query_vector: list[float],
        *,
        limit: int,
    ) -> list[tuple[str, float]]:
        dim = self.settings.embedding_dimensions
        vector = query_vector
        if len(vector) < dim:
            vector = vector + [0.0] * (dim - len(vector))
        elif len(vector) > dim:
            vector = vector[:dim]
        literal = "[" + ",".join(f"{v:.8f}" for v in vector) + "]"
        result = await self.session.execute(
            text(
                """
                SELECT id,
                       1 - (embedding_vector <=> CAST(:q AS vector)) AS score
                FROM document_chunks
                WHERE embedding_vector IS NOT NULL
                ORDER BY embedding_vector <=> CAST(:q AS vector)
                LIMIT :limit
                """
            ),
            {"q": literal, "limit": limit},
        )
        return [(str(row.id), float(row.score or 0.0)) for row in result]

    async def _fts_search_postgres(self, query: str, *, limit: int) -> list[tuple[str, float]]:
        # Convert user query into a plainto_tsquery-friendly expression.
        result = await self.session.execute(
            text(
                """
                SELECT id,
                       ts_rank_cd(search_vector, plainto_tsquery('english', :q)) AS score
                FROM document_chunks
                WHERE search_vector @@ plainto_tsquery('english', :q)
                ORDER BY score DESC
                LIMIT :limit
                """
            ),
            {"q": query, "limit": limit},
        )
        return [(str(row.id), float(row.score or 0.0)) for row in result]

    async def _vector_search_sqlite(
        self,
        query_vector: list[float],
        *,
        limit: int,
    ) -> list[tuple[str, float]]:
        result = await self.session.execute(
            select(DocumentChunk).where(DocumentChunk.embedding.is_not(None))
        )
        scored: list[tuple[str, float]] = []
        for chunk in result.scalars().all():
            if not chunk.embedding:
                continue
            score = EmbeddingService.cosine_similarity(query_vector, chunk.embedding)
            scored.append((chunk.id, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    async def _fts_search_sqlite(self, query: str, *, limit: int) -> list[tuple[str, float]]:
        """Lightweight lexical ranking for SQLite local mode (BM25-like TF scoring)."""
        terms = [t.lower() for t in re.findall(r"[a-zA-Z0-9_]{2,}", query)]
        if not terms:
            return []
        result = await self.session.execute(select(DocumentChunk))
        scored: list[tuple[str, float]] = []
        for chunk in result.scalars().all():
            text_l = chunk.content.lower()
            tf = 0.0
            for term in terms:
                count = text_l.count(term)
                if count:
                    tf += 1.0 + math.log(count)
            if tf > 0:
                scored.append((chunk.id, tf))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]
