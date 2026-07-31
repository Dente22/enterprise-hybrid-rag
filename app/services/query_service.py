"""Query orchestration: sanitize → hybrid retrieve → rerank → grounded answer."""

from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.schemas.documents import QueryRequest, QueryResponse, SourceCitation
from app.services.hybrid_search_service import HybridSearchService
from app.services.llm_service import LLMService, LLMServiceError
from app.services.sanitizer import sanitize_question

logger = logging.getLogger(__name__)


class QueryService:
    """End-to-end hybrid Q&A pipeline."""

    def __init__(self, session: AsyncSession, settings: Settings | None = None) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.search = HybridSearchService(session=session, settings=self.settings)

    async def ask(self, payload: QueryRequest) -> QueryResponse:
        """Answer a question using hybrid retrieval and Structured Outputs."""
        question = sanitize_question(payload.question)
        top_k = min(payload.top_k, self.settings.rerank_top_k)

        ranked, mode, candidates, reranked = await self.search.retrieve(
            query=question,
            top_k=top_k,
        )

        citations = [
            SourceCitation(
                document_id=item.document_id,
                chunk_id=item.chunk_id,
                source=item.source,
                excerpt=item.content[:400],
                score=float(item.rerank_score if item.rerank_score is not None else item.fused_score),
                metadata=item.metadata,
            )
            for item in ranked
        ]

        # Confidence proxy from best retrieval score (normalized-ish).
        best = max((c.score for c in citations), default=0.0)
        # Cross-encoder scores can be outside 0..1; squash roughly for gating.
        retrieval_confidence = _squash_score(best)

        if not ranked or retrieval_confidence < self.settings.min_confidence_threshold:
            fallback = LLMService.low_confidence_fallback(
                reason=(
                    "Retrieved context relevance is below the configured threshold "
                    f"({self.settings.min_confidence_threshold:.2f})."
                ),
                citations=citations,
            )
            return QueryResponse(
                answer=fallback.answer,
                confidence_score=fallback.confidence_score,
                sources=fallback.sources,
                model="n/a",
                provider="n/a",
                retrieval_mode=mode,
                reranked=reranked,
                low_confidence=True,
                candidates_considered=candidates,
            )

        context_blocks = [
            {
                "document_id": item.document_id,
                "chunk_id": item.chunk_id,
                "source": item.source,
                "score": citations[idx].score,
                "content": item.content,
                "metadata": item.metadata,
            }
            for idx, item in enumerate(ranked)
        ]

        async with LLMService(settings=self.settings) as llm:
            try:
                grounded, model, provider, _retries = await llm.answer(
                    question=question,
                    context_blocks=context_blocks,
                )
            except LLMServiceError:
                logger.exception("LLM answering failed")
                raise

        # Prefer pipeline citations if the model returned empty sources.
        sources = grounded.sources or citations
        low = grounded.confidence_score < self.settings.min_confidence_threshold
        if low:
            grounded = LLMService.low_confidence_fallback(
                reason="Model confidence was below the safety threshold.",
                citations=sources,
            )

        return QueryResponse(
            answer=grounded.answer,
            confidence_score=grounded.confidence_score,
            sources=grounded.sources or sources,
            model=model,
            provider=provider,
            retrieval_mode=mode,
            reranked=reranked,
            low_confidence=low,
            candidates_considered=candidates,
        )


def _squash_score(score: float) -> float:
    """Map arbitrary ranking scores into an approximate 0..1 confidence."""
    if score <= 0:
        return 0.0
    if score <= 1:
        return float(score)
    # Cross-encoder logits / unbounded ranks → logistic squash.
    import math

    return 1.0 / (1.0 + math.exp(-score))
