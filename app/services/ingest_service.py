"""Ingest pipeline: parse → chunk → embed → persist (vector + FTS)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.models.documents import Document, DocumentChunk
from app.schemas.documents import IngestResponse
from app.services.chunking import chunk_text
from app.services.embedding_service import EmbeddingService
from app.services.sanitizer import sanitize_user_text

logger = logging.getLogger(__name__)


class IngestService:
    """End-to-end document indexing pipeline."""

    def __init__(self, session: AsyncSession, settings: Settings | None = None) -> None:
        self.session = session
        self.settings = settings or get_settings()

    async def ingest_text(
        self,
        *,
        raw_text: str,
        source: str,
        content_type: str = "text/plain",
        metadata: dict[str, Any] | None = None,
    ) -> IngestResponse:
        """Chunk, embed, and index a text document."""
        cleaned = sanitize_user_text(raw_text, field_name="document text")
        meta = dict(metadata or {})
        meta.setdefault("ingestion_ts", datetime.now(UTC).isoformat())

        document = Document(
            source=source[:255],
            content_type=content_type,
            raw_text=cleaned,
            metadata_json=meta,
        )
        self.session.add(document)
        await self.session.flush()

        chunks = chunk_text(
            cleaned,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            base_metadata={
                "source": source,
                "document_id": document.id,
                "creation_date": document.created_at.isoformat()
                if document.created_at
                else meta.get("ingestion_ts"),
            },
        )
        if not chunks:
            raise ValueError("No indexable chunks produced from input text")

        provider_used = "unknown"
        async with EmbeddingService(settings=self.settings) as embedder:
            for piece in chunks:
                vector, provider_used = await embedder.embed(piece.content)
                record = DocumentChunk(
                    document_id=document.id,
                    chunk_index=piece.index,
                    content=piece.content,
                    token_estimate=piece.token_estimate,
                    metadata_json=piece.metadata,
                    embedding=vector,
                    embedding_dim=len(vector),
                    embedding_provider=provider_used,
                )
                self.session.add(record)
                await self.session.flush()

                if self.settings.is_postgres:
                    await self._sync_postgres_indexes(record.id, piece.content, vector)

        logger.info(
            "Ingested document=%s chunks=%s provider=%s",
            document.id,
            len(chunks),
            provider_used,
        )
        return IngestResponse(
            document_id=document.id,
            source=document.source,
            chunk_count=len(chunks),
            embedding_provider=provider_used,
            created_at=document.created_at or datetime.now(UTC),
        )

    async def _sync_postgres_indexes(
        self,
        chunk_id: str,
        content: str,
        vector: list[float],
    ) -> None:
        """Update pgvector column and tsvector for hybrid retrieval."""
        dim = self.settings.embedding_dimensions
        if len(vector) < dim:
            vector = vector + [0.0] * (dim - len(vector))
        elif len(vector) > dim:
            vector = vector[:dim]
        literal = "[" + ",".join(f"{v:.8f}" for v in vector) + "]"
        await self.session.execute(
            text(
                """
                UPDATE document_chunks
                SET embedding_vector = CAST(:vec AS vector),
                    search_vector = to_tsvector('english', :content)
                WHERE id = :chunk_id
                """
            ),
            {"vec": literal, "content": content, "chunk_id": chunk_id},
        )
