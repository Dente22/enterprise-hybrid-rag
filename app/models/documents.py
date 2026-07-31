"""Persistence models for documents and searchable chunks."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.db.base import Base


def _uuid_str() -> str:
    return str(uuid.uuid4())


class Document(Base):
    """Ingested source document (PDF / Markdown / raw text)."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    source: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    content_type: Mapped[str] = mapped_column(String(64), nullable=False, default="text/plain")
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSON().with_variant(JSONB, "postgresql"),
        nullable=False,
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    chunks: Mapped[list[DocumentChunk]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )


class DocumentChunk(Base):
    """
    Semantically chunked unit with dense embedding and FTS payload.

    - `embedding` JSON works on SQLite and Postgres.
    - `embedding_vector` (pgvector) and `search_vector` (tsvector) are
      maintained on PostgreSQL for hybrid ANN + FTS retrieval.
    """

    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_estimate: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSON().with_variant(JSONB, "postgresql"),
        nullable=False,
        default=dict,
    )
    embedding: Mapped[list[float] | None] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=True,
    )
    embedding_dim: Mapped[int | None] = mapped_column(Integer, nullable=True)
    embedding_provider: Mapped[str | None] = mapped_column(String(32), nullable=True)
    search_vector: Mapped[str | None] = mapped_column(
        TSVECTOR().with_variant(Text(), "sqlite"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    document: Mapped[Document] = relationship(back_populates="chunks")
