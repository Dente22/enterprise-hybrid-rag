"""Bootstrap schema for hybrid RAG.

Revision ID: 001_initial
Revises:
Create Date: 2026-07-31
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    is_postgres = bind.dialect.name == "postgresql"
    if is_postgres:
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    json_type = postgresql.JSONB() if is_postgres else sa.JSON()

    op.create_table(
        "documents",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("source", sa.String(255), nullable=False),
        sa.Column("content_type", sa.String(64), nullable=False),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("metadata", json_type, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_index("ix_documents_source", "documents", ["source"])

    op.create_table(
        "document_chunks",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("document_id", sa.String(36), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("token_estimate", sa.Integer(), nullable=False),
        sa.Column("metadata", json_type, nullable=False),
        sa.Column("embedding", json_type, nullable=True),
        sa.Column("embedding_dim", sa.Integer(), nullable=True),
        sa.Column("embedding_provider", sa.String(32), nullable=True),
        sa.Column("search_vector", postgresql.TSVECTOR() if is_postgres else sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_document_chunks_document_id", "document_chunks", ["document_id"])

    if is_postgres:
        op.execute("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS embedding_vector vector(768)")
        op.execute(
            """
            CREATE INDEX IF NOT EXISTS ix_document_chunks_embedding_vector
            ON document_chunks USING hnsw (embedding_vector vector_cosine_ops)
            """
        )
        op.execute(
            """
            CREATE INDEX IF NOT EXISTS ix_document_chunks_search_vector
            ON document_chunks USING gin (search_vector)
            """
        )


def downgrade() -> None:
    op.drop_table("document_chunks")
    op.drop_table("documents")
