"""Request/response contracts for document ingest and hybrid Q&A."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TextIngestRequest(BaseModel):
    """Ingest raw text as a document."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    text: str = Field(..., min_length=1, max_length=100_000)
    source: str = Field(default="raw", max_length=255)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    """Result of a successful ingest pipeline run."""

    document_id: str
    source: str
    chunk_count: int
    embedding_provider: str
    created_at: datetime


class QueryRequest(BaseModel):
    """Hybrid search + grounded answer request."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)
    filters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("question")
    @classmethod
    def _non_empty_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value.strip()


class SourceCitation(BaseModel):
    """A grounded evidence snippet returned to the client."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    chunk_id: str
    source: str
    excerpt: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class GroundedAnswer(BaseModel):
    """
    Structured LLM answer contract.

    Enforced via Pydantic so invalid JSON never reaches the client.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    answer: str = Field(..., min_length=1, max_length=8000)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    sources: list[SourceCitation] = Field(default_factory=list, max_length=10)


class QueryResponse(GroundedAnswer):
    """HTTP response for POST /query with pipeline diagnostics."""

    model: str
    provider: str
    retrieval_mode: str
    reranked: bool
    low_confidence: bool
    candidates_considered: int
