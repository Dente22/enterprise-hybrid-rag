"""
FastAPI application entrypoint for Hybrid-Search Enterprise Document Q&A.

Exposes:
- POST /api/v1/documents/ingest
- POST /api/v1/documents/ingest-file
- POST /api/v1/query
- GET  /api/v1/health
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.session import dispose_db, init_db


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Initialize logging, storage directories, and database schema."""
    settings = get_settings()
    configure_logging(debug=settings.debug)
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    if settings.is_sqlite:
        Path("./data").mkdir(parents=True, exist_ok=True)
    await init_db()
    yield
    await dispose_db()


def create_app() -> FastAPI:
    """Application factory used by Uvicorn and tests."""
    settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "Hybrid-Search Enterprise Document Q&A demonstrating pgvector + FTS "
            "Reciprocal Rank Fusion, cross-encoder reranking, and Structured Outputs."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(api_router, prefix=settings.api_v1_prefix)
    return application


app = create_app()
