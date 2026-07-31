"""Async SQLAlchemy engine and session management."""

from collections.abc import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Create (or return) the shared async engine."""
    global _engine, _session_factory
    if _engine is None:
        settings = get_settings()
        connect_args: dict[str, object] = {}
        if settings.is_sqlite:
            connect_args["check_same_thread"] = False
        _engine = create_async_engine(
            settings.database_url,
            echo=settings.debug,
            pool_pre_ping=True,
            connect_args=connect_args,
        )
        _session_factory = async_sessionmaker(
            bind=_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the shared async session factory."""
    if _session_factory is None:
        get_engine()
    assert _session_factory is not None
    return _session_factory


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an AsyncSession."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Initialize schema, extensions, and hybrid-search indexes."""
    from app.db.base import Base
    import app.models.documents  # noqa: F401

    engine = get_engine()
    settings = get_settings()

    async with engine.begin() as conn:
        if settings.is_postgres:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)

        if settings.is_postgres:
            dim = settings.embedding_dimensions
            await conn.execute(
                text(
                    f"""
                    ALTER TABLE document_chunks
                    ADD COLUMN IF NOT EXISTS embedding_vector vector({dim})
                    """
                )
            )
            await conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS ix_document_chunks_embedding_vector
                    ON document_chunks
                    USING hnsw (embedding_vector vector_cosine_ops)
                    """
                )
            )
            await conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS ix_document_chunks_search_vector
                    ON document_chunks
                    USING gin (search_vector)
                    """
                )
            )


async def dispose_db() -> None:
    """Dispose the engine on application shutdown."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
