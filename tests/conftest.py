"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Iterator
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import Settings, get_settings
from app.db.base import Base
from app.db.session import get_db_session
from app.main import create_app


@pytest.fixture
def test_settings(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Settings]:
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("API_KEYS", "test-api-key")
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path.as_posix()}")
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("RERANKER_ENABLED", "false")
    monkeypatch.setenv("MIN_CONFIDENCE_THRESHOLD", "0.2")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "8")
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


@pytest.fixture
async def session(test_settings: Settings) -> AsyncGenerator[AsyncSession, None]:
    engine = create_async_engine(test_settings.database_url, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as db:
        yield db
    await engine.dispose()


@pytest.fixture
async def client(
    test_settings: Settings,
    session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[AsyncClient, None]:
    monkeypatch.setattr("app.main.init_db", AsyncMock())
    monkeypatch.setattr("app.main.dispose_db", AsyncMock())
    app = create_app()

    async def _override_db() -> AsyncGenerator[AsyncSession, None]:
        yield session

    app.dependency_overrides[get_db_session] = _override_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()
