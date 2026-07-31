"""Embedding generation via Ollama or OpenAI."""

from __future__ import annotations

import logging
import math
from typing import Literal, Self

import httpx

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)
ProviderName = Literal["ollama", "openai"]


class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be produced."""


class EmbeddingService:
    """Create L2-normalized dense embeddings for indexing and queries."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: httpx.AsyncClient | None = None,
        provider: ProviderName | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._client = client
        self._owns_client = client is None
        self._preferred_provider = provider

    async def __aenter__(self) -> Self:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=10.0))
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("EmbeddingService HTTP client is not initialized")
        return self._client

    async def embed(self, text: str) -> tuple[list[float], ProviderName]:
        """Embed a single text string and return (vector, provider)."""
        provider = await self._resolve_provider()
        if provider == "ollama":
            vector = await self._embed_ollama(text)
        else:
            vector = await self._embed_openai(text)
        return self._normalize(vector), provider

    async def _resolve_provider(self) -> ProviderName:
        if self._preferred_provider is not None:
            return self._preferred_provider
        mode = self.settings.llm_provider
        if mode == "openai":
            if not self.settings.openai_api_key:
                raise EmbeddingError("OPENAI_API_KEY is required for embeddings")
            return "openai"
        if mode == "ollama":
            return "ollama"
        try:
            response = await self.client.get(f"{self.settings.ollama_base_url}/api/tags")
            if response.status_code == 200:
                return "ollama"
        except httpx.HTTPError:
            pass
        if self.settings.openai_api_key:
            return "openai"
        raise EmbeddingError("No embedding provider available")

    async def _embed_ollama(self, text: str) -> list[float]:
        response = await self.client.post(
            f"{self.settings.ollama_base_url}/api/embeddings",
            json={"model": self.settings.ollama_embed_model, "prompt": text},
        )
        response.raise_for_status()
        embedding = response.json().get("embedding")
        if not embedding:
            raise EmbeddingError("Ollama returned an empty embedding")
        return [float(x) for x in embedding]

    async def _embed_openai(self, text: str) -> list[float]:
        headers = {
            "Authorization": f"Bearer {self.settings.openai_api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, object] = {
            "model": self.settings.openai_embed_model,
            "input": text,
        }
        if self.settings.openai_embed_model.startswith("text-embedding-3"):
            payload["dimensions"] = self.settings.embedding_dimensions
        response = await self.client.post(
            f"{self.settings.openai_base_url}/embeddings",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        return [float(x) for x in response.json()["data"][0]["embedding"]]

    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            return vector
        return [v / norm for v in vector]

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        return float(sum(x * y for x, y in zip(a, b, strict=True)))
