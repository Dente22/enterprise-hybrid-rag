"""Application settings loaded from environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the hybrid RAG service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Hybrid-Search Enterprise Document Q&A"
    app_env: Literal["development", "staging", "production", "test"] = "development"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"

    api_keys: str = Field(default="dev-secret-key-change-me")

    max_text_length: int = 100_000
    max_upload_bytes: int = 10 * 1024 * 1024
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60

    database_url: str = "sqlite+aiosqlite:///./data/hybrid_rag.db"

    llm_provider: Literal["auto", "ollama", "openai"] = "auto"
    llm_max_retries: int = 2
    llm_temperature: float = 0.1

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3"
    ollama_embed_model: str = "nomic-embed-text"

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    openai_embed_model: str = "text-embedding-3-small"

    embedding_dimensions: int = 768
    chunk_size: int = 800
    chunk_overlap: int = 120

    hybrid_vector_weight: float = 0.55
    hybrid_fts_weight: float = 0.45
    hybrid_candidate_limit: int = 20
    rerank_top_k: int = 3
    min_confidence_threshold: float = 0.35

    reranker_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    upload_dir: str = "./uploads"

    @field_validator("api_keys", mode="before")
    @classmethod
    def _normalize_api_keys(cls, value: object) -> str:
        if value is None:
            return "dev-secret-key-change-me"
        return str(value)

    @property
    def api_key_set(self) -> set[str]:
        return {key.strip() for key in self.api_keys.split(",") if key.strip()}

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @property
    def is_postgres(self) -> bool:
        return self.database_url.startswith("postgresql")


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()
