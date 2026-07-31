"""LLM client for grounded Q&A with Structured Outputs."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal

import httpx
from pydantic import ValidationError

from app.core.config import Settings, get_settings
from app.schemas.documents import GroundedAnswer, SourceCitation

logger = logging.getLogger(__name__)
ProviderName = Literal["ollama", "openai"]

ANSWER_SYSTEM_PROMPT = """You are an enterprise document Q&A assistant.
Answer ONLY using the provided context chunks.
If the context is insufficient, say you cannot find enough evidence and set confidence_score low.

Return ONLY valid JSON matching:
{
  "answer": "string",
  "confidence_score": 0.0,
  "sources": [
    {
      "document_id": "string",
      "chunk_id": "string",
      "source": "string",
      "excerpt": "short quote from context",
      "score": 0.0,
      "metadata": {}
    }
  ]
}

Rules:
- Never invent facts outside the context.
- confidence_score must be between 0 and 1.
- Prefer citing the chunks you actually used.
- Do not include markdown fences.
"""


class LLMServiceError(RuntimeError):
    """Raised when the LLM cannot produce a valid grounded answer."""


class LLMService:
    """Async LLM orchestration with Ollama → OpenAI fallback and schema retries."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._client = client
        self._owns_client = client is None

    async def __aenter__(self) -> LLMService:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("LLMService HTTP client is not initialized")
        return self._client

    async def answer(
        self,
        *,
        question: str,
        context_blocks: list[dict[str, Any]],
    ) -> tuple[GroundedAnswer, str, ProviderName, int]:
        """
        Produce a schema-valid grounded answer.

        Returns:
            (answer, model, provider, retries)
        """
        provider = await self._resolve_provider()
        model = self._model_for(provider)
        context_json = json.dumps(context_blocks, ensure_ascii=False, default=str)
        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Context chunks (JSON):\n{context_json}\n\n"
            "Return the grounded JSON answer now."
        )
        feedback = ""
        last_error: Exception | None = None

        for attempt in range(self.settings.llm_max_retries + 1):
            try:
                prompt = user_prompt if not feedback else f"{user_prompt}\n\nCorrection:\n{feedback}"
                raw = await self._chat(provider=provider, model=model, prompt=prompt)
                parsed = self._parse_json_payload(raw)
                # Allow LLM to omit sources; we can attach them later.
                if "sources" not in parsed or parsed["sources"] is None:
                    parsed["sources"] = []
                answer = GroundedAnswer.model_validate(parsed)
                return answer, model, provider, attempt
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                feedback = f"Previous response failed validation: {exc}. Return corrected JSON only."
                logger.warning("LLM validation failed attempt=%s: %s", attempt, exc)
            except httpx.HTTPError as exc:
                last_error = exc
                if (
                    provider == "ollama"
                    and self.settings.llm_provider == "auto"
                    and self.settings.openai_api_key
                ):
                    provider = "openai"
                    model = self._model_for(provider)
                    feedback = ""
                    continue
                break

        raise LLMServiceError(f"Failed to obtain schema-valid answer: {last_error}")

    async def _resolve_provider(self) -> ProviderName:
        mode = self.settings.llm_provider
        if mode == "openai":
            if not self.settings.openai_api_key:
                raise LLMServiceError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
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
        raise LLMServiceError("No LLM provider available")

    def _model_for(self, provider: ProviderName) -> str:
        return self.settings.ollama_model if provider == "ollama" else self.settings.openai_model

    async def _chat(self, *, provider: ProviderName, model: str, prompt: str) -> str:
        if provider == "ollama":
            response = await self.client.post(
                f"{self.settings.ollama_base_url}/api/chat",
                json={
                    "model": model,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": self.settings.llm_temperature},
                    "messages": [
                        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            response.raise_for_status()
            content = response.json().get("message", {}).get("content")
            if not content:
                raise ValueError("Ollama returned empty content")
            return str(content)

        response = await self.client.post(
            f"{self.settings.openai_base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "temperature": self.settings.llm_temperature,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            },
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        if not content:
            raise ValueError("OpenAI returned empty content")
        return str(content)

    @staticmethod
    def _parse_json_payload(raw: str) -> dict[str, Any]:
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("LLM JSON root must be an object")
        return data

    @staticmethod
    def low_confidence_fallback(
        *,
        reason: str,
        citations: list[SourceCitation],
    ) -> GroundedAnswer:
        """Deterministic safe answer when retrieval confidence is too low."""
        return GroundedAnswer(
            answer=(
                "I do not have enough relevant context to answer confidently. "
                f"{reason}"
            ),
            confidence_score=0.15,
            sources=citations[:3],
        )
