"""Input sanitization to reduce prompt-injection risk."""

from __future__ import annotations

import re

# Common jailbreak / instruction-override patterns (defense-in-depth, not perfect).
_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
        r"disregard\s+(all\s+)?(previous|prior|above)",
        r"you\s+are\s+now\s+(dan|unfiltered|jailbroken)",
        r"system\s*prompt\s*:",
        r"<\s*/?\s*system\s*>",
        r"\[\s*INST\s*\]",
        r"do\s+not\s+follow\s+your\s+(system|developer)\s+prompt",
        r"reveal\s+(your|the)\s+(system|hidden)\s+prompt",
    )
)

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


class SanitizationError(ValueError):
    """Raised when input fails security validation."""


def sanitize_user_text(text: str, *, field_name: str = "text") -> str:
    """
    Normalize and lightly harden user-supplied text.

    Raises:
        SanitizationError: If high-confidence injection patterns are detected.
    """
    cleaned = _CONTROL_CHARS.sub("", text).strip()
    if not cleaned:
        raise SanitizationError(f"{field_name} must not be empty after sanitization")

    for pattern in _INJECTION_PATTERNS:
        if pattern.search(cleaned):
            raise SanitizationError(
                f"{field_name} rejected: potential prompt-injection pattern detected"
            )
    return cleaned


def sanitize_question(question: str) -> str:
    """Sanitize a retrieval question."""
    return sanitize_user_text(question, field_name="question")
