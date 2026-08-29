"""Redact secrets and customer payloads before they leave the process.

Logging is deny-by-default for prompt-like keys. Token *counts* are telemetry
and are kept.
"""

from __future__ import annotations

import re
from typing import Any

REDACTED = "[redacted]"

_SECRET_PARTS = (
    "secret",
    "token",
    "password",
    "credential",
    "private_key",
    "api_key",
    "apikey",
    "authorization",
    "service_account",
)

# Keys that are customer content or prompts. Do not log them raw.
_CONTENT_KEYS = frozenset(
    {
        "prompt",
        "system_prompt",
        "messages",
        "document",
        "content",
        "payload",
        "input",
        "output",
    }
)

_SAFE_KEYS = frozenset(
    {
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "max_tokens",
        "tokens_reported",
        "token_count",
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "typical_input_tokens",
        "typical_output_tokens",
    }
)

_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"sk-(?:proj-)?[A-Za-z0-9_\-]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{12,}"),
    re.compile(r"(?i)(postgres(?:ql)?(?:\+\w+)?://)[^:\s]+:[^@\s]+@"),
    re.compile(r"(?i)(redis(?:s)?://)[^:\s]+:[^@\s]+@"),
)

_MAX_STRING = 400
_MAX_DEPTH = 6
_MAX_LIST = 50


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    if lowered in _SAFE_KEYS:
        return False
    if lowered in _CONTENT_KEYS:
        return True
    return any(part in lowered for part in _SECRET_PARTS)


def redact_text(value: str) -> str:
    """Redact known secret shapes inside a free-text string."""
    out = value
    for pattern in _PATTERNS:
        if pattern.groups:
            out = pattern.sub(rf"\1{REDACTED}:{REDACTED}@", out)
        else:
            out = pattern.sub(REDACTED, out)
    if len(out) > _MAX_STRING:
        return out[:_MAX_STRING] + "…"
    return out


def redact(value: Any, _depth: int = 0) -> Any:
    """Recursively redact a structure before it reaches a log or trace sink."""
    if _depth > _MAX_DEPTH:
        return REDACTED
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {
            str(key): REDACTED if _is_secret_key(str(key)) else redact(item, _depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item, _depth + 1) for item in value][:_MAX_LIST]
    return value
