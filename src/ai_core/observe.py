"""Langfuse generation helper. Fail open. Never send raw customer content.

If Langfuse is missing or misconfigured, callers still get a record and the
underlying call is unaffected.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from ai_core.redact import redact, redact_text

PROMPT_PREVIEW_CHARS = 0


@dataclass
class GenerationRecord:
    """Filled in by the caller while the provider call is in flight."""

    model: str
    provider: str
    request_id: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float | None = None
    cost_status: str = "unknown"
    outcome: str = "ok"
    error: str | None = None
    latency_ms: int = 0
    output_chars: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_usage(self) -> bool:
        return self.input_tokens > 0 or self.output_tokens > 0

    def usage_details(self) -> dict[str, int] | None:
        if not self.has_usage:
            return None
        return {
            "input": self.input_tokens,
            "output": self.output_tokens,
            "total": self.input_tokens + self.output_tokens,
        }

    def cost_details(self) -> dict[str, float] | None:
        if self.cost_status != "known" or self.estimated_cost_usd is None:
            return None
        return {"total": float(self.estimated_cost_usd)}


def call_shape(system: str, prompt: str, max_tokens: int | None = None) -> dict[str, Any]:
    """What a prompt looked like, without what it said."""
    shape: dict[str, Any] = {
        "system_chars": len(system or ""),
        "prompt_chars": len(prompt or ""),
    }
    if max_tokens is not None:
        shape["max_tokens"] = max_tokens
    if PROMPT_PREVIEW_CHARS > 0:
        shape["prompt_preview"] = (prompt or "")[:PROMPT_PREVIEW_CHARS]
    return shape


@lru_cache(maxsize=1)
def get_langfuse() -> Any | None:
    """Return a Langfuse client, or None when it is not configured."""
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        return None
    try:
        from langfuse import Langfuse
    except ImportError:
        return None
    host = os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL")
    kwargs: dict[str, Any] = {"public_key": public_key, "secret_key": secret_key}
    if host:
        kwargs["host"] = host
    return Langfuse(**kwargs)


@contextmanager
def observe_generation(
    *,
    model: str,
    provider: str,
    request_id: str | None = None,
    shape: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    client_factory: Any = get_langfuse,
) -> Iterator[GenerationRecord]:
    """Wrap one provider attempt. Tracing failures never escape."""
    record = GenerationRecord(model=model, provider=provider, request_id=request_id)
    generation = None
    started = time.monotonic()

    try:
        client = client_factory()
        if client is not None and hasattr(client, "start_generation"):
            generation = client.start_generation(
                name=f"generation:{provider}/{model}",
                model=model,
                input=redact(shape or {}),
                metadata=redact(
                    {
                        "provider": provider,
                        "request_id": request_id,
                        **(metadata or {}),
                    }
                ),
            )
            if request_id is None:
                try:
                    record.request_id = getattr(client, "get_current_trace_id", lambda: None)()
                except Exception:
                    record.request_id = None
    except Exception:
        generation = None

    try:
        yield record
    except BaseException as exc:
        if record.outcome == "ok":
            record.outcome = "error"
        record.error = record.error or f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record.latency_ms = int((time.monotonic() - started) * 1000)
        if generation is not None:
            with suppress(Exception):
                _close(generation, record)


def _close(generation: Any, record: GenerationRecord) -> None:
    safe_error = redact_text(record.error) if record.error else None
    update: dict[str, Any] = {
        "output": {
            "outcome": record.outcome,
            "output_chars": record.output_chars,
            "error": (safe_error or "")[:500] or None,
        },
        "metadata": redact(
            {
                "provider": record.provider,
                "request_id": record.request_id,
                "outcome": record.outcome,
                "cost_status": record.cost_status,
                "generation_latency_ms": record.latency_ms,
                "tokens_reported": record.has_usage,
                **record.metadata,
            }
        ),
        "level": "ERROR" if record.outcome != "ok" else "DEFAULT",
    }
    if safe_error:
        update["status_message"] = safe_error[:500]
    usage = record.usage_details()
    if usage is not None:
        update["usage_details"] = usage
    cost = record.cost_details()
    if cost is not None:
        update["cost_details"] = cost
    generation.update(**update)
    generation.end()
