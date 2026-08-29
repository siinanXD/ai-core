"""Bounded timeout and retry for a single LLM call.

Job-queue retries are out of scope. This module retries one operation a
finite number of times and then stops with a clear error.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

_RETRYABLE_NAMES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
        "TimeoutError",
    }
)
_NON_RETRYABLE_NAMES = frozenset(
    {
        "AuthenticationError",
        "PermissionDeniedError",
        "BadRequestError",
        "NotFoundError",
        "UnprocessableEntityError",
    }
)
_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


class RetryableError(Exception):
    """Transient failure that is safe to retry."""


class RetryExhaustedError(Exception):
    """Every allowed attempt failed with a retryable error."""

    def __init__(self, message: str, *, attempts: int, last_error: BaseException) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Limits for one call. None of these may be unbounded."""

    max_attempts: int = 3
    initial_backoff_seconds: float = 0.2
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 8.0
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0")
        if self.initial_backoff_seconds < 0:
            raise ValueError("initial_backoff_seconds must be at least 0")
        if self.backoff_multiplier < 1:
            raise ValueError("backoff_multiplier must be at least 1")
        if self.max_backoff_seconds < 0:
            raise ValueError("max_backoff_seconds must be at least 0")


def is_retryable(error: BaseException) -> bool:
    """True only for errors that are known to be transient."""
    if isinstance(error, RetryableError):
        return True
    name = type(error).__name__
    if name in _NON_RETRYABLE_NAMES:
        return False
    if name in _RETRYABLE_NAMES or isinstance(error, (TimeoutError, asyncio.TimeoutError)):
        return True
    status = getattr(error, "status_code", None)
    return isinstance(status, int) and status in _RETRYABLE_STATUS


async def with_retry[T](operation: Callable[[], Awaitable[T]], *, policy: RetryPolicy) -> T:
    """Run `operation` with a per-attempt timeout and a bounded retry loop."""
    last_error: BaseException | None = None
    for attempt in range(1, policy.max_attempts + 1):
        try:
            async with asyncio.timeout(policy.timeout_seconds):
                return await operation()
        except Exception as exc:
            last_error = exc
            can_retry = is_retryable(exc)
            if not can_retry:
                raise
            if attempt == policy.max_attempts:
                raise RetryExhaustedError(
                    f"gave up after {attempt} attempts: {exc}",
                    attempts=attempt,
                    last_error=exc,
                ) from exc
            delay = min(
                policy.initial_backoff_seconds * (policy.backoff_multiplier ** (attempt - 1)),
                policy.max_backoff_seconds,
            )
            if delay:
                await asyncio.sleep(delay)

    raise RetryExhaustedError(
        "gave up without a recorded error",
        attempts=policy.max_attempts,
        last_error=last_error or RuntimeError("retry loop ended without an error"),
    )
