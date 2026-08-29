from __future__ import annotations

import pytest

from ai_core.provider import NonRetryableProviderError, RetryableProviderError
from ai_core.retry import RetryExhaustedError, RetryPolicy, is_retryable, with_retry
from ai_core.structured import StructuredOutputError


class _RateLimitError(Exception):
    """Name must match the OpenAI SDK exception the retry policy recognizes."""


_RateLimitError.__name__ = "RateLimitError"


class _AuthError(Exception):
    pass


_AuthError.__name__ = "AuthenticationError"


@pytest.mark.asyncio
async def test_retryable_failure_is_retried_then_succeeds() -> None:
    calls = {"n": 0}

    async def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RetryableProviderError("transient")
        return "ok"

    assert (
        await with_retry(
            flaky,
            policy=RetryPolicy(max_attempts=3, initial_backoff_seconds=0, timeout_seconds=1),
        )
        == "ok"
    )
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_retryable_sdk_error_is_retried() -> None:
    calls = {"n": 0}

    async def flaky() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise _RateLimitError("slow down")
        return "ok"

    assert (
        await with_retry(
            flaky,
            policy=RetryPolicy(max_attempts=2, initial_backoff_seconds=0, timeout_seconds=1),
        )
        == "ok"
    )
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_non_retryable_failure_is_not_retried() -> None:
    calls = {"n": 0}

    async def boom() -> str:
        calls["n"] += 1
        raise NonRetryableProviderError("bad key")

    with pytest.raises(NonRetryableProviderError, match="bad key"):
        await with_retry(
            boom,
            policy=RetryPolicy(max_attempts=4, initial_backoff_seconds=0, timeout_seconds=1),
        )
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_retry_exhaustion_raises_a_clear_error() -> None:
    calls = {"n": 0}

    async def always() -> str:
        calls["n"] += 1
        raise RetryableProviderError("still down")

    with pytest.raises(RetryExhaustedError, match="gave up after 3 attempts") as caught:
        await with_retry(
            always,
            policy=RetryPolicy(max_attempts=3, initial_backoff_seconds=0, timeout_seconds=1),
        )
    assert caught.value.attempts == 3
    assert isinstance(caught.value.last_error, RetryableProviderError)
    assert calls["n"] == 3


def test_structured_and_auth_errors_are_not_retryable() -> None:
    assert is_retryable(StructuredOutputError("no")) is False
    assert is_retryable(_AuthError("no")) is False
    assert is_retryable(RetryableProviderError("yes")) is True


def test_policy_rejects_unbounded_values() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        RetryPolicy(max_attempts=0)
    with pytest.raises(ValueError, match="timeout_seconds"):
        RetryPolicy(timeout_seconds=0)
