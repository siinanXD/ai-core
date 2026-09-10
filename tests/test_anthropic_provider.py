"""AnthropicProvider contract compliance.

Every external call is faked; no test may reach the network.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from pydantic import BaseModel

from ai_core.cost import ModelPricing
from ai_core.provider import (
    AnthropicProvider,
    NonRetryableProviderError,
    ProviderResponseError,
)
from ai_core.retry import RetryExhaustedError, RetryPolicy


class _Item(BaseModel):
    title: str


@dataclass
class _Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_input_tokens: int | None = None


@dataclass
class _TextBlock:
    text: str
    type: str = "text"


@dataclass
class _Response:
    content: list[_TextBlock] = field(default_factory=list)
    usage: _Usage | None = None
    id: str = "msg-test"
    stop_reason: str = "end_turn"


class _FakeMessages:
    def __init__(
        self,
        text: str | None = "answer",
        usage: _Usage | None = None,
        response_id: str = "msg-test",
        errors: list[Exception] | None = None,
        *,
        stop_reason: str = "end_turn",
        content: list[_TextBlock] | None = None,
    ) -> None:
        self.calls: list[dict] = []
        self._text = text
        self._usage = usage
        self._response_id = response_id
        self._errors = list(errors or [])
        self._stop_reason = stop_reason
        self._content = content

    async def create(self, model, max_tokens, system, messages, **kwargs):
        self.calls.append(
            {
                "model": model,
                "max_tokens": max_tokens,
                "system": system,
                "messages": messages,
            }
        )
        if self._errors:
            raise self._errors.pop(0)
        content = (
            self._content
            if self._content is not None
            else ([_TextBlock(text=self._text)] if self._text is not None else [])
        )
        return _Response(
            content=content,
            usage=self._usage,
            id=self._response_id,
            stop_reason=self._stop_reason,
        )


class _FakeAnthropicClient:
    def __init__(self, messages: _FakeMessages) -> None:
        self.messages = messages


class _RateLimitError(Exception):
    pass


_RateLimitError.__name__ = "RateLimitError"


class _AuthError(Exception):
    pass


_AuthError.__name__ = "AuthenticationError"


def _provider(messages: _FakeMessages, **kwargs) -> AnthropicProvider:
    retry = kwargs.pop("retry", RetryPolicy(max_attempts=1, timeout_seconds=2))
    return AnthropicProvider(_FakeAnthropicClient(messages), retry=retry, **kwargs)


@pytest.mark.asyncio
async def test_complete_returns_text_and_normalized_metadata() -> None:
    messages = _FakeMessages(
        text="grounded answer",
        usage=_Usage(input_tokens=11, output_tokens=4),
    )
    provider = _provider(messages, model="claude-haiku-4-5-20251001")

    result = await provider.complete("sys", "user")

    assert result.text == "grounded answer"
    assert result.provider == "anthropic"
    assert result.model == "claude-haiku-4-5-20251001"
    assert result.request_id == "msg-test"
    assert result.usage.input_tokens == 11
    assert result.usage.output_tokens == 4
    assert result.usage.total_tokens == 15
    assert result.latency_ms >= 0
    assert result.cost.status == "unknown"
    assert messages.calls[0]["system"] == "sys"
    assert messages.calls[0]["messages"] == [{"role": "user", "content": "user"}]


@pytest.mark.asyncio
async def test_complete_attaches_known_cost_when_pricing_is_supplied() -> None:
    messages = _FakeMessages(
        text="ok",
        usage=_Usage(input_tokens=1_000_000, output_tokens=0),
    )
    pricing = ModelPricing(
        "claude-haiku-4-5-20251001", input_usd_per_mtok=1.0, output_usd_per_mtok=5.0
    )
    provider = _provider(messages, pricing=pricing)

    result = await provider.complete("sys", "user")

    assert result.cost.status == "known"
    assert result.cost.estimated_cost_usd == 1.0


@pytest.mark.asyncio
async def test_missing_usage_reports_unknown_cost() -> None:
    provider = _provider(_FakeMessages(text="ok", usage=None))

    result = await provider.complete("sys", "user")

    assert result.usage.input_tokens is None
    assert result.usage.output_tokens is None
    assert result.cost.status == "unknown"


@pytest.mark.asyncio
async def test_empty_content_raises_provider_response_error() -> None:
    provider = _provider(_FakeMessages(content=[]))

    with pytest.raises(ProviderResponseError, match="no content blocks"):
        await provider.complete("sys", "user")


@pytest.mark.asyncio
async def test_non_success_stop_reason_raises_provider_response_error() -> None:
    provider = _provider(_FakeMessages(text="truncated", stop_reason="max_tokens"))

    with pytest.raises(ProviderResponseError, match="max_tokens"):
        await provider.complete("sys", "user")


@pytest.mark.asyncio
async def test_structured_output_recovers_from_text() -> None:
    messages = _FakeMessages(text='```json\n{"title": "Recovered"}\n```')
    provider = _provider(messages)

    result = await provider.complete_structured("sys", "user", _Item)

    assert result.parsed == _Item(title="Recovered")


@pytest.mark.asyncio
async def test_invalid_structured_output_raises() -> None:
    messages = _FakeMessages(text='{"nope": true}')
    provider = _provider(messages, model="claude-opus-5")

    with pytest.raises(ProviderResponseError, match="no parsable _Item"):
        await provider.complete_structured("sys", "user", _Item)


@pytest.mark.asyncio
async def test_retryable_provider_error_is_mapped_and_retried() -> None:
    messages = _FakeMessages(text="ok", errors=[_RateLimitError("slow down")])
    provider = _provider(
        messages,
        retry=RetryPolicy(max_attempts=2, initial_backoff_seconds=0, timeout_seconds=2),
    )

    result = await provider.complete("sys", "user")

    assert result.text == "ok"
    assert len(messages.calls) == 2


@pytest.mark.asyncio
async def test_non_retryable_provider_error_is_mapped() -> None:
    messages = _FakeMessages(errors=[_AuthError("bad key")])
    provider = _provider(
        messages,
        retry=RetryPolicy(max_attempts=3, initial_backoff_seconds=0, timeout_seconds=2),
    )

    with pytest.raises(NonRetryableProviderError, match="AuthenticationError"):
        await provider.complete("sys", "user")
    assert len(messages.calls) == 1


@pytest.mark.asyncio
async def test_retryable_exhaustion_through_the_provider() -> None:
    messages = _FakeMessages(errors=[_RateLimitError("a"), _RateLimitError("b")])
    provider = _provider(
        messages,
        retry=RetryPolicy(max_attempts=2, initial_backoff_seconds=0, timeout_seconds=2),
    )

    with pytest.raises(RetryExhaustedError):
        await provider.complete("sys", "user")


def test_provider_identity() -> None:
    provider = _provider(_FakeMessages(), model="claude-opus-5")
    assert provider.provider == "anthropic"
    assert provider.model == "claude-opus-5"


@pytest.mark.asyncio
async def test_default_max_tokens_is_sent() -> None:
    messages = _FakeMessages(text="ok")
    provider = _provider(messages)

    await provider.complete("sys", "user")

    assert messages.calls[0]["max_tokens"] == 4096


@pytest.mark.asyncio
async def test_custom_max_tokens_is_sent() -> None:
    messages = _FakeMessages(text="ok")
    provider = _provider(messages, max_tokens=256)

    await provider.complete("sys", "user")

    assert messages.calls[0]["max_tokens"] == 256
