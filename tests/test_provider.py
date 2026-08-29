"""Provider contract and the OpenAI implementation.

Every external call is faked; no test may reach the network.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from ai_core.cost import ModelPricing
from ai_core.provider import (
    NonRetryableProviderError,
    OpenAIProvider,
    ProviderResponseError,
)
from ai_core.retry import RetryExhaustedError, RetryPolicy


class _Item(BaseModel):
    title: str


@dataclass
class _Usage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class _FakeCompletions:
    def __init__(
        self,
        content: str | None = "answer",
        parsed: BaseModel | None = None,
        usage: _Usage | None = None,
        response_id: str = "chatcmpl-test",
        errors: list[Exception] | None = None,
    ) -> None:
        self.calls: list[dict] = []
        self._content = content
        self._parsed = parsed
        self._usage = usage
        self._response_id = response_id
        self._errors = list(errors or [])

    def _raise_or_return(self, payload: dict) -> object:
        if self._errors:
            raise self._errors.pop(0)
        message = type("Message", (), payload)()
        choice = type("Choice", (), {"message": message})()
        return type(
            "Response",
            (),
            {"choices": [choice], "usage": self._usage, "id": self._response_id},
        )()

    async def create(self, model, messages, **kwargs):
        self.calls.append({"method": "create", "model": model, "messages": messages})
        return self._raise_or_return({"content": self._content})

    async def parse(self, model, messages, response_format, **kwargs):
        self.calls.append(
            {
                "method": "parse",
                "model": model,
                "messages": messages,
                "schema": response_format,
            }
        )
        return self._raise_or_return({"content": self._content, "parsed": self._parsed})


class _FakeChatClient:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.chat = type("Chat", (), {"completions": completions})()


class _RateLimitError(Exception):
    pass


_RateLimitError.__name__ = "RateLimitError"


class _AuthError(Exception):
    pass


_AuthError.__name__ = "AuthenticationError"


def _provider(
    completions: _FakeCompletions,
    **kwargs,
) -> OpenAIProvider:
    retry = kwargs.pop("retry", RetryPolicy(max_attempts=1, timeout_seconds=2))
    return OpenAIProvider(_FakeChatClient(completions), retry=retry, **kwargs)


@pytest.mark.asyncio
async def test_complete_returns_text_and_normalized_metadata() -> None:
    completions = _FakeCompletions(
        content="grounded answer",
        usage=_Usage(prompt_tokens=11, completion_tokens=4, total_tokens=15),
    )
    provider = _provider(completions, model="gpt-4o-mini")

    result = await provider.complete("sys", "user")

    assert result.text == "grounded answer"
    assert result.provider == "openai"
    assert result.model == "gpt-4o-mini"
    assert result.request_id == "chatcmpl-test"
    assert result.usage.input_tokens == 11
    assert result.usage.output_tokens == 4
    assert result.usage.total_tokens == 15
    assert result.latency_ms >= 0
    assert result.cost is not None
    assert result.cost.status == "unknown"
    assert completions.calls[0]["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "user"},
    ]


@pytest.mark.asyncio
async def test_complete_attaches_known_cost_when_pricing_is_supplied() -> None:
    completions = _FakeCompletions(
        content="ok",
        usage=_Usage(prompt_tokens=1_000_000, completion_tokens=0, total_tokens=1_000_000),
    )
    pricing = ModelPricing("gpt-4o-mini", input_usd_per_mtok=0.15, output_usd_per_mtok=0.60)
    provider = _provider(completions, pricing=pricing)

    result = await provider.complete("sys", "user")

    assert result.cost is not None
    assert result.cost.status == "known"
    assert result.cost.estimated_cost_usd == 0.15


@pytest.mark.asyncio
async def test_missing_usage_stays_none() -> None:
    provider = _provider(_FakeCompletions(content="ok", usage=None))

    result = await provider.complete("sys", "user")

    assert result.usage.input_tokens is None
    assert result.usage.output_tokens is None
    assert result.cost is None


@pytest.mark.asyncio
async def test_structured_output_uses_sdk_parse() -> None:
    completions = _FakeCompletions(parsed=_Item(title="Contract"))
    provider = _provider(completions)

    result = await provider.complete_structured("sys", "user", _Item)

    assert result.parsed == _Item(title="Contract")
    assert completions.calls[0]["method"] == "parse"
    assert completions.calls[0]["schema"] is _Item


@pytest.mark.asyncio
async def test_structured_output_recovers_from_text_when_parse_is_empty() -> None:
    completions = _FakeCompletions(
        content='```json\n{"title": "Recovered"}\n```',
        parsed=None,
    )
    provider = _provider(completions)

    result = await provider.complete_structured("sys", "user", _Item)

    assert result.parsed == _Item(title="Recovered")


@pytest.mark.asyncio
async def test_invalid_structured_output_raises() -> None:
    completions = _FakeCompletions(content='{"nope": true}', parsed=None)
    provider = _provider(completions, model="gpt-4o")

    with pytest.raises(ProviderResponseError, match="no parsable _Item"):
        await provider.complete_structured("sys", "user", _Item)


@pytest.mark.asyncio
async def test_retryable_provider_error_is_mapped_and_retried() -> None:
    completions = _FakeCompletions(
        content="ok",
        errors=[_RateLimitError("slow down")],
    )
    provider = _provider(
        completions,
        retry=RetryPolicy(max_attempts=2, initial_backoff_seconds=0, timeout_seconds=2),
    )

    result = await provider.complete("sys", "user")

    assert result.text == "ok"
    assert len(completions.calls) == 2


@pytest.mark.asyncio
async def test_non_retryable_provider_error_is_mapped() -> None:
    completions = _FakeCompletions(errors=[_AuthError("bad key")])
    provider = _provider(
        completions,
        retry=RetryPolicy(max_attempts=3, initial_backoff_seconds=0, timeout_seconds=2),
    )

    with pytest.raises(NonRetryableProviderError, match="AuthenticationError"):
        await provider.complete("sys", "user")
    assert len(completions.calls) == 1


@pytest.mark.asyncio
async def test_retryable_exhaustion_through_the_provider() -> None:
    completions = _FakeCompletions(
        errors=[_RateLimitError("a"), _RateLimitError("b")],
    )
    provider = _provider(
        completions,
        retry=RetryPolicy(max_attempts=2, initial_backoff_seconds=0, timeout_seconds=2),
    )

    with pytest.raises(RetryExhaustedError):
        await provider.complete("sys", "user")


def test_provider_identity() -> None:
    provider = _provider(_FakeCompletions(), model="gpt-4o")
    assert provider.provider == "openai"
    assert provider.model == "gpt-4o"
