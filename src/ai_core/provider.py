"""OpenAI-only LLM contract.

Call sites depend on `LLMProvider`. The OpenAI SDK is imported by the
implementation and by `build_openai_client`, not by the rest of the package.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel

from ai_core.cost import CostEstimate, ModelPricing, estimate_cost
from ai_core.retry import RetryableError, RetryPolicy, with_retry
from ai_core.structured import StructuredOutputError, parse_model


class ProviderError(Exception):
    """A provider call failed in a way the caller must handle."""


class ProviderResponseError(ProviderError):
    """The provider returned a response the caller cannot use."""


class RetryableProviderError(ProviderError, RetryableError):
    """Transient provider failure. Safe to retry the same call."""


class NonRetryableProviderError(ProviderError):
    """Permanent provider failure. Retrying will not help."""


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class Generation:
    """Normalized result of one completion."""

    text: str
    provider: str
    model: str
    latency_ms: int
    usage: Usage
    cost: CostEstimate | None = None
    parsed: BaseModel | None = None
    request_id: str | None = None


class LLMProvider(Protocol):
    """Generates text, optionally constrained to a Pydantic schema."""

    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...

    async def complete(self, system: str, user: str) -> Generation: ...

    async def complete_structured(
        self, system: str, user: str, schema: type[BaseModel]
    ) -> Generation: ...


def build_openai_client(api_key: str, *, timeout_seconds: float) -> Any:
    """Construct a real OpenAI client. Never called from unit tests."""
    from openai import AsyncOpenAI

    return AsyncOpenAI(api_key=api_key, timeout=timeout_seconds, max_retries=0)


class OpenAIProvider:
    def __init__(
        self,
        client: Any,
        model: str = "gpt-4o-mini",
        *,
        retry: RetryPolicy | None = None,
        pricing: ModelPricing | dict[str, ModelPricing] | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._retry = retry or RetryPolicy()
        self._pricing = pricing

    @property
    def provider(self) -> str:
        return "openai"

    @property
    def model(self) -> str:
        return self._model

    async def complete(self, system: str, user: str) -> Generation:
        async def _once() -> Generation:
            started = time.monotonic()
            try:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=_messages(system, user),
                )
            except Exception as exc:
                raise _map_error(exc) from exc
            return self._generation(response, started, text=_message_text(response))

        return await with_retry(_once, policy=self._retry)

    async def complete_structured[T: BaseModel](
        self, system: str, user: str, schema: type[T]
    ) -> Generation:
        async def _once() -> Generation:
            started = time.monotonic()
            parse = getattr(self._client.chat.completions, "parse", None)
            try:
                if parse is not None:
                    response = await parse(
                        model=self._model,
                        messages=_messages(system, user),
                        response_format=schema,
                    )
                else:
                    response = await self._client.chat.completions.create(
                        model=self._model,
                        messages=_messages(system, user),
                    )
            except Exception as exc:
                raise _map_error(exc) from exc

            parsed = getattr(getattr(response.choices[0], "message", None), "parsed", None)
            text = _message_text(response)
            if parsed is None:
                try:
                    parsed = parse_model(text, schema)
                except StructuredOutputError as exc:
                    raise ProviderResponseError(
                        f"{self._model} returned no parsable {schema.__name__}: {exc}"
                    ) from exc
            return self._generation(response, started, text=text, parsed=parsed)

        return await with_retry(_once, policy=self._retry)

    def _generation(
        self,
        response: Any,
        started: float,
        *,
        text: str,
        parsed: BaseModel | None = None,
    ) -> Generation:
        usage = _usage(response)
        cost = None
        if usage.input_tokens is not None and usage.output_tokens is not None:
            cost = estimate_cost(
                self._model, usage.input_tokens, usage.output_tokens, self._pricing
            )
        return Generation(
            text=text,
            provider=self.provider,
            model=self._model,
            latency_ms=int((time.monotonic() - started) * 1000),
            usage=usage,
            cost=cost,
            parsed=parsed,
            request_id=getattr(response, "id", None),
        )


def _messages(system: str, user: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _message_text(response: Any) -> str:
    message = response.choices[0].message
    return getattr(message, "content", None) or ""


def _usage(response: Any) -> Usage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return Usage()
    input_tokens = _int_or_none(
        getattr(usage, "prompt_tokens", None), getattr(usage, "input_tokens", None)
    )
    output_tokens = _int_or_none(
        getattr(usage, "completion_tokens", None), getattr(usage, "output_tokens", None)
    )
    total = _int_or_none(getattr(usage, "total_tokens", None))
    if total is None and input_tokens is not None and output_tokens is not None:
        total = input_tokens + output_tokens
    return Usage(input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total)


def _int_or_none(*values: Any) -> int | None:
    for value in values:
        if value is None:
            continue
        return int(value)
    return None


def _map_error(exc: BaseException) -> ProviderError:
    if isinstance(exc, ProviderError):
        return exc
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    if name in {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
        "TimeoutError",
    } or (isinstance(status, int) and status in {408, 409, 429, 500, 502, 503, 504}):
        return RetryableProviderError(f"{name}: {exc}")
    if name in {
        "AuthenticationError",
        "PermissionDeniedError",
        "BadRequestError",
        "NotFoundError",
        "UnprocessableEntityError",
    }:
        return NonRetryableProviderError(f"{name}: {exc}")
    if isinstance(exc, (TimeoutError,)):
        return RetryableProviderError(f"{name}: {exc}")
    return NonRetryableProviderError(f"{name}: {exc}")
