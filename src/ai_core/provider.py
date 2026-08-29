"""OpenAI-only LLM contract.

Call sites depend on `LLMProvider`. The OpenAI SDK is imported by the
implementation and by `build_openai_client`, not by the rest of the package.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel

from ai_core.cost import CostEstimate, ModelPricing, estimate_cost
from ai_core.observe import GenerationRecord, call_shape, observe_generation
from ai_core.retry import RetryableError, RetryPolicy, is_retryable, with_retry
from ai_core.structured import StructuredOutputError, parse_model

_SUCCESS_FINISH_REASONS = frozenset({"stop", None})


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
    cached_input_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class Generation:
    """Normalized result of one completion."""

    text: str
    provider: str
    model: str
    latency_ms: int
    usage: Usage
    cost: CostEstimate
    parsed: BaseModel | None = None
    request_id: str | None = None
    finish_reason: str | None = None
    refusal: str | None = None


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
        started = time.monotonic()

        async def _once() -> Generation:
            with observe_generation(
                model=self._model,
                provider=self.provider,
                shape=call_shape(system, user),
            ) as record:
                try:
                    response = await self._client.chat.completions.create(
                        model=self._model,
                        messages=_messages(system, user),
                    )
                    generation = self._generation_from_response(response)
                    _populate_record(record, generation)
                    return generation
                except ProviderError:
                    raise
                except Exception as exc:
                    raise _map_error(exc) from exc

        generation = await with_retry(_once, policy=self._retry)
        return dataclasses.replace(
            generation,
            latency_ms=int((time.monotonic() - started) * 1000),
        )

    async def complete_structured[T: BaseModel](
        self, system: str, user: str, schema: type[T]
    ) -> Generation:
        started = time.monotonic()

        async def _once() -> Generation:
            with observe_generation(
                model=self._model,
                provider=self.provider,
                shape=call_shape(system, user),
            ) as record:
                try:
                    parse = getattr(self._client.chat.completions, "parse", None)
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
                    choice = _choice(response, self._model)
                    message = choice.message
                    parsed = getattr(message, "parsed", None)
                    text = _message_text(message)
                    if parsed is None:
                        try:
                            parsed = parse_model(text, schema)
                        except StructuredOutputError as exc:
                            raise ProviderResponseError(
                                f"{self._model} returned no parsable {schema.__name__}: {exc}"
                            ) from exc
                    generation = self._generation_from_response(
                        response,
                        text=text,
                        parsed=parsed,
                    )
                    _populate_record(record, generation)
                    return generation
                except ProviderError:
                    raise
                except Exception as exc:
                    raise _map_error(exc) from exc

        generation = await with_retry(_once, policy=self._retry)
        return dataclasses.replace(
            generation,
            latency_ms=int((time.monotonic() - started) * 1000),
        )

    def _generation_from_response(
        self,
        response: Any,
        *,
        text: str | None = None,
        parsed: BaseModel | None = None,
    ) -> Generation:
        choice = _choice(response, self._model)
        message = choice.message
        refusal = getattr(message, "refusal", None)
        if refusal:
            raise ProviderResponseError(f"{self._model} refused the request")
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason not in _SUCCESS_FINISH_REASONS:
            raise ProviderResponseError(f"{self._model} finished with reason {finish_reason!r}")
        resolved_text = text if text is not None else _message_text(message)
        usage = _usage(response)
        cost = _cost_for_usage(self._model, usage, self._pricing)
        return Generation(
            text=resolved_text,
            provider=self.provider,
            model=self._model,
            latency_ms=0,
            usage=usage,
            cost=cost,
            parsed=parsed,
            request_id=getattr(response, "id", None),
            finish_reason=finish_reason,
            refusal=None,
        )


def _messages(system: str, user: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _choice(response: Any, model: str) -> Any:
    choices = getattr(response, "choices", None)
    if not choices:
        raise ProviderResponseError(f"{model} returned no choices")
    return choices[0]


def _message_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if content is None:
        return ""
    if not isinstance(content, str):
        raise ProviderResponseError("message content was not text")
    return content


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
    cached = _cached_input_tokens(usage)
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total,
        cached_input_tokens=cached,
    )


def _cached_input_tokens(usage: Any) -> int | None:
    details = getattr(usage, "prompt_tokens_details", None) or getattr(
        usage, "input_tokens_details", None
    )
    if details is None:
        return None
    value = getattr(details, "cached_tokens", None)
    return _int_or_none(value)


def _int_or_none(*values: Any) -> int | None:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ProviderResponseError(f"invalid token count: {value!r}") from exc
    return None


def _cost_for_usage(
    model: str,
    usage: Usage,
    pricing: ModelPricing | dict[str, ModelPricing] | None,
) -> CostEstimate:
    if usage.input_tokens is None or usage.output_tokens is None:
        return CostEstimate(model, 0, 0, None, "unknown")
    return estimate_cost(
        model,
        usage.input_tokens,
        usage.output_tokens,
        pricing,
        cached_input_tokens=usage.cached_input_tokens or 0,
    )


def _populate_record(record: GenerationRecord, generation: Generation) -> None:
    if generation.usage.input_tokens is not None:
        record.input_tokens = generation.usage.input_tokens
    if generation.usage.output_tokens is not None:
        record.output_tokens = generation.usage.output_tokens
    record.request_id = generation.request_id
    record.output_chars = len(generation.text)
    if generation.cost.known:
        record.estimated_cost_usd = generation.cost.estimated_cost_usd
        record.cost_status = "known"


def _map_error(exc: BaseException) -> ProviderError:
    if isinstance(exc, ProviderError):
        return exc
    if isinstance(exc, (IndexError, KeyError, TypeError, ValueError)):
        return ProviderResponseError(f"{type(exc).__name__}: {exc}")
    name = type(exc).__name__
    message = f"{name}: {exc}"
    if is_retryable(exc):
        return RetryableProviderError(message)
    return NonRetryableProviderError(message)
