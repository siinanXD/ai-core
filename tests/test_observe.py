from __future__ import annotations

from typing import Any

from ai_core.observe import call_shape, get_langfuse, observe_generation


class _FakeGeneration:
    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []
        self.ended = False

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)

    def end(self) -> None:
        self.ended = True


class _FakeClient:
    def __init__(self) -> None:
        self.starts: list[dict[str, Any]] = []
        self.generations: list[_FakeGeneration] = []
        self.flushed = 0

    def start_generation(self, **kwargs: Any) -> _FakeGeneration:
        self.starts.append(kwargs)
        generation = _FakeGeneration()
        self.generations.append(generation)
        return generation

    def flush(self) -> None:
        self.flushed += 1


class _ExplodingClient:
    def start_generation(self, **kwargs: Any) -> None:
        raise RuntimeError("langfuse is down")


def test_observability_is_a_noop_when_unconfigured(monkeypatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    assert get_langfuse() is None
    with observe_generation(model="gpt-4o-mini", provider="openai") as record:
        record.input_tokens = 10
        record.outcome = "ok"

    assert record.model == "gpt-4o-mini"
    assert record.input_tokens == 10
    assert record.latency_ms >= 0


def test_observability_survives_a_backend_failure() -> None:
    with observe_generation(
        model="gpt-4o-mini",
        provider="openai",
        client_factory=_ExplodingClient,
    ) as record:
        record.outcome = "ok"

    assert record.outcome == "ok"


def test_observed_generation_carries_metadata_without_prompt_text() -> None:
    client = _FakeClient()
    with observe_generation(
        model="gpt-4o-mini",
        provider="openai",
        request_id="req-1",
        shape=call_shape("system secret", "customer prompt body", 128),
        client_factory=lambda: client,
    ) as record:
        record.input_tokens = 11
        record.output_tokens = 4
        record.estimated_cost_usd = 0.0123
        record.cost_status = "known"
        record.output_chars = len("private answer")

    payload = repr(client.starts) + repr(client.generations[0].updates)
    assert "customer prompt body" not in payload
    assert "system secret" not in payload
    assert "private answer" not in payload
    assert client.generations[0].ended is True
    assert client.flushed == 1
    update = client.generations[0].updates[0]
    assert update["usage_details"] == {"input": 11, "output": 4, "total": 15}
    assert update["cost_details"] == {"total": 0.0123}
    assert update["metadata"]["request_id"] == "req-1"


def test_unknown_cost_is_omitted_from_the_trace() -> None:
    client = _FakeClient()
    with observe_generation(
        model="mystery",
        provider="openai",
        client_factory=lambda: client,
    ) as record:
        record.estimated_cost_usd = None
        record.cost_status = "unknown"

    assert "cost_details" not in client.generations[0].updates[0]
