"""Optional live OpenAI call. Not run in normal CI."""

from __future__ import annotations

import os

import pytest
from pydantic import BaseModel

from ai_core.provider import OpenAIProvider, build_openai_client
from ai_core.retry import RetryPolicy

pytestmark = pytest.mark.integration


class _Ping(BaseModel):
    ok: bool


@pytest.mark.skipif(
    os.getenv("RUN_OPENAI_INTEGRATION") != "1" or not os.getenv("OPENAI_API_KEY"),
    reason="set RUN_OPENAI_INTEGRATION=1 and OPENAI_API_KEY to run",
)
@pytest.mark.asyncio
async def test_real_openai_structured_call() -> None:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = build_openai_client(os.environ["OPENAI_API_KEY"], timeout_seconds=30)
    provider = OpenAIProvider(
        client,
        model=model,
        retry=RetryPolicy(max_attempts=1, timeout_seconds=30),
    )

    result = await provider.complete_structured(
        "Reply with JSON only.",
        "Return ok=true.",
        _Ping,
    )

    assert result.parsed is not None
    assert result.parsed.ok is True
    assert result.provider == "openai"
    assert result.model == model
