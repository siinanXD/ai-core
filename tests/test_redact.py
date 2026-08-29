from __future__ import annotations

from ai_core.redact import REDACTED, redact, redact_text


def test_secret_keys_and_prompt_payloads_are_redacted() -> None:
    redacted = redact(
        {
            "api_key": "should-not-leak",
            "prompt": "customer document text",
            "input_tokens": 12,
            "model": "gpt-4o-mini",
        }
    )

    assert redacted["api_key"] == REDACTED
    assert redacted["prompt"] == REDACTED
    assert redacted["input_tokens"] == 12
    assert redacted["model"] == "gpt-4o-mini"


def test_compound_sensitive_keys_are_redacted() -> None:
    redacted = redact(
        {
            "customer_prompt": "secret instructions",
            "raw_messages": [{"role": "user", "content": "hello"}],
            "latency_ms": 12,
        }
    )

    assert redacted["customer_prompt"] == REDACTED
    assert redacted["raw_messages"] == REDACTED
    assert redacted["latency_ms"] == 12


def test_secret_shapes_in_free_text_are_redacted() -> None:
    token = "sk-" + ("a" * 24)
    password_url = "postgres://user:hunter2@localhost/db"

    assert REDACTED in redact_text(f"key={token}")
    assert token not in redact_text(f"key={token}")
    assert "hunter2" not in redact_text(password_url)


def test_long_strings_are_truncated() -> None:
    out = redact_text("n" * 1000)
    assert out.endswith("…")
    assert len(out) < 1000
