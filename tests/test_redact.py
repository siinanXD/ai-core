from __future__ import annotations

from pydantic import SecretBytes, SecretStr

from ai_core.redact import REDACTED, redact, redact_text, scrub_secret_patterns


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


def test_secret_wrapper_types_are_redacted_regardless_of_key() -> None:
    redacted = redact(
        {
            "db_password": SecretStr("hunter2"),
            "signing_key": SecretBytes(b"binary-secret"),
            "note": "not secret",
        }
    )

    assert redacted["db_password"] == REDACTED
    assert redacted["signing_key"] == REDACTED
    assert redacted["note"] == "not secret"


def test_cookie_keys_are_redacted() -> None:
    redacted = redact({"cookie": "session=abc123", "set-cookie": "session=abc123"})

    assert redacted["cookie"] == REDACTED
    assert redacted["set-cookie"] == REDACTED


def test_scrub_secret_patterns_does_not_truncate() -> None:
    token = "sk-" + ("a" * 24)
    long_text = f"prefix {token} " + ("n" * 1000)

    out = scrub_secret_patterns(long_text)

    assert token not in out
    assert REDACTED in out
    assert len(out) > 400
    assert not out.endswith("…")


def test_redact_text_composes_scrub_and_truncate() -> None:
    token = "sk-" + ("a" * 24)
    long_text = f"prefix {token} " + ("n" * 1000)

    assert redact_text(long_text) == scrub_secret_patterns(long_text)[:400] + "…"
