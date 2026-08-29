from __future__ import annotations

import pytest
from pydantic import BaseModel

from ai_core.structured import (
    StructuredOutputError,
    find_value,
    parse_model,
    parse_object,
    strip_fences,
)


class Item(BaseModel):
    title: str
    score: float | None = None


def test_a_json_fence_is_removed() -> None:
    assert strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_unfenced_text_is_untouched() -> None:
    assert strip_fences('  {"a": 1}  ') == '{"a": 1}'


def test_successful_structured_output() -> None:
    parsed = parse_model('{"title": "Contract", "score": 0.9}', Item)
    assert parsed == Item(title="Contract", score=0.9)


def test_invalid_structured_output_is_predictable() -> None:
    with pytest.raises(StructuredOutputError, match="did not match Item") as caught:
        parse_model('{"score": 0.9}', Item)
    assert caught.value.found == "invalid schema"


def test_malformed_json_recovers_the_first_balanced_object() -> None:
    parsed = parse_model(
        'Sure!\n\n```json\n{"title": "Contract", "score": 0.9}\n```\nHope that helps.',
        Item,
    )
    assert parsed.title == "Contract"


def test_the_first_complete_value_is_returned_not_the_widest_span() -> None:
    raw = 'Here it is: {"a": 1} and then some prose with a } in it.'
    assert find_value(raw, "{") == '{"a": 1}'


def test_empty_and_prose_failures_name_what_was_there() -> None:
    with pytest.raises(StructuredOutputError) as empty:
        parse_object("   ")
    assert empty.value.found == "empty"

    with pytest.raises(StructuredOutputError, match="prose with no JSON"):
        parse_object("I cannot help with that request.")


def test_unparseable_object_names_the_position() -> None:
    with pytest.raises(StructuredOutputError, match="could not be parsed"):
        parse_object('{"a": 1,,}')
