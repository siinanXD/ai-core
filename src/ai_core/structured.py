"""Get JSON out of a model answer and validate it as a Pydantic schema.

The recovery path is the proven one: strip a markdown fence, then take the
first balanced object. It is not a general parser.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ValidationError

FENCES = ("```json", "```JSON", "```")


class StructuredOutputError(ValueError):
    """The answer could not be read as the requested shape."""

    def __init__(self, message: str, *, raw: str = "", found: str | None = None) -> None:
        super().__init__(message)
        self.raw_length = len(raw or "")
        self.found = found
        self.preview = _preview(raw)


def _preview(raw: str, limit: int = 160) -> str:
    text = " ".join((raw or "").split())
    return text[:limit]


def strip_fences(raw: str) -> str:
    """Remove a surrounding markdown code fence, if there is one."""
    text = (raw or "").strip()
    for fence in FENCES:
        if text.startswith(fence):
            text = text[len(fence) :]
            break
    if text.endswith("```"):
        text = text[: -len("```")]
    return text.strip()


def _openers_outside_strings(raw: str, opener: str) -> list[int]:
    """Positions of `opener` that are not inside a JSON string literal."""
    positions: list[int] = []
    in_string = False
    escaped = False
    for index, char in enumerate(raw):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            positions.append(index)
    return positions


def find_value(raw: str, opener: str) -> str | None:
    """The first complete `{...}` or `[...]` in the text, brackets balanced."""
    closer = {"{": "}", "[": "]"}[opener]
    for start in _openers_outside_strings(raw, opener):
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(raw)):
            char = raw[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    return raw[start : index + 1]
    return None


def _describe(raw: str) -> str:
    text = strip_fences(raw)
    if not text:
        return "an empty response"
    if find_value(text, "[") is not None:
        return "a JSON array"
    obj = find_value(text, "{")
    if obj is not None:
        try:
            keys = list(json.loads(obj).keys())
            return f"a JSON object with keys {keys}"
        except Exception:
            return "something object-shaped that does not parse"
    for opener, shape in (("[", "an array"), ("{", "an object")):
        if _openers_outside_strings(text, opener):
            return f"{shape} that was cut off before it closed"
    return "prose with no JSON in it"


def parse_object(raw: str) -> dict[str, Any]:
    """Read the answer as a JSON object, or explain what it was instead."""
    text = strip_fences(raw)
    if not text:
        raise StructuredOutputError("the model returned an empty response", raw=raw, found="empty")
    candidate = find_value(text, "{")
    if candidate is None:
        raise StructuredOutputError(
            f"expected a JSON object, found {_describe(raw)}", raw=raw, found=_describe(raw)
        )
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise StructuredOutputError(
            f"the object found could not be parsed: {exc.msg} at position {exc.pos}",
            raw=raw,
            found="unparseable object",
        ) from exc
    if not isinstance(parsed, dict):
        raise StructuredOutputError(
            f"expected a JSON object, found a {type(parsed).__name__}", raw=raw
        )
    return parsed


def parse_model[T: BaseModel](raw: str, schema: type[T]) -> T:
    """Recover JSON and validate it as `schema`."""
    try:
        payload = parse_object(raw)
    except StructuredOutputError:
        raise
    try:
        return schema.model_validate(payload)
    except ValidationError as exc:
        raise StructuredOutputError(
            f"response did not match {schema.__name__}: {exc.error_count()} validation error(s)",
            raw=raw,
            found="invalid schema",
        ) from exc
