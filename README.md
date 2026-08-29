# ai-core

Small reusable AI runtime primitives. One OpenAI adapter, structured output, bounded retry, untrusted-content wrapping, redaction, optional Langfuse tracing, and explicit cost estimates.

This is a library. It has no web app, database, queue, or RAG.

## Install

```bash
pip install -e ".[dev]"
```

Langfuse is optional:

```bash
pip install -e ".[dev,observe]"
```

## Public API

| Module | Role |
| --- | --- |
| `ai_core.provider` | `LLMProvider` protocol and `OpenAIProvider` |
| `ai_core.structured` | Fence/balanced-JSON recovery and Pydantic validation |
| `ai_core.retry` | Per-attempt timeout and bounded retry |
| `ai_core.untrusted` | Wrap external text as data, not instructions |
| `ai_core.redact` | Redact secrets and prompt-like keys before logging |
| `ai_core.observe` | Fail-open Langfuse generation span |
| `ai_core.cost` | Cost metadata from caller-supplied prices |

## Tests

Unit tests use fakes. They do not call paid APIs.

```bash
pytest -m "not integration"
ruff check .
ruff format --check .
```

An optional live OpenAI path exists but is skipped unless `OPENAI_API_KEY` is set and the `integration` marker is selected:

```bash
pytest -m integration
```

## Not in this package

Extra providers, routing, embeddings, retrieval, tools, approvals, workers, Redis, LangGraph, FastAPI, and evaluation infrastructure.
