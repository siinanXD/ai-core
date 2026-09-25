# AGENTS.md – ai-core

Bibliothek mit wiederverwendbaren KI-Bausteinen: OpenAI- und Anthropic-Adapter
hinter einem `LLMProvider`-Protokoll, strukturierte Ausgabe, begrenzte Retries,
Wrapping von nicht vertrauenswürdigem Inhalt, Redaction, optionales
Langfuse-Tracing, Kostenschätzung. Details: `README.md`.

Workspace-Regeln gelten zusätzlich: `C:\Dev\CLAUDE.md` → `AI-Workspace\shared-rules\`.

## Harte Fakten

- Eine Bibliothek: keine Web-App, keine Datenbank, keine Queue, kein RAG.
  Was bewusst fehlt, steht in README „Not in this package“.
- Python `>=3.12`.
- Vendor-SDKs (`openai`, `anthropic`, `langfuse`) sind optional und werden nie auf
  Modulebene importiert. Diese Regel beibehalten.
- Wird von den Templates (`ai-starter`, `ai-template-*`) genutzt. Änderungen an der
  öffentlichen API wirken dort mit.

## Build & Test

```bash
pip install -e ".[dev]"
pytest -m "not integration"       # Unit-Tests mit Fakes, keine bezahlten API-Aufrufe
ruff check .
ruff format --check .
```

Live-Test (kostet Geld, nur bewusst): `RUN_OPENAI_INTEGRATION=1 pytest -m integration`.
