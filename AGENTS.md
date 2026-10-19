# AGENTS.md

## Structure

- `app/` — FastAPI app, query and document endpoints
- `ingestion/` — parse, chunk, embed, load into Qdrant (embedded or server)
- `retrieval/` — Retriever: dense, sparse, hybrid (RRF), rerank
- `rag/` — answer generation with citations, or "not found"
- `llm/` — LLMClient seam: OpenAI-compatible and Anthropic adapters
- `eval/` — test set and evaluation scripts
- `tests/` — pytest tests
- `docs/` — design and decisions
- `data/` — raw documents (not committed)

## Conventions

- Python 3.11, type hints on public functions, async for I/O in `app/`.
- Format and lint with ruff (line length 100).
- Config via environment variables (see `.env.example`); never hardcode secrets.
- One merge request per week; include eval results before/after.
- Answers must cite articles; return "not found" when there is no evidence.

## Commands

```bash
uv sync        # install deps
make up        # optional: Qdrant server in Docker (default is embedded)
make down      # stop services
make ingest    # load documents
make eval      # run evaluation
make test      # pytest
make lint      # ruff
```

## Agent skills

### Issue tracker

Issues và specs nằm ở GitHub Issues của `nvtoan0201-swe/legal-rag` (dùng `gh` CLI). See `docs/agents/issue-tracker.md`.

### Triage labels

Dùng 5 nhãn mặc định: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` và `docs/adr/` ở gốc repo. See `docs/agents/domain.md`.
