.PHONY: up down ingest ingest-all eval eval-table testset test lint serve mcp

up:
	docker compose up -d

down:
	docker compose down

# Parse -> chunk -> embed -> load Qdrant + Neo4j (bge-m3, ~2 GB model download).
ingest:
	uv run python -m ingestion

# Rebuild every collection from scratch (structure, article, fixed-512 baseline).
ingest-all:
	uv run python -m ingestion --granularity all --recreate

eval:
	uv run python -m eval --config all

eval-table:
	uv run python -m eval --table

# Rebuild eval/testset.jsonl from the corpus + external QA datasets.
testset:
	uv run python -m eval.build_testset

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

serve:
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

mcp:
	uv run python -m retrieval.mcp_server
