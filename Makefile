.PHONY: up down ingest eval test lint

up:
	docker compose up -d

down:
	docker compose down

ingest:
	uv run python -m ingestion  # TODO

eval:
	uv run python -m eval  # TODO

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .
