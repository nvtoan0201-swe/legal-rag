# legal-rag

Question answering and analysis over Vietnamese legal documents, built as **GraphRAG + Multi-Agent**.
Every answer cites specific articles; when there is no supporting evidence, the system answers "not found" instead of making things up.

## Corpus

- Labor Code 2019
- Land Law 2024
- Road Traffic Order and Safety Law 2024
- Decree 168/2024/ND-CP (traffic violation penalties; references the laws above)

## Stack

| Component | Technology |
|---|---|
| Parsing | Docling (fallback: PyMuPDF) |
| Embedding | bge-m3 (dense + sparse) |
| Vector DB | Qdrant (hybrid search) |
| Rerank | bge-reranker-v2-m3 |
| Graph | Neo4j |
| Agent | LangGraph (router + tools) |
| Tool server | MCP |
| LLM | Hosted API model vs. Qwen3-8B (vLLM / Ollama 4-bit) |
| Backend | FastAPI async, SSE streaming |
| Infra | docker-compose, Langfuse, GitLab CI |

## Run

```bash
cp .env.example .env
uv sync
make up       # Qdrant + Neo4j
make ingest   # TODO
make eval     # TODO
```

TODO: start the full system with a single `docker compose up`.

## Results

TODO: ablation table (Recall@5, MRR, Faithfulness, p95 latency) and demo video.

## Docs

- [docs/design.md](docs/design.md) — architecture, schemas, API
- [docs/decisions.md](docs/decisions.md) — technical decisions
- [eval/README.md](eval/README.md) — test set and metrics
