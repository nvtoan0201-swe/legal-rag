# legal-rag

Question answering and analysis over Vietnamese legal documents, built as
**GraphRAG + a LangGraph router agent**. Every answer cites specific articles;
when there is no supporting evidence the system answers "not found" instead of
making things up.

**Corpus** (four documents, 624 articles, 2,740 structure chunks):

| doc_id | Document | Number | Articles | Effective |
|---|---|---|---|---|
| `blld2019` | Bộ luật Lao động 2019 | 45/2019/QH14 | 220 | 2021-01-01 |
| `luadd2024` | Luật Đất đai 2024 | 31/2024/QH15 | 260 | 2025-01-01 |
| `luattatgt2024` | Luật Trật tự, an toàn giao thông đường bộ 2024 | 36/2024/QH15 | 89 | 2025-01-01 |
| `nd168_2024` | Nghị định 168/2024/NĐ-CP | 168/2024/NĐ-CP | 55 | 2025-01-01 |

## Architecture

```
Client ──HTTP/SSE──> FastAPI (app/)
                        │
                     LangGraph agent (agent/graph.py)
                        │  router: rules + LLM
        ┌───────────────┼────────────────┐
   vector_search     graph_query     compare_documents
        └───────────────┴────────────────┘
                  MCP server (retrieval/mcp_server.py)
              ┌─────────┴─────────┐
           Qdrant               Neo4j
       (dense+sparse,       (Document, Article,
        bge-m3, rerank)      REFERENCES, DETAILS)
                        │
              LLM (llm/): hosted API | Qwen3-8B (vLLM/Ollama)
                        │
                  Langfuse (observability.py, optional)
```

**Ingestion:** text/PDF/DOCX → Docling (fallback PyMuPDF) → regex split by
Chapter/Article/Clause → metadata per `docs/design.md` → bge-m3 dense + sparse →
Qdrant; documents, articles and cross-references → Neo4j.

**Query:** question → router → tool(s) → hybrid search → rerank
(bge-reranker-v2-m3) → optional graph expansion → answer with citations → SSE
stream. No evidence → "not found".

## Quickstart

```bash
cp .env.example .env
uv sync                 # Python 3.11 environment
make up                 # Qdrant + Neo4j (docker compose)
make ingest             # parse -> chunk -> embed -> load (~2 GB model download)
make test               # hermetic test suite (no services needed)
make eval               # ablation table over the test set
make serve              # FastAPI on :8000
```

Ask a question (the LLM is configured in `.env`: the hosted OpenCode Go gateway
with `mimo-v2.6-flash`, or `LLM_PROVIDER=local` for the self-hosted Qwen3-8B):

```bash
curl -s localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"question": "Người lao động đơn phương chấm dứt hợp đồng không xác định thời hạn phải báo trước bao nhiêu ngày?"}' | jq
```

Streaming (SSE events: `meta`, `sources`, `token`, `done`):

```bash
curl -N localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"question": "So sánh Nghị định 168/2024 và Luật Trật tự, an toàn giao thông đường bộ 2024 về mức phạt?", "stream": true}'
```

Everything in Docker:

```bash
docker compose up -d                   # qdrant + neo4j + one-shot ingest + app
docker compose --profile llm up -d     # optional Ollama for the self-hosted model
docker compose --profile tracing up -d # optional Langfuse at :3000
```

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/query` | Answer with citations; `use_agent=false` returns chunks only, `stream=true` streams SSE |
| GET | `/health` | Qdrant points, LLM reachability, configured backends |
| POST | `/documents` | Upload a document; processing runs in the background |
| GET | `/documents` | Corpus + uploaded documents with status |
| GET | `/documents/{id}/status` | Processing status of one document |

## Results

Ablation over the evaluation test set (187 questions filtered from public
Vietnamese legal QA datasets plus LLM-generated grounded questions, graded
against gold article ids). Measured on an RTX 4050 (6 GB), bge-m3 dense+sparse,
bge-reranker-v2-m3; reproduce with `make eval`:

| Config | Chunks | Search | Recall@5 | MRR | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| `baseline_fixed512_dense` | fixed 512 tokens | dense | 0.588 | 0.375 | 18 | 24 |
| `structure_dense` | Chapter/Article/Clause | dense | 0.872 | 0.776 | 18 | 20 |
| `structure_hybrid` | structure | dense+sparse RRF | 0.866 | 0.785 | 20 | 23 |
| `structure_hybrid_rerank` | structure | hybrid + rerank | **0.893** | **0.811** | 1770 | 2442 |
| `article_hybrid_rerank` | article-level | hybrid + rerank | 0.840 | 0.741 | 2413 | 6521 |

Per-document Recall@5:

| Gold document | fixed-512 dense | structure dense | structure hybrid | structure + rerank | article + rerank |
|---|---|---|---|---|---|
| `blld2019` (Labor Code) | 0.314 | 0.829 | 0.800 | 0.857 | 0.886 |
| `luadd2024` (Land Law) | 0.679 | 0.929 | 0.875 | 0.911 | 0.911 |
| `luattatgt2024` (Traffic Order Law) | 0.558 | 0.923 | 0.904 | 0.942 | 0.904 |
| `nd168_2024` (Decree 168) | 0.727 | 0.773 | 0.864 | 0.841 | 0.636 |

Reading the table:

* Structure-aware chunking is the single biggest win: +28.4 Recall@5 points and
  +0.40 MRR over the fixed-512 baseline, the hypothesis in
  `docs/decisions.md`.
* Sparse vectors improve ranking quality (MRR 0.776 -> 0.785) at a negligible
  Recall cost, because legal queries contain exact terms and article numbers.
  They help Decree 168 the most (0.773 -> 0.864).
* Reranking adds +2.7 Recall@5 and +0.026 MRR but costs ~1.8 s p50 on this GPU;
  the "not found" cutoff lives here (`RERANK_THRESHOLD`).
* Clause-level chunks beat article-level chunks (0.893 vs 0.840); Decree 168
  collapses to 0.636 at article level because its sanctions are spread across
  many short clauses.
* Decree 168 stays the hardest document: its fine tables are flattened to text
  and its questions are numeric ("how much is the fine?").

### Test set

`eval/testset.jsonl` (see `eval/README.md`): 187 questions — 123 filtered from
public Vietnamese legal QA datasets (mapped to our article ids,
`<doc_id>_d<article>`) and 64 LLM-generated questions grounded in the corpus.
`eval/testset_hard.jsonl` holds 12 hand-written comparison/multi-step questions
with two gold articles each. The Zalo AI 2021 legal retrieval corpus is mirrored
in `eval/zalo_raw/` for reference, but none of its gold passages belong to these
four documents (it was collected before 2024), so it is not part of the test set
— a documented deviation from `plans/week-2-eval-baseline.md`.

### Agent evaluation

End-to-end run of the LangGraph agent on a 6-question sample with the local
Qwen3-8B (`uv run python -m eval.agent_eval --limit 6`; embedding and reranking
on CPU because the 6 GB card is shared with the model). Full rows in
`eval/results/agent.json`:

| Metric | Value |
|---|---|
| Recall@5 (retrieved context) | 0.833 |
| Faithfulness (citation check + LLM judge) | 0.833 |
| Token F1 vs. reference answer | 0.539 |
| Answer latency p50 / p95 | 85 s / 225 s |
| Recall@5 after graph expansion | 1.000 (+0.167) |

The last row is the "+graph" ablation entry: expanding the top-5 hits with Neo4j
neighbours recovers every gold article that plain retrieval missed in this
sample, at the cost of extra context. Latency is dominated by the 4-bit model
splitting between GPU and CPU, not by the pipeline: retrieval alone is ~18-20 ms
p50, or ~1.8 s p50 with the reranker.

### Model comparison

| Model | Serving | Faithfulness | Token F1 | Answer p50 | Cost |
|---|---|---|---|---|---|
| Qwen3-8B Q4_K_M | Ollama, RTX 4050 6 GB (53/47 GPU/CPU) | 0.833 | 0.539 | 85 s | 0 (local hardware) |
| MiMo-V2.6-Flash | OpenCode Go gateway (`LLM_PROVIDER=api`) | not run — no API key in this environment | — | — | per token |

The hosted path is wired the same way (see `.env.example`,
`app/settings.py`, and `llm/openai_backend.py`, which sends
`LLM_API_HEADERS` such as `x-opencode-session`) but the gateway key expired
before this run, so only the local row carries measurements.

## Evaluation

```bash
make eval          # run every config, write eval/results/*.json
make eval-table    # print the table from saved results
make testset       # rebuild the test set (needs an LLM for generated questions)
```

Metrics: Recall@5/10, MRR, token-F1 against reference answers, LLM-judge
faithfulness (citation check + judge verdict), latency p50/p95.

## Project layout

```
app/          FastAPI app, SSE streaming, document endpoints
agent/        LangGraph router agent
ingestion/    registry, parse (Docling/PyMuPDF), chunk, embed, load, graph
retrieval/    search (dense/sparse/hybrid/rerank), tools, MCP server, graph queries
llm/          OpenAI-compatible client (hosted API or Qwen3-8B)
eval/         test set builder, metrics, runner, LLM judge
docs/         design and decisions
data/raw/     source documents (not committed; see data/raw/SOURCES.md)
```

## Decisions and limitations

* `docs/decisions.md` — structure-aware chunking, bge-m3, Qdrant hybrid search,
  reranking, Neo4j references, LangGraph + MCP, local vs. hosted model.
* `docs/design.md` — schemas and API surface.
* Limitations: the land-law appendix is not part of the source text; reference
  extraction is regex-based and conservative (bare "Điều N" resolves to the same
  document); the self-hosted model runs 4-bit on a 6 GB GPU, so answer latency is
  seconds rather than milliseconds.
