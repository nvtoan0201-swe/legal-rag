# legal-rag

Question answering over Vietnamese legal documents with **cited answers** and an
explicit "not found" when the corpus has no evidence. Runs on a personal machine:
Qdrant is embedded (no Docker), and the LLM is any OpenAI-compatible server or the
Anthropic API, chosen by configuration.

**Corpus** (four documents, 624 articles, 2,740 structure chunks):

| doc_id | Document | Number | Articles | Effective |
|---|---|---|---|---|
| `blld2019` | Bộ luật Lao động 2019 | 45/2019/QH14 | 220 | 2021-01-01 |
| `luadd2024` | Luật Đất đai 2024 | 31/2024/QH15 | 260 | 2025-01-01 |
| `luattatgt2024` | Luật Trật tự, an toàn giao thông đường bộ 2024 | 36/2024/QH15 | 89 | 2025-01-01 |
| `nd168_2024` | Nghị định 168/2024/NĐ-CP | 168/2024/NĐ-CP | 55 | 2025-01-01 |

## Architecture

```
question ──> rag.answer ──> retrieval (Retriever) ──> Qdrant (embedded)
                │              dense + sparse, RRF, optional rerank
                │                  ▲
                │            ingestion: parse → chunk → embed → upsert
                ▼
           llm/ (LLMClient)  ── openai | anthropic   (LLM_PROVIDER)
                │
           cited answer, or "not found"
```

* **Ingestion** (`ingestion/`): text/PDF → regex split by Chương/Điều/Khoản with
  metadata per `docs/design.md` → bge-m3 dense + sparse vectors → Qdrant.
* **Retrieval** (`retrieval/`): hybrid search (dense + sparse, RRF fusion), then
  bge-reranker-v2-m3 when enabled. A score below the threshold returns no hits.
* **Answer** (`rag/answer.py`): numbers the retrieved chunks `[1]`, `[2]`, asks the
  LLM to cite them, and returns "not found" when there is no evidence or no
  citation points at a real chunk. The LLM is trusted for wording only.
* **LLM** (`llm/`): one `LLMClient` seam. `LLM_PROVIDER=openai` covers any
  OpenAI-compatible server (OpenAI, OpenCode Go, Ollama, vLLM, LM Studio);
  `LLM_PROVIDER=anthropic` uses the Messages API. Switching is configuration only.

## Quickstart

```bash
cp .env.example .env    # set LLM_PROVIDER, LLM_API_BASE, LLM_MODEL, LLM_API_KEY
uv sync                 # Python 3.11 environment
make ingest             # parse -> chunk -> embed -> Qdrant (data/qdrant), ~10 min on GPU
make test               # hermetic test suite (no services, no network)
make eval               # retrieval ablation over the test set
make serve              # FastAPI on :8000
```

`QDRANT_URL` empty (the default) keeps Qdrant embedded at `QDRANT_PATH`. Embedded
Qdrant locks its folder to one process, so the API and an eval run cannot share it at
the same time. Set `QDRANT_URL=http://localhost:6333` to use a Qdrant server instead
(`docker compose up -d qdrant`).

If the LLM call hangs for minutes on a machine with a blackholed IPv6 route, set
`LLM_IPV4_ONLY=true` (see `llm/net.py`).

Ask a question:

```bash
curl -s localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"question": "Người lao động đơn phương chấm dứt hợp đồng không xác định thời hạn phải báo trước bao nhiêu ngày?"}' | jq
```

Retrieval only, without calling the LLM:

```bash
curl -s localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"question": "Phải báo trước bao nhiêu ngày?", "answer": false}' | jq
```

Docker (optional): `docker compose up -d` runs Qdrant, one-shot ingestion and the app.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/query` | `answer: true` (default) returns a cited answer; `answer: false` returns chunks only |
| GET | `/health` | Qdrant reachability and point count, LLM configuration and reachability |
| POST | `/documents` | Upload a document; processing runs in the background |
| GET | `/documents` | Corpus + uploaded documents with status |
| GET | `/documents/{id}/status` | Processing status of one document |

Response fields of `/query`: `found`, `answer`, `cited` (chunk ids the answer
cites), `results` (retrieved chunks with citations), `elapsed_ms`.

## Results

Numbers below were measured with Qdrant on a server (phase 0). Phase 1 runs the same
chunks and vectors embedded in Qdrant; see "Local run" for the comparison.


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

### Local run (phase 1, embedded Qdrant)

Same 187 questions, same chunks and vectors, Qdrant embedded at `data/qdrant`
(`QDRANT_URL` empty). Run with `uv run python -m eval --config all --out-dir eval/results/local`:

| Config | Search | Recall@5 local | Recall@5 server (phase 0) | MRR local | MRR server | p50 ms local |
|---|---|---|---|---|---|---|
| `baseline_fixed512_dense` | fixed 512 tokens, dense | 0.588 | 0.588 | 0.375 | 0.375 | 25 |
| `structure_dense` | structure, dense | 0.872 | 0.872 | 0.776 | 0.776 | 31 |
| `structure_hybrid` | structure, dense+sparse RRF | 0.872 | 0.866 | 0.785 | 0.785 | 105 |
| `structure_hybrid_rerank` | structure, hybrid + rerank | 0.898 | 0.893 | 0.813 | 0.811 | 2268 |
| `article_hybrid_rerank` | article-level, hybrid + rerank | 0.840 | 0.840 | 0.741 | 0.741 | 3153 |

Four of five configs reproduce the server numbers exactly. The two hybrid configs
differ by one question out of 187 on Recall@5. Rerank latency is higher locally because
the GPU is shared between the embedder and the reranker.

### Test set

`eval/testset.jsonl` (see `eval/README.md`): 187 questions — 123 filtered from
public Vietnamese legal QA datasets (mapped to our article ids,
`<doc_id>_d<article>`) and 64 LLM-generated questions grounded in the corpus.
`eval/testset_hard.jsonl` holds 12 hand-written comparison/multi-step questions
with two gold articles each. The Zalo AI 2021 legal retrieval corpus is mirrored
in `eval/zalo_raw/` for reference, but none of its gold passages belong to these
four documents (it was collected before 2024), so it is not part of the test set
— a documented deviation from the original evaluation plan.

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
app/          FastAPI app, query and document endpoints
rag/          answer generation with citations and "not found"
retrieval/    Retriever (dense, sparse, hybrid, rerank) and hit model
ingestion/    registry, parse (text/PDF), chunk, embed, load Qdrant
llm/          LLMClient seam: openai-compatible and anthropic adapters
eval/         test set builder, metrics, retrieval runner, LLM judge
docs/         design and decisions
data/raw/     source documents (not committed; see data/raw/SOURCES.md)
```

## Decisions and limitations

* `docs/decisions.md` — structure-aware chunking, bge-m3, Qdrant hybrid search,
  reranking, hosted model.
* `docs/design.md` — schemas and API surface.
* Limitations: the land-law appendix is not part of the source text; the
  "Điều N" reference resolution used by phase 0's graph is no longer part of the
  query path; Qdrant embedded mode is single-process.
* The Docker path (Dockerfile + compose) is configured but the image build was not
  verified on the development host (constrained root disk).
