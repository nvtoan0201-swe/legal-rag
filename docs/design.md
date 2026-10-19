# Design

## Architecture

```
question ──> rag.answer.answer ──> Retriever.search ──> Qdrant (embedded or server)
                  │                    dense + sparse, RRF, optional rerank
                  │
                  ▼
             LLMClient.chat ── openai | anthropic   (LLM_PROVIDER)
                  │
             cited answer, or "not found"
```

Phase 1 is the local RAG core. The graph (Neo4j, cross-reference edges), the
LangGraph router and the MCP server from phase 0 are out of the query path; the
graph returns in a later phase, measured against this baseline.

## Data flow

**Ingestion:** text/PDF → regex split by Chapter/Article/Clause → metadata → bge-m3
dense + sparse → Qdrant. `python -m ingestion --granularity all` builds three
collections (`legal_docs`, `legal_docs_article`, `legal_docs_baseline`).

**Query:** question → `Retriever.search` (mode, top_k, threshold) → numbered
contexts `[1]…[k]` → LLM → validation → cited answer. No hit above the threshold,
no answer, or no valid citation → "not found".

## Answer validation (`rag/answer.py`)

1. No hits: return not found without calling the LLM.
2. The model replies `NOT_FOUND`, or replies with nothing: not found.
3. The reply must cite at least one context number that exists (`[1]` to `[k]`).
   Cited numbers beyond `k` are ignored; if none remain, the answer is not found.

The LLM is trusted for wording only. Grounding is checked by the code, not by the
model.

## LLM seam (`llm/`)

`LLMClient` is the only interface the answer path sees: `chat`, `achat`, `stream`,
`astream`, `available`, and `model`. Two adapters implement it:

- `OpenAICompatibleClient` for `LLM_PROVIDER=openai` (alias `api`): any server that
  speaks `/v1/chat/completions`.
- `AnthropicClient` for `LLM_PROVIDER=anthropic`: the Messages API over httpx, with
  the system prompt in the top-level `system` field.

`llm/factory.get_llm()` picks the adapter from settings. Tests use a scripted fake
and `httpx.MockTransport`, so no test needs a network or a model.

## Storage (Qdrant)

`QDRANT_URL` empty: embedded storage at `QDRANT_PATH` (one process at a time).
`QDRANT_URL` set: the server at that URL. The collection layout is the same in both.

## Chunk schema (Qdrant payload)

```json
{
  "chunk_id": "blld2019_d35_k1",
  "doc_id": "blld2019",
  "doc_title": "Bộ luật Lao động 2019",
  "doc_number": "45/2019/QH14",
  "doc_type": "law",
  "article_id": "blld2019_d35",
  "article_no": 35,
  "article_title": "Quyền đơn phương chấm dứt hợp đồng lao động của người lao động",
  "chapter": "Chương III. Hợp đồng lao động",
  "section": "",
  "clause_no": 1,
  "text": "...",
  "effective_date": "2021-01-01",
  "source_file": "data/raw/blld2019.txt",
  "granularity": "structure"
}
```

Vectors: `dense` (bge-m3, 1024-d, cosine) and `sparse` (bge-m3 lexical weights,
Qdrant sparse index). Baseline chunks use the same schema with
`granularity="fixed512"` and the nearest preceding article id so Recall can still
be computed.

## Retrieval modes

`dense` | `sparse` | `hybrid` (RRF fusion) | `hybrid_rerank` (default). The
reranker takes `prefetch` (default 30) candidates and returns `top_k`.

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/query` | Cited answer (`answer: true`, default) or chunks only (`answer: false`) |
| GET | `/health` | Qdrant reachability and point count; LLM provider, model and reachability |
| POST | `/documents` | Upload a document; processing runs in a background task |
| GET | `/documents` | Corpus documents + uploads with status |
| GET | `/documents/{id}/status` | Processing status |

### `POST /query`

Request: `{question, top_k, mode, prefetch, threshold, include_text, answer}`.
Response:

```json
{
  "question": "...",
  "found": true,
  "mode": "hybrid_rerank",
  "answer": "Theo [1], phải báo trước 45 ngày.",
  "cited": ["blld2019_d35_k2"],
  "results": [{"chunk_id": "blld2019_d35_k2", "citation": "Điều 35 khoản 2, Bộ luật Lao động 2019", "score": 0.91, "text": "..."}],
  "elapsed_ms": 1234.5
}
```

`cited` lists the chunk ids the answer cites; `results` lists every retrieved
chunk, cited or not.
