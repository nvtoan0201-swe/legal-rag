# Design

## Architecture

```
Client ──HTTP/SSE──> FastAPI (app/)
                        │
                     LangGraph agent (agent/)
                        │ router: rules + LLM
        ┌───────────────┼────────────────┐
   vector_search     graph_query     compare_documents
        └───────────────┴────────────────┘
                  MCP server (retrieval/)
              ┌─────────┴─────────┐
           Qdrant               Neo4j
                        │
                   LLM (llm/): API model | Qwen3-8B
                        │
                     Langfuse (optional)
```

## Data flow

**Ingestion:** text/PDF/DOCX → Docling (fallback PyMuPDF) → regex split by
Chapter/Article/Clause → metadata → bge-m3 dense+sparse → Qdrant; articles and
references → Neo4j. `python -m ingestion --granularity all` builds three Qdrant
collections (`legal_docs`, `legal_docs_article`, `legal_docs_baseline`) and the
graph.

**Query:** question → router → tool(s) → hybrid search → rerank → optional graph
expansion → LLM answer with citations → SSE stream. No evidence → "not found".

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

## Graph schema (Neo4j)

Nodes:
- `(:Document {doc_id, title, doc_type, number, effective_date})`
- `(:Article {article_id, doc_id, no, title, chapter})`

Relationships:
- `(:Document)-[:HAS_ARTICLE]->(:Article)` — 624 edges
- `(:Article)-[:REFERENCES]->(:Article)` — explicit citation / bare "Điều N"
- `(:Article)-[:DETAILS]->(:Article)` — decree article implementing a law article

Counts after ingesting the four documents: 4 Documents, 624 Articles, 624
HAS_ARTICLE, 635 REFERENCES, 4 DETAILS.

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/query` | Answer with citations. `use_agent=false` → retrieval only; `stream=true` → SSE |
| GET | `/health` | Qdrant point count, LLM reachability, configure backends |
| POST | `/documents` | Upload a document; processing runs in a background task |
| GET | `/documents` | Corpus documents + uploads with status |
| GET | `/documents/{id}/status` | Processing status |

### `POST /query`

Request: `{question, top_k, mode, prefetch, threshold, include_text, use_agent, stream}`.
Response (JSON):

```json
{
  "question": "...",
  "found": true,
  "mode": "agent",
  "route": "vector_search",
  "answer": "Theo Điều 35 khoản 2, Bộ luật Lao động 2019 ...",
  "results": [{"chunk_id": "...", "citation": "Điều 35 khoản 2, Bộ luật Lao động 2019", "score": 0.91, "text": "..."}],
  "elapsed_ms": 1234
}
```

SSE events when `stream=true`: `meta` (question, route), `sources` (citations
only), `token` (answer deltas), `done` (answer, found, elapsed_ms).

## Retrieval modes

`dense` | `sparse` | `hybrid` (RRF fusion) | `hybrid_rerank` (default). The
reranker takes `prefetch` (default 30) candidates and returns `top_k`.
