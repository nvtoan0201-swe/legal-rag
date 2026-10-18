# Design

## Architecture

```
Client ──HTTP/SSE──> FastAPI (app/)
                        │
                     LangGraph agent (agent/)
                        │ router: classify question
        ┌───────────────┼────────────────┐
   vector_search     graph_query     compare_docs
        └───────────────┴────────────────┘
                  MCP server (retrieval/)
              ┌─────────┴─────────┐
           Qdrant               Neo4j
                        │
                   LLM (llm/): API model | Qwen3-8B
                        │
                     Langfuse (tracing)
```

## Data flow

**Ingestion:** PDF/DOCX → Docling (fallback PyMuPDF) → regex split by Chapter/Article/Clause → attach metadata → bge-m3 dense+sparse → Qdrant; articles and references → Neo4j.

**Query:** question → router → tool(s) → hybrid search → rerank (bge-reranker-v2-m3) → optional graph expansion → LLM answer with citations → SSE stream. No evidence → "not found".

## Chunk schema (Qdrant payload)

```json
{
  "chunk_id": "blld2019_d35_k1",
  "doc_id": "blld2019",
  "doc_title": "Bộ luật Lao động 2019",
  "chapter": "Chương III",
  "article_no": 35,
  "article_title": "Quyền đơn phương chấm dứt hợp đồng lao động của người lao động",
  "clause_no": 1,
  "text": "...",
  "effective_date": "2021-01-01",
  "source_file": "data/raw/..."
}
```

Vectors: `dense` (bge-m3, 1024-d) and `sparse` (bge-m3 lexical weights).

## Graph schema (Neo4j)

Nodes:
- `(:Document {doc_id, title, type, number, effective_date})`
- `(:Article {article_id, doc_id, no, title})`

Relationships:
- `(:Document)-[:HAS_ARTICLE]->(:Article)`
- `(:Article)-[:REFERENCES]->(:Article)` — explicit citation
- `(:Article)-[:DETAILS]->(:Article)` — decree article that implements a law article

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/documents` | Upload a document; processing runs in background |
| GET | `/documents` | List documents |
| GET | `/documents/{id}/status` | Processing status |
| POST | `/query` | Answer with citations, streamed via SSE |
| GET | `/health` | Service health |
