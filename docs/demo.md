# Demo script

A scripted walkthrough of the demo moments: a cited answer, streaming (SSE),
the "not found" fallback, a cross-document question, document upload, and the
graph/MCP tools. Recording the video itself needs a human at a browser;
everything below runs against `make serve` with Qdrant, Neo4j and the local
Qwen3-8B configured in `.env`.

## 1. Health

```bash
curl -s localhost:8000/health | jq
```

## 2. A cited answer

```bash
curl -s localhost:8000/query -H 'Content-Type: application/json' -d '{
  "question": "Người lao động đơn phương chấm dứt hợp đồng lao động không xác định thời hạn phải báo trước bao nhiêu ngày?"
}' | jq '{found, route, answer, citations: [.results[].citation]}'
```

Expected shape: `found: true`, `route: "vector_search"`, an answer citing
"Điều 35, Bộ luật Lao động 2019", and the same article in `results`.

## 3. Streaming answer (SSE)

```bash
curl -N localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"question": "Thời giờ làm việc bình thường tối đa trong một tuần là bao nhiêu?", "stream": true}'
```

Events: `meta` (route), `sources` (citations before the answer), `token`
(answer deltas), `done` (final answer, found, elapsed_ms).

## 4. "Not found" instead of hallucination

```bash
curl -s localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"question": "Thủ tục thành lập trường đại học tư thục cần những giấy tờ gì?"}' | jq '{found, answer}'
```

The corpus has no rule on this topic, so the retriever returns no hits above the
threshold and the agent answers `"Không tìm thấy quy định phù hợp trong cơ sở dữ liệu."`
without calling the LLM.

## 5. Cross-document question (decree ↔ law)

```bash
curl -s localhost:8000/query -H 'Content-Type: application/json' -d '{
  "question": "So sánh quy định của Nghị định 168/2024/NĐ-CP và Luật Trật tự, an toàn giao thông đường bộ 2024 về hành vi vi phạm bị xử phạt?"
}' | jq '{route, found, answer, docs: [.results[].doc_id] | unique}'
```

`route` should be `compare_documents` and `docs` should contain both
`nd168_2024` and `luattatgt2024`.

## 6. Upload a document

```bash
curl -s -F file=@data/raw/nd168_2024.txt -F doc_id=nd168_copy localhost:8000/documents | jq
curl -s localhost:8000/documents/nd168_copy/status | jq
```

Processing (parse → chunk → embed → Qdrant → Neo4j) runs in the background; the
status endpoint reports `pending`, `processing`, then `done` with chunk and
article counts.

## 7. Graph and MCP tools

```bash
uv run python -c "
from retrieval.graph import GraphStore
from retrieval.tools import graph_query
store = GraphStore(); store.verify()
print(graph_query(store, article_id='nd168_2024_d6'))
"
uv run python -m retrieval.mcp_server   # stdio MCP server: vector_search, graph_query, compare_documents
```
