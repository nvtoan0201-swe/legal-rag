# Demo script

A scripted walkthrough of the phase-1 demo moments: health, a cited answer, the
"not found" fallback, retrieval only, and document upload. Recording the video
needs a human at a browser; everything below runs against `make serve` with the
local Qdrant index (`make ingest`) and the LLM configured in `.env`.

## 1. Health

```bash
curl -s localhost:8000/health | jq
```

`qdrant.reachable` is true, `qdrant.points` is 2740 for the four-document corpus,
and `llm.provider` shows the active adapter.

## 2. A cited answer

```bash
curl -s localhost:8000/query -H 'Content-Type: application/json' -d '{
  "question": "Người lao động đơn phương chấm dứt hợp đồng lao động không xác định thời hạn phải báo trước bao nhiêu ngày?"
}' | jq '{found, answer, cited, citations: [.results[].citation]}'
```

Expected: `found: true`, an answer that cites `[1]` (or another number that
exists in `results`), `cited` holding the chunk id of that result, and an
"Điều 35, Bộ luật Lao động 2019" citation in `results`.

## 3. "Not found" instead of hallucination

```bash
curl -s localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"question": "Thủ tục thành lập trường đại học tư thục cần những giấy tờ gì?"}' | jq '{found, answer}'
```

The corpus has no rule on this topic. Either no hit clears the threshold, or the
model answers `NOT_FOUND`, or its answer cites nothing valid. In each case the
response is `found: false` with the fixed "not found" text.

## 4. Retrieval only

```bash
curl -s localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"question": "Phải báo trước bao nhiêu ngày?", "answer": false}' | jq '{found, answer, docs: [.results[].doc_id] | unique}'
```

`answer` is null and no LLM call is made. Useful for checking retrieval on its own.

## 5. Upload a document

```bash
curl -s -F file=@data/raw/nd168_2024.txt -F doc_id=nd168_copy localhost:8000/documents | jq
curl -s localhost:8000/documents/nd168_copy/status | jq
```

Processing (parse → chunk → embed → Qdrant) runs in the background; the status
endpoint reports `pending`, `processing`, then `done` with chunk and article counts.
