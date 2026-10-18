"""API tests with stub retriever/agent: no models, no Qdrant, no LLM required."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from agent.graph import AgentResult
from app.main import app
from retrieval.search import Hit

PAYLOAD = {
    "chunk_id": "blld2019_d35_k2",
    "doc_id": "blld2019",
    "doc_title": "Bộ luật Lao động 2019",
    "article_id": "blld2019_d35",
    "article_no": 35,
    "article_title": "Quyền đơn phương chấm dứt hợp đồng lao động",
    "clause_no": 2,
    "text": "Phải báo trước ít nhất 45 ngày.",
}


class FakeRetriever:
    def __init__(self, hits=None):
        self.hits = hits if hits is not None else [Hit("blld2019_d35_k2", 0.91, PAYLOAD, "rerank")]
        self.calls: list[tuple] = []

    def search(self, question, mode, limit, prefetch, threshold):
        self.calls.append((question, mode, limit, prefetch, threshold))
        return list(self.hits)


class FakeAgent:
    def __init__(self, hits=None, answer="Theo Điều 35, phải báo trước 45 ngày."):
        self.hits = hits if hits is not None else [Hit("blld2019_d35_k2", 0.91, PAYLOAD, "rerank")]
        self.answer = answer
        self.questions: list[str] = []

    def run(self, question):
        self.questions.append(question)
        return AgentResult(
            question=question,
            route="vector_search",
            answer=self.answer,
            found=True,
            hits=list(self.hits),
            contexts=list(self.hits),
            elapsed_ms=12.5,
        )

    async def astream(self, question):
        yield {"event": "meta", "data": {"question": question, "route": "vector_search"}}
        yield {"event": "sources", "data": [hit.to_dict(include_text=False) for hit in self.hits]}
        yield {"event": "token", "data": {"text": self.answer}}
        yield {"event": "done", "data": {"answer": self.answer, "found": True, "elapsed_ms": 12.5}}


def test_query_uses_agent_and_returns_citations():
    with TestClient(app) as client:
        fake = FakeAgent()
        app.state.agent = fake
        response = client.post("/query", json={"question": "Phải báo trước bao nhiêu ngày?"})
        assert response.status_code == 200
        body = response.json()
        assert body["mode"] == "agent"
        assert body["route"] == "vector_search"
        assert body["found"] is True
        assert body["answer"].startswith("Theo Điều 35")
        assert body["results"][0]["citation"] == "Điều 35 khoản 2, Bộ luật Lao động 2019"
        assert fake.questions == ["Phải báo trước bao nhiêu ngày?"]


def test_query_retrieval_only_mode():
    with TestClient(app) as client:
        fake = FakeRetriever()
        app.state.retriever = fake
        body = client.post(
            "/query", json={"question": "Phải báo trước bao nhiêu ngày?", "use_agent": False}
        ).json()
        assert body["found"] is True
        assert body["mode"] == "hybrid_rerank"
        assert body["answer"] is None
        assert body["results"][0]["citation"] == "Điều 35 khoản 2, Bộ luật Lao động 2019"
        assert fake.calls[0][0] == "Phải báo trước bao nhiêu ngày?"


def test_query_retrieval_only_not_found():
    with TestClient(app) as client:
        app.state.retriever = FakeRetriever(hits=[])
        body = client.post(
            "/query", json={"question": "Câu hỏi không có căn cứ", "use_agent": False}
        ).json()
        assert body["found"] is False
        assert body["results"] == []


def test_query_streams_sse_events():
    with TestClient(app) as client:
        app.state.agent = FakeAgent()
        with client.stream(
            "POST", "/query", json={"question": "Báo trước bao nhiêu ngày?", "stream": True}
        ) as response:
            assert response.status_code == 200
            text = "".join(response.iter_text())
        assert "event: meta" in text
        assert "event: sources" in text
        assert "event: token" in text
        assert "event: done" in text
        done_payload = [
            line for line in text.splitlines() if line.startswith("data:") and '"found"' in line
        ][-1]
        assert json.loads(done_payload.split("data:", 1)[1])["found"] is True


def test_query_validates_input():
    with TestClient(app) as client:
        app.state.agent = FakeAgent()
        assert client.post("/query", json={"question": "a"}).status_code == 422
        assert client.post("/query", json={"question": "hợp lệ?", "mode": "sai"}).status_code == 422
        assert client.post("/query", json={"question": "hợp lệ?", "top_k": 0}).status_code == 422


def test_health_reports_services():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert set(body) >= {
            "status",
            "qdrant",
            "llm",
            "embed_backend",
            "rerank_backend",
            "llm_provider",
        }
        assert body["status"] in {"ok", "degraded"}


def test_documents_listing_and_status():
    with TestClient(app) as client:
        listing = client.get("/documents").json()
        assert {item["doc_id"] for item in listing["corpus"]} >= {"blld2019", "luadd2024"}
        assert client.get("/documents/blld2019/status").status_code == 200
        assert client.get("/documents/khong-ton-tai/status").status_code == 404


def test_upload_queues_processing(monkeypatch):
    calls: list[tuple] = []

    def fake_process(path, doc_id, **kwargs):
        calls.append((path, doc_id))

    monkeypatch.setattr("app.main.process_document", fake_process)
    with TestClient(app) as client:
        response = client.post(
            "/documents",
            files={
                "file": (
                    "nghi-dinh-moi.txt",
                    "Điều 1. Phạm vi\nNội dung.\n".encode(),
                    "text/plain",
                )
            },
            data={"doc_id": "nghi_dinh_moi", "title": "Nghị định mới"},
        )
        assert response.status_code == 202
        assert response.json() == {"doc_id": "nghi_dinh_moi", "status": "pending"}
        assert calls and calls[0][1] == "nghi_dinh_moi"
        listing = client.get("/documents").json()
        assert any(record["doc_id"] == "nghi_dinh_moi" for record in listing["uploads"])
        assert client.get("/documents/nghi_dinh_moi/status").json()["status"] == "pending"
