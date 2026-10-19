"""API tests with stub retriever and LLM: no models, no Qdrant, no network required."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from llm.base import Message
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

    def search(self, question, mode, limit, prefetch, threshold, doc_ids=None):
        self.calls.append((question, mode, limit, prefetch, threshold))
        return list(self.hits)


class ScriptedLLM:
    model = "scripted"

    def __init__(self, reply="Theo [1], phải báo trước 45 ngày."):
        self.reply = reply
        self.received: list[Message] = []

    def chat(self, messages, temperature=0.0, max_tokens=800):
        self.received = list(messages)
        return self.reply


def test_query_returns_cited_answer():
    with TestClient(app) as client:
        app.state.retriever = FakeRetriever()
        llm = ScriptedLLM()
        app.state.llm = llm
        body = client.post("/query", json={"question": "Phải báo trước bao nhiêu ngày?"}).json()
        assert body["mode"] == "hybrid_rerank"
        assert body["found"] is True
        assert body["answer"] == "Theo [1], phải báo trước 45 ngày."
        assert body["cited"] == ["blld2019_d35_k2"]
        assert body["results"][0]["citation"] == "Điều 35 khoản 2, Bộ luật Lao động 2019"
        assert "Phải báo trước bao nhiêu ngày?" in llm.received[-1].content


def test_query_not_found_when_model_declines():
    with TestClient(app) as client:
        app.state.retriever = FakeRetriever()
        app.state.llm = ScriptedLLM(reply="NOT_FOUND")
        body = client.post("/query", json={"question": "Câu hỏi không có căn cứ?"}).json()
        assert body["found"] is False
        assert body["cited"] == []


def test_query_retrieval_only_mode():
    with TestClient(app) as client:
        fake = FakeRetriever()
        app.state.retriever = fake
        llm = ScriptedLLM()
        app.state.llm = llm
        body = client.post(
            "/query", json={"question": "Phải báo trước bao nhiêu ngày?", "answer": False}
        ).json()
        assert body["found"] is True
        assert body["answer"] is None
        assert body["results"][0]["citation"] == "Điều 35 khoản 2, Bộ luật Lao động 2019"
        assert fake.calls[0][0] == "Phải báo trước bao nhiêu ngày?"
        assert llm.received == []


def test_query_retrieval_only_not_found():
    with TestClient(app) as client:
        app.state.retriever = FakeRetriever(hits=[])
        body = client.post(
            "/query", json={"question": "Câu hỏi không có căn cứ", "answer": False}
        ).json()
        assert body["found"] is False
        assert body["results"] == []


def test_query_validates_input():
    with TestClient(app) as client:
        app.state.retriever = FakeRetriever()
        app.state.llm = ScriptedLLM()
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
