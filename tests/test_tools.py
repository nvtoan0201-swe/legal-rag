"""Tests for the shared tool implementations (no MCP transport)."""

from __future__ import annotations

from retrieval.search import Hit
from retrieval.tools import (
    compare_documents,
    graph_query,
    mentioned_doc_ids,
    vector_search,
)


class FakeRetriever:
    def __init__(self):
        self.calls: list[dict] = []

    def search(self, question, mode, limit, prefetch, threshold=None, doc_ids=None):
        self.calls.append({"mode": mode, "limit": limit, "doc_ids": doc_ids, "question": question})
        doc_id = (doc_ids or ["blld2019"])[0]
        return [
            Hit(
                f"{doc_id}_d1_k1",
                0.9,
                {
                    "chunk_id": f"{doc_id}_d1_k1",
                    "doc_id": doc_id,
                    "doc_title": doc_id,
                    "article_id": f"{doc_id}_d1",
                    "article_no": 1,
                    "clause_no": 1,
                    "text": "nội dung",
                },
            )
        ]


class FakeStore:
    ready = True

    def article(self, article_id):
        return {"article_id": article_id, "no": 6, "title": "Xử phạt"}

    def references(self, article_id, limit=10):
        return [{"article_id": "luattatgt2024_d9", "no": 9}]

    def neighbors(self, article_id, limit=10):
        return self.references(article_id, limit)

    def details_from(self, doc_id, limit=50):
        return [{"source_id": "nd168_2024_d6", "target_id": "luattatgt2024_d9"}]

    def cross_document_pairs(self, limit=50):
        return [{"source_id": "nd168_2024_d6", "target_id": "luattatgt2024_d9"}]


def test_mentioned_doc_ids_follows_question_order():
    question = (
        "Nghị định 168/2024 quy định thế nào so với Luật Trật tự, an toàn giao thông đường bộ?"
    )
    assert mentioned_doc_ids(question) == ["nd168_2024", "luattatgt2024"]
    assert mentioned_doc_ids("Bộ luật Lao động quy định gì về nghỉ hằng năm?") == ["blld2019"]


def test_vector_search_serialises_hits():
    retriever = FakeRetriever()
    results = vector_search(retriever, "câu hỏi", top_k=3)
    assert results[0]["citation"] == "Điều 1 khoản 1, blld2019"
    assert retriever.calls[0]["mode"] == "hybrid_rerank"
    assert retriever.calls[0]["limit"] == 3


def test_graph_query_returns_references_for_article():
    result = graph_query(FakeStore(), article_id="nd168_2024_d6")
    assert result["found"] is True
    assert result["article"]["no"] == 6
    assert result["references"][0]["article_id"] == "luattatgt2024_d9"


def test_graph_query_without_store_is_unavailable():
    assert graph_query(None, article_id="x")["available"] is False


def test_compare_documents_searches_each_document():
    retriever = FakeRetriever()
    result = compare_documents(retriever, "so sánh quy định", ["nd168_2024", "luattatgt2024"])
    assert set(result["by_document"]) == {"nd168_2024", "luattatgt2024"}
    assert len(result["merged"]) == 2
    assert [call["doc_ids"] for call in retriever.calls] == [["nd168_2024"], ["luattatgt2024"]]


def test_compare_documents_infers_docs_from_question():
    retriever = FakeRetriever()
    result = compare_documents(retriever, "So sánh Luật Đất đai 2024 và Bộ luật Lao động 2019", [])
    assert set(result["by_document"]) == {"blld2019", "luadd2024"}
