"""Tests for rag.answer with a fake retriever and a scripted LLM (no services)."""

from __future__ import annotations

from llm.base import Message
from rag.answer import NOT_FOUND, answer
from retrieval.search import Hit


def make_hit(chunk_id: str, article_no: int, text: str) -> Hit:
    return Hit(
        chunk_id,
        0.9,
        {
            "chunk_id": chunk_id,
            "doc_id": "blld2019",
            "doc_title": "Bộ luật Lao động 2019",
            "article_id": f"blld2019_d{article_no}",
            "article_no": article_no,
            "article_title": "Tiêu đề",
            "clause_no": None,
            "text": text,
        },
        "rerank",
    )


class FakeRetriever:
    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits
        self.calls: list[dict] = []

    def search(self, question, mode, limit, prefetch, threshold=None, doc_ids=None) -> list[Hit]:
        self.calls.append({"question": question, "mode": mode, "limit": limit, "doc_ids": doc_ids})
        hits = self.hits
        if doc_ids:
            hits = [hit for hit in hits if hit.doc_id in doc_ids]
        return hits[:limit]


class ScriptedLLM:
    model = "scripted"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.received: list[Message] = []

    def chat(self, messages, temperature: float = 0.0, max_tokens: int = 800) -> str:
        self.received = list(messages)
        return self.reply


HITS = [
    make_hit("c1", 35, "Người lao động có quyền đơn phương chấm dứt hợp đồng."),
    make_hit("c2", 36, "Người sử dụng lao động có quyền đơn phương chấm dứt hợp đồng."),
]


def test_no_hits_returns_not_found_without_calling_llm():
    llm = ScriptedLLM("bất kỳ")
    result = answer("câu hỏi?", FakeRetriever([]), llm)
    assert result.found is False
    assert result.text == NOT_FOUND
    assert llm.received == []


def test_cited_answer_returns_only_cited_hits():
    llm = ScriptedLLM("Người lao động được quyền đơn phương chấm dứt [1].")
    result = answer("ai có quyền?", FakeRetriever(HITS), llm)
    assert result.found is True
    assert [hit.chunk_id for hit in result.citations] == ["c1"]


def test_context_numbers_appear_in_prompt():
    llm = ScriptedLLM("[1][2]")
    answer("ai có quyền?", FakeRetriever(HITS), llm)
    user_prompt = llm.received[-1].content
    assert "[1] Điều 35, Bộ luật Lao động 2019" in user_prompt
    assert "[2] Điều 36, Bộ luật Lao động 2019" in user_prompt


def test_model_declares_not_found_is_respected():
    result = answer("câu hỏi?", FakeRetriever(HITS), ScriptedLLM("NOT_FOUND"))
    assert result.found is False
    assert result.text == NOT_FOUND


def test_answer_without_valid_citation_becomes_not_found():
    # [9] points past the 2 contexts, and the rest has no citation at all.
    result = answer("câu hỏi?", FakeRetriever(HITS), ScriptedLLM("Có [9] và không rõ."))
    assert result.found is False


def test_answer_without_any_citation_becomes_not_found():
    result = answer("câu hỏi?", FakeRetriever(HITS), ScriptedLLM("Người lao động có quyền."))
    assert result.found is False


def test_retriever_gets_hybrid_rerank_mode_and_top_k():
    retriever = FakeRetriever(HITS)
    answer("câu hỏi?", retriever, ScriptedLLM("[1]"), top_k=2)
    assert retriever.calls == [
        {"question": "câu hỏi?", "mode": "hybrid_rerank", "limit": 2, "doc_ids": None}
    ]


def make_doc_hit(chunk_id: str, doc_id: str, doc_title: str, text: str) -> Hit:
    hit = make_hit(chunk_id, 1, text)
    hit.payload.update({"doc_id": doc_id, "doc_title": doc_title})
    return hit


def test_mentioned_doc_ids_matches_title_and_number():
    from rag.answer import mentioned_doc_ids

    question = "So sánh Nghị định 168/2024/NĐ-CP và Luật Trật tự, an toàn giao thông đường bộ 2024"
    assert set(mentioned_doc_ids(question)) == {"nd168_2024", "luattatgt2024"}
    assert mentioned_doc_ids("Người lao động phải báo trước bao nhiêu ngày?") == []
    assert mentioned_doc_ids("Nghị định 168/2024 quy định gì?") == ["nd168_2024"]
    assert mentioned_doc_ids("Mã 1168/2024 không phải văn bản nào") == []


def test_comparison_question_gets_contexts_from_each_named_document():
    from rag.answer import retrieve

    hits = [
        make_doc_hit("labour-1", "blld2019", "Bộ luật Lao động 2019", "a"),
        make_doc_hit("labour-2", "blld2019", "Bộ luật Lao động 2019", "b"),
        make_doc_hit("decree-1", "nd168_2024", "Nghị định 168/2024/NĐ-CP", "c"),
    ]
    retriever = FakeRetriever(hits)
    question = "So sánh Nghị định 168/2024/NĐ-CP với Bộ luật Lao động 2019 về xử phạt?"
    got = retrieve(question, retriever, top_k=2)
    assert {hit.chunk_id for hit in got} == {"decree-1", "labour-1", "labour-2"}
    assert sorted(tuple(call["doc_ids"]) for call in retriever.calls) == [
        ("blld2019",),
        ("nd168_2024",),
    ]
