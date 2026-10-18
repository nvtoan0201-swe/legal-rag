"""Tests for the faithfulness judge (no LLM server needed)."""

from __future__ import annotations

from eval.judge import (
    LLMJudge,
    citation_valid,
    extract_cited_articles,
    is_not_found,
    parse_verdict,
)
from retrieval.search import Hit


def hit(article_no, chunk_id="c1"):
    return Hit(
        chunk_id,
        0.8,
        {
            "chunk_id": chunk_id,
            "doc_title": "Bộ luật Lao động 2019",
            "article_no": article_no,
            "article_id": f"blld2019_d{article_no}",
            "text": "nội dung điều luật",
        },
    )


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def chat(self, messages, temperature=0.0, max_tokens=800):
        self.calls.append(messages)
        return self.response


def test_extract_and_validate_citations():
    assert extract_cited_articles("Theo Điều 35 và Điều 35, khoản 2 Điều 10") == [35, 10]
    assert citation_valid("Theo Điều 35, BLLĐ 2019", [hit(35)]) is True
    assert citation_valid("Theo Điều 35, BLLĐ 2019", [hit(10)]) is False
    assert citation_valid("Không có trích dẫn", [hit(35)]) is False


def test_is_not_found():
    assert is_not_found("Không tìm thấy quy định phù hợp.") is True
    assert is_not_found("Theo Điều 35, phải báo trước 45 ngày.") is False


def test_parse_verdict():
    assert parse_verdict("CÓ") is True
    assert parse_verdict("không") is False
    assert parse_verdict("Yes, supported") is True
    assert parse_verdict("maybe") is False


def test_judge_uses_llm_for_supported_answer():
    judge = LLMJudge(llm=FakeLLM("CÓ"))
    assert (
        judge("Báo trước bao nhiêu ngày?", "Theo Điều 35, phải báo trước 45 ngày.", [hit(35)])
        is True
    )
    assert judge.llm.calls, "judge must call the LLM"


def test_judge_rejects_unretrieved_citation_without_calling_llm():
    fake = FakeLLM("CÓ")
    judge = LLMJudge(llm=fake)
    assert judge("câu hỏi", "Theo Điều 99, ...", [hit(35)]) is False
    assert fake.calls == []


def test_judge_scores_not_found_against_evidence():
    judge = LLMJudge(llm=FakeLLM("KHÔNG"))
    assert judge("câu hỏi", "Không tìm thấy quy định.", []) is True
    assert judge("câu hỏi", "Không tìm thấy quy định.", [hit(35)]) is False
