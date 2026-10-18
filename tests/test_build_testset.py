"""Tests for test-set building helpers (no LLM needed)."""

from __future__ import annotations

import json

from eval.build_testset import (
    dedupe_rows,
    parse_json_object,
    sample_articles,
    validate_row,
    verify_testset,
)


def article(article_id, doc_id, no, text):
    return {
        "article_id": article_id,
        "doc_id": doc_id,
        "doc_title": "Bộ luật Lao động 2019",
        "article_no": no,
        "article_title": "Tiêu đề",
        "chapter": "",
        "text": text,
    }


def test_parse_json_object_tolerates_fences_and_prose():
    assert parse_json_object('{"question": "q", "answer": "a"}') == {"question": "q", "answer": "a"}
    assert (
        parse_json_object(
            "Đây là kết quả:\n"
            + chr(96) * 3
            + 'json\n{"question": "q", "answer": "a"}\n'
            + chr(96) * 3
        )["question"]
        == "q"
    )
    assert parse_json_object("không có json") == {}


def test_validate_row_rules():
    known = {"blld2019_d35"}
    good = {
        "question": "Người lao động phải báo trước bao nhiêu ngày?",
        "article_id": "blld2019_d35",
        "answer": "Ít nhất 45 ngày.",
    }
    assert validate_row(good, known) is None
    assert (
        validate_row({**good, "article_id": "zzz_d1"}, known) == "unknown gold article ['zzz_d1']"
    )
    assert validate_row({**good, "question": "quá ngắn?"}, known) == "question length"
    assert (
        validate_row({**good, "question": "Không có dấu hỏi."}, known) == "question must end with ?"
    )
    assert validate_row({**good, "answer": "ngắn"}, known) == "answer too short"
    assert validate_row({**good, "article_id": []}, known) == "no gold article"


def test_sample_articles_is_deterministic_and_skips_short():
    by_doc = {
        "blld2019": [article(f"blld2019_d{i}", "blld2019", i, "x" * 300) for i in range(10)]
        + [article("blld2019_d99", "blld2019", 99, "ngắn")],
        "luadd2024": [article(f"luadd2024_d{i}", "luadd2024", i, "x" * 300) for i in range(10)],
    }
    first = sample_articles(by_doc, 3, seed=1)
    second = sample_articles(by_doc, 3, seed=1)
    assert [item["article_id"] for item in first] == [item["article_id"] for item in second]
    assert len(first) == 6
    assert all(item["article_id"] != "blld2019_d99" for item in first)


def test_dedupe_and_verify(tmp_path):
    rows = [
        {"question": "Câu hỏi giống nhau?", "article_id": "blld2019_d1"},
        {"question": "câu   hỏi GIỐNG nhau?", "article_id": "blld2019_d1"},
        {"question": "Câu hỏi khác?", "article_id": "blld2019_d2"},
    ]
    assert len(dedupe_rows(rows)) == 2

    processed = tmp_path / "processed"
    processed.mkdir()
    payload = {
        "chunk_id": "blld2019_d1_k1",
        "doc_id": "blld2019",
        "doc_title": "Bộ luật Lao động 2019",
        "article_id": "blld2019_d1",
        "article_no": 1,
        "article_title": "Tiêu đề",
        "chapter": "Chương I",
        "clause_no": 1,
        "text": "nội dung",
    }
    (processed / "blld2019.structure.jsonl").write_text(
        json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    testset = tmp_path / "testset.jsonl"
    testset.write_text(
        json.dumps({"question": "q1?", "article_id": "blld2019_d1"})
        + "\n"
        + json.dumps({"question": "q2?", "article_id": "blld2019_d404"})
        + "\n",
        encoding="utf-8",
    )
    report = verify_testset(testset, processed)
    assert report["total"] == 2
    assert len(report["bad"]) == 1
    assert report["bad"][0]["missing"] == ["blld2019_d404"]
