"""Tests for retrieval/answer metrics on tiny hand-made cases."""

from __future__ import annotations

from eval.metrics import (
    dedupe,
    first_gold_rank,
    per_document,
    percentile,
    recall_at_k,
    reciprocal_rank,
    summarize,
    token_f1,
)


def test_first_gold_rank_and_recall():
    ranked = ["a_d1", "b_d2", "c_d3", "d_d4", "e_d5", "f_d6"]
    assert first_gold_rank(ranked, ["c_d3"]) == 3
    assert first_gold_rank(ranked, ["zz"]) is None
    assert recall_at_k(ranked, ["e_d5"], k=5) == 1.0
    assert recall_at_k(ranked, ["f_d6"], k=5) == 0.0
    assert recall_at_k(ranked, ["f_d6"], k=10) == 1.0
    assert recall_at_k(ranked, [], k=5) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b"], ["b"]) == 0.5
    assert reciprocal_rank(["a", "b"], ["c"]) == 0.0
    assert reciprocal_rank([], ["a"]) == 0.0


def test_dedupe_keeps_first_order():
    assert dedupe(["b_d2", "a_d1", "b_d2", "", "a_d1"]) == ["b_d2", "a_d1"]


def test_percentile_interpolates():
    assert percentile([], 95) == 0.0
    assert percentile([10.0], 95) == 10.0
    assert percentile([0.0, 10.0], 50) == 5.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5


def test_token_f1():
    assert token_f1("Báo trước 45 ngày", "báo trước 45 ngày") == 1.0
    assert token_f1("", "abc") == 0.0
    assert 0 < token_f1("báo trước 45 ngày", "phải báo trước ít nhất 45 ngày") < 1


def test_summarize_and_per_document():
    rows = [
        {
            "gold": ["blld2019_d35"],
            "retrieved": ["blld2019_d35", "x"],
            "latency_ms": 10,
            "faithful": True,
        },
        {
            "gold": ["luadd2024_d10"],
            "retrieved": ["x", "y"],
            "latency_ms": 30,
            "faithful": False,
        },
    ]
    summary = summarize(rows)
    assert summary["n"] == 2
    assert summary["recall@5"] == 0.5
    assert summary["mrr"] == 0.5
    assert summary["faithfulness"] == 0.5
    assert summary["p50_ms"] == 20.0
    assert summary["p95_ms"] == 29.0

    by_doc = per_document(rows)
    assert set(by_doc) == {"blld2019", "luadd2024"}
    assert by_doc["blld2019"]["recall@5"] == 1.0
    assert by_doc["luadd2024"]["recall@5"] == 0.0
