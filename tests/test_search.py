"""Tests for retrieval modes, citations, and the not-found threshold."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from qdrant_client import models

from ingestion.embed import HashingEmbedder
from retrieval.search import HashingReranker, Hit, Retriever


def scored_point(chunk_id: str, score: float, **payload_overrides):
    payload = {
        "chunk_id": chunk_id,
        "doc_id": "blld2019",
        "doc_title": "Bộ luật Lao động 2019",
        "article_id": "blld2019_d35",
        "article_no": 35,
        "article_title": "Quyền đơn phương chấm dứt hợp đồng lao động",
        "clause_no": 2,
        "text": "Phải báo trước ít nhất 45 ngày.",
    }
    payload.update(payload_overrides)
    return models.ScoredPoint(id=1, version=0, score=score, payload=payload)


class FakeClient:
    def __init__(self, points):
        self.points = points
        self.calls: list[dict] = []

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(points=self.points)


def build(points, **kwargs):
    return Retriever(
        client=FakeClient(points),
        embedder=HashingEmbedder(),
        reranker=HashingReranker(),
        **kwargs,
    )


def test_dense_search_maps_payload_to_citation():
    retriever = build([scored_point("blld2019_d35_k2", 0.8)])
    hits = retriever.dense_search("báo trước bao nhiêu ngày", limit=3)
    assert len(hits) == 1
    hit = hits[0]
    assert hit.citation == "Điều 35 khoản 2, Bộ luật Lao động 2019"
    assert hit.retrieval == "dense"
    assert retriever.client.calls[0]["using"] == "dense"
    data = hit.to_dict()
    assert data["score"] == 0.8
    assert data["text"].startswith("Phải báo trước")


def test_citation_without_clause_or_article():
    build([])  # exercise construction without a live client
    assert (
        Hit("x", 0.0, {"doc_title": "Nghị định 168/2024/NĐ-CP"}).citation
        == "Nghị định 168/2024/NĐ-CP"
    )
    assert Hit("x", 0.0, {"doc_title": "Luật Đất đai 2024", "article_no": 10}).citation == (
        "Điều 10, Luật Đất đai 2024"
    )


def test_not_found_threshold_returns_no_hits():
    retriever = build([scored_point("a", 0.5)])
    assert retriever.search("câu hỏi", mode="dense", threshold=0.9) == []
    assert len(retriever.search("câu hỏi", mode="dense", threshold=0.1)) == 1


def test_hybrid_uses_prefetch_and_fusion():
    retriever = build([scored_point("blld2019_d35_k2", 0.03)])
    hits = retriever.hybrid_search("báo trước", limit=30, prefetch=40)
    call = retriever.client.calls[0]
    assert "using" not in call
    assert len(call["prefetch"]) == 2
    assert [p.using for p in call["prefetch"]] == ["dense", "sparse"]
    assert hits[0].retrieval == "hybrid"


def test_rerank_rescores_and_sorts():
    points = [
        scored_point("blld2019_d1_k1", 0.9, text="quy định chung về lao động", article_no=1),
        scored_point("blld2019_d35_k2", 0.5, text="báo trước ít nhất 45 ngày"),
    ]
    retriever = build(points)
    hits = retriever.search("báo trước 45 ngày", mode="hybrid_rerank", limit=2)
    assert hits[0].chunk_id == "blld2019_d35_k2"
    assert hits[0].retrieval == "rerank"
    assert hits[0].score >= hits[1].score


def test_unknown_mode_rejected():
    retriever = build([])
    with pytest.raises(ValueError):
        retriever.search("câu hỏi", mode="magic")


def test_hashing_reranker_scores_overlap():
    reranker = HashingReranker()
    scores = reranker.score("báo trước 45 ngày", ["báo trước 45 ngày", "nội dung khác"])
    assert scores[0] > scores[1]
