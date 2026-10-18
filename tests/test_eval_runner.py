"""Tests for the eval runner with a stub retriever (no models, no Qdrant)."""

from __future__ import annotations

import json

from eval.run import CONFIGS, Config, evaluate, format_table, load_testset, run_config
from retrieval.search import Hit


class FakeRetriever:
    def __init__(self, hits_by_question=None, collection="legal_docs"):
        self.hits_by_question = hits_by_question or {}
        self.collection = collection
        self.calls: list[dict] = []

    def search(self, question, mode, limit, prefetch, threshold=None):
        self.calls.append(
            {"question": question, "mode": mode, "limit": limit, "prefetch": prefetch}
        )
        return list(self.hits_by_question.get(question, []))


def make_hit(chunk_id, article_id, score=0.5):
    return Hit(
        chunk_id,
        score,
        {
            "chunk_id": chunk_id,
            "doc_id": "blld2019",
            "doc_title": "Bộ luật Lao động 2019",
            "article_id": article_id,
            "article_no": 35,
            "text": "nội dung",
        },
    )


def test_load_testset_accepts_string_and_list(tmp_path):
    path = tmp_path / "testset.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"question": "q1", "article_id": "blld2019_d35", "answer": "a"}),
                json.dumps({"question": "q2", "article_id": ["blld2019_d1", "luadd2024_d2"]}),
                "",
            ]
        ),
        encoding="utf-8",
    )
    rows = load_testset(path)
    assert rows[0]["gold"] == ["blld2019_d35"]
    assert rows[1]["gold"] == ["blld2019_d1", "luadd2024_d2"]


def test_evaluate_records_retrieved_ids_and_latency():
    testset = [{"question": "q1", "gold": ["blld2019_d35"], "answer": "báo trước 45 ngày"}]
    retriever = FakeRetriever({"q1": [make_hit("blld2019_d35_k2", "blld2019_d35")]})
    config = Config(name="structure_dense", collection="legal_docs", mode="dense")
    rows = evaluate(testset, retriever, config)
    assert rows[0]["retrieved"] == ["blld2019_d35"]
    assert rows[0]["chunks"] == ["blld2019_d35_k2"]
    assert rows[0]["latency_ms"] >= 0
    assert rows[0]["faithful"] is None


def test_evaluate_calls_judge_when_answer_provided():
    testset = [{"question": "q1", "gold": ["blld2019_d35"], "answer": "ref"}]
    retriever = FakeRetriever({"q1": [make_hit("blld2019_d35_k2", "blld2019_d35")]})
    config = Config(name="x", collection="legal_docs", mode="dense")
    rows = evaluate(
        testset,
        retriever,
        config,
        answer_fn=lambda question, hits: "câu trả lời",
        judge=lambda question, answer, hits: True,
    )
    assert rows[0]["answer"] == "câu trả lời"
    assert rows[0]["faithful"] is True


def test_run_config_writes_json_and_sets_collection(tmp_path):
    testset = [{"question": "q1", "gold": ["blld2019_d35"], "answer": "ref"}]
    retriever = FakeRetriever({"q1": [make_hit("blld2019_d35_k2", "blld2019_d35")]})
    result = run_config("structure_hybrid_rerank", testset, retriever, out_dir=tmp_path)
    assert retriever.collection == CONFIGS["structure_hybrid_rerank"].collection
    assert result["summary"]["recall@5"] == 1.0
    saved = json.loads((tmp_path / "structure_hybrid_rerank.json").read_text(encoding="utf-8"))
    assert saved["config"] == "structure_hybrid_rerank"
    table = format_table([result])
    assert "structure_hybrid_rerank" in table
    assert "Recall@5" in table
