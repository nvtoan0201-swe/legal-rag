"""Tests for the agent-level evaluation helpers (no services)."""

from __future__ import annotations

from eval.agent_eval import evaluate_agent, graph_expansion_gain, question_type
from retrieval.search import Hit


def hit(article_id, doc_id="blld2019"):
    return Hit(
        f"{article_id}_k1",
        0.9,
        {
            "chunk_id": f"{article_id}_k1",
            "doc_id": doc_id,
            "doc_title": "Bộ luật Lao động 2019",
            "article_id": article_id,
            "article_no": int(article_id.split("_d")[1]),
            "text": "nội dung",
        },
    )


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits

    def search(self, question, mode, limit, prefetch, threshold=None, doc_ids=None):
        return list(self.hits)


class FakeStore:
    ready = True

    def article(self, article_id):
        return {"article_id": article_id}

    def references(self, article_id, limit=10):
        return [{"article_id": "blld2019_d36"}]

    def neighbors(self, article_id, limit=10):
        return [{"article_id": "blld2019_d36", "no": 36, "doc_id": "blld2019"}]

    def details_from(self, doc_id, limit=10):
        return []

    def cross_document_pairs(self, limit=10):
        return []


class FakeResult:
    def __init__(self, question, answer="Theo Điều 35, báo trước 45 ngày."):
        self.question = question
        self.route = "vector_search"
        self.answer = answer
        self.found = True
        self.hits = [hit("blld2019_d35")]
        self.contexts = [hit("blld2019_d35")]


class FakeRunner:
    def run(self, question):
        return FakeResult(question)


def test_question_type():
    assert question_type("Người lao động phải báo trước bao nhiêu ngày?") == "simple"
    assert question_type("So sánh Nghị định 168/2024 và Luật Đất đai 2024?") == "comparison"
    assert question_type("Điều 35 dẫn chiếu tới điều nào?") == "cross_reference"


def test_evaluate_agent_summarises_and_scores():
    testset = [
        {"question": "Báo trước bao nhiêu ngày?", "gold": ["blld2019_d35"], "answer": "45 ngày"}
    ]
    report = evaluate_agent(
        testset, FakeRunner(), judge=lambda question, answer, hits: True, limit=None
    )
    assert report["summary"]["recall@5"] == 1.0
    assert report["summary"]["faithfulness"] == 1.0
    assert report["summary"]["answer_f1"] > 0
    assert report["per_type"]["simple"]["n"] == 1


def test_graph_expansion_gain_counts_neighbours():
    testset = [
        {"question": "Điều 35 dẫn chiếu điều nào?", "gold": ["blld2019_d36"], "answer": "Điều 36"}
    ]
    gain = graph_expansion_gain(FakeRetriever([hit("blld2019_d35")]), FakeStore(), testset)
    assert gain["recall@k"] == 0.0
    assert gain["recall@k+graph"] == 1.0
    assert gain["gain"] == 1.0
