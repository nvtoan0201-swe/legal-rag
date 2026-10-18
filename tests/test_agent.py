"""Tests for the LangGraph agent with fake retriever/LLM (no services)."""

from __future__ import annotations

from agent.graph import (
    NOT_FOUND_ANSWER,
    AgentRunner,
    format_contexts,
    looks_not_found,
)
from retrieval.search import Hit


def make_hit(chunk_id, article_id, doc_id, doc_title, article_no, text, score=0.9):
    return Hit(
        chunk_id,
        score,
        {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "doc_title": doc_title,
            "article_id": article_id,
            "article_no": article_no,
            "article_title": "Tiêu đề",
            "clause_no": 1,
            "text": text,
        },
        "rerank",
    )


class FakeRetriever:
    def __init__(self, hits=None, by_doc=None):
        self.hits = hits or []
        self.by_doc = by_doc or {}
        self.calls: list[dict] = []

    def search(self, question, mode, limit, prefetch, threshold=None, doc_ids=None):
        self.calls.append({"doc_ids": doc_ids, "mode": mode})
        if doc_ids:
            return list(self.by_doc.get(doc_ids[0], []))
        return list(self.hits)

    def article_texts(self, article_ids):
        return {
            article_id: {
                "article_id": article_id,
                "doc_id": article_id.split("_d")[0],
                "doc_title": "Luật",
                "article_no": int(article_id.split("_d")[1]),
                "article_title": "Liên quan",
                "text": "nội dung điều liên quan",
            }
            for article_id in article_ids
        }


class FakeLLM:
    def __init__(
        self,
        answer="Theo Điều 35, Bộ luật Lao động 2019, phải báo trước 45 ngày.",
        route="vector_search",
    ):
        self.answer = answer
        self.route = route
        self.calls: list[list] = []

    def chat(self, messages, temperature=0.0, max_tokens=800):
        self.calls.append(messages)
        if "bộ định tuyến" in messages[0].content.lower():
            return self.route
        return self.answer

    async def astream(self, messages, temperature=0.0, max_tokens=800):
        for piece in self.answer.split(" "):
            yield piece + " "


class FakeStore:
    ready = True

    def article(self, article_id):
        return {"article_id": article_id}

    def references(self, article_id, limit=10):
        return [{"article_id": "blld2019_d36"}]

    def neighbors(self, article_id, limit=10):
        return [
            {
                "article_id": "blld2019_d36",
                "no": 36,
                "doc_id": "blld2019",
                "doc_title": "Bộ luật Lao động 2019",
                "relation": "REFERENCES",
            }
        ]


def test_vector_route_produces_cited_answer():
    hits = [
        make_hit(
            "blld2019_d35_k2",
            "blld2019_d35",
            "blld2019",
            "Bộ luật Lao động 2019",
            35,
            "báo trước 45 ngày",
        )
    ]
    runner = AgentRunner(
        FakeRetriever(hits=hits), llm=FakeLLM(route="vector_search"), use_llm_router=False
    )
    result = runner.run("Người lao động phải báo trước bao nhiêu ngày?")
    assert result.route == "vector_search"
    assert result.found is True
    assert "Điều 35" in result.answer
    assert result.contexts[0].citation == "Điều 35 khoản 1, Bộ luật Lao động 2019"


def test_no_evidence_returns_not_found_without_answer_llm_call():
    llm = FakeLLM()
    runner = AgentRunner(FakeRetriever(hits=[]), llm=llm, use_llm_router=False)
    result = runner.run("Câu hỏi không có căn cứ?")
    assert result.found is False
    assert result.answer == NOT_FOUND_ANSWER
    assert llm.calls == []


def test_comparison_question_uses_compare_tool():
    by_doc = {
        "nd168_2024": [
            make_hit(
                "nd168_2024_d6_k1",
                "nd168_2024_d6",
                "nd168_2024",
                "Nghị định 168/2024/NĐ-CP",
                6,
                "mức phạt",
            )
        ],
        "luattatgt2024": [
            make_hit(
                "luattatgt2024_d9_k1",
                "luattatgt2024_d9",
                "luattatgt2024",
                "Luật TTATGT 2024",
                9,
                "hành vi bị cấm",
            )
        ],
    }
    retriever = FakeRetriever(by_doc=by_doc)
    runner = AgentRunner(retriever, llm=FakeLLM(), use_llm_router=False)
    result = runner.run(
        "So sánh quy định của Nghị định 168/2024/NĐ-CP và "
        "Luật Trật tự, an toàn giao thông đường bộ 2024?"
    )
    assert result.route == "compare_documents"
    assert {hit.doc_id for hit in result.contexts} == {"nd168_2024", "luattatgt2024"}
    assert [call["doc_ids"] for call in retriever.calls] == [["nd168_2024"], ["luattatgt2024"]]


def test_graph_route_expands_neighbours():
    hits = [
        make_hit(
            "blld2019_d35_k1", "blld2019_d35", "blld2019", "Bộ luật Lao động 2019", 35, "dẫn chiếu"
        )
    ]
    runner = AgentRunner(
        FakeRetriever(hits=hits), graph_store=FakeStore(), llm=FakeLLM(), use_llm_router=False
    )
    result = runner.run("Điều 35 dẫn chiếu tới điều nào?")
    assert result.route == "graph_query"
    assert any(hit.retrieval == "graph" for hit in result.contexts)
    assert any(hit.article_id == "blld2019_d36" for hit in result.contexts)


def test_llm_router_label_is_used_when_no_rule_matches():
    runner = AgentRunner(
        FakeRetriever(hits=[]), llm=FakeLLM(route="graph_query"), use_llm_router=True
    )
    assert runner.route_node({"question": "Điều này liên quan gì?"})["route"] == "graph_query"


def test_format_contexts_and_not_found_detection():
    hit = make_hit("x", "blld2019_d35", "blld2019", "Bộ luật Lao động 2019", 35, "nội dung")
    formatted = format_contexts([hit])
    assert formatted.startswith("[Điều 35 khoản 1, Bộ luật Lao động 2019]")
    assert looks_not_found("Không tìm thấy quy định phù hợp.") is True
    assert looks_not_found("Theo Điều 35...") is False


async def test_astream_emits_sources_tokens_and_done():
    hits = [
        make_hit(
            "blld2019_d35_k2",
            "blld2019_d35",
            "blld2019",
            "Bộ luật Lao động 2019",
            35,
            "báo trước 45 ngày",
        )
    ]
    runner = AgentRunner(FakeRetriever(hits=hits), llm=FakeLLM(), use_llm_router=False)
    events = [event async for event in runner.astream("Báo trước bao nhiêu ngày?")]
    assert [event["event"] for event in events][0] == "meta"
    assert events[1]["event"] == "sources"
    assert any(event["event"] == "token" for event in events)
    done = events[-1]
    assert done["event"] == "done"
    assert done["data"]["found"] is True
