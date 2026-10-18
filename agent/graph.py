"""LangGraph router agent: classify -> tool -> cited answer, or "not found".

The graph has five nodes:

    START -> router -> {vector_search | graph_query | compare_documents} -> answer -> END

The router is rule-first and falls back to the LLM; the rules cover the two
unambiguous shapes (comparison between two named documents, explicit reference
questions) so the router stays cheap for the common case. The answer node is the
only place that talks to the LLM about content, and it must cite articles; when no
evidence was retrieved it returns "not found" without calling the LLM.

Streaming (SSE) reuses the same router/tool nodes and streams only the answer
node's tokens; see AgentRunner.astream.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from llm.base import strip_thinking, system, user
from retrieval import tools as tool_impl
from retrieval.search import Hit

log = logging.getLogger("agent")

NOT_FOUND_ANSWER = "Không tìm thấy quy định phù hợp trong cơ sở dữ liệu."
NOT_FOUND_MARKERS = (
    "không tìm thấy",
    "không có đủ căn cứ",
    "không đủ căn cứ",
    "không có thông tin",
)

ROUTER_SYSTEM = (
    "Bạn là bộ định tuyến của hệ thống hỏi đáp pháp luật Việt Nam. "
    "Phân loại câu hỏi vào đúng MỘT nhãn sau:\n"
    "- vector_search: tra cứu quy định trong một văn bản\n"
    "- graph_query: hỏi về quan hệ dẫn chiếu giữa các điều luật\n"
    "- compare_documents: so sánh quy định giữa hai văn bản\n"
    "Chỉ trả lời đúng nhãn, không giải thích."
)

ANSWER_SYSTEM = (
    "Bạn là trợ lý pháp luật Việt Nam. Chỉ sử dụng NGỮ CẢNH được cung cấp dưới đây. "
    "Mỗi khẳng định phải kèm trích dẫn dạng (Điều X, tên văn bản). "
    f"Nếu ngữ cảnh không đủ căn cứ, trả lời đúng một câu: {NOT_FOUND_ANSWER} "
    "Tuyệt đối không bịa quy định ngoài ngữ cảnh."
)

COMPARISON_RE = re.compile(
    r"so sánh|khác nhau|khác gì|giống nhau|phân biệt|đối chiếu", re.IGNORECASE
)
GRAPH_RE = re.compile(
    r"dẫn chiếu|tham chiếu|viện dẫn|dẫn tới điều|quy định tại điều nào", re.IGNORECASE
)


class AgentState(TypedDict, total=False):
    question: str
    route: str
    hits: list
    contexts: list
    answer: str
    found: bool


@dataclass
class AgentResult:
    question: str
    route: str
    answer: str
    found: bool
    hits: list = field(default_factory=list)
    contexts: list = field(default_factory=list)
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "route": self.route,
            "answer": self.answer,
            "found": self.found,
            "sources": [hit.to_dict() for hit in self.contexts],
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


def format_contexts(contexts: list[Hit]) -> str:
    blocks = []
    for hit in contexts:
        if not hit.text:
            continue
        blocks.append(f"[{hit.citation}]\n{hit.text}")
    return "\n\n".join(blocks)


def looks_not_found(answer: str) -> bool:
    lowered = (answer or "").lower()
    return any(marker in lowered for marker in NOT_FOUND_MARKERS)


class AgentRunner:
    def __init__(
        self,
        retriever,
        graph_store=None,
        llm=None,
        top_k: int = 5,
        use_llm_router: bool = True,
    ) -> None:
        self.retriever = retriever
        self.graph_store = graph_store
        self.llm = llm
        self.top_k = top_k
        self.use_llm_router = use_llm_router and llm is not None
        self.graph = self._build_graph()

    # -- graph wiring -----------------------------------------------------
    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("router", self.route_node)
        builder.add_node("vector_search", self.vector_node)
        builder.add_node("graph_query", self.graph_node)
        builder.add_node("compare_documents", self.compare_node)
        builder.add_node("answer", self.answer_node)
        builder.add_edge(START, "router")
        builder.add_conditional_edges(
            "router",
            lambda state: state.get("route") or "vector_search",
            {
                "vector_search": "vector_search",
                "graph_query": "graph_query",
                "compare_documents": "compare_documents",
            },
        )
        builder.add_edge("vector_search", "answer")
        builder.add_edge("graph_query", "answer")
        builder.add_edge("compare_documents", "answer")
        builder.add_edge("answer", END)
        return builder.compile()

    # -- nodes ------------------------------------------------------------
    def route_node(self, state: AgentState) -> dict:
        question = state["question"]
        if COMPARISON_RE.search(question) and len(tool_impl.mentioned_doc_ids(question)) >= 2:
            return {"route": "compare_documents"}
        if GRAPH_RE.search(question):
            return {"route": "graph_query"}
        if self.use_llm_router:
            try:
                # Budget must also cover the model's reasoning tokens, if any
                # (a 12-token cap starves hybrid-thinking models like MiMo).
                response = self.llm.chat(
                    [system(ROUTER_SYSTEM), user(question)], temperature=0.0, max_tokens=128
                )
                label = response.strip().lower()
                for candidate in ("compare_documents", "graph_query", "vector_search"):
                    if candidate in label:
                        return {"route": candidate}
            except Exception as exc:  # pragma: no cover - depends on the LLM
                log.warning("router LLM failed, using vector_search: %s", exc)
        return {"route": "vector_search"}

    def vector_node(self, state: AgentState) -> dict:
        hits = self.retriever.search(
            state["question"], mode="hybrid_rerank", limit=self.top_k, prefetch=30
        )
        return {"hits": hits, "contexts": list(hits)}

    def _expand(self, article_ids: list[str], relation_rows: list[dict]) -> list[Hit]:
        """Fetch article texts for graph neighbours and turn them into Hits."""
        if not article_ids or not hasattr(self.retriever, "article_texts"):
            return []
        texts = self.retriever.article_texts(article_ids)
        hits: list[Hit] = []
        for article_id, payload in texts.items():
            hits.append(
                Hit(
                    chunk_id=f"graph:{article_id}",
                    score=0.0,
                    payload={
                        **payload,
                        "article_id": article_id,
                        "text": payload.get("text", ""),
                    },
                    retrieval="graph",
                )
            )
        return hits

    def graph_node(self, state: AgentState) -> dict:
        hits = self.retriever.search(state["question"], mode="hybrid_rerank", limit=2, prefetch=20)
        if not hits or self.graph_store is None:
            return {"hits": hits, "contexts": list(hits)}
        seeds = [hit.article_id for hit in hits if hit.article_id]
        neighbours: list[dict] = []
        neighbour_ids: list[str] = []
        for article_id in seeds:
            payload = tool_impl.graph_query(self.graph_store, article_id=article_id)
            for row in payload.get("neighbors", []) or []:
                neighbours.append(row)
                if row.get("article_id"):
                    neighbour_ids.append(row["article_id"])
        expanded = self._expand(list(dict.fromkeys(neighbour_ids)), neighbours)
        return {"hits": hits, "contexts": list(hits) + expanded}

    def compare_node(self, state: AgentState) -> dict:
        question = state["question"]
        doc_ids = tool_impl.mentioned_doc_ids(question)
        if len(doc_ids) < 2:
            return self.vector_node(state)
        payload = tool_impl.compare_documents(
            self.retriever, question, doc_ids, top_k=max(2, self.top_k // 2)
        )
        contexts = [
            Hit(
                chunk_id=item["chunk_id"],
                score=float(item.get("score", 0.0)),
                payload={
                    "chunk_id": item["chunk_id"],
                    "doc_id": item.get("doc_id", ""),
                    "doc_title": item.get("doc_title", ""),
                    "article_id": item.get("article_id", ""),
                    "article_no": item.get("article_no", 0),
                    "article_title": item.get("article_title", ""),
                    "clause_no": item.get("clause_no"),
                    "text": item.get("text", ""),
                },
                retrieval="compare",
            )
            for item in payload.get("merged", [])
        ]
        return {"hits": contexts, "contexts": contexts}

    def answer_node(self, state: AgentState) -> dict:
        contexts: list[Hit] = state.get("contexts") or []
        contexts = [hit for hit in contexts if hit.text]
        if not contexts:
            return {"answer": NOT_FOUND_ANSWER, "found": False}
        prompt = (
            f"Câu hỏi: {state['question']}\n\n"
            f"Ngữ cảnh:\n{format_contexts(contexts)}\n\n"
            "Hãy trả lời ngắn gọn, nêu rõ điều luật và tên văn bản cho mỗi khẳng định."
        )
        try:
            answer = self.llm.chat(
                [system(ANSWER_SYSTEM), user(prompt)], temperature=0.0, max_tokens=700
            )
        except Exception as exc:  # pragma: no cover - depends on the LLM
            log.error("answer LLM failed: %s", exc)
            return {"answer": NOT_FOUND_ANSWER, "found": False}
        if not answer:
            return {"answer": NOT_FOUND_ANSWER, "found": False}
        return {"answer": answer, "found": not looks_not_found(answer)}

    # -- public API -------------------------------------------------------
    def run(self, question: str) -> AgentResult:
        from observability import log_event, trace

        started = time.perf_counter()
        with trace("agent.run", question=question) as span:
            state = self.graph.invoke({"question": question})
            elapsed_ms = (time.perf_counter() - started) * 1000
            result = AgentResult(
                question=question,
                route=state.get("route") or "vector_search",
                answer=state.get("answer") or NOT_FOUND_ANSWER,
                found=bool(state.get("found")),
                hits=list(state.get("hits") or []),
                contexts=list(state.get("contexts") or []),
                elapsed_ms=elapsed_ms,
            )
            log_event(
                span,
                "answer",
                route=result.route,
                found=result.found,
                sources=len(result.contexts),
                answer=result.answer[:500],
            )
        return result

    def prepare(self, question: str) -> AgentResult:
        """Run router + tool nodes only (used by the streaming path)."""
        route = self.route_node({"question": question}).get("route", "vector_search")
        node = {
            "vector_search": self.vector_node,
            "graph_query": self.graph_node,
            "compare_documents": self.compare_node,
        }[route]
        state = node({"question": question})
        return AgentResult(
            question=question,
            route=route,
            answer="",
            found=bool(state.get("contexts")),
            hits=list(state.get("hits") or []),
            contexts=list(state.get("contexts") or []),
        )

    async def astream(self, question: str) -> AsyncIterator[dict]:
        """Yield SSE-ready events: meta, sources, token, done."""
        started = time.perf_counter()
        prepared = await asyncio.to_thread(self.prepare, question)
        yield {
            "event": "meta",
            "data": {"question": question, "route": prepared.route},
        }
        yield {
            "event": "sources",
            "data": [hit.to_dict(include_text=False) for hit in prepared.contexts],
        }
        contexts = [hit for hit in prepared.contexts if hit.text]
        if not contexts:
            answer = NOT_FOUND_ANSWER
            yield {"event": "token", "data": {"text": answer}}
            yield {
                "event": "done",
                "data": {
                    "answer": answer,
                    "found": False,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
                },
            }
            return

        prompt = (
            f"Câu hỏi: {question}\n\n"
            f"Ngữ cảnh:\n{format_contexts(contexts)}\n\n"
            "Hãy trả lời ngắn gọn, nêu rõ điều luật và tên văn bản cho mỗi khẳng định."
        )
        messages = [system(ANSWER_SYSTEM), user(prompt)]
        parts: list[str] = []
        try:
            async for token in self.llm.astream(messages, temperature=0.0, max_tokens=700):
                if not token:
                    continue
                parts.append(token)
                yield {"event": "token", "data": {"text": token}}
        except Exception as exc:  # pragma: no cover - depends on the LLM
            log.error("streaming LLM failed: %s", exc)
        answer = strip_thinking("".join(parts)) or NOT_FOUND_ANSWER
        yield {
            "event": "done",
            "data": {
                "answer": answer,
                "found": not looks_not_found(answer),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        }
