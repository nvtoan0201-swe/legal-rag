"""Answer a question from retrieved chunks, with citations or an explicit "not found".

The LLM is only trusted for wording. Whether an answer is grounded is decided here:
no retrieved evidence above the threshold -> not found; an answer that cites no valid
context number -> not found.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from ingestion.registry import REGISTRY
from llm.base import LLMClient, system, user
from retrieval.search import Hit

NOT_FOUND = "Không tìm thấy căn cứ trong các văn bản hiện có để trả lời câu hỏi này."
NOT_FOUND_TOKEN = "NOT_FOUND"

SYSTEM_PROMPT = f"""Bạn trả lời câu hỏi về văn bản pháp luật Việt Nam.
Chỉ dùng thông tin trong các đoạn tham khảo được đánh số [1], [2], ...
Mỗi ý phải kèm số đoạn làm căn cứ, ví dụ [1] hoặc [2][3].
Nếu câu hỏi có nhiều phần, trả lời phần nào có căn cứ và nói rõ phần nào không có
trong các đoạn tham khảo. Nếu không có phần nào có căn cứ, chỉ trả lời đúng một
từ: {NOT_FOUND_TOKEN}."""

_CITATION_RE = re.compile(r"\[(\d+)\]")


class Searcher(Protocol):
    def search(
        self,
        question: str,
        mode: str,
        limit: int,
        prefetch: int,
        threshold: float | None = None,
        doc_ids: list[str] | None = None,
    ) -> list[Hit]: ...


@dataclass
class Answer:
    text: str
    found: bool
    citations: list[Hit] = field(default_factory=list)


def format_contexts(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{number}] {hit.citation}\n{hit.text}" for number, hit in enumerate(hits, 1)
    )


_NUMBER_STEM_RE = re.compile(r"\d+/\d{4}")


def mentioned_doc_ids(question: str) -> list[str]:
    """Documents the question names by title, short title, or number stem (e.g. 168/2024).

    The number stem matters because users write "Nghị định 168/2024", while the registry
    holds "168/2024/NĐ-CP". Matching the stem with boundaries avoids "1168/2024" and similar.
    """
    lowered = question.lower()
    found = []
    for spec in REGISTRY.values():
        names = [spec.title, spec.short_title]
        stem = _NUMBER_STEM_RE.match(spec.number)
        if any(name and name.lower() in lowered for name in names):
            found.append(spec.doc_id)
        elif stem and re.search(rf"(?<![\d/]){re.escape(stem.group())}(?![\d/])", lowered):
            found.append(spec.doc_id)
    return found


def retrieve(
    question: str,
    retriever: Searcher,
    mode: str = "hybrid_rerank",
    top_k: int = 5,
    prefetch: int = 30,
    threshold: float | None = None,
) -> list[Hit]:
    """Retrieve contexts. A question that names two or more documents gets top_k from each.

    A single global top_k can be filled by one document, which hides the other one in a
    comparison question. Searching each named document separately keeps both in play.
    """
    doc_ids = mentioned_doc_ids(question)
    if len(doc_ids) < 2:
        return retriever.search(
            question, mode=mode, limit=top_k, prefetch=prefetch, threshold=threshold
        )
    merged: dict[str, Hit] = {}
    for doc_id in doc_ids:
        for hit in retriever.search(
            question,
            mode=mode,
            limit=top_k,
            prefetch=prefetch,
            threshold=threshold,
            doc_ids=[doc_id],
        ):
            merged.setdefault(hit.chunk_id, hit)
    return list(merged.values())


def answer(
    question: str,
    retriever: Searcher,
    llm: LLMClient,
    top_k: int = 5,
    mode: str = "hybrid_rerank",
    max_tokens: int = 800,
) -> Answer:
    hits = retrieve(question, retriever, mode=mode, top_k=top_k)
    return answer_from_hits(question, hits, llm, max_tokens=max_tokens)


def answer_from_hits(
    question: str, hits: list[Hit], llm: LLMClient, max_tokens: int = 800
) -> Answer:
    """Generate a cited answer from already-retrieved hits (no second retrieval)."""
    if not hits:
        return Answer(text=NOT_FOUND, found=False)

    prompt = f"Câu hỏi: {question}\n\nĐoạn tham khảo:\n\n{format_contexts(hits)}"
    raw = llm.chat([system(SYSTEM_PROMPT), user(prompt)], temperature=0.0, max_tokens=max_tokens)
    text = raw.strip()
    if not text or text.strip(" .").upper() == NOT_FOUND_TOKEN:
        return Answer(text=NOT_FOUND, found=False)

    cited = sorted({int(number) for number in _CITATION_RE.findall(text)})
    valid = [number for number in cited if 1 <= number <= len(hits)]
    if not valid:
        return Answer(text=NOT_FOUND, found=False)
    return Answer(text=text, found=True, citations=[hits[number - 1] for number in valid])
