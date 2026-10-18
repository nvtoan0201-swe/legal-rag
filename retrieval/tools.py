"""Tool implementations shared by the MCP server and the LangGraph agent.

Keeping the logic here (plain functions over a Retriever/GraphStore) makes the
three tools in plans/week-4-graph-agent-mcp.md testable without an MCP transport.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

log = logging.getLogger(__name__)

DOC_MENTION_PATTERNS = (
    ("blld2019", re.compile(r"bộ luật lao động|luật lao động|bllđ", re.IGNORECASE)),
    (
        "luattatgt2024",
        re.compile(
            r"luật trật tự[,\s]+an toàn giao thông đường bộ|"
            r"trật tự, an toàn giao thông đường bộ|luật ttatgt",
            re.IGNORECASE,
        ),
    ),
    ("luadd2024", re.compile(r"luật đất đai", re.IGNORECASE)),
    ("nd168_2024", re.compile(r"nghị định\s*(?:số\s*)?168/2024|nghị định 168", re.IGNORECASE)),
)


def mentioned_doc_ids(question: str) -> list[str]:
    """Document ids explicitly named in a question, in mention order."""
    found: list[tuple[int, str]] = []
    for doc_id, pattern in DOC_MENTION_PATTERNS:
        match = pattern.search(question)
        if match:
            found.append((match.start(), doc_id))
    return [doc_id for _position, doc_id in sorted(found)]


def vector_search(
    retriever,
    question: str,
    top_k: int = 5,
    mode: str = "hybrid_rerank",
    doc_ids: Sequence[str] | None = None,
) -> list[dict]:
    hits = retriever.search(
        question,
        mode=mode,
        limit=top_k,
        prefetch=max(30, top_k * 6),
        doc_ids=list(doc_ids) if doc_ids else None,
    )
    return [hit.to_dict() for hit in hits]


def graph_query(
    store,
    article_id: str | None = None,
    doc_id: str | None = None,
    limit: int = 10,
) -> dict:
    """Neighbours of an article, or the decree→law edge list for a document."""
    if store is None or not getattr(store, "ready", True):
        return {"available": False, "error": "graph store not configured"}
    try:
        if article_id:
            article = store.article(article_id)
            if article is None:
                return {"available": True, "found": False, "article_id": article_id}
            return {
                "available": True,
                "found": True,
                "article": article,
                "references": store.references(article_id, limit=limit),
                "neighbors": store.neighbors(article_id, limit=limit),
            }
        if doc_id:
            return {
                "available": True,
                "found": True,
                "doc_id": doc_id,
                "details": store.details_from(doc_id, limit=limit),
            }
        return {"available": True, "cross_document": store.cross_document_pairs(limit=limit)}
    except Exception as exc:  # pragma: no cover - depends on Neo4j
        log.warning("graph_query failed: %s", exc)
        return {"available": False, "error": str(exc)}


def compare_documents(
    retriever,
    question: str,
    doc_ids: Sequence[str],
    top_k: int = 3,
) -> dict:
    """Retrieve evidence separately for each named document."""
    if len(doc_ids) < 2:
        doc_ids = list(dict.fromkeys(list(doc_ids) + mentioned_doc_ids(question)))
    results: dict[str, list[dict]] = {}
    for doc_id in doc_ids:
        hits = retriever.search(
            question, mode="hybrid_rerank", limit=top_k, prefetch=30, doc_ids=[doc_id]
        )
        results[doc_id] = [hit.to_dict() for hit in hits]
    merged: list[dict] = []
    seen: set[str] = set()
    for doc_id in doc_ids:
        for hit in results.get(doc_id, []):
            if hit["chunk_id"] in seen:
                continue
            seen.add(hit["chunk_id"])
            merged.append(hit)
    return {"doc_ids": list(doc_ids), "by_document": results, "merged": merged}
