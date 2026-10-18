"""Agent-level evaluation: answer quality, faithfulness, and graph expansion.

Retrieval-only metrics live in eval/run.py. This module measures the end-to-end
agent (plans/week-4 task 8): token-F1 against reference answers, LLM-judge
faithfulness, per-question-type breakdown, and the recall gain from expanding the
top-k hits with Neo4j neighbours (the "+graph" row of the ablation table).

Usage:
    uv run python -m eval.agent_eval --limit 25
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from eval import metrics
from eval.run import RESULTS_DIR, Config, load_testset

log = logging.getLogger("eval.agent")

DEFAULT_LIMIT = 25
PREFIXES = ("so sánh", "khác nhau", "dẫn chiếu", "tham chiếu")


def question_type(question: str) -> str:
    lowered = question.lower()
    if any(prefix in lowered for prefix in PREFIXES):
        return "comparison" if "sánh" in lowered or "khác" in lowered else "cross_reference"
    return "simple"


def graph_expansion_gain(retriever, store, testset: list[dict], top_k: int = 5, limit=None) -> dict:
    """Recall@k of plain retrieval vs. retrieval + graph neighbours."""
    from retrieval.tools import graph_query

    rows = testset[:limit] if limit else testset
    plain = 0.0
    expanded = 0.0
    counted = 0
    for item in rows:
        gold = item.get("gold") or []
        if not gold:
            continue
        counted += 1
        hits = retriever.search(item["question"], mode="hybrid_rerank", limit=top_k, prefetch=30)
        retrieved = metrics.dedupe([hit.article_id for hit in hits])
        neighbor_ids: list[str] = []
        if store is not None:
            for hit in hits:
                if not hit.article_id:
                    continue
                payload = graph_query(store, article_id=hit.article_id)
                for row in payload.get("neighbors", []) or []:
                    if row.get("article_id"):
                        neighbor_ids.append(row["article_id"])
        expanded_ids = metrics.dedupe(retrieved + neighbor_ids)
        plain += metrics.recall_at_k(retrieved, gold, k=top_k)
        expanded += metrics.recall_at_k(expanded_ids, gold, k=len(expanded_ids) or top_k)
    if not counted:
        return {"n": 0, "recall@k": 0.0, "recall@k+graph": 0.0, "gain": 0.0}
    return {
        "n": counted,
        "recall@k": plain / counted,
        "recall@k+graph": expanded / counted,
        "gain": (expanded - plain) / counted,
    }


def evaluate_agent(
    testset: list[dict],
    runner,
    judge=None,
    limit: int | None = DEFAULT_LIMIT,
    dimensions: int = 5,
) -> dict:
    rows: list[dict] = []
    questions = testset[:limit] if limit else testset
    for index, item in enumerate(questions, start=1):
        started = time.perf_counter()
        result = runner.run(item["question"])
        latency_ms = (time.perf_counter() - started) * 1000
        retrieved = metrics.dedupe([hit.article_id for hit in result.hits])
        context_ids = metrics.dedupe([hit.article_id for hit in result.contexts])
        row = {
            "question": item["question"],
            "type": question_type(item["question"]),
            "gold": item.get("gold") or [],
            "retrieved": retrieved,
            "context": context_ids,
            "chunks": [hit.chunk_id for hit in result.hits],
            "answer": result.answer,
            "reference": item.get("answer") or "",
            "route": result.route,
            "found": result.found,
            "latency_ms": round(latency_ms, 2),
            "faithful": None,
        }
        if judge is not None:
            row["faithful"] = bool(judge(item["question"], result.answer, result.contexts))
        rows.append(row)
        if index % 5 == 0:
            log.info("agent: %d/%d evaluated", index, len(questions))

    by_type: dict[str, list[dict]] = {}
    for row in rows:
        by_type.setdefault(row["type"], []).append(row)
    return {
        "rows": rows,
        "summary": metrics.summarize(rows, k=dimensions),
        "per_document": metrics.per_document(rows, k=dimensions),
        "per_type": {
            name: metrics.summarize(group, k=dimensions) for name, group in by_type.items()
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eval.agent_eval", description=__doc__)
    parser.add_argument("--testset", type=Path, default=Path("eval/testset.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    from eval.judge import build_judge
    from eval.run import build_retriever
    from llm.factory import get_llm
    from retrieval.graph import GraphStore

    testset = load_testset(args.testset)
    log.info("loaded %d questions", len(testset))

    config = Config(name="agent", collection="legal_docs", mode="hybrid_rerank")
    retriever = build_retriever(config)
    llm = get_llm()
    store = None if args.no_graph else GraphStore()

    from agent.graph import AgentRunner

    runner = AgentRunner(retriever=retriever, graph_store=store, llm=llm, top_k=5)
    judge = build_judge(llm)
    report = evaluate_agent(testset, runner, judge=judge, limit=args.limit)

    if store is not None and store.verify():
        report["graph_expansion"] = graph_expansion_gain(
            retriever, store, testset, limit=args.limit
        )
    report["config"] = "agent_qwen3_8b"
    report["ran_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "agent.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log.info("wrote %s", path)
    print(json.dumps({k: report[k] for k in ("summary", "per_type")}, ensure_ascii=False, indent=2))
    if "graph_expansion" in report:
        print("graph expansion:", json.dumps(report["graph_expansion"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
