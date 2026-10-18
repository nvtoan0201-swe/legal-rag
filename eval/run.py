"""Run the evaluation test set against named retrieval configs.

Each config maps to a row of the ablation table in plans/week-3-retrieval.md:
collection (fixed-512 baseline vs structure vs article) x search mode
(dense / sparse / hybrid / hybrid+rerank). Results are written to
eval/results/<config>.json and printed as a table.

Usage:
    uv run python -m eval --config all
    uv run python -m eval --config structure_hybrid_rerank --limit 20 --judge
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from eval import metrics

log = logging.getLogger("eval")

TESTSET_PATH = Path("eval/testset.jsonl")
RESULTS_DIR = Path("eval/results")

DEFAULT_TOP_K = 5


@dataclass(frozen=True)
class Config:
    name: str
    collection: str
    mode: str
    top_k: int = DEFAULT_TOP_K
    prefetch: int = 30
    rerank: bool = True
    description: str = ""


CONFIGS: dict[str, Config] = {
    config.name: config
    for config in (
        Config(
            name="baseline_fixed512_dense",
            collection="legal_docs_baseline",
            mode="dense",
            rerank=False,
            description="week-2 baseline: fixed 512-token chunks, dense only",
        ),
        Config(
            name="structure_dense",
            collection="legal_docs",
            mode="dense",
            rerank=False,
            description="structure chunks, dense only",
        ),
        Config(
            name="structure_hybrid",
            collection="legal_docs",
            mode="hybrid",
            rerank=False,
            description="structure chunks, dense+sparse RRF fusion",
        ),
        Config(
            name="structure_hybrid_rerank",
            collection="legal_docs",
            mode="hybrid_rerank",
            rerank=True,
            description="structure chunks, hybrid + bge-reranker-v2-m3",
        ),
        Config(
            name="article_hybrid_rerank",
            collection="legal_docs_article",
            mode="hybrid_rerank",
            rerank=True,
            description="article-level chunks, hybrid + rerank",
        ),
    )
}


def load_testset(path: Path = TESTSET_PATH) -> list[dict]:
    """Read a JSONL test set; article_id may be a string or a list."""
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            gold = row.get("article_id", row.get("gold"))
            if isinstance(gold, str):
                gold = [gold] if gold else []
            row["gold"] = [str(item) for item in (gold or [])]
            if not row.get("question"):
                raise ValueError(f"{path}:{line_no}: missing question")
            rows.append(row)
    return rows


def evaluate(
    testset: list[dict],
    retriever,
    config: Config,
    answer_fn: Callable[[str, list], str] | None = None,
    judge: Callable[[str, str, list], bool] | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Retrieve (and optionally answer) for every question; return row records."""
    from retrieval.search import Retriever

    assert isinstance(retriever, Retriever) or hasattr(retriever, "search")
    rows: list[dict] = []
    questions = testset[:limit] if limit else testset
    for index, item in enumerate(questions, start=1):
        question = item["question"]
        started = time.perf_counter()
        hits = retriever.search(
            question,
            mode=config.mode,
            limit=config.top_k,
            prefetch=config.prefetch,
        )
        answer = answer_fn(question, hits) if answer_fn else None
        latency_ms = (time.perf_counter() - started) * 1000
        row = {
            "question": question,
            "gold": item["gold"],
            "retrieved": metrics.dedupe([hit.article_id for hit in hits]),
            "chunks": [hit.chunk_id for hit in hits],
            "latency_ms": round(latency_ms, 2),
            "answer": answer,
            "reference": item.get("answer") or "",
            "faithful": None,
        }
        if judge is not None and answer is not None:
            row["faithful"] = bool(judge(question, answer, hits))
        rows.append(row)
        if index % 25 == 0:
            log.info("%d/%d evaluated", index, len(questions))
    return rows


def run_config(
    name: str,
    testset: list[dict],
    retriever,
    answer_fn=None,
    judge=None,
    limit: int | None = None,
    out_dir: Path = RESULTS_DIR,
) -> dict:
    config = (
        CONFIGS[name]
        if name in CONFIGS
        else Config(name=name, collection=name, mode="hybrid_rerank")
    )
    log.info(
        "running config %s (collection=%s, mode=%s)", config.name, config.collection, config.mode
    )
    if hasattr(retriever, "collection"):
        retriever.collection = config.collection
    rows = evaluate(testset, retriever, config, answer_fn=answer_fn, judge=judge, limit=limit)
    summary = metrics.summarize(rows, k=config.top_k)
    result = {
        "config": config.name,
        "collection": config.collection,
        "mode": config.mode,
        "top_k": config.top_k,
        "ran_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": summary,
        "per_document": metrics.per_document(rows, k=config.top_k),
        "rows": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{config.name}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log.info("wrote %s", out_path)
    return result


def format_table(results: list[dict]) -> str:
    header = (
        f"{'config':<28} {'Recall@5':>9} {'MRR':>7} {'Faithful':>9} "
        f"{'F1':>7} {'p50ms':>8} {'p95ms':>8} {'n':>4}"
    )
    lines = [header, "-" * len(header)]
    for result in results:
        summary = result["summary"]
        faithful = summary.get("faithfulness")
        f1 = summary.get("answer_f1")
        lines.append(
            f"{result['config']:<28} {summary['recall@5']:>9.3f} {summary['mrr']:>7.3f} "
            f"{(f'{faithful:.3f}' if faithful is not None else '-'):>9} "
            f"{(f'{f1:.3f}' if f1 is not None else '-'):>7} "
            f"{summary['p50_ms']:>8.1f} {summary['p95_ms']:>8.1f} {summary['n']:>4}"
        )
    return "\n".join(lines)


def load_results(out_dir: Path = RESULTS_DIR) -> list[dict]:
    results: list[dict] = []
    for path in sorted(out_dir.glob("*.json")):
        results.append(json.loads(path.read_text(encoding="utf-8")))
    return results


def build_retriever(config: Config, settings=None, embedder=None, reranker=None):
    from app.settings import get_settings
    from ingestion.load import get_client
    from retrieval.search import Retriever

    settings = settings or get_settings()
    client = get_client(settings.qdrant_url, settings.qdrant_api_key)
    return Retriever(
        client=client,
        collection=config.collection,
        embedder=embedder,
        reranker=reranker if config.rerank else None,
        dense_threshold=settings.dense_threshold,
        rerank_threshold=settings.rerank_threshold,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eval", description=__doc__)
    parser.add_argument("--config", default="structure_hybrid_rerank", help="config name or 'all'")
    parser.add_argument("--testset", type=Path, default=TESTSET_PATH)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--judge", action="store_true", help="score faithfulness with the LLM judge"
    )
    parser.add_argument(
        "--table", action="store_true", help="print the ablation table from saved results"
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.table:
        results = load_results(args.out_dir)
        print(format_table(results) if results else "no results yet")
        return 0

    testset = load_testset(args.testset)
    log.info("loaded %d questions from %s", len(testset), args.testset)
    names = list(CONFIGS) if args.config == "all" else [args.config]

    embedder = None
    reranker = None
    if "baseline" not in args.config and any(
        CONFIGS.get(name, Config("", "", "")).mode != "dense" for name in names
    ):
        from ingestion.embed import get_embedder

        embedder = get_embedder()
    if any(CONFIGS.get(name, Config("", "", "", rerank=False)).rerank for name in names):
        from retrieval.search import get_reranker

        reranker = get_reranker()

    judge = None
    if args.judge:
        from eval.judge import build_judge

        judge = build_judge()

    results = []
    for name in names:
        config = CONFIGS.get(name, Config(name=name, collection=name, mode="hybrid_rerank"))
        retriever = build_retriever(config, embedder=embedder, reranker=reranker)
        results.append(
            run_config(
                name, testset, retriever, judge=judge, limit=args.limit, out_dir=args.out_dir
            )
        )

    print(format_table(results))
    stats = [row["latency_ms"] for result in results for row in result["rows"]]
    if stats:
        print(f"\nlatency: mean {statistics.mean(stats):.1f} ms over {len(stats)} questions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
