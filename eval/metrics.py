"""Retrieval and answer metrics for the ablation table.

Definitions follow eval/README.md:
  Recall@5  - fraction of questions with at least one gold article in the top 5
  MRR       - mean of 1 / rank of the first gold article (0 when absent)
  F1        - token overlap between generated and reference answer
  p50/p95   - latency percentiles
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "").lower()
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def dedupe(items: Iterable[str]) -> list[str]:
    """Keep first occurrence order (an article retrieved twice counts once)."""
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def first_gold_rank(ranked: Sequence[str], gold: Sequence[str]) -> int | None:
    gold_set = {item for item in gold if item}
    for rank, item in enumerate(ranked, start=1):
        if item in gold_set:
            return rank
    return None


def recall_at_k(ranked: Sequence[str], gold: Sequence[str], k: int = 5) -> float:
    """1.0 when a gold article is inside the top k, else 0.0."""
    if not gold:
        return 0.0
    return 1.0 if first_gold_rank(list(ranked)[:k], gold) is not None else 0.0


def reciprocal_rank(ranked: Sequence[str], gold: Sequence[str]) -> float:
    rank = first_gold_rank(ranked, gold)
    return 1.0 / rank if rank else 0.0


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolation percentile (p in 0..100); 0.0 for empty input."""
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (p / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style token F1 on normalised Vietnamese text."""
    predicted = normalize_text(prediction).split()
    expected = normalize_text(reference).split()
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    from collections import Counter

    common = Counter(predicted) & Counter(expected)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def _faithfulness_rate(rows: Sequence[dict]) -> float | None:
    judged = [row["faithful"] for row in rows if row.get("faithful") is not None]
    if not judged:
        return None
    return sum(1.0 for value in judged if value) / len(judged)


def _answer_f1(rows: Sequence[dict]) -> float | None:
    """Token F1 over answered rows only; None when no answer was generated."""
    scored = [
        token_f1(row["answer"], row["reference"])
        for row in rows
        if row.get("reference") and row.get("answer")
    ]
    if not scored:
        return None
    return sum(scored) / len(scored)


def summarize(rows: Sequence[dict], k: int = 5) -> dict:
    """Aggregate eval rows: each row needs gold, retrieved, and optionally latency."""
    scored = [row for row in rows if row.get("gold")]
    latencies = [float(row["latency_ms"]) for row in rows if row.get("latency_ms") is not None]
    result = {
        "n": len(rows),
        "n_scored": len(scored),
        "recall@5": sum(recall_at_k(row["retrieved"], row["gold"], 5) for row in scored)
        / len(scored)
        if scored
        else 0.0,
        "recall@10": sum(recall_at_k(row["retrieved"], row["gold"], 10) for row in scored)
        / len(scored)
        if scored
        else 0.0,
        "mrr": sum(reciprocal_rank(row["retrieved"], row["gold"]) for row in scored) / len(scored)
        if scored
        else 0.0,
        "faithfulness": _faithfulness_rate(rows),
        "answer_f1": _answer_f1(rows),
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
    }
    if k != 5:
        result[f"recall@{k}"] = (
            sum(recall_at_k(row["retrieved"], row["gold"], k) for row in scored) / len(scored)
            if scored
            else 0.0
        )
    return result


def per_document(rows: Sequence[dict], k: int = 5) -> dict[str, dict]:
    """Break the summary down by the gold document id prefix (e.g. blld2019)."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        for gold_id in row.get("gold", []) or [""]:
            doc_id = str(gold_id).split("_d")[0] if gold_id else "unknown"
            groups.setdefault(doc_id, []).append(row)
    return {doc_id: summarize(group, k=k) for doc_id, group in sorted(groups.items())}
