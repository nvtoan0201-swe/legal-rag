"""Build eval/testset.jsonl.

The Zalo AI legal corpus does not contain any of our four documents (their gold
passages never map to our articles), so the test set is built from our own index:

  * grounded  - one LLM-written question per sampled article (gold = that article)
  * hard      - LLM-written cross-article questions (comparison / multi-step)
  * external  - rows from eval/external_*.jsonl and eval/zalo_raw/qa_candidates.jsonl
  * seed      - hand-written rows from eval/testset_seed.jsonl

Every row is validated against the ingested chunks; rows whose gold article ids do
not exist in the corpus are dropped and reported (plans/week-2, "Checks").
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from llm.base import Message, system, user

log = logging.getLogger("build_testset")

PROCESSED_DIR = Path("data/processed")
TESTSET_PATH = Path("eval/testset.jsonl")
SEED_PATH = Path("eval/testset_seed.jsonl")
EXTERNAL_GLOBS = ("eval/external_*.jsonl", "eval/zalo_raw/qa_candidates.jsonl")

GROUNDED_SYSTEM = (
    "Bạn là chuyên gia soạn đề thi pháp luật Việt Nam. "
    "Nhiệm vụ: đọc một điều luật và viết MỘT câu hỏi tình huống tự nhiên bằng tiếng Việt "
    "mà câu trả lời nằm hoàn toàn trong điều luật đó, kèm câu trả lời ngắn (1-3 câu). "
    "Không nhắc số điều trong câu hỏi. "
    'Chỉ trả về JSON: {"question": "...", "answer": "..."}'
)

HARD_SYSTEM = (
    "Bạn là chuyên gia soạn đề thi pháp luật Việt Nam. "
    "Nhiệm vụ: đọc HAI điều luật và viết MỘT câu hỏi khó bằng tiếng Việt cần kết hợp cả hai "
    "(so sánh, phân biệt, hoặc tình huống nhiều bước), kèm câu trả lời ngắn (1-4 câu). "
    "Không nhắc số điều trong câu hỏi. "
    'Chỉ trả về JSON: {"question": "...", "answer": "..."}'
)


# ---------------------------------------------------------------- corpus
def load_articles(processed_dir: Path = PROCESSED_DIR) -> dict[str, list[dict]]:
    """Group structure chunks into one record per article, keyed by doc_id."""
    by_doc: dict[str, dict[str, dict]] = defaultdict(dict)
    for path in sorted(processed_dir.glob("*.structure.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                article_id = payload.get("article_id")
                if not article_id:
                    continue
                record = by_doc[payload["doc_id"]].setdefault(
                    article_id,
                    {
                        "article_id": article_id,
                        "doc_id": payload["doc_id"],
                        "doc_title": payload.get("doc_title", ""),
                        "article_no": payload.get("article_no", 0),
                        "article_title": payload.get("article_title", ""),
                        "chapter": payload.get("chapter", ""),
                        "text": "",
                    },
                )
                record["text"] = (record["text"] + "\n" + payload.get("text", "")).strip()
    return {doc_id: list(articles.values()) for doc_id, articles in by_doc.items()}


def sample_articles(
    by_doc: dict[str, list[dict]],
    per_doc: int,
    min_chars: int = 220,
    seed: int = 13,
) -> list[dict]:
    """Deterministic sample of substantial articles, one document at a time."""
    rng = random.Random(seed)
    picked: list[dict] = []
    for doc_id in sorted(by_doc):
        candidates = [item for item in by_doc[doc_id] if len(item["text"]) >= min_chars]
        rng.shuffle(candidates)
        picked.extend(candidates[:per_doc])
    return picked


def article_context(article: dict, limit: int = 1600) -> str:
    text = article["text"][:limit]
    return (
        f"Văn bản: {article['doc_title']}\n"
        f"Điều {article['article_no']}. {article['article_title']}\n\n{text}"
    )


# ---------------------------------------------------------------- parsing
def parse_json_object(text: str) -> dict:
    """Extract the first JSON object from an LLM answer (tolerates code fences)."""
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^'''+.*?'''+$", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"^\x60\x60\x60(?:json)?|^\x60\x60\x60", "", cleaned, flags=re.MULTILINE)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def validate_row(row: dict, known_ids: set[str]) -> str | None:
    """Return a rejection reason, or None when the row is usable."""
    question = (row.get("question") or "").strip()
    answer = (row.get("answer") or "").strip()
    gold = row.get("article_id") or row.get("gold") or []
    if isinstance(gold, str):
        gold = [gold]
    gold = [str(item) for item in gold if item]
    if not (15 <= len(question) <= 400):
        return "question length"
    if not question.endswith("?"):
        return "question must end with ?"
    if len(answer) < 10:
        return "answer too short"
    if not gold:
        return "no gold article"
    missing = [item for item in gold if item not in known_ids]
    if missing:
        return f"unknown gold article {missing}"
    return None


# ---------------------------------------------------------------- generation
def _ask(llm, messages: list[Message], retries: int = 2) -> dict:
    for attempt in range(retries + 1):
        try:
            response = llm.chat(messages, temperature=0.3, max_tokens=500)
        except Exception as exc:  # pragma: no cover - depends on the LLM server
            log.warning("LLM call failed (attempt %d): %s", attempt + 1, exc)
            continue
        parsed = parse_json_object(response)
        if parsed.get("question") and parsed.get("answer"):
            return parsed
        log.debug("unparseable LLM output: %s", response[:200])
    return {}


def checkpoint_key(row: dict) -> tuple[str, ...]:
    gold = row.get("article_id")
    gold = gold if isinstance(gold, list) else [gold]
    return tuple(sorted(str(item) for item in gold if item))


def load_checkpoint(path: Path | None) -> list[dict]:
    """Reuse questions generated by a previous run (the local LLM is slow)."""
    if not path or not Path(path).exists():
        return []
    rows: list[dict] = []
    for line in Path(path).open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def append_checkpoint(path: Path | None, row: dict) -> None:
    if not path:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_grounded(
    articles: list[dict],
    llm,
    source: str = "grounded",
    checkpoint: Path | None = None,
    covered: set = frozenset(),
) -> list[dict]:
    rows: list[dict] = []
    for index, article in enumerate(articles, start=1):
        if (article["article_id"],) in covered:
            continue
        prompt = (
            "Điều luật:\n\n"
            + article_context(article)
            + "\n\nHãy viết câu hỏi và câu trả lời theo đúng định dạng JSON."
        )
        parsed = _ask(llm, [system(GROUNDED_SYSTEM), user(prompt)])
        if not parsed:
            log.warning("no question generated for %s", article["article_id"])
            continue
        row = {
            "question": parsed["question"].strip(),
            "article_id": article["article_id"],
            "answer": parsed["answer"].strip(),
            "source": source,
        }
        rows.append(row)
        append_checkpoint(checkpoint, row)
        if index % 10 == 0:
            log.info("grounded: %d/%d", index, len(articles))
    return rows


def generate_hard(
    articles: list[dict],
    llm,
    count: int,
    seed: int = 13,
    checkpoint: Path | None = None,
    covered: set = frozenset(),
) -> list[dict]:
    rng = random.Random(seed + 1)
    rows: list[dict] = []
    pairs: list[tuple[dict, dict]] = []
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for article in articles:
        by_doc[article["doc_id"]].append(article)
    doc_ids = sorted(by_doc)
    for _ in range(count * 3):
        if len(doc_ids) >= 2 and rng.random() < 0.4:
            first_doc, second_doc = rng.sample(doc_ids, 2)
        else:
            first_doc = second_doc = rng.choice(doc_ids)
        first = rng.choice(by_doc[first_doc])
        second = rng.choice(by_doc[second_doc])
        if first["article_id"] != second["article_id"]:
            pairs.append((first, second))
        if len(pairs) >= count:
            break
    for first, second in pairs:
        if tuple(sorted((first["article_id"], second["article_id"]))) in covered:
            continue
        prompt = (
            article_context(first, limit=1100)
            + "\n\n---\n\n"
            + article_context(second, limit=1100)
            + "\n\nHãy viết MỘT câu hỏi khó cần cả hai điều luật và câu trả lời, "
            "theo đúng định dạng JSON."
        )
        parsed = _ask(llm, [system(HARD_SYSTEM), user(prompt)])
        if not parsed:
            log.warning("no hard question for %s + %s", first["article_id"], second["article_id"])
            continue
        gold = [first["article_id"], second["article_id"]]
        row = {
            "question": parsed["question"].strip(),
            "article_id": gold,
            "answer": parsed["answer"].strip(),
            "source": "hard",
        }
        rows.append(row)
        append_checkpoint(checkpoint, row)
    return rows


# ---------------------------------------------------------------- merging
def load_external(globs: tuple[str, ...] = EXTERNAL_GLOBS) -> list[dict]:
    rows: list[dict] = []
    for pattern in globs:
        for path in sorted(Path(".").glob(pattern)):
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    row.setdefault("source", path.stem)
                    rows.append(row)
            log.info("loaded %d external rows from %s", len(rows), path)
    return rows


def sample_external(rows: list[dict], per_doc: int = 25, seed: int = 13) -> list[dict]:
    """Stratified sample of external questions, capped per gold document."""
    rng = random.Random(seed + 3)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        gold = row.get("article_id") or row.get("gold")
        gold = gold[0] if isinstance(gold, list) and gold else gold
        doc_id = str(gold).split("_d")[0] if gold else "unknown"
        groups[doc_id].append(row)
    picked: list[dict] = []
    for doc_id in sorted(groups):
        items = list(groups[doc_id])
        rng.shuffle(items)
        picked.extend(items[:per_doc])
    return picked


def dedupe_rows(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for row in rows:
        key = re.sub(r"\W+", " ", (row.get("question") or "").lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def build(
    grounded_per_doc: int = 35,
    hard: int = 20,
    external_per_doc: int = 25,
    seed: int = 13,
    use_llm: bool = True,
    llm=None,
    processed_dir: Path = PROCESSED_DIR,
    checkpoint: Path | None = None,
) -> dict:
    by_doc = load_articles(processed_dir)
    known_ids = {article["article_id"] for items in by_doc.values() for article in items}
    log.info("corpus: %d documents, %d articles", len(by_doc), len(known_ids))

    rows: list[dict] = []
    for path in (SEED_PATH,):
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                seed_rows = [json.loads(line) for line in handle if line.strip()]
            for row in seed_rows:
                row.setdefault("source", "seed")
            rows.extend(seed_rows)
            log.info("seed rows: %d", len(seed_rows))

    external = sample_external(load_external(), per_doc=external_per_doc, seed=seed)
    log.info("external rows sampled: %d", len(external))
    rows.extend(external)

    cached = load_checkpoint(checkpoint)
    if cached:
        log.info("reusing %d cached generated questions", len(cached))
        rows.extend(cached)
    covered = {checkpoint_key(row) for row in cached}

    if use_llm:
        if grounded_per_doc > 0:
            sample = sample_articles(by_doc, grounded_per_doc, seed=seed)
            log.info("generating %d grounded questions", len(sample))
            rows.extend(generate_grounded(sample, llm, checkpoint=checkpoint, covered=covered))
        if hard > 0:
            hard_sample_size = max(2, grounded_per_doc // 3 or 2)
            hard_articles = sample_articles(by_doc, hard_sample_size, seed=seed + 7)
            log.info("generating %d hard questions", hard)
            rows.extend(
                generate_hard(
                    hard_articles,
                    llm,
                    hard,
                    seed=seed,
                    checkpoint=checkpoint,
                    covered=covered,
                )
            )

    accepted: list[dict] = []
    rejected: list[tuple[str, str]] = []
    for row in dedupe_rows(rows):
        reason = validate_row(row, known_ids)
        if reason:
            rejected.append((str(row.get("question", ""))[:60], reason))
            continue
        gold = row.get("article_id")
        accepted.append(
            {
                "question": row["question"].strip(),
                "article_id": gold if isinstance(gold, list) else str(gold),
                "answer": (row.get("answer") or "").strip(),
                "source": row.get("source", "unknown"),
            }
        )

    random.Random(seed).shuffle(accepted)
    by_source = defaultdict(int)
    for row in accepted:
        by_source[row["source"]] += 1
    return {
        "rows": accepted,
        "rejected": rejected,
        "by_source": dict(by_source),
        "known_ids": known_ids,
    }


def verify_testset(path: Path = TESTSET_PATH, processed_dir: Path = PROCESSED_DIR) -> dict:
    by_doc = load_articles(processed_dir)
    known_ids = {article["article_id"] for items in by_doc.values() for article in items}
    bad: list[dict] = []
    total = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            total += 1
            row = json.loads(line)
            gold = row.get("article_id")
            gold = [gold] if isinstance(gold, str) else (gold or [])
            missing = [item for item in gold if item not in known_ids]
            if missing:
                bad.append({"question": row.get("question", "")[:60], "missing": missing})
    return {"total": total, "bad": bad, "ok": total - len(bad)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="build_testset", description=__doc__)
    parser.add_argument("--out", type=Path, default=TESTSET_PATH)
    parser.add_argument("--grounded-per-doc", type=int, default=35)
    parser.add_argument("--hard", type=int, default=20)
    parser.add_argument("--external-per-doc", type=int, default=25)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=Path("eval/generated_questions.jsonl"))
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.verify_only:
        report = verify_testset(args.out, args.processed_dir)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1 if report["bad"] else 0

    llm = None
    if not args.no_llm:
        from llm.factory import get_llm

        llm = get_llm()
    result = build(
        grounded_per_doc=args.grounded_per_doc,
        hard=args.hard,
        external_per_doc=args.external_per_doc,
        seed=args.seed,
        use_llm=not args.no_llm,
        llm=llm,
        processed_dir=args.processed_dir,
        checkpoint=args.checkpoint,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in result["rows"]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "written": len(result["rows"]),
                "by_source": result["by_source"],
                "rejected": len(result["rejected"]),
                "rejected_examples": result["rejected"][:10],
                "out": str(args.out),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
