"""Ingestion CLI: parse raw documents, chunk, embed, and load Qdrant.

Usage:
    uv run python -m ingestion                 # full pipeline, structure chunks
    uv run python -m ingestion --granularity all --recreate
    uv run python -m ingestion --skip-parse --limit 20
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from ingestion import chunk as chunk_mod
from ingestion import load as load_mod
from ingestion import parse as parse_mod
from ingestion.registry import PROCESSED_DIR, RAW_DIR, source_files

log = logging.getLogger("ingestion")

DEFAULT_COLLECTION = load_mod.DEFAULT_COLLECTION
GRANULARITIES = ("structure", "article", "fixed512")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ingestion", description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--granularity", choices=(*GRANULARITIES, "all"), default="structure")
    parser.add_argument("--collection", default=None, help="override collection name")
    parser.add_argument("--recreate", action="store_true", help="drop and recreate collections")
    parser.add_argument("--skip-parse", action="store_true", help="reuse data/processed text")
    parser.add_argument("--skip-embed", action="store_true", help="chunk only, do not embed/load")
    parser.add_argument("--docs", default=None, help="comma-separated doc_ids to process")
    parser.add_argument("--limit", type=int, default=None, help="max chunks per document (debug)")
    parser.add_argument("--embed-backend", default=None, help="bge-m3 (default) or hashing")
    parser.add_argument(
        "--skip-if-loaded",
        action="store_true",
        help="exit early when the target collections already hold points (docker compose)",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def _qdrant_client():
    from app.settings import get_settings

    settings = get_settings()
    return load_mod.get_client(settings.qdrant_url, settings.qdrant_api_key, settings.qdrant_path)


def _write_jsonl(path: Path, chunks) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in chunks:
            handle.write(json.dumps(item.payload(), ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _chunks_for(text: str, spec, granularity: str, source_file: str):
    if granularity == "structure":
        return chunk_mod.chunk_document(text, spec, source_file)
    if granularity == "article":
        return chunk_mod.article_chunks(chunk_mod.chunk_document(text, spec, source_file), spec)
    return chunk_mod.fixed_token_chunks(text, spec, source_file)


def _chunk_from_payload(payload: dict) -> chunk_mod.Chunk:
    return chunk_mod.Chunk(**payload)


def run(args: argparse.Namespace) -> int:
    started = time.time()
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(args.docs.split(",")) if args.docs else None
    specs = [
        (spec, path)
        for spec, path in source_files(args.raw_dir)
        if not wanted or spec.doc_id in wanted
    ]
    if not specs:
        log.error(
            "no recognised files in %s (expected names matching the registry aliases)", args.raw_dir
        )
        return 1

    granularities = GRANULARITIES if args.granularity == "all" else (args.granularity,)

    if args.skip_if_loaded and not args.recreate:
        try:
            client = _qdrant_client()
            loaded = [
                collection
                for granularity in granularities
                if (collection := (args.collection or load_mod.collection_for(granularity)))
                and client.collection_exists(collection)
                and load_mod.count_points(client, collection) > 0
            ]
            if len(loaded) == len(granularities):
                log.info("collections already loaded (%s); nothing to do", ", ".join(loaded))
                return 0
        except Exception as exc:
            log.warning("cannot check existing collections (%s); running ingestion", exc)
    manifest_path = args.processed_dir / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    manifest.setdefault("documents", {})

    # 1) parse
    texts: dict[str, str] = {}
    for spec, path in specs:
        text_path = args.processed_dir / f"{spec.doc_id}.txt"
        if args.skip_parse and text_path.exists():
            texts[spec.doc_id] = text_path.read_text(encoding="utf-8")
            log.info("%s: reusing %s", spec.doc_id, text_path)
            continue
        parsed = parse_mod.parse_document(path, spec)
        text_path.write_text(parsed.text, encoding="utf-8")
        texts[spec.doc_id] = parsed.text
        entry = manifest["documents"].setdefault(spec.doc_id, {})
        entry.update(
            {
                "title": spec.title,
                "number": spec.number,
                "source_file": str(path),
                "parser": parsed.parser,
                "max_article": parsed.max_article,
                "heading_count": parsed.heading_count,
                "chars": len(parsed.text),
                "warnings": parsed.warnings,
            }
        )
        for warning in parsed.warnings:
            log.warning("%s: %s", spec.doc_id, warning)
        log.info(
            "%s: parsed with %s, %d chars, max article %d (%d headings)",
            spec.doc_id,
            parsed.parser,
            len(parsed.text),
            parsed.max_article,
            parsed.heading_count,
        )

    # 2) chunk
    chunked: dict[str, list[chunk_mod.Chunk]] = {}
    for granularity in granularities:
        for spec, _path in specs:
            text = texts.get(spec.doc_id, "")
            if not text:
                continue
            chunks = _chunks_for(text, spec, granularity, str(_path))
            if args.limit:
                chunks = chunks[: args.limit]
            key = f"{spec.doc_id}::{granularity}"
            chunked[key] = chunks
            _write_jsonl(args.processed_dir / f"{spec.doc_id}.{granularity}.jsonl", chunks)
            stats = chunk_mod.chunk_stats(chunks)
            entry = manifest["documents"].setdefault(spec.doc_id, {})
            entry.setdefault("chunks", {})[granularity] = stats
            log.info(
                "%s: %s chunks -> %s (articles=%d, clauses=%d)",
                spec.doc_id,
                granularity,
                stats["chunks"],
                stats["articles"],
                stats["clause_chunks"],
            )

    manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if args.skip_embed:
        log.info("chunk-only run finished in %.1fs", time.time() - started)
        return 0

    # 3) embed + load
    from ingestion.embed import get_embedder

    embedder = get_embedder(args.embed_backend)
    client = _qdrant_client()
    total = 0
    for granularity in granularities:
        collection = args.collection or load_mod.collection_for(granularity)
        load_mod.ensure_collection(
            client, collection, dense_dim=getattr(embedder, "dim", 1024), recreate=args.recreate
        )
        for spec, _path in specs:
            chunks = chunked.get(f"{spec.doc_id}::{granularity}", [])
            if not chunks:
                continue
            embedded: list[tuple[chunk_mod.Chunk, object]] = []
            batch_size = 16
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start : start + batch_size]
                vectors = embedder.encode([item.embed_text for item in batch])
                embedded.extend(zip(batch, vectors, strict=True))
            loaded = load_mod.upsert_points(client, collection, embedded)
            total += loaded
            log.info("%s: loaded %d points into %s", spec.doc_id, loaded, collection)
        log.info(
            "collection %s now holds %d points",
            collection,
            load_mod.count_points(client, collection),
        )

    log.info("ingestion finished: %d points in %.1fs", total, time.time() - started)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return run(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
