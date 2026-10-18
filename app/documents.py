"""Uploaded-document processing: status store + parse/chunk/embed/load pipeline.

The store is a small JSON file guarded by a lock because Starlette runs sync
background tasks in a threadpool. Processing never raises: failures are recorded
on the document record so GET /documents/{id}/status can report them.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from ingestion import chunk as chunk_mod
from ingestion import load as load_mod
from ingestion import parse as parse_mod
from ingestion.registry import DocSpec

log = logging.getLogger("app.documents")

UPLOAD_DIR = Path("data/uploads")
STORE_PATH = Path("data/processed/uploads.json")
PROCESSED_DIR = Path("data/processed")


def slugify(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_")
    return cleaned or "document"


def document_spec(doc_id: str, title: str = "", doc_type: str = "law") -> DocSpec:
    return DocSpec(
        doc_id=doc_id,
        title=title or doc_id,
        short_title=title or doc_id,
        doc_type=doc_type,
        number="",
        issued_date="",
        effective_date="",
        aliases=(doc_id,),
        expected_articles=None,
    )


@dataclass
class DocumentRecord:
    doc_id: str
    filename: str
    title: str = ""
    status: str = "pending"  # pending | processing | done | error
    message: str = ""
    chunks: int = 0
    articles: int = 0
    path: str = ""
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class DocumentStore:
    def __init__(self, path: Path = STORE_PATH) -> None:
        self.path = Path(path)
        self._records: dict[str, DocumentRecord] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("cannot read %s: %s", self.path, exc)
            return
        for doc_id, row in (payload.get("documents") or {}).items():
            known = DocumentRecord.__dataclass_fields__
            self._records[doc_id] = DocumentRecord(**{k: v for k, v in row.items() if k in known})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "documents": {doc_id: record.to_dict() for doc_id, record in self._records.items()}
        }
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def put(self, record: DocumentRecord) -> DocumentRecord:
        with self._lock:
            self._records[record.doc_id] = record
            self.save()
        return record

    def get(self, doc_id: str) -> DocumentRecord | None:
        with self._lock:
            return self._records.get(doc_id)

    def list(self) -> list[DocumentRecord]:
        with self._lock:
            return sorted(self._records.values(), key=lambda record: record.doc_id)

    def update(self, doc_id: str, **fields) -> DocumentRecord | None:
        with self._lock:
            record = self._records.get(doc_id)
            if record is None:
                return None
            for key, value in fields.items():
                if hasattr(record, key):
                    setattr(record, key, value)
            record.updated_at = time.strftime("%Y-%m-%dT%H:%M:%S")
            self.save()
            return record


def process_document(
    path: Path,
    doc_id: str,
    store: DocumentStore | None = None,
    title: str = "",
    doc_type: str = "law",
    settings=None,
    client=None,
    embedder=None,
    collection: str | None = None,
    load_neo4j: bool = True,
    processed_dir: Path = PROCESSED_DIR,
) -> DocumentRecord:
    """Parse, chunk, embed, and load one uploaded document into Qdrant."""
    store = store or DocumentStore()
    record = store.get(doc_id) or store.put(
        DocumentRecord(doc_id=doc_id, filename=path.name, title=title or doc_id, path=str(path))
    )
    store.update(doc_id, status="processing", message="")
    try:
        spec = document_spec(doc_id, title or record.title or doc_id, doc_type)
        parsed = parse_mod.parse_document(Path(path), spec)
        processed_dir.mkdir(parents=True, exist_ok=True)
        (processed_dir / f"{doc_id}.txt").write_text(parsed.text, encoding="utf-8")
        chunks = chunk_mod.chunk_document(parsed.text, spec, str(path))
        with (processed_dir / f"{doc_id}.structure.jsonl").open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(json.dumps(chunk.payload(), ensure_ascii=False) + "\n")

        if settings is None:
            from app.settings import get_settings

            settings = get_settings()
        if embedder is None:
            from ingestion.embed import get_embedder

            embedder = get_embedder(settings.embed_backend, model_name=settings.embed_model)
        if client is None:
            client = load_mod.get_client(settings.qdrant_url, settings.qdrant_api_key)
        collection = collection or settings.collection
        load_mod.ensure_collection(client, collection, dense_dim=getattr(embedder, "dim", 1024))

        embedded = []
        for start in range(0, len(chunks), 16):
            batch = chunks[start : start + 16]
            vectors = embedder.encode([item.embed_text for item in batch])
            embedded.extend(zip(batch, vectors, strict=True))
        loaded = load_mod.upsert_points(client, collection, embedded)

        if load_neo4j:
            try:
                from ingestion import graph as graph_mod

                driver = graph_mod.get_driver()
                graph_mod.ensure_constraints(driver)
                graph_mod.load_graph(driver, [spec], {spec.doc_id: chunks})
                driver.close()
            except Exception as exc:  # pragma: no cover - Neo4j optional
                log.warning("graph load skipped for %s: %s", doc_id, exc)

        return store.update(
            doc_id,
            status="done",
            chunks=loaded,
            articles=chunk_mod.chunk_stats(chunks)["articles"],
            message=f"parser={parsed.parser}, max_article={parsed.max_article}",
            path=str(path),
        )
    except Exception as exc:  # pragma: no cover - depends on runtime services
        log.exception("processing failed for %s", doc_id)
        return store.update(doc_id, status="error", message=str(exc))
