"""FastAPI application: health, cited RAG query, document endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from app.documents import UPLOAD_DIR, DocumentRecord, DocumentStore, process_document, slugify
from app.settings import Settings, get_settings
from rag.answer import answer_from_hits, retrieve
from retrieval.search import Retriever, get_reranker

log = logging.getLogger("app")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, description="Câu hỏi tiếng Việt")
    top_k: int = Field(default=5, ge=1, le=50)
    mode: Literal["dense", "sparse", "hybrid", "hybrid_rerank"] = "hybrid_rerank"
    prefetch: int = Field(default=30, ge=5, le=200)
    threshold: float | None = Field(default=None, description="Ghi đè ngưỡng not-found")
    include_text: bool = True
    answer: bool = Field(default=True, description="Sinh câu trả lời có trích dẫn từ LLM")


class QueryResponse(BaseModel):
    question: str
    found: bool
    mode: str
    answer: str | None = None
    cited: list[str] = Field(default_factory=list, description="chunk_id được trích dẫn")
    results: list[dict] = Field(default_factory=list)
    elapsed_ms: float = 0.0


class HealthResponse(BaseModel):
    status: str
    qdrant: dict
    llm: dict
    embed_backend: str
    rerank_backend: str


def build_retriever(settings: Settings) -> Retriever:
    from ingestion.embed import get_embedder
    from ingestion.load import get_client

    return Retriever(
        client=get_client(settings.qdrant_url, settings.qdrant_api_key, settings.qdrant_path),
        collection=settings.collection,
        embedder=get_embedder(settings.embed_backend, model_name=settings.embed_model),
        reranker=get_reranker(settings.rerank_backend, model_name=settings.rerank_model),
        dense_threshold=settings.dense_threshold,
        rerank_threshold=settings.rerank_threshold,
    )


def get_retriever(request: Request) -> Retriever:
    app = request.app
    if getattr(app.state, "retriever", None) is None:
        app.state.retriever = build_retriever(app.state.settings)
    return app.state.retriever


def get_llm_client(request: Request):
    app = request.app
    if getattr(app.state, "llm", None) is None:
        from llm.factory import get_llm

        app.state.llm = get_llm(app.state.settings)
    return app.state.llm


def get_documents(request: Request) -> DocumentStore:
    return request.app.state.documents


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = get_settings()
    app.state.retriever = None
    app.state.llm = None
    app.state.documents = DocumentStore()
    log.info("legal-rag API starting")
    yield
    log.info("legal-rag API stopping")


app = FastAPI(title="legal-rag", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    settings: Settings = request.app.state.settings
    qdrant_info: dict = {"url": settings.qdrant_url, "reachable": False}
    try:
        from ingestion.load import count_points, get_client

        client = get_client(settings.qdrant_url, settings.qdrant_api_key, settings.qdrant_path)
        qdrant_info["reachable"] = True
        qdrant_info["collection"] = settings.collection
        qdrant_info["points"] = count_points(client, settings.collection)
    except Exception as exc:  # pragma: no cover - depends on runtime services
        qdrant_info["error"] = str(exc)

    llm_info: dict = {"provider": settings.llm_provider}
    try:
        from llm.factory import get_llm

        llm = get_llm(settings)
        llm_info["model"] = llm.model
        llm_info["reachable"] = llm.available()
    except Exception as exc:  # pragma: no cover
        llm_info["reachable"] = False
        llm_info["error"] = str(exc)

    return HealthResponse(
        status="ok" if qdrant_info["reachable"] else "degraded",
        qdrant=qdrant_info,
        llm=llm_info,
        embed_backend=settings.embed_backend,
        rerank_backend=settings.rerank_backend,
    )


@app.post("/query", response_model=QueryResponse)
async def query(payload: QueryRequest, request: Request) -> QueryResponse:
    started = time.perf_counter()
    retriever = get_retriever(request)
    try:
        hits = await asyncio.to_thread(
            retrieve,
            payload.question,
            retriever,
            payload.mode,
            payload.top_k,
            payload.prefetch,
            payload.threshold,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    results = [hit.to_dict(include_text=payload.include_text) for hit in hits]
    if not payload.answer:
        return QueryResponse(
            question=payload.question,
            found=bool(hits),
            mode=payload.mode,
            results=results,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    llm = get_llm_client(request)
    result = await asyncio.to_thread(answer_from_hits, payload.question, hits, llm)
    return QueryResponse(
        question=payload.question,
        found=result.found,
        mode=payload.mode,
        answer=result.text,
        cited=[hit.chunk_id for hit in result.citations],
        results=results,
        elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
    )


@app.post("/documents", status_code=202)
async def upload_document(
    request: Request,
    background: BackgroundTasks,
    file: UploadFile = File(...),
    doc_id: str | None = Form(default=None),
    title: str = Form(default=""),
) -> dict:
    store = get_documents(request)
    identifier = slugify(doc_id or Path(file.filename or "document").stem)
    suffix = Path(file.filename or "").suffix or ".bin"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    destination = UPLOAD_DIR / f"{identifier}{suffix}"
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="empty upload")
    destination.write_bytes(content)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    store.put(
        DocumentRecord(
            doc_id=identifier,
            filename=file.filename or destination.name,
            title=title or identifier,
            status="pending",
            path=str(destination),
            created_at=timestamp,
            updated_at=timestamp,
        )
    )
    settings = request.app.state.settings
    background.add_task(
        process_document,
        destination,
        identifier,
        store=store,
        title=title or identifier,
        settings=settings,
    )
    return {"doc_id": identifier, "status": "pending"}


@app.get("/documents")
def list_documents(request: Request) -> dict:
    from ingestion.registry import REGISTRY, source_files

    manifest: dict = {}
    manifest_path = Path("data/processed/manifest.json")
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")).get("documents", {})
        except json.JSONDecodeError:
            manifest = {}

    known = {spec.doc_id for spec, _path in source_files()}
    corpus = []
    for spec in REGISTRY.values():
        entry = manifest.get(spec.doc_id, {})
        corpus.append(
            {
                "doc_id": spec.doc_id,
                "title": spec.title,
                "number": spec.number,
                "effective_date": spec.effective_date,
                "doc_type": spec.doc_type,
                "status": "ingested" if spec.doc_id in known or entry else "not_ingested",
                "articles": entry.get("max_article"),
                "chunks": (entry.get("chunks") or {}).get("structure", {}).get("chunks"),
            }
        )
    return {
        "corpus": corpus,
        "uploads": [record.to_dict() for record in get_documents(request).list()],
    }


@app.get("/documents/{doc_id}/status")
def document_status(doc_id: str, request: Request) -> dict:
    record = get_documents(request).get(doc_id)
    if record is not None:
        return record.to_dict()

    from ingestion.registry import REGISTRY

    spec = REGISTRY.get(doc_id)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"unknown document {doc_id}")
    manifest_path = Path("data/processed/manifest.json")
    entry = {}
    if manifest_path.exists():
        try:
            entry = (
                json.loads(manifest_path.read_text(encoding="utf-8"))
                .get("documents", {})
                .get(doc_id, {})
            )
        except json.JSONDecodeError:
            entry = {}
    return {
        "doc_id": doc_id,
        "title": spec.title,
        "status": "ingested" if entry else "not_ingested",
        "articles": entry.get("max_article"),
        "chunks": entry.get("chunks"),
        "source_file": entry.get("source_file"),
    }
