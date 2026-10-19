"""Retrieval: Qdrant dense/sparse/hybrid search with optional reranking.

Modes map to the ablation table in plans/week-3-retrieval.md:
  dense          -> week-1 baseline endpoint behaviour
  sparse         -> bge-m3 lexical weights only
  hybrid         -> dense + sparse fused in Qdrant (RRF)
  hybrid_rerank  -> hybrid top-N reranked with bge-reranker-v2-m3 (default)

The "not found" rule (plans/week-3, task 5) is a score threshold: when the best
hit scores below it, search returns no hits so the agent can answer "not found".
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from qdrant_client import QdrantClient, models

from ingestion.load import DEFAULT_COLLECTION, DENSE_VECTOR, SPARSE_VECTOR

log = logging.getLogger(__name__)

DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_DENSE_THRESHOLD = 0.45
DEFAULT_RERANK_THRESHOLD = 0.20
SEARCH_MODES = ("dense", "sparse", "hybrid", "hybrid_rerank")


@dataclass
class Hit:
    """One retrieved chunk with its citation metadata."""

    chunk_id: str
    score: float
    payload: dict = field(default_factory=dict)
    retrieval: str = "dense"

    @property
    def doc_id(self) -> str:
        return str(self.payload.get("doc_id", ""))

    @property
    def doc_title(self) -> str:
        return str(self.payload.get("doc_title", ""))

    @property
    def article_id(self) -> str:
        return str(self.payload.get("article_id", ""))

    @property
    def article_no(self) -> int:
        return int(self.payload.get("article_no") or 0)

    @property
    def article_title(self) -> str:
        return str(self.payload.get("article_title", ""))

    @property
    def clause_no(self) -> int | None:
        value = self.payload.get("clause_no")
        return int(value) if value is not None else None

    @property
    def text(self) -> str:
        return str(self.payload.get("text", ""))

    @property
    def citation(self) -> str:
        doc_title = self.doc_title or self.doc_id
        if not self.article_no:
            return doc_title
        citation = f"Điều {self.article_no}"
        if self.clause_no is not None:
            citation += f" khoản {self.clause_no}"
        return f"{citation}, {doc_title}"

    def to_dict(self, include_text: bool = True) -> dict:
        data = {
            "chunk_id": self.chunk_id,
            "score": round(float(self.score), 4),
            "retrieval": self.retrieval,
            "citation": self.citation,
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "article_id": self.article_id,
            "article_no": self.article_no,
            "article_title": self.article_title,
            "clause_no": self.clause_no,
        }
        if include_text:
            data["text"] = self.text
        return data


def hits_from_points(points, retrieval: str) -> list[Hit]:
    return [
        Hit(
            chunk_id=str(point.payload.get("chunk_id", point.id)),
            score=float(point.score),
            payload=dict(point.payload or {}),
            retrieval=retrieval,
        )
        for point in points
    ]


class Reranker:
    """bge-reranker-v2-m3 wrapper (FlagEmbedding)."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANK_MODEL,
        device: str | None = None,
        use_fp16: bool | None = None,
        max_length: int = 1024,
    ) -> None:
        self.model_name = model_name
        self.device = device or os.environ.get("RERANK_DEVICE") or None
        self.use_fp16 = use_fp16
        self.max_length = max_length
        self._model = None

    def _load(self):
        if self._model is None:
            from FlagEmbedding import FlagReranker

            kwargs = {"use_fp16": bool(self.use_fp16) if self.use_fp16 is not None else True}
            if self.device:
                kwargs["devices"] = self.device
            log.info("loading reranker %s", self.model_name)
            try:
                self._model = FlagReranker(self.model_name, **kwargs)
            except TypeError:
                kwargs.pop("devices", None)
                self._model = FlagReranker(self.model_name, **kwargs)
        return self._model

    def score(self, question: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        model = self._load()
        pairs = [[question, text] for text in texts]
        scores = model.compute_score(pairs, normalize=True, max_length=self.max_length)
        if isinstance(scores, (int, float)):
            return [float(scores)]
        return [float(score) for score in scores]


class HashingReranker:
    """Token-overlap stand-in for tests/offline runs (no semantic quality)."""

    def score(self, question: str, texts: list[str]) -> list[float]:
        question_tokens = {token.lower() for token in question.split()}
        scores: list[float] = []
        for text in texts:
            text_tokens = {token.lower() for token in text.split()}
            if not text_tokens or not question_tokens:
                scores.append(0.0)
                continue
            scores.append(len(question_tokens & text_tokens) / len(question_tokens | text_tokens))
        return scores


def get_reranker(backend: str | None = None, **kwargs):
    backend = (backend or os.environ.get("RERANK_BACKEND") or "bge").lower()
    if backend in {"hashing", "hash", "offline"}:
        return HashingReranker()
    return Reranker(**kwargs)


class Retriever:
    """Search facade used by the API and the eval harness."""

    def __init__(
        self,
        client: QdrantClient,
        collection: str = DEFAULT_COLLECTION,
        embedder=None,
        reranker=None,
        dense_threshold: float = DEFAULT_DENSE_THRESHOLD,
        rerank_threshold: float = DEFAULT_RERANK_THRESHOLD,
    ) -> None:
        self.client = client
        self.collection = collection
        self.embedder = embedder
        self.reranker = reranker
        self.dense_threshold = dense_threshold
        self.rerank_threshold = rerank_threshold

    def _embedder_or_load(self):
        if self.embedder is None:
            from ingestion.embed import get_embedder

            self.embedder = get_embedder()
        return self.embedder

    def embed_query(self, question: str):
        vectors = self._embedder_or_load().encode([question], max_length=512)
        return vectors[0]

    @staticmethod
    def _doc_filter(doc_ids):
        if not doc_ids:
            return None
        return models.Filter(
            must=[models.FieldCondition(key="doc_id", match=models.MatchAny(any=list(doc_ids)))]
        )

    def _query_dense(self, dense: list[float], limit: int, query_filter=None):
        return self.client.query_points(
            collection_name=self.collection,
            query=dense,
            using=DENSE_VECTOR,
            limit=limit,
            with_payload=True,
            query_filter=query_filter,
        ).points

    def _query_sparse(self, indices: list[int], values: list[float], limit: int, query_filter=None):
        return self.client.query_points(
            collection_name=self.collection,
            query=models.SparseVector(indices=indices, values=values),
            using=SPARSE_VECTOR,
            limit=limit,
            with_payload=True,
            query_filter=query_filter,
        ).points

    def _query_hybrid(
        self,
        dense: list[float],
        indices: list[int],
        values: list[float],
        limit: int,
        prefetch: int,
        query_filter=None,
    ):
        return self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                models.Prefetch(
                    query=dense, using=DENSE_VECTOR, limit=prefetch, filter=query_filter
                ),
                models.Prefetch(
                    query=models.SparseVector(indices=indices, values=values),
                    using=SPARSE_VECTOR,
                    limit=prefetch,
                    filter=query_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
            query_filter=query_filter,
        ).points

    def dense_search(self, question: str, limit: int = 5, doc_ids=None) -> list[Hit]:
        vector = self.embed_query(question)
        return hits_from_points(
            self._query_dense(vector.dense, limit, self._doc_filter(doc_ids)), "dense"
        )

    def sparse_search(self, question: str, limit: int = 5, doc_ids=None) -> list[Hit]:
        vector = self.embed_query(question)
        return hits_from_points(
            self._query_sparse(
                vector.sparse_indices, vector.sparse_values, limit, self._doc_filter(doc_ids)
            ),
            "sparse",
        )

    def hybrid_search(
        self, question: str, limit: int = 30, prefetch: int = 40, doc_ids=None
    ) -> list[Hit]:
        vector = self.embed_query(question)
        return hits_from_points(
            self._query_hybrid(
                vector.dense,
                vector.sparse_indices,
                vector.sparse_values,
                limit,
                prefetch,
                self._doc_filter(doc_ids),
            ),
            "hybrid",
        )

    def rerank(self, question: str, hits: list[Hit], top_k: int = 5) -> list[Hit]:
        if not hits:
            return []
        if self.reranker is None:
            self.reranker = get_reranker()
        scores = self.reranker.score(question, [hit.text for hit in hits])
        rescored = [
            Hit(chunk_id=hit.chunk_id, score=float(score), payload=hit.payload, retrieval="rerank")
            for hit, score in zip(hits, scores, strict=True)
        ]
        rescored.sort(key=lambda hit: hit.score, reverse=True)
        return rescored[:top_k]

    def search(
        self,
        question: str,
        mode: str = "hybrid_rerank",
        limit: int = 5,
        prefetch: int = 30,
        threshold: float | None = None,
        doc_ids: list[str] | None = None,
    ) -> list[Hit]:
        """Return ranked hits, or [] when nothing clears the threshold."""
        if mode not in SEARCH_MODES:
            raise ValueError(f"unknown search mode {mode!r}; expected one of {SEARCH_MODES}")
        if mode == "dense":
            hits = self.dense_search(question, limit=limit, doc_ids=doc_ids)
            cutoff = self.dense_threshold if threshold is None else threshold
        elif mode == "sparse":
            hits = self.sparse_search(question, limit=limit, doc_ids=doc_ids)
            cutoff = 0.0 if threshold is None else threshold
        elif mode == "hybrid":
            hits = self.hybrid_search(question, limit=limit, prefetch=prefetch, doc_ids=doc_ids)
            cutoff = 0.0 if threshold is None else threshold
        else:
            candidates = self.hybrid_search(
                question,
                limit=max(prefetch, limit),
                prefetch=max(prefetch, limit),
                doc_ids=doc_ids,
            )
            hits = self.rerank(question, candidates, top_k=limit)
            cutoff = self.rerank_threshold if threshold is None else threshold
        if cutoff and hits and hits[0].score < cutoff:
            return []
        return hits
