"""Load chunks + vectors into Qdrant."""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Iterable, Sequence

from qdrant_client import QdrantClient, models

from ingestion.chunk import Chunk
from ingestion.embed import Vectorized

log = logging.getLogger(__name__)

DEFAULT_URL = "http://localhost:6333"
DEFAULT_COLLECTION = "legal_docs"
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"
POINT_NAMESPACE = uuid.UUID("0f6a1f2e-6d5a-4f3b-9f1e-2b7c8d9a0e11")

COLLECTION_BY_GRANULARITY = {
    "structure": "legal_docs",
    "article": "legal_docs_article",
    "fixed512": "legal_docs_baseline",
}


def collection_for(granularity: str) -> str:
    return COLLECTION_BY_GRANULARITY.get(granularity, DEFAULT_COLLECTION)


def get_client(url: str | None = None, api_key: str | None = None) -> QdrantClient:
    url = url or os.environ.get("QDRANT_URL", DEFAULT_URL)
    return QdrantClient(
        url=url,
        api_key=api_key or os.environ.get("QDRANT_API_KEY") or None,
        timeout=120,
    )


def ensure_collection(
    client: QdrantClient,
    collection: str = DEFAULT_COLLECTION,
    dense_dim: int = 1024,
    recreate: bool = False,
) -> None:
    exists = client.collection_exists(collection)
    if exists and recreate:
        log.info("recreating collection %s", collection)
        client.delete_collection(collection)
        exists = False
    if not exists:
        log.info("creating collection %s (dense=%d + sparse)", collection, dense_dim)
        client.create_collection(
            collection_name=collection,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(size=dense_dim, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={
                SPARSE_VECTOR: models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            },
        )


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, chunk_id))


def upsert_points(
    client: QdrantClient,
    collection: str,
    items: Sequence[tuple[Chunk, Vectorized]],
    batch_size: int = 64,
) -> int:
    total = 0
    batch: list[models.PointStruct] = []

    def flush() -> None:
        nonlocal total, batch
        if not batch:
            return
        client.upsert(collection_name=collection, points=batch, wait=True)
        total += len(batch)
        batch = []

    for chunk, vector in items:
        batch.append(
            models.PointStruct(
                id=point_id(chunk.chunk_id),
                vector={
                    DENSE_VECTOR: vector.dense,
                    SPARSE_VECTOR: models.SparseVector(
                        indices=vector.sparse_indices, values=vector.sparse_values
                    ),
                },
                payload=chunk.payload(),
            )
        )
        if len(batch) >= batch_size:
            flush()
    flush()
    return total


def count_points(client: QdrantClient, collection: str) -> int:
    return int(client.count(collection_name=collection, exact=True).count)


def iter_batches(items: Iterable, size: int):
    batch: list = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
