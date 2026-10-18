"""bge-m3 embeddings: dense (1024-d) + sparse lexical weights.

One model produces both vectors (see docs/decisions.md). The hashing backend is a
deterministic offline stand-in used by tests and by any environment that cannot
download the 2 GB model; it is never used for evaluation numbers.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_DENSE_DIM = 1024
_TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)


@dataclass
class Vectorized:
    dense: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - torch missing or broken
        return False


def resolve_device(device: str | None = None) -> str:
    if device:
        return device
    env = os.environ.get("EMBED_DEVICE")
    if env:
        return env
    return "cuda" if cuda_available() else "cpu"


class BGEM3Embedder:
    """Wrapper around FlagEmbedding's BGEM3FlagModel."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        use_fp16: bool | None = None,
        max_length: int = 1024,
        batch_size: int = 8,
    ) -> None:
        self.model_name = model_name
        self.device = resolve_device(device)
        self.use_fp16 = cuda_available() if use_fp16 is None else use_fp16
        self.max_length = max_length
        self.batch_size = batch_size
        self._model = None
        self.dim = DEFAULT_DENSE_DIM

    def _load(self):
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel

            log.info("loading %s on %s (fp16=%s)", self.model_name, self.device, self.use_fp16)
            kwargs = {"use_fp16": self.use_fp16, "devices": self.device}
            try:
                self._model = BGEM3FlagModel(self.model_name, **kwargs)
            except TypeError:
                kwargs.pop("devices", None)
                self._model = BGEM3FlagModel(self.model_name, **kwargs)
        return self._model

    def encode(
        self,
        texts: list[str],
        batch_size: int | None = None,
        max_length: int | None = None,
    ) -> list[Vectorized]:
        if not texts:
            return []
        model = self._load()
        output = model.encode(
            list(texts),
            batch_size=batch_size or self.batch_size,
            max_length=max_length or self.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense_vectors = output["dense_vecs"]
        lexical_weights = output["lexical_weights"]
        results: list[Vectorized] = []
        for index in range(len(texts)):
            sparse = lexical_weights[index]
            indices: list[int] = []
            values: list[float] = []
            for token_id, weight in sparse.items():
                value = float(weight)
                if value <= 0:
                    continue
                indices.append(int(token_id))
                values.append(value)
            results.append(
                Vectorized(
                    dense=[float(value) for value in dense_vectors[index]],
                    sparse_indices=indices,
                    sparse_values=values,
                )
            )
        return results


class HashingEmbedder:
    """Deterministic offline embedder (tests/CI only, no semantic quality)."""

    def __init__(self, dim: int = DEFAULT_DENSE_DIM, batch_size: int = 64, **_ignored) -> None:
        self.dim = dim
        self.batch_size = batch_size
        self.device = "cpu"
        self.model_name = "hashing"

    def encode(
        self,
        texts: list[str],
        batch_size: int | None = None,
        max_length: int | None = None,
    ) -> list[Vectorized]:
        results: list[Vectorized] = []
        for text in texts:
            dense = [0.0] * self.dim
            sparse: dict[int, float] = {}
            tokens = [token.lower() for token in _TOKEN_RE.findall(text)]
            for token in tokens:
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "big") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                dense[index] += sign
                sparse[index] = sparse.get(index, 0.0) + 1.0
            norm = math.sqrt(sum(value * value for value in dense)) or 1.0
            results.append(
                Vectorized(
                    dense=[value / norm for value in dense],
                    sparse_indices=sorted(sparse),
                    sparse_values=[sparse[index] for index in sorted(sparse)],
                )
            )
        return results


def get_embedder(backend: str | None = None, **kwargs):
    """Return the embedder selected by EMBED_BACKEND (default bge-m3)."""
    backend = (backend or os.environ.get("EMBED_BACKEND") or "bge-m3").lower()
    if backend in {"hashing", "hash", "offline"}:
        return HashingEmbedder(**kwargs)
    return BGEM3Embedder(**kwargs)
