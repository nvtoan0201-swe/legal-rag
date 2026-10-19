"""Hermetic test environment: no model downloads and no live services required."""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("EMBED_BACKEND", "hashing")
os.environ.setdefault("RERANK_BACKEND", "hashing")
