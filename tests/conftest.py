"""Hermetic test environment: no model downloads and no live services required."""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("EMBED_BACKEND", "hashing")
os.environ.setdefault("RERANK_BACKEND", "hashing")
os.environ.setdefault("LOCAL_LLM_BASE_URL", "http://127.0.0.1:9/v1")
os.environ.setdefault("LOCAL_LLM_MODEL", "test-model")
