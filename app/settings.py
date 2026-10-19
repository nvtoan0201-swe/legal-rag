"""Application settings, loaded from environment variables / .env."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Empty qdrant_url = embedded local storage at qdrant_path (no server needed).
    qdrant_url: str = ""
    qdrant_path: str = "data/qdrant"
    qdrant_api_key: str = ""

    collection: str = "legal_docs"
    baseline_collection: str = "legal_docs_baseline"
    article_collection: str = "legal_docs_article"

    embed_backend: str = "bge-m3"
    embed_model: str = "BAAI/bge-m3"
    rerank_backend: str = "bge"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    dense_threshold: float = 0.45
    rerank_threshold: float = 0.20

    # "openai" (any OpenAI-compatible server: OpenAI, OpenCode Go, Ollama, vLLM, LM Studio)
    # or "anthropic" (Messages API). LLM_API_BASE must match the provider.
    llm_provider: str = "openai"
    # Bind LLM sockets to IPv4 to skip a blackholed IPv6 route (see llm/net.py).
    llm_ipv4_only: bool = False
    llm_api_key: str = ""
    llm_api_base: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    # Extra request headers for the hosted API, as JSON in the env (.env), e.g.
    # {"x-opencode-session": "legal-rag", "User-Agent": "legal-rag/0.1.0"}
    llm_api_headers: dict[str, str] = {}
    # Optional override for the /models endpoint used by the health check.
    llm_models_url: str = ""
    # Extra request body fields for the hosted API, as JSON in the env (.env),
    # e.g. {"chat_template_kwargs": {"enable_thinking": false}} to turn off
    # reasoning so it does not eat the answer's max_tokens budget.
    llm_extra_body: dict = {}
    llm_temperature: float = 0.0
    llm_max_tokens: int = 800

    data_dir: str = "data"


@lru_cache
def get_settings() -> Settings:
    return Settings()
