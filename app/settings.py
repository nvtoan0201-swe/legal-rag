"""Application settings, loaded from environment variables / .env."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "change-me"

    collection: str = "legal_docs"
    baseline_collection: str = "legal_docs_baseline"
    article_collection: str = "legal_docs_article"

    embed_backend: str = "bge-m3"
    embed_model: str = "BAAI/bge-m3"
    rerank_backend: str = "bge"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    dense_threshold: float = 0.45
    rerank_threshold: float = 0.20

    llm_provider: str = "local"  # "local" (Qwen3-8B via vLLM/Ollama) or "api" (hosted)
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
    local_llm_base_url: str = "http://localhost:8000/v1"
    local_llm_model: str = "Qwen/Qwen3-8B"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 800

    langfuse_host: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    data_dir: str = "data"


@lru_cache
def get_settings() -> Settings:
    return Settings()
