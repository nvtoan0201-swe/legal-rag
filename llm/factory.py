"""Build the LLM client from settings: LLM_PROVIDER picks the adapter."""

from __future__ import annotations

from app.settings import Settings, get_settings
from llm.anthropic_backend import AnthropicClient
from llm.base import LLMClient
from llm.openai_backend import OpenAICompatibleClient


def get_llm(settings: Settings | None = None) -> LLMClient:
    settings = settings or get_settings()
    if settings.llm_provider == "anthropic":
        return AnthropicClient(
            base_url=settings.llm_api_base,
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            ipv4_only=settings.llm_ipv4_only,
        )
    if settings.llm_provider in {"openai", "api"}:
        return OpenAICompatibleClient(
            base_url=settings.llm_api_base,
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            default_headers=settings.llm_api_headers or None,
            models_url=settings.llm_models_url or None,
            extra_body=settings.llm_extra_body or None,
            ipv4_only=settings.llm_ipv4_only,
        )
    raise ValueError(
        f"unknown LLM_PROVIDER {settings.llm_provider!r}; use 'openai' (alias 'api') or 'anthropic'"
    )
