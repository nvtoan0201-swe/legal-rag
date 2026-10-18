"""Select the LLM backend from settings (LLM_PROVIDER=local|api)."""

from __future__ import annotations

from app.settings import Settings, get_settings
from llm.base import LLMClient
from llm.openai_backend import OpenAICompatibleClient


def get_llm(settings: Settings | None = None, provider: str | None = None) -> LLMClient:
    settings = settings or get_settings()
    provider = (provider or settings.llm_provider or "local").lower()
    if provider in {"api", "hosted", "openai"}:
        return OpenAICompatibleClient(
            base_url=settings.llm_api_base,
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            disable_thinking=False,
            default_headers=settings.llm_api_headers or None,
            models_url=settings.llm_models_url or None,
        )
    model = settings.local_llm_model
    return OpenAICompatibleClient(
        base_url=settings.local_llm_base_url,
        model=model,
        api_key="not-needed",
        disable_thinking="qwen3" in model.lower(),
    )
