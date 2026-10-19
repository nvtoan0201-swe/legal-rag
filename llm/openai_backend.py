"""OpenAI-compatible LLM backend for the hosted API.

The differences between gateways are only base_url, model name, API key, and
(for some gateways) extra headers and request body fields.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator, Sequence
from urllib.request import Request, urlopen

import httpx

from llm.base import Message, strip_thinking
from llm.net import async_transport, sync_transport

log = logging.getLogger(__name__)


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "not-needed",
        timeout: float = 180.0,
        max_retries: int = 2,
        default_headers: dict[str, str] | None = None,
        models_url: str | None = None,
        extra_body: dict | None = None,
        ipv4_only: bool = False,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key or "not-needed"
        self.timeout = timeout
        self.max_retries = max_retries
        self.default_headers = default_headers or {}
        self.models_url = models_url
        self.extra_body = extra_body
        self.ipv4_only = ipv4_only
        self._sync = None
        self._async = None

    # -- lazily created SDK clients -------------------------------------
    def _sync_http_client(self):
        if not self.ipv4_only:
            return None
        return httpx.Client(transport=sync_transport(True), timeout=self.timeout)

    def _async_http_client(self):
        if not self.ipv4_only:
            return None
        return httpx.AsyncClient(transport=async_transport(True), timeout=self.timeout)

    @property
    def sync(self):
        if self._sync is None:
            from openai import OpenAI

            self._sync = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=self.max_retries,
                default_headers=self.default_headers or None,
                http_client=self._sync_http_client(),
            )
        return self._sync

    @property
    def async_client(self):
        if self._async is None:
            from openai import AsyncOpenAI

            self._async = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=self.max_retries,
                default_headers=self.default_headers or None,
                http_client=self._async_http_client(),
            )
        return self._async

    # -- blocking --------------------------------------------------------
    def chat(
        self, messages: Sequence[Message], temperature: float = 0.0, max_tokens: int = 800
    ) -> str:
        response = self.sync.chat.completions.create(
            model=self.model,
            messages=[message.to_dict() for message in messages],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=self.extra_body,
        )
        return strip_thinking(response.choices[0].message.content or "")

    def stream(
        self, messages: Sequence[Message], temperature: float = 0.0, max_tokens: int = 800
    ) -> Iterator[str]:
        stream = self.sync.chat.completions.create(
            model=self.model,
            messages=[message.to_dict() for message in messages],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=self.extra_body,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            piece = getattr(delta, "content", None)
            if piece:
                # Raw deltas: the whitespace belongs to the tokens; the agent
                # strips the accumulated answer once it is complete.
                yield piece

    # -- async -----------------------------------------------------------
    async def achat(
        self, messages: Sequence[Message], temperature: float = 0.0, max_tokens: int = 800
    ) -> str:
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=[message.to_dict() for message in messages],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=self.extra_body,
        )
        return strip_thinking(response.choices[0].message.content or "")

    async def astream(
        self, messages: Sequence[Message], temperature: float = 0.0, max_tokens: int = 800
    ) -> AsyncIterator[str]:
        stream = await self.async_client.chat.completions.create(
            model=self.model,
            messages=[message.to_dict() for message in messages],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=self.extra_body,
            stream=True,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            piece = getattr(delta, "content", None)
            if piece:
                # Raw deltas: the whitespace belongs to the tokens; the agent
                # strips the accumulated answer once it is complete.
                yield piece

    def available(self) -> bool:
        if self.models_url:
            # Some gateways expose the model list at a non-standard path.
            try:
                request = Request(
                    self.models_url,
                    headers={"Authorization": f"Bearer {self.api_key}", **self.default_headers},
                )
                with urlopen(request, timeout=30) as response:
                    return 200 <= response.status < 300
            except Exception as exc:  # pragma: no cover - depends on the server
                log.debug("LLM %s not available: %s", self.models_url, exc)
                return False
        try:
            self.sync.models.list()
            return True
        except Exception as exc:  # pragma: no cover - depends on the server
            log.debug("LLM %s not available: %s", self.base_url, exc)
            return False
