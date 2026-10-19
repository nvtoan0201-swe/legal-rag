"""Anthropic Messages API backend, over plain httpx (no SDK dependency).

Implements the same LLMClient surface as OpenAICompatibleClient, so the rest of the
pipeline does not know which provider answers. System messages are sent in the
top-level `system` field, as the Messages API requires.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Iterator, Sequence

import httpx

from llm.base import Message
from llm.net import async_transport as async_transport_for
from llm.net import sync_transport

log = logging.getLogger(__name__)

API_VERSION = "2023-06-01"


class AnthropicClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        timeout: float = 180.0,
        transport: httpx.BaseTransport | None = None,
        async_transport: httpx.AsyncBaseTransport | None = None,
        ipv4_only: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self._transport = transport or sync_transport(ipv4_only)
        self._async_transport = async_transport or async_transport_for(ipv4_only)

    # -- request building -----------------------------------------------
    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }

    def _body(
        self,
        messages: Sequence[Message],
        temperature: float,
        max_tokens: int,
        stream: bool,
    ) -> dict:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        turns = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
        body: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": turns,
        }
        if system:
            body["system"] = system
        if stream:
            body["stream"] = True
        return body

    @staticmethod
    def _text_of(payload: dict) -> str:
        return "".join(
            block.get("text", "")
            for block in payload.get("content", [])
            if block.get("type") == "text"
        )

    @staticmethod
    def _delta_of(line: str) -> str:
        """Return the text delta of one SSE `data:` line, or '' for other events."""
        if not line.startswith("data:"):
            return ""
        event = json.loads(line[len("data:") :].strip() or "{}")
        if event.get("type") != "content_block_delta":
            return ""
        delta = event.get("delta", {})
        return delta.get("text", "") if delta.get("type") == "text_delta" else ""

    # -- blocking --------------------------------------------------------
    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.base_url,
            headers=self._headers(),
            timeout=self.timeout,
            transport=self._transport,
        )

    def chat(
        self, messages: Sequence[Message], temperature: float = 0.0, max_tokens: int = 800
    ) -> str:
        with self._client() as client:
            response = client.post(
                "/v1/messages", json=self._body(messages, temperature, max_tokens, stream=False)
            )
            response.raise_for_status()
            return self._text_of(response.json())

    def stream(
        self, messages: Sequence[Message], temperature: float = 0.0, max_tokens: int = 800
    ) -> Iterator[str]:
        with self._client() as client:
            with client.stream(
                "POST",
                "/v1/messages",
                json=self._body(messages, temperature, max_tokens, stream=True),
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    piece = self._delta_of(line)
                    if piece:
                        yield piece

    # -- async -----------------------------------------------------------
    def _async_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.base_url,
            headers=self._headers(),
            timeout=self.timeout,
            transport=self._async_transport,
        )

    async def achat(
        self, messages: Sequence[Message], temperature: float = 0.0, max_tokens: int = 800
    ) -> str:
        async with self._async_client() as client:
            response = await client.post(
                "/v1/messages", json=self._body(messages, temperature, max_tokens, stream=False)
            )
            response.raise_for_status()
            return self._text_of(response.json())

    async def astream(
        self, messages: Sequence[Message], temperature: float = 0.0, max_tokens: int = 800
    ) -> AsyncIterator[str]:
        async with self._async_client() as client:
            async with client.stream(
                "POST",
                "/v1/messages",
                json=self._body(messages, temperature, max_tokens, stream=True),
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    piece = self._delta_of(line)
                    if piece:
                        yield piece

    # -- health ----------------------------------------------------------
    def available(self) -> bool:
        try:
            with self._client() as client:
                response = client.get("/v1/models")
                return response.is_success
        except httpx.HTTPError as exc:  # pragma: no cover - depends on the server
            log.debug("Anthropic API %s not available: %s", self.base_url, exc)
            return False
