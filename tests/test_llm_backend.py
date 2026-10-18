"""Streaming passthrough tests for the OpenAI-compatible client (no network)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from llm.base import user
from llm.openai_backend import OpenAICompatibleClient

PIECES = ["Người", " lao", " động", " được", " nghỉ", " 12", " ngày."]
EXPECTED = "Người lao động được nghỉ 12 ngày."


def _chunk(content: str):
    delta = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


class _FakeSync:
    def __init__(self, pieces):
        self._pieces = pieces

    @property
    def chat(self):
        def create(**kwargs):
            return iter(_chunk(piece) for piece in self._pieces)

        completions = SimpleNamespace(create=create)
        return SimpleNamespace(completions=completions)


class _FakeAsync:
    def __init__(self, pieces):
        self._pieces = pieces

    @property
    def chat(self):
        async def create(**kwargs):
            async def chunks():
                for piece in self._pieces:
                    yield _chunk(piece)

            return chunks()

        completions = SimpleNamespace(create=create)
        return SimpleNamespace(completions=completions)


def _client() -> OpenAICompatibleClient:
    return OpenAICompatibleClient(base_url="http://test", model="fake")


def test_sync_stream_preserves_inter_token_spaces():
    client = _client()
    client._sync = _FakeSync(PIECES)
    assert "".join(client.stream([user("x")])) == EXPECTED


def test_async_stream_preserves_inter_token_spaces():
    client = _client()
    client._async = _FakeAsync(PIECES)

    async def collect():
        return "".join([piece async for piece in client.astream([user("x")])])

    assert asyncio.run(collect()) == EXPECTED
