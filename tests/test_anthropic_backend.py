"""Tests for the Anthropic Messages adapter with httpx.MockTransport (no network)."""

from __future__ import annotations

import asyncio
import json

import httpx

from llm.anthropic_backend import AnthropicClient
from llm.base import system, user


def _delta(text: str) -> str:
    event = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": text}}
    return f"event: content_block_delta\ndata: {json.dumps(event, ensure_ascii=False)}"


SSE_BODY = "\n".join(
    [
        'event: message_start\ndata: {"type": "message_start"}',
        _delta("Người"),
        _delta(" lao động"),
        'event: message_stop\ndata: {"type": "message_stop"}',
        "",
    ]
)


def _client(handler) -> AnthropicClient:
    transport = httpx.MockTransport(handler)
    return AnthropicClient(
        base_url="https://api.example.test",
        model="claude-test",
        api_key="sk-test",
        transport=transport,
        async_transport=httpx.MockTransport(handler),
    )


def test_chat_sends_system_field_and_parses_text_blocks():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = request.headers
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "content": [
                    {"type": "text", "text": "Xin "},
                    {"type": "text", "text": "chào [1]."},
                ]
            },
        )

    client = _client(handler)
    reply = client.chat([system("Luật sư ảo."), user("Hỏi?")], temperature=0.0, max_tokens=64)

    assert reply == "Xin chào [1]."
    assert seen["url"] == "https://api.example.test/v1/messages"
    assert seen["headers"]["x-api-key"] == "sk-test"
    assert seen["headers"]["anthropic-version"] == "2023-06-01"
    assert seen["body"]["system"] == "Luật sư ảo."
    assert seen["body"]["messages"] == [{"role": "user", "content": "Hỏi?"}]
    assert seen["body"]["max_tokens"] == 64
    assert "stream" not in seen["body"]


def test_stream_yields_only_text_deltas():
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, text=SSE_BODY, headers={"content-type": "text/event-stream"})

    pieces = list(_client(handler).stream([user("Hỏi?")]))
    assert pieces == ["Người", " lao động"]


def test_astream_matches_sync_stream():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=SSE_BODY, headers={"content-type": "text/event-stream"})

    async def collect():
        client = _client(handler)
        return [piece async for piece in client.astream([user("Hỏi?")])]

    assert asyncio.run(collect()) == ["Người", " lao động"]


def test_available_checks_models_endpoint():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": []})

    assert _client(handler).available() is True


def test_available_false_on_error_status():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "unauthorized"})

    assert _client(handler).available() is False
