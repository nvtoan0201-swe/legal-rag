"""Transport selection for IPv4-only LLM connections (no network)."""

from __future__ import annotations

import httpx

from llm.net import async_transport, sync_transport


def test_default_leaves_transport_to_the_client():
    assert sync_transport(False) is None
    assert async_transport(False) is None


def test_ipv4_only_binds_an_ipv4_local_address():
    transport = sync_transport(True)
    assert isinstance(transport, httpx.HTTPTransport)
    assert isinstance(async_transport(True), httpx.AsyncHTTPTransport)
