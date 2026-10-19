"""Transport options shared by the LLM adapters.

Some hosts publish an IPv6 address whose route is blackholed. The client then waits
for the IPv6 connect to time out before falling back. Binding the socket to an IPv4
local address skips that wait. Enable with LLM_IPV4_ONLY=true; off by default.
"""

from __future__ import annotations

import httpx


def sync_transport(ipv4_only: bool) -> httpx.BaseTransport | None:
    if not ipv4_only:
        return None
    return httpx.HTTPTransport(local_address="0.0.0.0")


def async_transport(ipv4_only: bool) -> httpx.AsyncBaseTransport | None:
    if not ipv4_only:
        return None
    return httpx.AsyncHTTPTransport(local_address="0.0.0.0")
