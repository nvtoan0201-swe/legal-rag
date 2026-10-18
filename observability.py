"""Optional Langfuse tracing: a no-op unless LANGFUSE_* keys are configured.

The agent wraps each request in a trace so retrieval/answer steps show up in
Langfuse when the tracing profile from docker-compose is running.
"""

from __future__ import annotations

import contextlib
import logging
import os

log = logging.getLogger(__name__)
_client = None
_initialised = False


def tracing_enabled() -> bool:
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))


def get_client():
    global _client, _initialised
    if _initialised:
        return _client
    _initialised = True
    if not tracing_enabled():
        return None
    try:
        from langfuse import Langfuse

        _client = Langfuse(
            host=os.environ.get("LANGFUSE_HOST") or None,
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        )
    except Exception as exc:  # pragma: no cover - depends on Langfuse
        log.warning("Langfuse init failed: %s", exc)
        _client = None
    return _client


@contextlib.contextmanager
def trace(name: str, **metadata):
    """Yield a Langfuse trace object, or None when tracing is off."""
    client = get_client()
    if client is None:
        yield None
        return
    span = None
    try:  # pragma: no cover - depends on Langfuse
        span = client.trace(name=name, metadata=metadata or None)
    except Exception as exc:
        log.warning("Langfuse trace failed: %s", exc)
    try:
        yield span
    finally:
        if span is not None:
            try:
                client.flush()
            except Exception:  # pragma: no cover
                pass


def log_event(span, name: str, **payload) -> None:
    """Attach an event to a trace when tracing is enabled."""
    if span is None:
        return
    try:  # pragma: no cover - depends on Langfuse
        span.event(name=name, metadata=payload or None)
    except Exception:
        pass
