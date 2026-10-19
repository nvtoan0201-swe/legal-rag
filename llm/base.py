"""Common LLM interface.

The hosted API is OpenAI-compatible, so one client class covers it; see
docs/decisions.md.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

Role = Literal["system", "user", "assistant"]

_THINK_BLOCK = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)
_THINK_PREFIX = re.compile(r"^.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)


@dataclass
class Message:
    role: Role
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


def system(content: str) -> Message:
    return Message("system", content)


def user(content: str) -> Message:
    return Message("user", content)


def strip_thinking(text: str) -> str:
    """Drop reasoning blocks (<think>...</think>) from model output."""
    cleaned = _THINK_BLOCK.sub("", text or "")
    if "</think>" in cleaned.lower():
        cleaned = _THINK_PREFIX.sub("", cleaned)
    return cleaned.strip()


class LLMClient(Protocol):
    model: str

    def chat(
        self, messages: Sequence[Message], temperature: float = 0.0, max_tokens: int = 800
    ) -> str: ...

    async def achat(
        self, messages: Sequence[Message], temperature: float = 0.0, max_tokens: int = 800
    ) -> str: ...

    def stream(
        self, messages: Sequence[Message], temperature: float = 0.0, max_tokens: int = 800
    ) -> Iterator[str]: ...

    def astream(
        self, messages: Sequence[Message], temperature: float = 0.0, max_tokens: int = 800
    ) -> AsyncIterator[str]: ...

    def available(self) -> bool: ...
