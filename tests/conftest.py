"""Shared pytest fixtures. No real models, network or large datasets are touched."""

from __future__ import annotations

import os
from collections import deque
from typing import Any, Iterable, List

import pytest

# Keep test runs from appending to logs/trajectories.jsonl.
os.environ.setdefault("VERISCAN_TRAJECTORY_LOG", "0")


class FakeLLM:
    """Scripted stand-in for ``models.local_llm.LocalLLM``: returns queued replies in order."""

    def __init__(self, replies: Iterable[str] = ()):
        self.replies = deque(replies)
        self.prompts: List[str] = []

    def generate(self, prompt: str, max_tokens: int = 250, temp: float = 0.0) -> str:
        self.prompts.append(prompt)
        if not self.replies:
            return '{"action":"final","answer":"done"}'
        reply = self.replies.popleft()
        if isinstance(reply, Exception):
            raise reply
        return reply

    def generate_chat(
        self, messages: list[dict[str, Any]], max_tokens: int = 512, temp: float = 0.7, **_: Any
    ) -> str:
        return self.generate(messages[-1]["content"], max_tokens=max_tokens, temp=temp)


@pytest.fixture
def fake_llm_factory():
    return FakeLLM
