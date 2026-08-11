"""Common harness interface, with uniform per-call metrics."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from .metrics import Metrics


class Agent(ABC):
    """Async context-managed agent: enter once, ask any number of prompts, exit."""

    def __init__(self, *, name: str | None = None, model: str | None = None) -> None:
        self.name = name or type(self).__name__
        self.model = model
        self.metrics: list[Metrics] = []

    async def __aenter__(self) -> "Agent":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        pass

    async def ask(self, prompt: str) -> str:
        start = time.monotonic()
        answer, extra = await self._ask(prompt)
        self.metrics.append(Metrics(
            harness=self.name,
            prompt=prompt,
            latency_s=time.monotonic() - start,
            answer_chars=len(answer),
            **extra,
        ))
        return answer

    @abstractmethod
    async def _ask(self, prompt: str) -> tuple[str, dict[str, Any]]: ...
