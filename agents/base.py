"""Common harness interface, with uniform per-call metrics."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from .metrics import Metrics

_DONE_MARKER = "DONE:"
_FINALIZATION_HINT = (
    f"\n\nWhen you have fully finished, end your reply with a line starting "
    f"with '{_DONE_MARKER}' followed by your answer. Anything else is "
    f"treated as unfinished, and you'll be asked to continue."
)


def _continue_prompt(original_prompt: str, last_reply: str) -> str:
    return (
        f"Original task: {original_prompt}\n\n"
        f"Your last reply didn't confirm you were finished: {last_reply!r}\n\n"
        f"If the task isn't complete, keep working on it. Once it is, reply "
        f"with a line starting with '{_DONE_MARKER}' followed by your answer."
    )


class Agent(ABC):
    """Async context-managed agent: enter once, ask any number of prompts, exit."""

    def __init__(
        self,
        *,
        name: str | None = None,
        model: str | None = None,
        max_continuations: int = 10,
    ) -> None:
        self.name = name or type(self).__name__
        self.model = model
        self.metrics: list[Metrics] = []
        self._max_continuations = max_continuations

    async def __aenter__(self) -> "Agent":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        pass

    async def ask(self, prompt: str) -> str:
        start = time.monotonic()
        answer, extra, continuations = await self._ask_until_done(prompt)
        self.metrics.append(Metrics(
            harness=self.name,
            prompt=prompt,
            latency_s=time.monotonic() - start,
            answer_chars=len(answer),
            continuations=continuations,
            **extra,
        ))
        return answer

    async def _ask_until_done(self, prompt: str) -> tuple[str, dict[str, Any], int]:
        answer, extra = await self._ask(prompt + _FINALIZATION_HINT)
        continuations = 0
        while _DONE_MARKER not in answer and continuations < self._max_continuations:
            continuations += 1
            reply, round_extra = await self._ask(_continue_prompt(prompt, answer))
            answer = reply
            extra = self._merge_extra(extra, round_extra)
        if _DONE_MARKER in answer:
            answer = answer.split(_DONE_MARKER, 1)[1].strip()
        return answer, extra, continuations

    @staticmethod
    def _merge_extra(total: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
        merged = dict(total)
        for key, value in extra.items():
            if value is None:
                continue
            merged[key] = merged.get(key, 0) + value
        return merged

    @abstractmethod
    async def _ask(self, prompt: str) -> tuple[str, dict[str, Any]]: ...
