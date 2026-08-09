"""Common harness interface."""
from __future__ import annotations

from abc import ABC, abstractmethod


class Agent(ABC):
    """Async context-managed agent: enter once, ask any number of prompts, exit."""

    async def __aenter__(self) -> "Agent":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        pass

    @abstractmethod
    async def ask(self, prompt: str) -> str: ...
