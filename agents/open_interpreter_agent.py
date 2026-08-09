"""Agent backed by Open Interpreter"""
from __future__ import annotations

import asyncio
import os
import tempfile

from .base import Agent

DEFAULT_MODEL = "ollama/qwen3.6:latest"


class OpenInterpreterAgent(Agent):
    def __init__(self, *, model: str = DEFAULT_MODEL) -> None:
        self._model = model
        self._tmp: tempfile.TemporaryDirectory[str] | None = None

    async def __aenter__(self) -> "OpenInterpreterAgent":
        self._tmp = tempfile.TemporaryDirectory(
            prefix="open-interpreter-bench-")
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()

    async def ask(self, prompt: str) -> str:
        if self._tmp is None:
            raise RuntimeError(
                "OpenInterpreterAgent must be entered with 'async with' before use")
        return await asyncio.to_thread(self._chat, prompt, self._tmp.name)

    def _chat(self, prompt: str, workdir: str) -> str:
        from interpreter import interpreter  # heavy import; deferred until used

        interpreter.offline = True
        interpreter.auto_run = True
        interpreter.llm.model = self._model
        previous_cwd = os.getcwd()
        os.chdir(workdir)
        try:
            messages = interpreter.chat(prompt, display=False)
        finally:
            os.chdir(previous_cwd)
        for message in reversed(messages):
            if message.get("role") == "assistant" and message.get("content"):
                return message["content"]
        return ""
