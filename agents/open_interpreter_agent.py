"""Agent backed by Open Interpreter"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from typing import Any
from interpreter import interpreter

from .base import Agent

DEFAULT_MODEL = "ollama/qwen3.6:latest"


class OpenInterpreterAgent(Agent):
    def __init__(self, *, name: str | None = None, model: str | None = None) -> None:
        super().__init__(name=name, model=model or DEFAULT_MODEL)
        self._tmp: tempfile.TemporaryDirectory[str] | None = None

    async def __aenter__(self) -> "OpenInterpreterAgent":
        self._tmp = tempfile.TemporaryDirectory(
            prefix="open-interpreter-bench-")
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        # Kill all Jupyter notebook runtimes.
        interpreter.computer.terminal.terminate()
        if self._tmp is not None:
            self._tmp.cleanup()

    async def _ask(self, prompt: str) -> tuple[str, dict[str, Any]]:
        if self._tmp is None:
            raise RuntimeError(
                "OpenInterpreterAgent must be entered with 'async with' before use")
        return await asyncio.to_thread(self._chat, prompt, self._tmp.name)

    def _chat(self, prompt: str, workdir: str) -> tuple[str, dict[str, Any]]:
        from interpreter import interpreter  # heavy import; deferred until used

        interpreter.offline = True
        interpreter.auto_run = True
        interpreter.llm.model = self.model
        previous_cwd = os.getcwd()
        os.chdir(workdir)
        messages_before = len(interpreter.messages)
        try:
            self._print_progress(interpreter.chat(
                prompt, display=False, stream=True))
        finally:
            os.chdir(previous_cwd)
        messages = interpreter.messages[messages_before:]

        answer = next(
            (m["content"] for m in reversed(messages)
             if m.get("role") == "assistant" and m.get("content")),
            "",
        )
        extra = {
            "turns": sum(1 for m in messages if m.get("role") == "assistant"),
            "tool_calls": sum(1 for m in messages if m.get("type") == "code"),
        }
        return answer, extra

    @staticmethod
    def _print_progress(chunks: Any) -> None:
        """Streams each step (model text, generated code, execution output) to
        stderr as it happens, instead of staying silent until the final reply
        — otherwise a slow local model looks hung with no visible progress."""
        active_type = None
        for chunk in chunks:
            chunk_type = chunk.get("type")
            if chunk_type != active_type:
                label = f"{chunk_type}:{chunk['format']}" if chunk.get(
                    "format") else chunk_type
                print(f"\n[{label}] ", end="", file=sys.stderr)
                active_type = chunk_type
            content = chunk.get("content")
            if content:
                print(content, end="", file=sys.stderr, flush=True)
        print(file=sys.stderr)
