"""Agent backed by the Codex CLI."""
from __future__ import annotations

import asyncio
import tempfile

from .base import Agent

DEFAULT_MODEL = "gpt-4o"


class CodexAgent(Agent):
    def __init__(self, *, model: str = DEFAULT_MODEL) -> None:
        self._model = model
        self._tmp: tempfile.TemporaryDirectory[str] | None = None

    async def __aenter__(self) -> "CodexAgent":
        self._tmp = tempfile.TemporaryDirectory(prefix="codex-bench-")
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()

    async def ask(self, prompt: str) -> str:
        if self._tmp is None:
            raise RuntimeError(
                "CodexAgent must be entered with 'async with' before use")
        cmd = ["codex", "-q", "--full-auto", "--writable-root", self._tmp.name,
               "--model", self._model, prompt]
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=self._tmp.name,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"codex exited {proc.returncode}: {stderr.decode().strip()}")
        return stdout.decode().strip()
