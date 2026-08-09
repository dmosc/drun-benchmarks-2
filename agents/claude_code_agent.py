"""Agent backed by Claude Code."""
from __future__ import annotations

import asyncio
import json
import tempfile

from .base import Agent


class ClaudeCodeAgent(Agent):
    def __init__(self, *, model: str | None = None) -> None:
        self._model = model
        self._tmp: tempfile.TemporaryDirectory[str] | None = None

    async def __aenter__(self) -> "ClaudeCodeAgent":
        self._tmp = tempfile.TemporaryDirectory(prefix="claude-code-bench-")
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()

    async def ask(self, prompt: str) -> str:
        if self._tmp is None:
            raise RuntimeError(
                "ClaudeCodeAgent must be entered with 'async with' before use")
        cmd = ["claude", "-p", prompt, "--output-format",
               "json", "--dangerously-skip-permissions"]
        if self._model:
            cmd += ["--model", self._model]
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=self._tmp.name,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"claude exited {proc.returncode}: {stderr.decode().strip()}")
        return json.loads(stdout)["result"]
