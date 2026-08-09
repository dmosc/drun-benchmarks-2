"""Agent backed by the Codex CLI.

Runs in --full-auto mode (edits and commands auto-approved) with the writable
root pinned to a fresh scratch directory, mirroring ClaudeCodeAgent's isolation.
Quiet mode only prints the final answer, so no turns/tokens/tool_calls are
observable here — those fields stay None.
"""
from __future__ import annotations

import asyncio
import tempfile
from typing import Any

from .base import Agent

DEFAULT_MODEL = "gpt-4o"


class CodexAgent(Agent):
    def __init__(self, *, name: str | None = None, model: str | None = None) -> None:
        super().__init__(name=name, model=model or DEFAULT_MODEL)
        self._tmp: tempfile.TemporaryDirectory[str] | None = None

    async def __aenter__(self) -> "CodexAgent":
        self._tmp = tempfile.TemporaryDirectory(prefix="codex-bench-")
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()

    async def _ask(self, prompt: str) -> tuple[str, dict[str, Any]]:
        if self._tmp is None:
            raise RuntimeError(
                "CodexAgent must be entered with 'async with' before use")
        cmd = ["codex", "-q", "--full-auto", "--writable-root", self._tmp.name,
               "--model", self.model, prompt]
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=self._tmp.name,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"codex exited {proc.returncode}: {stderr.decode().strip()}")
        return stdout.decode().strip(), {}
