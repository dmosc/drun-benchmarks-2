"""Agent backed by the real Claude Code CLI and its own native tools (Bash, Write, Edit, ...).

Runs unattended via --dangerously-skip-permissions, so tool calls execute with
no approval prompt. Unlike DrunAgent's sandboxed session, Bash here has full
host access; a fresh scratch directory only limits where files land, not what
commands can do.
"""
from __future__ import annotations

import asyncio
import json
import tempfile
from typing import Any

from .base import Agent


class ClaudeCodeAgent(Agent):
    def __init__(self, *, model: str | None = None) -> None:
        super().__init__()
        self._model = model
        self._tmp: tempfile.TemporaryDirectory[str] | None = None

    async def __aenter__(self) -> "ClaudeCodeAgent":
        self._tmp = tempfile.TemporaryDirectory(prefix="claude-code-bench-")
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()

    async def _ask(self, prompt: str) -> tuple[str, dict[str, Any]]:
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
        data = json.loads(stdout)
        usage = data.get("usage", {})
        extra = {
            "turns": data.get("num_turns"),
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "cost_usd": data.get("total_cost_usd"),
        }
        return data["result"], extra
