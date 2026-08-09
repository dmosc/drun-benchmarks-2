"""Agent backed by drun."""
from __future__ import annotations

from typing import Any

from drun.chat import ChatAgent
from drun.mcp_bridge import DrunMcpBridge

from .base import Agent

DEFAULT_MCP_URL = "http://127.0.0.1:7273/mcp"
DEFAULT_MODEL = "ollama_chat/qwen3.6:latest"


class _CountingBridge:
    """Wraps a Bridge to count tool calls, without touching drun-py internals."""

    def __init__(self, inner: DrunMcpBridge) -> None:
        self._inner = inner
        self.call_count = 0

    @property
    def default_system_prompt(self) -> str:
        return self._inner.default_system_prompt

    async def tools(self) -> list[dict[str, Any]]:
        return await self._inner.tools()

    async def call(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        self.call_count += 1
        return await self._inner.call(name, arguments)


class DrunAgent(Agent):
    def __init__(
        self,
        *,
        mcp_url: str = DEFAULT_MCP_URL,
        model: str = DEFAULT_MODEL,
        **chat_kwargs: Any,
    ) -> None:
        super().__init__()
        self._bridge = DrunMcpBridge(mcp_url)
        self._model = model
        self._chat_kwargs = chat_kwargs
        self._counting_bridge: _CountingBridge | None = None
        self._chat_agent: ChatAgent | None = None

    async def __aenter__(self) -> "DrunAgent":
        await self._bridge.__aenter__()
        self._counting_bridge = _CountingBridge(self._bridge)
        self._chat_agent = ChatAgent(
            self._counting_bridge, model=self._model, **self._chat_kwargs)
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self._bridge.__aexit__(*exc_info)

    async def _ask(self, prompt: str) -> tuple[str, dict[str, Any]]:
        if self._chat_agent is None or self._counting_bridge is None:
            raise RuntimeError(
                "DrunAgent must be entered with 'async with' before use")
        calls_before = self._counting_bridge.call_count
        answer = await self._chat_agent.run(prompt)
        tool_calls = self._counting_bridge.call_count - calls_before
        return answer, {"tool_calls": tool_calls}
