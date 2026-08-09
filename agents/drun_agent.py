"""Agent backed by drun."""
from __future__ import annotations

from typing import Any

from drun.chat import ChatAgent
from drun.mcp_bridge import DrunMcpBridge

from .base import Agent

DEFAULT_MCP_URL = "http://127.0.0.1:7273/mcp"
DEFAULT_MODEL = "ollama_chat/qwen3.6:latest"


class DrunAgent(Agent):
    def __init__(
        self,
        *,
        mcp_url: str = DEFAULT_MCP_URL,
        model: str = DEFAULT_MODEL,
        **chat_kwargs: Any,
    ) -> None:
        self._bridge = DrunMcpBridge(mcp_url)
        self._model = model
        self._chat_kwargs = chat_kwargs
        self._chat_agent: ChatAgent | None = None

    async def __aenter__(self) -> "DrunAgent":
        await self._bridge.__aenter__()
        self._chat_agent = ChatAgent(
            self._bridge, model=self._model, **self._chat_kwargs)
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self._bridge.__aexit__(*exc_info)

    async def ask(self, prompt: str) -> str:
        if self._chat_agent is None:
            raise RuntimeError(
                "DrunAgent must be entered with 'async with' before use")
        return await self._chat_agent.run(prompt)
