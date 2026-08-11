"""Tracks token usage across every harness whose LLM calls route through litellm."""
from __future__ import annotations

from typing import Any

_METER: Any | None = None


def install() -> Any:
    """Registers the shared token meter with litellm"""
    global _METER
    if _METER is not None:
        return _METER

    import litellm
    from litellm.integrations.custom_logger import CustomLogger

    class _TokenMeter(CustomLogger):
        def __init__(self) -> None:
            super().__init__()
            self.input_tokens = 0
            self.output_tokens = 0

        def mark(self) -> tuple[int, int]:
            return (self.input_tokens, self.output_tokens)

        def delta(self, mark: tuple[int, int]) -> tuple[int, int]:
            return (self.input_tokens - mark[0], self.output_tokens - mark[1])

        def _record(self, response_obj: Any) -> None:
            usage = getattr(response_obj, "usage", None)
            if usage is None:
                return
            self.input_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.output_tokens += getattr(usage, "completion_tokens", 0) or 0

        def log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
            self._record(response_obj)

        async def async_log_success_event(
            self, kwargs, response_obj, start_time, end_time
        ) -> None:
            self._record(response_obj)

    _METER = _TokenMeter()
    litellm.callbacks.append(_METER)
    return _METER
