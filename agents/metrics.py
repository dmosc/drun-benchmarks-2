"""Per-`ask()` execution metrics, collected uniformly across harnesses."""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Iterable

_NUMERIC_FIELDS = (
    "latency_s", "answer_chars", "turns", "tool_calls",
    "input_tokens", "output_tokens", "cost_usd",
)


@dataclass(slots=True)
class Metrics:
    """One `ask()` call's stats. A harness that can't report a field leaves it None."""

    harness: str
    prompt: str
    latency_s: float
    answer_chars: int
    turns: int | None = None
    tool_calls: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None

    def __str__(self) -> str:
        values = ((name, getattr(self, name)) for name in _NUMERIC_FIELDS)
        return ", ".join(f"{name}={v}" for name, v in values if v is not None)


@dataclass(slots=True)
class MetricsLog:
    """Accumulates Metrics records across harnesses and questions."""

    records: list[Metrics] = field(default_factory=list)

    def extend(self, records: Iterable[Metrics]) -> None:
        self.records.extend(records)

    def by_harness(self) -> dict[str, list[Metrics]]:
        grouped: dict[str, list[Metrics]] = {}
        for record in self.records:
            grouped.setdefault(record.harness, []).append(record)
        return grouped

    def summary(self) -> dict[str, dict[str, float]]:
        return {harness: self._averages(records)
                for harness, records in self.by_harness().items()}

    @staticmethod
    def _averages(records: list[Metrics]) -> dict[str, float]:
        averages: dict[str, float] = {"calls": len(records)}
        for name in _NUMERIC_FIELDS:
            values = [v for r in records if (v := getattr(r, name)) is not None]
            if values:
                averages[f"avg_{name}"] = round(mean(values), 3)
        return averages
