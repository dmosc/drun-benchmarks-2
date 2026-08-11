"""Latency/token/recall stress benchmark.

Runs a needle-in-a-haystack search task at a ladder of increasing input
sizes, repeated per size, against a sandboxed harness (DrunAgent) and an
unsandboxed one (OpenInterpreterAgent) side by side. Unlike basic_qa.py,
this isn't about coverage of distinct skills — it's a handful of trials,
rerun several times per size, to trace how each harness's latency and
token spend actually scale as the task grows, and where the two harnesses'
cost curves cross.

The task itself carries no hint about method (same reasoning as the
search-efficiency question in basic_qa.py): a brute-force approach costs
more in latency/tokens as size grows, and that shows up in the curves
rather than in whether the agent followed an explicit instruction to be
fast.
"""
from __future__ import annotations

import asyncio
import random
import statistics
from dataclasses import dataclass

from dotenv import load_dotenv

from agents import Agent, DrunAgent, OpenInterpreterAgent

load_dotenv(override=True)

# Number of files to create per trial. Geometric so the curve covers both
# cheap and stressed-out scales. Pushing this further is just extending the
# list -- DrunAgent's max_iterations below is already sized with headroom
# for a one-tool-call-per-file worst case at the largest size.
SIZES = [10, 20, 40, 80, 160]
REPEATS = 5
SEED = 0  # fixed so reruns generate the same needle positions/codes

MODEL = "qwen3.6:latest"

HARNESSES: list[Agent] = [
    DrunAgent(name="drun (sandboxed)",
              model=f"ollama_chat/{MODEL}", max_iterations=250),
    OpenInterpreterAgent(
        name="open interpreter (unsandboxed)", model=f"ollama/{MODEL}"),
]


@dataclass(slots=True)
class Trial:
    size: int
    repeat: int
    question: str
    code: str


@dataclass(slots=True)
class Sample:
    harness: str
    size: int
    repeat: int
    latency_s: float
    input_tokens: int | None
    output_tokens: int | None
    tool_calls: int | None
    correct: bool


def make_trials() -> list[Trial]:
    rng = random.Random(SEED)
    trials = []
    for size in SIZES:
        width = max(2, len(str(size)))
        for repeat in range(REPEATS):
            needle = rng.randint(1, size)
            code = f"{rng.randint(1000, 9999)}-{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}"
            first = f"entry_{1:0{width}}.txt"
            last = f"entry_{size:0{width}}.txt"
            needle_name = f"entry_{needle:0{width}}.txt"
            question = (
                f"Create {size} files named {first} through {last}, each with "
                f"a short unrelated sentence, except {needle_name}, which must "
                f"contain the exact line 'ACCESS CODE: {code}'. Report which "
                f"file contains that line and the access code itself."
            )
            trials.append(Trial(size=size, repeat=repeat,
                          question=question, code=code))
    return trials


async def run_harness(agent: Agent, trials: list[Trial], samples: list[Sample]) -> None:
    async with agent:
        print(f"\n=== {agent.name} ===")
        for trial in trials:
            print(f"\n[size={trial.size} repeat={trial.repeat}] {agent.name}")
            answer = await agent.ask(trial.question)
            metrics = agent.metrics[-1]
            correct = trial.code in answer
            print(f"  correct={correct} ({metrics})")
            samples.append(Sample(
                harness=agent.name,
                size=trial.size,
                repeat=trial.repeat,
                latency_s=metrics.latency_s,
                input_tokens=metrics.input_tokens,
                output_tokens=metrics.output_tokens,
                tool_calls=metrics.tool_calls,
                correct=correct,
            ))


def summarize(samples: list[Sample]) -> dict[tuple[str, int], dict[str, float]]:
    grouped: dict[tuple[str, int], list[Sample]] = {}
    for sample in samples:
        grouped.setdefault((sample.harness, sample.size), []).append(sample)

    summary: dict[tuple[str, int], dict[str, float]] = {}
    for key, group in grouped.items():
        latencies = [s.latency_s for s in group]
        input_tokens = [
            s.input_tokens for s in group if s.input_tokens is not None]
        output_tokens = [
            s.output_tokens for s in group if s.output_tokens is not None]
        tool_calls = [s.tool_calls for s in group if s.tool_calls is not None]

        avg_latency = statistics.mean(latencies)
        stats: dict[str, float] = {
            "avg_latency_s": round(avg_latency, 3),
            "recall": round(statistics.mean(s.correct for s in group), 3),
        }
        if len(latencies) > 1:
            # Coefficient of variation: how consistent latency is at this size,
            # independent of its absolute scale, so it's comparable across sizes.
            stats["latency_cv"] = round(
                statistics.stdev(latencies) / avg_latency, 3)
        if input_tokens and output_tokens:
            avg_input, avg_output = statistics.mean(
                input_tokens), statistics.mean(output_tokens)
            avg_tokens = avg_input + avg_output
            stats["avg_tokens"] = round(avg_tokens, 1)
            stats["output_token_ratio"] = round(avg_output / avg_tokens, 3)
            stats["tokens_per_second"] = round(avg_tokens / avg_latency, 1)
        if tool_calls:
            stats["avg_tool_calls"] = round(statistics.mean(tool_calls), 2)
        if "avg_tokens" in stats and stats.get("avg_tool_calls"):
            stats["avg_tokens_per_call"] = round(
                stats["avg_tokens"] / stats["avg_tool_calls"], 1)
        summary[key] = stats

    _add_growth_rates(summary)
    return summary


def _add_growth_rates(summary: dict[tuple[str, int], dict[str, float]]) -> None:
    """Marginal tokens per additional file between consecutive sizes, per
    harness — whether cost grows *faster* with size, not just how big it is."""
    harnesses = sorted({harness for harness, _ in summary})
    for harness in harnesses:
        sizes = sorted(size for h, size in summary if h == harness)
        for prev, curr in zip(sizes, sizes[1:]):
            prev_tokens = summary[(harness, prev)].get("avg_tokens")
            curr_tokens = summary[(harness, curr)].get("avg_tokens")
            if prev_tokens is None or curr_tokens is None:
                continue
            summary[(harness, curr)]["token_growth_rate"] = round(
                (curr_tokens - prev_tokens) / (curr - prev), 2)


_PANELS = [
    ("avg_latency_s", "Latency (s)"),
    ("avg_tokens", "Tokens (input + output)"),
    ("recall", "Recall (fraction correct)"),
    ("avg_tool_calls", "Tool calls"),
    ("avg_tokens_per_call", "Tokens per tool call"),
    ("token_growth_rate", "Marginal tokens / additional file"),
    ("tokens_per_second", "Tokens / second"),
    ("output_token_ratio", "Output token share"),
    ("latency_cv", "Latency variability (stdev / mean)"),
]


def plot_metrics(summary: dict[tuple[str, int], dict[str, float]], path: str, ncols: int = 3) -> None:
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    harnesses = sorted({harness for harness, _ in summary})
    nrows = math.ceil(len(_PANELS) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    axes = axes.flatten()

    for ax, (field, label) in zip(axes, _PANELS):
        for harness in harnesses:
            sizes = sorted(size for h, size in summary if h == harness)
            values = [summary[(harness, size)].get(field) for size in sizes]
            points = [(s, v) for s, v in zip(sizes, values) if v is not None]
            if not points:
                continue
            xs, ys = zip(*points)
            ax.plot(xs, ys, marker="o", label=harness)
        ax.set_xlabel("Input size (files)")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend()
    for ax in axes[len(_PANELS):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


async def main() -> None:
    trials = make_trials()
    samples: list[Sample] = []
    for agent in HARNESSES:
        await run_harness(agent, trials, samples)

    print("\n=== per-size summary ===")
    summary = summarize(samples)
    for (harness, size), stats in sorted(summary.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        print(f"{harness} @ size={size}: {stats}")

    plot_metrics(summary, "perf_curves.png")
    print("\nWrote curves to perf_curves.png")


if __name__ == "__main__":
    asyncio.run(main())
