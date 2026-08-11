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
        tokens = [
            s.input_tokens + s.output_tokens for s in group
            if s.input_tokens is not None and s.output_tokens is not None
        ]
        stats = {
            "avg_latency_s": round(statistics.mean(s.latency_s for s in group), 3),
            "recall": round(statistics.mean(s.correct for s in group), 3),
        }
        if tokens:
            stats["avg_tokens"] = round(statistics.mean(tokens), 1)
        summary[key] = stats
    return summary


def plot_curves(summary: dict[tuple[str, int], dict[str, float]], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    harnesses = sorted({harness for harness, _ in summary})
    panels = [
        ("avg_latency_s", "Latency (s)"),
        ("avg_tokens", "Tokens (input + output)"),
        ("recall", "Recall (fraction correct)"),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4.5))
    for ax, (field, label) in zip(axes, panels):
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

    plot_curves(summary, "perf_curves.png")
    print("\nWrote curves to perf_curves.png")


if __name__ == "__main__":
    asyncio.run(main())
