"""Hello-world benchmark: runs a fixed Q/A set through every harness and prints replies."""
from __future__ import annotations

import asyncio

from dotenv import load_dotenv

from agents import (
    Agent,
    ClaudeCodeAgent,
    CodexAgent,
    DrunAgent,
    MetricsLog,
    OpenInterpreterAgent,
)

load_dotenv(override=True)

QUESTIONS = [
    "What is 7 * 6?",
    "What operating system is this sandbox running on? Check with a shell command.",
    "Write a file named hello.txt containing 'hello world', then read it back to confirm.",
]

HARNESSES: list[type[Agent]] = [
    DrunAgent,
    ClaudeCodeAgent,
    # CodexAgent,
    OpenInterpreterAgent,
]


async def run_harness(harness: type[Agent], log: MetricsLog) -> None:
    print(f"\n=== {harness.__name__} ===")
    async with harness() as agent:
        for question in QUESTIONS:
            print(f"\nQ: {question}")
            answer = await agent.ask(question)
            print(f"A: {answer}")
            print(f"  ({agent.metrics[-1]})")
    log.extend(agent.metrics)


async def main() -> None:
    log = MetricsLog()
    for harness in HARNESSES:
        await run_harness(harness, log)

    print("\n=== summary ===")
    for harness, stats in log.summary().items():
        print(f"{harness}: {stats}")


if __name__ == "__main__":
    asyncio.run(main())
