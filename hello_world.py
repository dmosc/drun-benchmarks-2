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

HARNESSES: list[Agent] = [
    DrunAgent(name="drun open model", model="ollama_chat/qwen3.6:latest"),
    OpenInterpreterAgent(name="open model", model="ollama/qwen3.6:latest"),
    # DrunAgent(name="drun claude", model="claude-sonnet-5"),
    # ClaudeCodeAgent(name="claude"),
    # DrunAgent(name="drun chatgpt", model="gpt-4o"),
    # CodexAgent(name="chatgpt"),
]


async def run_harness(agent: Agent, log: MetricsLog) -> None:
    async with agent:
        print(f"\n=== {agent.name} ===")
        for question in QUESTIONS:
            print(f"\nQ: {question}")
            answer = await agent.ask(question)
            print(f"A: {answer}")
            print(f"  ({agent.metrics[-1]})")
    log.extend(agent.metrics)


async def main() -> None:
    log = MetricsLog()
    for agent in HARNESSES:
        await run_harness(agent, log)

    print("\n=== summary ===")
    for harness, stats in log.summary().items():
        print(f"{harness}: {stats}")


if __name__ == "__main__":
    asyncio.run(main())
