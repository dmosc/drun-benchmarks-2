"""Hello-world benchmark: runs a fixed Q/A set through Agent and prints replies."""
from __future__ import annotations

import asyncio

from agent import Agent

QUESTIONS = [
    "What is 7 * 6?",
    "What operating system is this sandbox running on? Check with a shell command.",
    "Write a file named hello.txt containing 'hello world', then read it back to confirm.",
]


async def main() -> None:
    async with Agent() as agent:
        for question in QUESTIONS:
            print(f"\nQ: {question}")
            answer = await agent.ask(question)
            print(f"A: {answer}")


if __name__ == "__main__":
    asyncio.run(main())
