"""Basic Q/A benchmark: runs a fixed Q/A set through every harness and prints replies."""
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
    # Health check.
    "What is 7 * 6?",
    # Environment awareness and basic operations.
    "What operating system is this sandbox running on? Check with a shell command.",
    # Mutation skills; showcasing standard utilities.
    "Write a file named hello.txt containing 'hello world', then read it back to confirm.",
    # Fetch and process information. The drun sandbox exposes a handful of tool
    # calls for efficient data traversal.
    "Download the contents of https://arxiv.org/pdf/1803.07199 and list me out all the Fibonacci algorithms documented in there.",
    # Rollback: undo a change and prove the prior state came back; this
    # showcases drun checkpointing ability.
    "Create a file named counter.txt containing the number 1. Update it to 2 "
    "and confirm the change. Then undo that last edit so the file reads 1 "
    "again, and show its final contents to prove the rollback worked.",
    # Speed: many small sequential operations, timed.
    "Create 10 files named part_01.txt through part_10.txt, in order, each "
    "containing only its own two-digit number, verifying each write before "
    "starting the next. Report how long the whole batch took.",
    # Search efficiency: find a needle among many small files. No hint on
    # method — a brute-force read-every-file approach costs 25x the tool
    # calls of a single search, which shows up in the tool_calls/latency
    # metrics rather than in whether the agent obeyed an instruction to be fast.
    "Create 25 files named entry_01.txt through entry_25.txt, each with a "
    "short unrelated sentence, except entry_18.txt, which must contain the "
    "exact line 'ACCESS CODE: 4471-Q'. Report which file contains that line "
    "and the access code itself.",
    # Targeted reading: same idea over one large file instead of many small
    # ones — reading it all costs proportionally more tokens/latency than
    # jumping straight to the line, again left to the metrics to show.
    "Generate a file named big_log.txt with 500 lines formatted as 'LINE "
    "<n>: ok', except line 342, which should read 'LINE 342: FATAL disk "
    "failure'. Report that line's exact number and content.",
    # History/diff: recall an intermediate state, not just the latest change.
    "Create a file named report.txt containing 'draft'. Update it to "
    "'reviewed', then update it again to 'final'. Report exactly what "
    "changed between the first version and the second version ('draft' to "
    "'reviewed'), not the most recent change.",
    # Error recovery: fail first, then resolve it within the same task.
    "Try to read a file named missing.txt that does not exist yet, observe "
    "the failure, then create that file with the content 'recovered' and "
    "read it back to confirm you completed the task despite the initial "
    "failure.",
]

HARNESSES: list[Agent] = [
    DrunAgent(name="drun open model", model="ollama_chat/qwen3.6:latest"),
    OpenInterpreterAgent(name="open model", model="ollama/qwen3.6:latest"),
    # Commenting out the following entries because testing them requires an API
    # key which can get costly.
    #
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
