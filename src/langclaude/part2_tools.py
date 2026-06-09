"""Part 2 — Tools: giving the agent hands.

Deep Agents ships a full built-in tool suite (read_file, ls, glob, grep,
write_file, edit_file, execute, task, write_todos). You add custom tools on
top with the ``@tool`` decorator — the docstring is the model's instruction
manual for when and how to call it.
"""

from __future__ import annotations

import subprocess

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool

from langclaude import DEFAULT_MODEL

# Token budgeting applies to YOUR tools too, not just the built-in read tools:
# a failing suite can emit megabytes of tracebacks — exactly the context flood
# a purpose-built tool exists to prevent. pytest puts failures at the end, so
# truncation keeps the tail.
MAX_OUTPUT_CHARS = 20_000  # roughly 5k tokens
TEST_TIMEOUT_SECONDS = 300


def clip_output(output: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Keep a tool result inside the context budget (keeps the tail)."""
    if len(output) <= limit:
        return output
    return f"[... output truncated to the last {limit} characters ...]\n" + output[-limit:]


@tool
def run_tests(path: str = ".") -> str:
    """Run the project's pytest suite and return its output.

    Output is truncated to the last 20k characters (failures appear at the
    end). The run is killed after 5 minutes.
    """
    try:
        result = subprocess.run(
            ["pytest", path],
            capture_output=True,
            text=True,
            check=False,
            timeout=TEST_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return f"pytest timed out after {TEST_TIMEOUT_SECONDS}s"
    return clip_output(result.stdout + result.stderr)


def build_agent(model: str | BaseChatModel = DEFAULT_MODEL):
    """An agent with a custom tool merged alongside the built-ins."""
    from deepagents import create_deep_agent

    return create_deep_agent(
        model=model,
        tools=[run_tests],
        system_prompt="You are a coding assistant. Always run tests after editing.",
    )
