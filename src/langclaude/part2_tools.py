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


@tool
def run_tests(path: str = ".") -> str:
    """Run the project's pytest suite and return the output."""
    result = subprocess.run(
        ["pytest", path], capture_output=True, text=True, check=False
    )
    return result.stdout + result.stderr


def build_agent(model: str | BaseChatModel = DEFAULT_MODEL):
    """An agent with a custom tool merged alongside the built-ins."""
    from deepagents import create_deep_agent

    return create_deep_agent(
        model=model,
        tools=[run_tests],
        system_prompt="You are a coding assistant. Always run tests after editing.",
    )
