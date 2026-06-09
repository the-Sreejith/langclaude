"""Putting it all together — every part in one agent.

Loop (Part 1) + custom tool (Part 2) + planning (Part 3) + context management
(Part 4) + subagent (Part 5) + shell/safety (Part 6) + persistence (Part 7).
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from langclaude import DEFAULT_MODEL
from langclaude.part2_tools import run_tests
from langclaude.part5_subagents import CODE_SEARCHER

SYSTEM_PROMPT = """You are a careful coding assistant.
Workflow:
1. Plan the task as a to-do list before doing anything.
2. Use your built-in read, grep, and glob tools to explore - never the raw
   shell equivalents like cat or grep.
3. Make focused edits.
4. ALWAYS run the tests after editing, and fix anything that breaks.
5. Delegate broad codebase searches to the code-searcher subagent.
"""


def build_agent(
    model: str | BaseChatModel = DEFAULT_MODEL,
    *,
    root_dir: str = ".",
    checkpointer: BaseCheckpointSaver | None = None,
    with_backend: bool = True,
):
    """Assemble the full agent.

    ``with_backend=False`` skips the LocalShellBackend (handy for fake-model
    tests that don't need real shell access).
    """
    from deepagents import create_deep_agent

    kwargs: dict = dict(
        model=model,
        tools=[run_tests],
        system_prompt=SYSTEM_PROMPT,
        subagents=[CODE_SEARCHER],
        checkpointer=checkpointer or InMemorySaver(),
    )
    if with_backend:
        from deepagents.backends import LocalShellBackend

        kwargs["backend"] = LocalShellBackend(root_dir=root_dir, virtual_mode=False)

    return create_deep_agent(**kwargs)
