"""Part 3 — Planning: thinking before doing.

The ``write_todos`` tool and its middleware are built in. The default system
prompt teaches the agent to write a to-do list before acting and to keep it
updated across the loop. The to-do list is stored in graph state under the
``todos`` key.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from langclaude import DEFAULT_MODEL


def build_agent(model: str | BaseChatModel = DEFAULT_MODEL):
    """Planning is on by default — nothing extra to configure."""
    from deepagents import create_deep_agent

    return create_deep_agent(
        model=model,
        tools=[],
        system_prompt="You are a careful coding assistant. Plan before you act.",
    )
