"""Part 7 — Memory and persistence: remembering across turns.

Because Deep Agents runs on LangGraph, plugging in a checkpointer makes the
agent's state survive across ``.invoke`` calls. A ``thread_id`` ties messages
into one ongoing conversation. ``InMemorySaver`` lasts for the life of the
process; swap in a database-backed checkpointer for durability across reboots.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from langclaude import DEFAULT_MODEL


def build_agent(
    model: str | BaseChatModel = DEFAULT_MODEL,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
):
    """An agent that remembers earlier turns within a thread."""
    from deepagents import create_deep_agent

    return create_deep_agent(
        model=model,
        tools=[],
        system_prompt="You are a coding assistant.",
        checkpointer=checkpointer or InMemorySaver(),
    )
