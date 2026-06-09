"""Part 1 — The loop: the engine that drives everything.

``create_deep_agent`` returns a compiled graph with the agent loop already
wired up: ask the model, run any requested tool, feed the result back, repeat
until the model replies with plain text.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from langclaude import DEFAULT_MODEL


def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    return f"It's always sunny in {city}!"


def build_agent(model: str | BaseChatModel = DEFAULT_MODEL):
    """The simplest possible Deep Agent — just the loop plus one tool."""
    from deepagents import create_deep_agent

    return create_deep_agent(
        model=model,
        tools=[get_weather],
        system_prompt="You are a helpful assistant.",
    )
