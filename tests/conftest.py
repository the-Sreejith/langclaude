"""Shared fixtures.

The whole point of these helpers is to validate the *harness mechanics*
(the loop, tool execution, planning, delegation, interrupts, persistence)
deterministically and for free — no Anthropic API key, no token cost.

``FakeLoopModel`` plays back a scripted list of AIMessages. Give it an
``AIMessage`` with ``tool_calls`` to make the agent execute a tool, and a
plain-text ``AIMessage`` to make the loop exit.
"""

from __future__ import annotations

import inspect
import os
from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage


class FakeLoopModel(GenericFakeChatModel):
    """A scripted chat model that accepts (and ignores) ``bind_tools``."""

    def bind_tools(self, *args, **kwargs):  # noqa: D102 - harness calls this
        return self


class RecordingModel(FakeLoopModel):
    """A scripted model that also records every prompt it receives.

    Lets a test prove *what the model was shown* — e.g. that compaction
    replaced old turns with a summary before the model call.
    """

    recorded: list[list[Any]] = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.recorded.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def scripted_model(messages: Sequence[AIMessage]) -> FakeLoopModel:
    """Build a fake model that emits ``messages`` in order across loop turns."""
    return FakeLoopModel(messages=iter(messages))


def tool_names(agent) -> set[str]:
    """Names of all tools wired into the agent's tool node."""
    node = agent.nodes["tools"]
    bound = getattr(node, "bound", node)
    return set(getattr(bound, "tools_by_name", {}).keys())


def installed_middleware(agent) -> dict[str, Any]:
    """Map middleware class name -> instance wired into the agent's model node.

    Most deepagents middleware hooks ``wrap_model_call``, so it never appears
    as a graph node. The instances live in the closures of the composed model
    node function — walk them.
    """
    node = agent.nodes["model"]
    bound = getattr(node, "bound", node)
    found: dict[str, Any] = {}
    seen: set[int] = set()

    def walk(func: Any, depth: int = 0) -> None:
        try:
            closure = inspect.getclosurevars(func)
        except TypeError:
            return
        for value in closure.nonlocals.values():
            if id(value) in seen:
                continue
            seen.add(id(value))
            if "Middleware" in type(value).__name__:
                found[type(value).__name__] = value
            elif hasattr(value, "__self__") and "Middleware" in type(value.__self__).__name__:
                found[type(value.__self__).__name__] = value.__self__
            elif callable(value) and depth < 8 and getattr(value, "__closure__", None):
                walk(value, depth + 1)

    walk(bound.func)
    return found


requires_api_key = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set; skipping live API test",
)
