"""Shared fixtures.

The whole point of these helpers is to validate the *harness mechanics*
(the loop, tool execution, planning, delegation, interrupts, persistence)
deterministically and for free — no Anthropic API key, no token cost.

``FakeLoopModel`` plays back a scripted list of AIMessages. Give it an
``AIMessage`` with ``tool_calls`` to make the agent execute a tool, and a
plain-text ``AIMessage`` to make the loop exit.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage


class FakeLoopModel(GenericFakeChatModel):
    """A scripted chat model that accepts (and ignores) ``bind_tools``."""

    def bind_tools(self, *args, **kwargs):  # noqa: D102 - harness calls this
        return self


def scripted_model(messages: Sequence[AIMessage]) -> FakeLoopModel:
    """Build a fake model that emits ``messages`` in order across loop turns."""
    return FakeLoopModel(messages=iter(messages))


requires_api_key = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set; skipping live API test",
)
