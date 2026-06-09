"""Part 0 — the hand-rolled loop: model -> tool -> result -> exit, no framework."""

from __future__ import annotations

from itertools import repeat
from types import SimpleNamespace
from typing import Any

import pytest

from langclaude.part0_scratch_loop import run_agent_loop


class FakeMessagesAPI:
    """Mimics ``anthropic.Anthropic().messages`` with scripted responses."""

    def __init__(self, responses):
        self._responses = iter(responses)
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return next(self._responses)


class FakeClient:
    def __init__(self, responses):
        self.messages = FakeMessagesAPI(responses)


def _tool_use(name: str, args: dict, call_id: str) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", name=name, input=args, id=call_id)


def _text(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def test_loop_runs_tool_then_exits_on_plain_text():
    client = FakeClient(
        [
            SimpleNamespace(content=[_tool_use("get_weather", {"city": "Paris"}, "t1")]),
            SimpleNamespace(content=[_text("Sunny in Paris.")]),
        ]
    )

    final, transcript = run_agent_loop(client, "Weather in Paris?")

    assert final == "Sunny in Paris."
    # The real tool function executed and its result was fed back to the model.
    second_request = client.messages.requests[1]["messages"]
    tool_results = [
        block
        for msg in second_request
        if isinstance(msg.get("content"), list)
        for block in msg["content"]
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]
    assert tool_results == [
        {"type": "tool_result", "tool_use_id": "t1", "content": "It's always sunny in Paris!"}
    ]
    assert transcript[0] == {"role": "user", "content": "Weather in Paris?"}


def test_loop_raises_if_it_never_terminates():
    # A model that calls a tool forever must hit the max_turns backstop.
    endless = repeat(SimpleNamespace(content=[_tool_use("get_weather", {"city": "X"}, "t")]))
    with pytest.raises(RuntimeError, match="did not finish"):
        run_agent_loop(FakeClient(endless), "loop forever", max_turns=3)
