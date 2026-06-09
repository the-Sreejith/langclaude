"""Part 0 — the loop from scratch: no framework, just the raw Messages API.

This is the whole engine of a coding agent in ~40 lines. Everything else in
this repo (and in deepagents) is this loop with batteries: ask the model, run
any tool it requests, feed the result back, repeat until it answers in plain
text. Seeing it bare makes the rest of the harness legible.

The ``client`` is injectable: pass ``anthropic.Anthropic()`` to run live, or a
scripted fake (see ``tests/test_part0_scratch_loop.py``) to exercise the loop
deterministically without an API key.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

ToolFunction = Callable[..., str]

WEATHER_TOOL_SCHEMA: dict[str, Any] = {
    "name": "get_weather",
    "description": "Get the weather for a given city.",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    return f"It's always sunny in {city}!"


DEFAULT_TOOLS: list[dict[str, Any]] = [WEATHER_TOOL_SCHEMA]
DEFAULT_TOOL_FUNCTIONS: dict[str, ToolFunction] = {"get_weather": get_weather}


def run_agent_loop(
    client: Any,
    user_message: str,
    *,
    model: str = "claude-sonnet-4-6",
    tools: list[dict[str, Any]] | None = None,
    tool_functions: dict[str, ToolFunction] | None = None,
    max_turns: int = 20,
) -> tuple[str, list[dict[str, Any]]]:
    """The agent loop, by hand. Returns (final_text, full_transcript)."""
    tools = tools if tools is not None else DEFAULT_TOOLS
    tool_functions = tool_functions if tool_functions is not None else DEFAULT_TOOL_FUNCTIONS

    messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
    for _ in range(max_turns):
        # 1. Ask the model what to do next.
        response = client.messages.create(
            model=model, max_tokens=1024, tools=tools, messages=messages
        )
        messages = [*messages, {"role": "assistant", "content": response.content}]

        # 2. Plain text and no tool request? The job is done — exit the loop.
        tool_uses = [block for block in response.content if block.type == "tool_use"]
        if not tool_uses:
            final = "".join(block.text for block in response.content if block.type == "text")
            return final, messages

        # 3. Run each requested tool and hand the results back. 4. Repeat.
        results = [
            {
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": tool_functions[call.name](**call.input),
            }
            for call in tool_uses
        ]
        messages = [*messages, {"role": "user", "content": results}]

    raise RuntimeError(f"agent loop did not finish within {max_turns} turns")
