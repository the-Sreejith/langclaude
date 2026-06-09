"""Part 1 — the loop fires end-to-end: model -> tool -> result -> exit."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from conftest import requires_api_key, scripted_model
from langclaude.part1_loop import build_agent


def test_loop_runs_tool_then_exits_on_plain_text():
    model = scripted_model(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_weather",
                        "args": {"city": "San Francisco"},
                        "id": "call_1",
                    }
                ],
            ),
            AIMessage(content="It's always sunny in San Francisco!"),
        ]
    )
    agent = build_agent(model)

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Weather in San Francisco?"}]}
    )

    final = result["messages"][-1]
    assert final.type == "ai"
    assert "sunny" in final.content
    # The real tool output must appear in the transcript (proves it executed).
    tool_msgs = [m for m in result["messages"] if m.type == "tool"]
    assert any("sunny in San Francisco" in m.content for m in tool_msgs)


def test_agent_is_a_compiled_graph():
    from langgraph.graph.state import CompiledStateGraph

    agent = build_agent(scripted_model([AIMessage(content="hi")]))
    assert isinstance(agent, CompiledStateGraph)


@requires_api_key
def test_loop_live():
    """Live smoke test against the real model (only with an API key set)."""
    agent = build_agent()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What's the weather in Paris?"}]}
    )
    assert result["messages"][-1].content
