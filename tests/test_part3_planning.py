"""Part 3 — write_todos is built in and writes the plan into graph state."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from conftest import scripted_model
from langclaude.part3_planning import build_agent


def test_write_todos_populates_state():
    todos = [
        {"content": "Read the failing test", "status": "pending"},
        {"content": "Fix the bug", "status": "pending"},
    ]
    model = scripted_model(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "write_todos", "args": {"todos": todos}, "id": "t1"}],
            ),
            AIMessage(content="Plan written."),
        ]
    )
    agent = build_agent(model)

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "fix the login bug"}]}
    )

    assert result["todos"] == todos


def test_todo_middleware_node_present():
    agent = build_agent(scripted_model([AIMessage(content="ok")]))
    nodes = list(agent.get_graph().nodes.keys())
    assert any("Todo" in n for n in nodes)
