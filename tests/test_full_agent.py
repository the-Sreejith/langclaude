"""Putting it all together — the full agent assembles and the loop runs."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from conftest import requires_api_key, scripted_model
from langclaude.full_agent import build_agent


def _tool_names(agent) -> set[str]:
    node = agent.nodes["tools"]
    bound = getattr(node, "bound", node)
    return set(getattr(bound, "tools_by_name", {}).keys())


def test_full_agent_has_every_capability_wired():
    agent = build_agent(scripted_model([AIMessage(content="ok")]))
    names = _tool_names(agent)
    # planning, files, shell, delegation, and the custom tool are all present
    assert {"write_todos", "read_file", "execute", "task", "run_tests"} <= names


def test_full_agent_loop_runs_through_planning():
    # Use write_todos (a safe built-in) so the loop runs without shelling out
    # to a real, recursive pytest invocation via the custom run_tests tool.
    model = scripted_model(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_todos",
                        "args": {"todos": [{"content": "Investigate", "status": "pending"}]},
                        "id": "r1",
                    }
                ],
            ),
            AIMessage(content="Planned the work."),
        ]
    )
    agent = build_agent(model)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "fix the failing login tests"}]},
        {"configurable": {"thread_id": "full"}},
    )
    assert result["messages"][-1].type == "ai"
    assert result["todos"] == [{"content": "Investigate", "status": "pending"}]


@requires_api_key
def test_full_agent_live():
    agent = build_agent()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "List the tools you have available."}]},
        {"configurable": {"thread_id": "live-full"}},
    )
    assert result["messages"][-1].content
