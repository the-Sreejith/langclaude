"""Part 6 — the brakes: the loop pauses for approval before a risky tool."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from conftest import scripted_model
from langclaude.part6_safety import build_agent


def test_interrupt_pauses_before_execute():
    model = scripted_model(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "execute", "args": {"command": "rm -rf build"}, "id": "e1"}
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    agent = build_agent(model, root_dir=".")
    config = {"configurable": {"thread_id": "hil-test"}}

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "clean the build dir"}]}, config
    )

    # The harness interrupted BEFORE running the shell command.
    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    actions = [a["name"] for a in payload["action_requests"]]
    assert "execute" in actions
    decisions = payload["review_configs"][0]["allowed_decisions"]
    assert {"approve", "reject"} <= set(decisions)


def test_local_shell_backend_is_not_a_sandbox():
    # Documenting reality: LocalShellBackend runs on the host, no isolation.
    from deepagents.backends import LocalShellBackend

    backend = LocalShellBackend(root_dir=".", virtual_mode=False)
    # It exposes an execute() that runs on the host — there is no sandbox class
    # boundary here. This test simply asserts the type the blog points at.
    assert backend.__class__.__name__ == "LocalShellBackend"
