"""Part 6 — the brakes: pause before a risky tool, then approve or reject."""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.types import Command

from conftest import scripted_model
from langclaude.part6_safety import build_agent


def _execute_call(command: str, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "execute", "args": {"command": command}, "id": call_id}],
    )


def test_interrupt_pauses_before_execute():
    model = scripted_model([_execute_call("rm -rf build", "e1"), AIMessage(content="done")])
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


def test_approve_resumes_and_actually_runs_the_command(tmp_path):
    model = scripted_model(
        [_execute_call("echo approved-and-ran", "e1"), AIMessage(content="done")]
    )
    agent = build_agent(model, root_dir=str(tmp_path))
    config = {"configurable": {"thread_id": "hil-approve"}}

    agent.invoke({"messages": [{"role": "user", "content": "echo something"}]}, config)
    result = agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}), config
    )

    tool_msgs = [m for m in result["messages"] if m.type == "tool"]
    assert any("approved-and-ran" in str(m.content) for m in tool_msgs)
    assert result["messages"][-1].content == "done"


def test_reject_resumes_without_running_the_command(tmp_path):
    model = scripted_model(
        [
            _execute_call("echo should-never-run", "e1"),
            AIMessage(content="understood, not running it"),
        ]
    )
    agent = build_agent(model, root_dir=str(tmp_path))
    config = {"configurable": {"thread_id": "hil-reject"}}

    agent.invoke({"messages": [{"role": "user", "content": "run it"}]}, config)
    result = agent.invoke(
        Command(resume={"decisions": [{"type": "reject"}]}), config
    )

    tool_msgs = [m for m in result["messages"] if m.type == "tool"]
    assert not any("should-never-run" in str(m.content) for m in tool_msgs)
    assert any("rejected" in str(m.content).lower() for m in tool_msgs)
    assert result["messages"][-1].content == "understood, not running it"


def test_local_shell_backend_is_not_a_sandbox(tmp_path):
    # Documenting reality, behaviorally: even with root_dir set, execute() runs
    # on the host and can read files far outside root_dir. This is the claim
    # the blog corrects — the backend is a tool provider, not an OS boundary.
    from deepagents.backends import LocalShellBackend

    backend = LocalShellBackend(root_dir=str(tmp_path), virtual_mode=False)
    response = backend.execute("cat /etc/hosts")
    assert "localhost" in response.output
