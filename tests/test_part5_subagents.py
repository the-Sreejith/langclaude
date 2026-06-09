"""Part 5 — delegation: the `task` tool runs a subagent's own loop."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from conftest import scripted_model
from langclaude.part5_subagents import CODE_SEARCHER, build_agent


def test_subagent_dict_uses_system_prompt_key():
    # Regression guard for the blog's `prompt` -> `system_prompt` correction.
    assert {"name", "description", "system_prompt"} <= set(CODE_SEARCHER)
    assert "prompt" not in CODE_SEARCHER or "system_prompt" in CODE_SEARCHER


def test_delegation_runs_subagent_and_returns_summary():
    model = scripted_model(
        [
            # main agent delegates
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "args": {
                            "description": "find auth logic",
                            "subagent_type": "code-searcher",
                        },
                        "id": "k1",
                    }
                ],
            ),
            # subagent's own loop answers
            AIMessage(content="Auth lives in src/auth/login.py"),
            # main agent wraps up
            AIMessage(content="The auth logic is in src/auth/login.py"),
        ]
    )
    agent = build_agent(model)

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "where is the auth logic?"}]}
    )

    # A `task` ToolMessage proves the subagent ran and reported back.
    assert any(getattr(m, "name", None) == "task" for m in result["messages"])
    assert "login.py" in result["messages"][-1].content
