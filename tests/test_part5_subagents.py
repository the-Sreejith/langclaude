"""Part 5 — delegation: the `task` tool runs a subagent's own loop."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from conftest import installed_middleware, scripted_model
from langclaude.part5_subagents import CODE_SEARCHER, build_agent


def test_subagent_dict_uses_system_prompt_key():
    # Regression guard for the blog's `prompt` -> `system_prompt` correction.
    assert {"name", "description", "system_prompt"} <= set(CODE_SEARCHER)


def test_prompt_key_is_rejected_by_deepagents():
    # The other half of the regression guard: the original post's `prompt` key
    # must actually fail. If a future deepagents starts accepting it, this
    # test flags that the correction is obsolete.
    from deepagents import create_deep_agent

    wrong_key = {
        "name": "code-searcher",
        "description": CODE_SEARCHER["description"],
        "prompt": CODE_SEARCHER["system_prompt"],  # the key the post used
    }
    with pytest.raises(KeyError, match="system_prompt"):
        create_deep_agent(
            model=scripted_model([AIMessage(content="ok")]),
            tools=[],
            system_prompt="x",
            subagents=[wrong_key],
        )


def test_subagents_cannot_spawn_subagents():
    # The depth limit the blog claims: a subagent's middleware stack contains
    # no SubAgentMiddleware, so it has no `task` tool and cannot nest helpers.
    agent = build_agent(scripted_model([AIMessage(content="ok")]))
    sub_mw = installed_middleware(agent)["SubAgentMiddleware"]
    searcher = next(s for s in sub_mw._subagents if s["name"] == "code-searcher")
    sub_stack = {type(m).__name__ for m in searcher["middleware"]}
    assert "SubAgentMiddleware" not in sub_stack


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
