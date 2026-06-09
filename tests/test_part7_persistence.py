"""Part 7 — persistence: state survives across invokes on the same thread_id."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from conftest import scripted_model
from langclaude.part7_persistence import build_agent


def test_checkpointer_remembers_earlier_turn():
    model = scripted_model(
        [
            AIMessage(content="Starting the refactor."),
            AIMessage(content="Now updating the tests, as discussed."),
        ]
    )
    agent = build_agent(model)
    config = {"configurable": {"thread_id": "project-alpha"}}

    agent.invoke(
        {"messages": [{"role": "user", "content": "Start refactoring the auth module."}]},
        config,
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Now update the tests too."}]},
        config,
    )

    # Both user turns are present -> the first turn was checkpointed and reloaded.
    human_turns = [m for m in result["messages"] if m.type == "human"]
    assert len(human_turns) == 2


def test_separate_threads_do_not_share_state():
    model = scripted_model([AIMessage(content="a"), AIMessage(content="b")])
    agent = build_agent(model)

    r1 = agent.invoke(
        {"messages": [{"role": "user", "content": "thread one"}]},
        {"configurable": {"thread_id": "t1"}},
    )
    r2 = agent.invoke(
        {"messages": [{"role": "user", "content": "thread two"}]},
        {"configurable": {"thread_id": "t2"}},
    )
    assert len([m for m in r1["messages"] if m.type == "human"]) == 1
    assert len([m for m in r2["messages"] if m.type == "human"]) == 1
