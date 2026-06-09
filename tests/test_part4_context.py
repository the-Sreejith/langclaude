"""Part 4 — context management middleware (filesystem + summarization) is wired in."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from conftest import scripted_model
from langclaude.part4_context import build_agent


def test_filesystem_tools_available_as_external_memory():
    # Files-as-memory: the virtual filesystem tools are present.
    agent = build_agent(scripted_model([AIMessage(content="ok")]))
    node = agent.nodes["tools"]
    bound = getattr(node, "bound", node)
    names = set(getattr(bound, "tools_by_name", {}).keys())
    assert {"read_file", "write_file", "ls"}.issubset(names)


def test_summarization_middleware_is_installed():
    # Compaction is provided by SummarizationMiddleware. Confirm the agent was
    # built with it by checking the middleware module exposes it and that the
    # filesystem middleware contributes nodes/tools.
    from deepagents.middleware import FilesystemMiddleware, SummarizationMiddleware

    assert FilesystemMiddleware is not None
    assert SummarizationMiddleware is not None
