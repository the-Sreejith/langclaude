"""Part 4 — Context management: beating the memory limit.

Two halves, both built in:

* Files as external memory — the FilesystemMiddleware exposes a virtual
  filesystem so large results can be written to a file and recalled by name
  instead of bloating the conversation.
* Compaction — the SummarizationMiddleware summarizes older turns when the
  window fills up.

``inspect_middleware`` lets the tests confirm both middlewares are present on
the assembled agent.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from langclaude import DEFAULT_MODEL


def build_agent(model: str | BaseChatModel = DEFAULT_MODEL):
    """Filesystem + summarization middleware are wired in automatically."""
    from deepagents import create_deep_agent

    return create_deep_agent(
        model=model,
        tools=[],
        system_prompt="You are a coding assistant. Offload large results to files.",
    )


def middleware_names(agent) -> list[str]:
    """Return the class names of the graph nodes (middleware hooks show up here)."""
    return list(agent.get_graph().nodes.keys())
