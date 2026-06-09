"""Part 4 — Context management: beating the memory limit.

Two halves, both built in:

* Files as external memory — the FilesystemMiddleware exposes a virtual
  filesystem so large results can be written to a file and recalled by name
  instead of bloating the conversation.
* Compaction — the SummarizationMiddleware summarizes older turns when the
  window fills up.

Neither middleware appears as a graph node (they hook ``wrap_model_call``),
so the tests verify them by walking the model node's closures — see
``tests/conftest.py::installed_middleware`` — and by actually making
compaction fire with a low trigger (``tests/test_part4_context.py``).
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
