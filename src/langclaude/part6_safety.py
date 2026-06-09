"""Part 6 — Safety and human-in-the-loop: the brakes.

Real safety lives outside the model, in the harness:

* ``interrupt_on={"<tool>": True}`` pauses the loop *before* a risky tool runs
  (LangGraph native interrupts). A checkpointer is required for interrupts.
  Any tool can be gated by name — here the shell AND the file-writing tools.
* A backend is required to enable the ``execute`` shell tool.
* Resume the paused loop with
  ``agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config)``
  (or ``"reject"`` / ``"edit"`` / ``"respond"``). Verified in
  ``tests/test_part6_safety.py``.

IMPORTANT (differs from the blog's framing): ``LocalShellBackend`` is NOT a
sandbox. Its own deprecation warning states it "provides no sandboxing
(execute() runs commands on the host)". For OS-level isolation use a real
sandbox backend (e.g. LangSmithSandbox). See docs/FINDINGS.md.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver

from langclaude import DEFAULT_MODEL

# The destructive tools: the shell, plus anything that changes files.
INTERRUPT_ON = {"execute": True, "write_file": True, "edit_file": True}


def build_agent(model: str | BaseChatModel = DEFAULT_MODEL, *, root_dir: str = "."):
    """An agent that pauses for approval before any destructive tool runs."""
    from deepagents import create_deep_agent
    from deepagents.backends import LocalShellBackend

    return create_deep_agent(
        model=model,
        system_prompt="You are a coding assistant working inside this project.",
        # Not a sandbox — see module docstring. virtual_mode set explicitly to
        # silence the 0.6.x default-change warning.
        backend=LocalShellBackend(root_dir=root_dir, virtual_mode=False),
        interrupt_on=INTERRUPT_ON,
        checkpointer=InMemorySaver(),  # interrupts require a checkpointer
    )
