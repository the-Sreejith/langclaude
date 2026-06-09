"""Part 5 — Subagents: divide and conquer.

The built-in ``task`` tool spawns a helper that runs its own loop in a fresh
context window and reports back a concise summary. You declare specialists as
dicts.

NOTE (differs from the blog post): in deepagents 0.6.8 the subagent dict key
for the helper's instructions is ``system_prompt``, not ``prompt``. The
required keys are ``name``, ``description`` and ``system_prompt``. See
docs/FINDINGS.md.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from langclaude import DEFAULT_MODEL

CODE_SEARCHER: dict[str, str] = {
    "name": "code-searcher",
    "description": (
        "Searches the codebase to find where specific logic lives. "
        "Use this for any open-ended 'where is X?' question."
    ),
    "system_prompt": (
        "You are an expert at navigating codebases. Use the grep and glob "
        "tools to locate relevant files, then report a concise summary of "
        "what you found and where. Do not make any edits."
    ),
}


def build_agent(model: str | BaseChatModel = DEFAULT_MODEL):
    """A main agent with a dedicated code-searcher specialist available."""
    from deepagents import create_deep_agent

    return create_deep_agent(
        model=model,
        tools=[],
        system_prompt="You are a coding assistant. Delegate broad searches.",
        subagents=[CODE_SEARCHER],
    )
