"""langclaude — a verified, runnable companion to the blog post
"Build Your Own Claude Code Using LangChain: A Deep Dive into Deep Agents".

Each module maps to one part of the post:

    part0_scratch_loop  The loop by hand — raw Messages API, no framework
    part1_loop          The agent loop (the engine)
    part2_tools         Custom tools (the hands)
    part3_planning      write_todos planning
    part4_context       Context management (filesystem + summarization)
    part5_subagents     Delegation to subagents
    part6_safety        Human-in-the-loop / permissions
    part7_persistence   Checkpointing across turns
    full_agent          All parts assembled

Every ``build_*`` function takes an optional ``model`` argument. Pass a real
model string like ``"anthropic:claude-sonnet-4-6"`` to run live, or inject a
deterministic fake model (see ``tests/conftest.py``) to exercise the harness
without an API key.
"""

DEFAULT_MODEL = "anthropic:claude-sonnet-4-6"

__all__ = ["DEFAULT_MODEL"]
