"""Part 2 — built-in tool suite is present and custom tools merge in."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from conftest import scripted_model, tool_names
from langclaude.part2_tools import MAX_OUTPUT_CHARS, build_agent, clip_output, run_tests

# The NINE built-ins (the blog's "ten" counts the custom run_tests on top).
BUILTIN_TOOLS = {
    "read_file",
    "ls",
    "glob",
    "grep",
    "write_file",
    "edit_file",
    "execute",
    "task",
    "write_todos",
}


def test_all_nine_builtin_tools_present_plus_custom():
    agent = build_agent(scripted_model([AIMessage(content="ok")]))
    names = tool_names(agent)
    assert BUILTIN_TOOLS.issubset(names)
    assert "run_tests" in names  # the custom tool merged alongside built-ins


def test_custom_tool_has_docstring_schema():
    # The docstring is the model's instruction manual -> it becomes the schema.
    assert run_tests.description
    assert "pytest" in run_tests.description.lower()


def test_tool_output_is_token_budgeted():
    # The post's own lesson applied to the custom tool: a huge result must not
    # flood the context window, and the tail (where pytest puts failures) wins.
    huge = "x" * (MAX_OUTPUT_CHARS * 5) + "FAILED tests/test_x.py - the part that matters"
    clipped = clip_output(huge)
    assert len(clipped) < len(huge)
    assert clipped.startswith("[... output truncated")
    assert clipped.endswith("the part that matters")
    assert clip_output("short output") == "short output"
