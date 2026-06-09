"""Part 2 — built-in tool suite is present and custom tools merge in."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from conftest import scripted_model
from langclaude.part2_tools import build_agent, run_tests

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


def _tool_names(agent) -> set[str]:
    node = agent.nodes["tools"]
    bound = getattr(node, "bound", node)
    return set(getattr(bound, "tools_by_name", {}).keys())


def test_all_builtin_tools_present_plus_custom():
    agent = build_agent(scripted_model([AIMessage(content="ok")]))
    names = _tool_names(agent)
    assert BUILTIN_TOOLS.issubset(names)
    assert "run_tests" in names  # the custom tool merged alongside built-ins


def test_custom_tool_has_docstring_schema():
    # The docstring is the model's instruction manual -> it becomes the schema.
    assert run_tests.description
    assert "pytest" in run_tests.description.lower()
