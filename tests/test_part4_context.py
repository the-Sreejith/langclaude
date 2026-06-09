"""Part 4 — context management: middleware is wired in AND compaction fires."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from conftest import RecordingModel, installed_middleware, scripted_model, tool_names
from langclaude.part4_context import build_agent


def test_filesystem_tools_available_as_external_memory():
    # Files-as-memory: the virtual filesystem tools are present.
    agent = build_agent(scripted_model([AIMessage(content="ok")]))
    assert {"read_file", "write_file", "ls"}.issubset(tool_names(agent))


def test_both_context_middlewares_installed_on_the_assembled_agent():
    # Neither middleware is a graph node (they hook wrap_model_call), so we
    # walk the model node's closures to prove they're wired into THIS agent.
    agent = build_agent(scripted_model([AIMessage(content="ok")]))
    names = set(installed_middleware(agent))
    assert "FilesystemMiddleware" in names
    assert any("Summarization" in n for n in names)


def test_compaction_actually_fires_when_the_window_fills(tmp_path):
    # Drive the same SummarizationMiddleware class deepagents installs, with a
    # low trigger so a fake conversation overflows it. Proves the mechanism:
    # the summarizer model is called, the next model call sees the summary
    # instead of the old turns, and the evicted history is offloaded to a file.
    from deepagents.backends import FilesystemBackend
    from deepagents.middleware import SummarizationMiddleware
    from langchain.agents import create_agent

    model = RecordingModel(
        messages=iter(
            [
                AIMessage(content="SUMMARY-OF-OLD-TURNS"),  # the summarizer call
                AIMessage(content="final answer"),  # the compacted main call
            ]
        ),
        recorded=[],
    )
    summarizer = SummarizationMiddleware(
        model=model,
        backend=FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True),
        trigger=("tokens", 200),
        keep=("messages", 1),
    )
    agent = create_agent(model=model, tools=[], middleware=[summarizer])

    big = "lorem ipsum dolor " * 400  # well past the 200-token trigger
    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": big},
                {"role": "assistant", "content": "noted"},
                {"role": "user", "content": "now answer briefly"},
            ]
        }
    )

    assert result["messages"][-1].content == "final answer"
    # The main model call saw the summary, NOT the original bloated turns.
    final_call = " ".join(str(m.content) for m in model.recorded[-1])
    assert "SUMMARY-OF-OLD-TURNS" in final_call
    assert "lorem ipsum" not in final_call
    # The evicted history was offloaded to a file (files as external memory).
    offloaded = list(tmp_path.rglob("*.md"))
    assert offloaded, "evicted history should land in conversation_history/*.md"
    assert "lorem ipsum" in offloaded[0].read_text()
