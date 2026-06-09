# langclaude — Build Your Own Claude Code with LangChain Deep Agents

A **verified, runnable** companion to the blog post *"Build Your Own Claude
Code Using LangChain: A Deep Dive into Deep Agents."*

Every concept in the post is implemented here against the real
[`deepagents`](https://pypi.org/project/deepagents/) library and backed by a
test suite. The harness mechanics — the loop, tools, planning, context
management, delegation, the human-in-the-loop brake, and persistence — are
proven to actually fire **without needing an API key**, by driving the agent
with a deterministic scripted model.

> **TL;DR of the verification:** the post's concepts are all correct. Two
> snippets need a small fix to run on `deepagents 0.6.8` (subagent key
> `prompt` → `system_prompt`; `LocalShellBackend` is *not* a sandbox). A
> second-revision code review then fixed the repo's own gaps — most notably,
> the capstone agent claimed a human-approval gate it never wired up. Full
> details in [`docs/FINDINGS.md`](docs/FINDINGS.md).

## The mental model

Claude Code isn't the model — it's the **harness** wrapped around it. A bare
LLM only emits text. The harness gives it a loop, hands (tools), a plan,
memory management, helpers, and brakes:

```
user ──▶ [ model decides next step ] ──▶ tool call? ──yes──▶ run tool ──┐
              ▲                              │                          │
              └──────────── result fed back ◀┘                          │
                                             │                          │
                                            no                          │
                                             ▼                          │
                                       plain-text answer ◀──────────────┘
```

The loop exits the moment the model replies with plain text instead of a tool
call. Everything else (planning, subagents, compaction, the permission/sandbox
layer) wraps this loop without changing it.

## The eight parts

Each part of the post is a small module under `src/langclaude/`, and each has a
test proving it works.

| Part | Concept | Module | Proven by |
| --- | --- | --- | --- |
| 0 | The loop from scratch | `part0_scratch_loop.py` | the bare loop runs against a scripted raw-API client, plus the `max_turns` backstop |
| 1 | The loop | `part1_loop.py` | model → tool → result → exit fires end-to-end |
| 2 | Tools (the hands) | `part2_tools.py` | all 9 built-ins present + custom `run_tests` merged; tool output is token-budgeted |
| 3 | Planning | `part3_planning.py` | `write_todos` writes the plan into state |
| 4 | Context management | `part4_context.py` | both middlewares wired into the agent **and compaction actually fires** (summary replaces old turns, history offloaded to a file) |
| 5 | Subagents | `part5_subagents.py` | `task` delegates and reports back; wrong `prompt` key raises; subagents **cannot** spawn subagents |
| 6 | Safety / human-in-the-loop | `part6_safety.py` | loop **pauses for approval**; **approve** really runs the command, **reject** really blocks it; `LocalShellBackend` proven to escape `root_dir` |
| 7 | Persistence | `part7_persistence.py` | state survives across turns on one `thread_id` |
| ∑ | All together | `full_agent.py` | every capability wired into one agent, **including the approval gate** |

## Quickstart

```bash
# 1. Create the environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 2. Run the full test suite — NO API key required
pytest
#   25 passed, 2 skipped   (the 2 skipped are the live API tests)

# Lint (same check CI runs)
ruff check src tests
```

### Run the live demo (optional, costs tokens)

```bash
cp .env.example .env        # then put your key in it
export ANTHROPIC_API_KEY=sk-ant-...
python examples/run_full_agent.py
```

With a key set, the two `@live` tests also run: `pytest -m live`.

## How the no-key verification works

Calling the real model costs money and is non-deterministic — a poor base for
"does the harness work?" tests. Instead, `tests/conftest.py` provides a
`FakeLoopModel` that plays back a scripted list of `AIMessage`s. Feed it an
`AIMessage` with `tool_calls` and the harness executes the real tool; feed it
plain text and the loop exits. This exercises the *actual* deepagents loop,
tool node, middleware, interrupts, and checkpointer — deterministically and for
free. The live model is only used for the optional end-to-end smoke tests.

## Using it as a library

```python
from langclaude.full_agent import build_agent

agent = build_agent()  # defaults to anthropic:claude-sonnet-4-6
result = agent.invoke(
    {"messages": [{"role": "user", "content": "The login tests are failing. Fix them."}]},
    {"configurable": {"thread_id": "my-project"}},
)
print(result["messages"][-1].content)
```

Every `build_*` function accepts a `model` argument so you can inject a fake
model in tests or swap providers.

## Project layout

```
langclaude/
├── README.md
├── requirements.txt          # pinned, verified versions
├── pyproject.toml            # package + pytest + ruff config
├── .env.example
├── .github/workflows/ci.yml  # lint + tests on every push
├── src/langclaude/           # one module per part of the post
│   ├── part0_scratch_loop.py   # the loop by hand — raw API, no framework
│   ├── part1_loop.py … part7_persistence.py
│   └── full_agent.py
├── examples/run_full_agent.py  # live demo (needs API key)
├── tests/                    # deterministic + live-marked tests
└── docs/
    ├── BLOG_POST.md          # the full post, with corrections applied inline
    └── FINDINGS.md           # blog vs. reality, the corrections summarized
```

## Verified environment

`deepagents 0.6.8`, `langchain 1.3.4`, `langchain-core 1.4.2`,
`langgraph 1.2.4`, `langgraph-checkpoint 4.1.1`, `langchain-anthropic 1.4.4`,
`anthropic 0.108.0`, Python 3.12. See `requirements.txt` for the full pin set.

## The honest part

What the harness gives you for free (~80%): the loop, the tool suite, planning,
context management, delegation, persistence. What's still on you (the hard
20%): the system prompt, a *real* sandbox for shell execution
(`LocalShellBackend` is not one — see `docs/FINDINGS.md`), and good tools.
Agents also cost money, are bounded by the model, and can be overkill for a
quick question.
