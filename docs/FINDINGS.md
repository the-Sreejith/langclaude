# Verification findings: blog post vs. `deepagents==0.6.8`

Every code idea in the post was executed against the real library
(`deepagents 0.6.8`, `langchain 1.3.4`, `langgraph 1.2.4`). The concepts all
hold up. Below are the exact points where the published snippets need a small
correction to run, plus the things that were confirmed working as written.

## ✅ Confirmed accurate

| Claim in the post | Verified |
| --- | --- |
| `create_deep_agent(model=, tools=, system_prompt=)` API | ✔ signature matches |
| `model="anthropic:claude-sonnet-4-6"` resolves | ✔ builds fine |
| `from deepagents.backends import LocalShellBackend` (post flagged as "verify path") | ✔ correct in 0.6.8 |
| Built-in tools: read/ls/glob/grep, write/edit, execute, task, write_todos | ✔ all 10 present |
| `@tool` custom tools merge alongside built-ins | ✔ `run_tests` shows up |
| Planning (`write_todos`) is built in and writes to state `["todos"]` | ✔ |
| Context mgmt: FilesystemMiddleware + SummarizationMiddleware auto-installed | ✔ both present |
| Subagents via the `task` tool run their own loop and report back | ✔ |
| `interrupt_on=` pauses **before** a risky tool (needs a checkpointer) | ✔ |
| `checkpointer=InMemorySaver()` persists state across `.invoke` on one `thread_id` | ✔ |

## ⚠️ Corrections needed to make the snippets run

### 1. Subagent dict key is `system_prompt`, not `prompt`

The post writes:

```python
code_searcher = {
    "name": "code-searcher",
    "description": "...",
    "prompt": "You are an expert at navigating codebases...",   # ← wrong key
}
```

In 0.6.8 the `SubAgent` schema requires `name`, `description`, and
**`system_prompt`**. Passing `prompt` raises:

```
KeyError: 'system_prompt'
```

This repo uses the correct key (`src/langclaude/part5_subagents.py`) and a
regression test guards it (`tests/test_part5_subagents.py`).

### 2. `LocalShellBackend` is **not** a sandbox

The post's safety section implies the backend gives you OS-level isolation
("a sandbox can stop a dangerous shell command at the operating-system
level"). The principle is right, but `LocalShellBackend` is **not** that
sandbox. Its own runtime warning states:

> `LocalShellBackend` provides no sandboxing (`execute()` runs commands on the
> host; `virtual_mode` does not restrict shell execution). Leaving
> `virtual_mode=False` allows absolute paths and `'..'` to bypass `root_dir`.

So:

* `LocalShellBackend` enables the `execute` tool but runs commands **on your
  host**. Use `interrupt_on={"execute": True}` (human-in-the-loop) as the real
  brake, and/or a true sandbox backend (e.g. `LangSmithSandbox`) for isolation.
* Set `virtual_mode` explicitly — the default is changing across the 0.6.x
  line and an unset value emits a `LangChainDeprecationWarning`.

### 3. The interrupt payload shape

When `interrupt_on` fires, `result["__interrupt__"][0].value` is a dict:

```python
{
  "action_requests": [{"name": "execute", "args": {...}, "description": "..."}],
  "review_configs":  [{"action_name": "execute",
                       "allowed_decisions": ["approve", "edit", "reject", "respond"]}],
}
```

You resume with a LangGraph `Command(resume=...)`.

## How this was verified without an API key

The harness mechanics (loop, tool execution, planning, delegation, interrupts,
persistence) are validated **deterministically** by injecting a scripted fake
chat model (`tests/conftest.py::FakeLoopModel`) that plays back a fixed list of
`AIMessage`s. This proves the wiring fires for free and reproducibly. The two
`@live` tests additionally hit the real model when `ANTHROPIC_API_KEY` is set.
