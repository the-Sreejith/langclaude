"""Live demo of the full agent. Requires ANTHROPIC_API_KEY.

    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/run_full_agent.py

This actually calls the model and may invoke the LocalShellBackend's `execute`
tool on your machine. Run it inside a throwaway directory you don't mind the
agent touching.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set. See .env.example.", file=sys.stderr)
        return 1

    from langclaude.full_agent import build_agent

    agent = build_agent()
    config = {"configurable": {"thread_id": "demo"}}

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "List the tools you have available, then stop.",
                }
            ]
        },
        config,
    )
    print(result["messages"][-1].content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
