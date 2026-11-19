"""State strategies: how the shared transcript becomes model context.

M1 implements only `full_history`. `sliding_window(k)` and `rolling_summary`
are M4 work.
"""

from __future__ import annotations

from app.state import Message, Strategy


def select_context(messages: list[Message], strategy: Strategy) -> list[Message]:
    if strategy == "full_history":
        return list(messages)
    raise NotImplementedError(f"strategy {strategy!r} is not implemented yet (planned for M4)")


def render_transcript(messages: list[Message]) -> str:
    lines = []
    for m in messages:
        who = m.get("name") or m.get("role", "unknown")
        lines.append(f"[{who}] {m.get('content', '')}")
    return "\n".join(lines)


def context_block(messages: list[Message], strategy: Strategy) -> str:
    """Transcript text handed to a node, after the strategy is applied."""
    return render_transcript(select_context(messages, strategy))
