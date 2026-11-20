"""State strategies: how the shared transcript becomes model context.

Every node calls `build_context(state, keep_recent)` and gets back the
transcript text plus any state updates to merge (only `rolling_summary`
produces updates).

- full_history:     every message so far.
- sliding_window:   the last `window_k` messages.
- rolling_summary:  a running summary of older messages + the unsummarized
                    tail. When the tail grows past 2 x keep_recent, everything
                    except the last keep_recent messages is folded into the
                    summary with one `summarizer` LLM call (which is traced and
                    counted in the request's cost).

The question, and for answer/critic the retrieved passages, are always given
to nodes separately, so strategies only change how much transcript is carried.
"""

from __future__ import annotations

from app import llm
from app.state import GraphState, Message

SUMMARIZER_SYSTEM = (
    "You maintain a running summary of a multi-agent question-answering session. "
    "Merge the existing summary and the new transcript lines into one concise summary "
    "(at most 150 words). Keep sub-queries tried, document IDs found, key facts, and critic feedback. "
    "Reply with the summary text only."
)


def render_transcript(messages: list[Message]) -> str:
    lines = []
    for m in messages:
        who = m.get("name") or m.get("role", "unknown")
        lines.append(f"[{who}] {m.get('content', '')}")
    return "\n".join(lines)


def summarize(previous: str, messages: list[Message]) -> str:
    prompt = (
        f"Existing summary:\n{previous or '(none)'}\n\n"
        f"New transcript lines:\n{render_transcript(messages)}\n\n"
        "Updated summary:"
    )
    resp = llm.generate("summarizer", SUMMARIZER_SYSTEM, [{"role": "user", "content": prompt}])
    return resp.text.strip()


def build_context(state: GraphState, keep_recent: int) -> tuple[str, dict]:
    messages = state["messages"]
    strategy = state["strategy"]

    if strategy == "full_history":
        return render_transcript(messages), {}

    if strategy == "sliding_window":
        k = state.get("window_k") or 1
        return render_transcript(messages[-k:]), {}

    if strategy == "rolling_summary":
        upto = state.get("summarized_upto", 0)
        summary = state.get("summary", "")
        updates: dict = {}
        if len(messages) - upto > 2 * keep_recent:
            fold_end = len(messages) - keep_recent
            summary = summarize(summary, messages[upto:fold_end])
            upto = fold_end
            updates = {"summary": summary, "summarized_upto": upto}
        tail = render_transcript(messages[upto:])
        text = f"[summary of earlier messages] {summary}\n{tail}" if summary else tail
        return text, updates

    raise ValueError(f"unknown strategy {strategy!r}")
