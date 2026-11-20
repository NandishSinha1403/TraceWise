"""Answer step: drafts the final answer from retrieved passages, with citations."""

from __future__ import annotations

from typing import Callable

from app import llm
from app.state import GraphState, Message
from app.strategies import build_context
from app.tools.search import format_results

SYSTEM = (
    "You write the final answer in a retrieval-augmented question-answering system. "
    "Answer only from the passages provided. If they do not contain the answer, say so. "
    'Respond with JSON only: {"answer": "<answer text>", "cited_doc_ids": ["<doc_id>", ...]}. '
    "Cite only doc_ids that appear in the passages."
)


def build_prompt(state: GraphState, transcript: str) -> str:
    return "\n".join(
        [
            "Transcript so far:",
            transcript,
            "",
            f"Question: {state['question']}",
            "Passages:",
            format_results(state.get("retrieved", [])),
        ]
    )


def make_answer_node(keep_recent: int) -> Callable[[GraphState], dict]:
    def answer_node(state: GraphState) -> dict:
        transcript, ctx_updates = build_context(state, keep_recent)
        return {**ctx_updates, **draft(state, transcript)}

    return answer_node


def draft(state: GraphState, transcript: str) -> dict:
    resp = llm.generate("answer", SYSTEM, [{"role": "user", "content": build_prompt(state, transcript)}], json_output=True)
    data = llm.parse_json(resp.text)
    if data is None:
        answer, cited = resp.text.strip(), []
    else:
        answer = str(data.get("answer") or "").strip()
        cited = [str(d) for d in data.get("cited_doc_ids") or []]
    known = {p["doc_id"] for p in state.get("retrieved", [])}
    cited = list(dict.fromkeys(d for d in cited if d in known))  # drop invented IDs, keep order
    return {
        "draft_answer": answer,
        "cited_doc_ids": cited,
        "messages": [Message(role="assistant", name="answer", content=answer)],
        "steps": ["answer"],
    }
