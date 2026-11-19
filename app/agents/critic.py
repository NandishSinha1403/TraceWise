"""Critic: checks grounding and decides accept / retry_planner / retry_retriever.

The critic is the only node that can send work back. The retry cap is
enforced here: once `critic_retries` reaches `max_retries`, a retry verdict
ends the graph with `capped=True` and the latest draft as the answer.
"""

from __future__ import annotations

from typing import Callable

from app import llm
from app.state import CriticVerdict, GraphState, Message
from app.strategies import context_block
from app.tools.search import format_results

SYSTEM = (
    "You are the critic in a retrieval-augmented question-answering system. "
    "Check whether the draft answer (1) answers the question and (2) is grounded in the passages, "
    "with correct citations. "
    'Respond with JSON only: {"verdict": "accept" | "retry_planner" | "retry_retriever", "reason": "<short reason>"}. '
    "Use retry_retriever when the plan is fine but evidence is missing; "
    "use retry_planner when the plan or sub-queries are wrong."
)

RETRY_TARGETS = {"retry_planner": "planner", "retry_retriever": "retriever"}


def build_prompt(state: GraphState) -> str:
    return "\n".join(
        [
            "Transcript so far:",
            context_block(state["messages"], state["strategy"]),
            "",
            f"Question: {state['question']}",
            "Passages:",
            format_results(state.get("retrieved", [])),
            "",
            f"Draft answer: {state.get('draft_answer', '')}",
            f"Cited doc_ids: {state.get('cited_doc_ids', [])}",
        ]
    )


def parse_verdict(text: str) -> CriticVerdict:
    data = llm.parse_json(text)
    if data is None:
        return CriticVerdict(verdict="invalid", reason="critic output was not valid JSON")
    verdict = str(data.get("verdict", "")).strip().lower()
    reason = str(data.get("reason", "")).strip()
    if verdict not in ("accept", *RETRY_TARGETS):
        return CriticVerdict(verdict="invalid", reason=f"unknown verdict {verdict!r}")
    return CriticVerdict(verdict=verdict, reason=reason)


def decide(verdict: CriticVerdict, retries: int, max_retries: int) -> tuple[str, int, bool]:
    """Return (route, new_retry_count, capped). Pure function; tested directly."""
    if verdict["verdict"] in RETRY_TARGETS:
        if retries < max_retries:
            return RETRY_TARGETS[verdict["verdict"]], retries + 1, False
        return "end", retries, True
    return "end", retries, False  # accept, or invalid output: stop rather than loop


def make_critic_node(max_retries: int) -> Callable[[GraphState], dict]:
    def critic_node(state: GraphState) -> dict:
        resp = llm.generate("critic", SYSTEM, [{"role": "user", "content": build_prompt(state)}], json_output=True)
        verdict = parse_verdict(resp.text)
        route, retries, capped = decide(verdict, state.get("critic_retries", 0), max_retries)
        return {
            "critic_verdict": verdict,
            "critic_retries": retries,
            "capped": capped,
            "route": route,
            "messages": [
                Message(role="assistant", name="critic", content=f"{verdict['verdict']}: {verdict['reason']}")
            ],
            "steps": ["critic"],
        }

    return critic_node
