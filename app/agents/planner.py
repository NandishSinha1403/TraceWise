"""Planner: turns the question (and any critic feedback) into sub-queries."""

from __future__ import annotations

from typing import Callable

from app import llm
from app.state import GraphState, Message
from app.strategies import build_context

SYSTEM = (
    "You are the planner in a retrieval-augmented question-answering system. "
    "Write a short plan for answering the user's question from a document collection. "
    'Respond with JSON only: {"plan": "<one or two sentences>", "sub_queries": ["<search query>", ...]}. '
    "Use 1 to 4 focused sub-queries. If critic feedback is present, address it."
)


def build_prompt(state: GraphState, transcript: str) -> str:
    parts = [
        "Transcript so far:",
        transcript,
        "",
        f"Question: {state['question']}",
    ]
    verdict = state.get("critic_verdict")
    if verdict and verdict["verdict"] == "retry_planner":
        parts.append(f"Critic feedback: {verdict['reason']}")
    return "\n".join(parts)


def make_planner_node(keep_recent: int) -> Callable[[GraphState], dict]:
    def planner_node(state: GraphState) -> dict:
        transcript, ctx_updates = build_context(state, keep_recent)
        return {**ctx_updates, **plan(state, transcript)}

    return planner_node


def plan(state: GraphState, transcript: str) -> dict:
    resp = llm.generate("planner", SYSTEM, [{"role": "user", "content": build_prompt(state, transcript)}], json_output=True)
    data = llm.parse_json(resp.text) or {}
    plan_text = str(data.get("plan") or "").strip()
    sub_queries = [str(q).strip() for q in data.get("sub_queries") or [] if str(q).strip()]
    if not sub_queries:
        # Model gave no usable sub-queries; fall back to searching the question itself.
        sub_queries = [state["question"]]
    if not plan_text:
        plan_text = "Search for: " + "; ".join(sub_queries)
    summary = plan_text + "\nSub-queries:\n" + "\n".join(f"- {q}" for q in sub_queries)
    return {
        "plan": plan_text,
        "sub_queries": sub_queries,
        "messages": [Message(role="assistant", name="planner", content=summary)],
        "steps": ["planner"],
    }
