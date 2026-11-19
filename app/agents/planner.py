"""Planner: turns the question (and any critic feedback) into sub-queries."""

from __future__ import annotations

from app import llm
from app.state import GraphState, Message
from app.strategies import context_block

SYSTEM = (
    "You are the planner in a retrieval-augmented question-answering system. "
    "Write a short plan for answering the user's question from a document collection. "
    'Respond with JSON only: {"plan": "<one or two sentences>", "sub_queries": ["<search query>", ...]}. '
    "Use 1 to 4 focused sub-queries. If critic feedback is present, address it."
)


def build_prompt(state: GraphState) -> str:
    parts = [
        "Transcript so far:",
        context_block(state["messages"], state["strategy"]),
        "",
        f"Question: {state['question']}",
    ]
    verdict = state.get("critic_verdict")
    if verdict and verdict["verdict"] == "retry_planner":
        parts.append(f"Critic feedback: {verdict['reason']}")
    return "\n".join(parts)


def planner_node(state: GraphState) -> dict:
    resp = llm.generate("planner", SYSTEM, [{"role": "user", "content": build_prompt(state)}], json_output=True)
    data = llm.parse_json(resp.text) or {}
    plan = str(data.get("plan") or "").strip()
    sub_queries = [str(q).strip() for q in data.get("sub_queries") or [] if str(q).strip()]
    if not sub_queries:
        # Model gave no usable sub-queries; fall back to searching the question itself.
        sub_queries = [state["question"]]
    if not plan:
        plan = "Search for: " + "; ".join(sub_queries)
    summary = plan + "\nSub-queries:\n" + "\n".join(f"- {q}" for q in sub_queries)
    return {
        "plan": plan,
        "sub_queries": sub_queries,
        "messages": [Message(role="assistant", name="planner", content=summary)],
        "steps": ["planner"],
    }
