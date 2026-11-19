"""LangGraph definition.

planner -> retriever -> answer -> critic -> (accept: END)
                                          | (retry_planner: planner)
                                          | (retry_retriever: retriever)
"""

from __future__ import annotations

import uuid
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.answer import answer_node
from app.agents.critic import make_critic_node
from app.agents.planner import planner_node
from app.agents.retriever import make_retriever_node
from app.config import Settings
from app.state import GraphState, initial_state
from app.tools.search import SearchDocuments
from app.vectorstore import VectorStore


def _route_after_critic(state: GraphState) -> str:
    return state.get("route") or "end"


def build_graph(settings: Settings, store: VectorStore):
    search = SearchDocuments(store, settings.search_top_k)

    g = StateGraph(GraphState)
    g.add_node("planner", planner_node)
    g.add_node("retriever", make_retriever_node(search, settings.max_retriever_steps))
    g.add_node("answer", answer_node)
    g.add_node("critic", make_critic_node(settings.max_critic_retries))

    g.add_edge(START, "planner")
    g.add_edge("planner", "retriever")
    g.add_edge("retriever", "answer")
    g.add_edge("answer", "critic")
    g.add_conditional_edges(
        "critic",
        _route_after_critic,
        {"planner": "planner", "retriever": "retriever", "end": END},
    )
    return g.compile()


def run(graph, settings: Settings, question: str, request_id: str | None = None) -> dict[str, Any]:
    request_id = request_id or uuid.uuid4().hex
    # Worst case is 4 nodes per pass; leave headroom above it.
    recursion_limit = 4 * (settings.max_critic_retries + 1) + 5
    final = graph.invoke(initial_state(question, request_id), config={"recursion_limit": recursion_limit})
    verdict = final.get("critic_verdict") or {"verdict": "invalid", "reason": "critic did not run"}
    return {
        "request_id": request_id,
        "question": question,
        "strategy": final["strategy"],
        "answer": final.get("draft_answer", ""),
        "cited_doc_ids": final.get("cited_doc_ids", []),
        "critic_retries": final.get("critic_retries", 0),
        "capped": final.get("capped", False),
        "critic_verdict": verdict,
        "steps": final.get("steps", []),
    }
