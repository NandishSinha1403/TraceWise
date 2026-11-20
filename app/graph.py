"""LangGraph definition.

planner -> retriever -> answer -> critic -> (accept: END)
                                          | (retry_planner: planner)
                                          | (retry_retriever: retriever)
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from app.agents.answer import make_answer_node
from app.agents.critic import make_critic_node
from app.agents.planner import make_planner_node
from app.agents.retriever import make_retriever_node
from app.config import Settings
from app.state import GraphState, initial_state
from app.tools.search import SearchDocuments
from app.tracing import middleware as tracing
from app.tracing.metrics import rollup
from app.vectorstore import VectorStore


def _route_after_critic(state: GraphState) -> str:
    return state.get("route") or "end"


def _traced(name: str, fn: Callable[[GraphState], dict]) -> Callable[[GraphState], dict]:
    def node(state: GraphState) -> dict:
        with tracing.node_scope(name):
            return fn(state)

    node.__name__ = f"{name}_node"
    return node


def build_graph(settings: Settings, store: VectorStore):
    search = SearchDocuments(store, settings.search_top_k)
    keep = settings.summary_keep_recent

    g = StateGraph(GraphState)
    g.add_node("planner", _traced("planner", make_planner_node(keep)))
    g.add_node("retriever", _traced("retriever", make_retriever_node(search, settings.max_retriever_steps, keep)))
    g.add_node("answer", _traced("answer", make_answer_node(keep)))
    g.add_node("critic", _traced("critic", make_critic_node(settings.max_critic_retries, keep)))

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


def strategy_label(settings: Settings) -> str:
    return f"sliding_window_k{settings.window_k}" if settings.strategy == "sliding_window" else settings.strategy


def run(
    graph,
    settings: Settings,
    question: str,
    request_id: str | None = None,
    keep_context: bool = False,
) -> dict[str, Any]:
    """Run one request. Returns the API response dict (plus `_context` if keep_context)."""
    request_id = request_id or uuid.uuid4().hex
    label = strategy_label(settings)
    # Worst case is 4 nodes per pass; leave headroom above it.
    recursion_limit = 4 * (settings.max_critic_retries + 1) + 5
    state = initial_state(question, request_id, settings.strategy, settings.window_k)

    final: dict = {}
    error: Exception | None = None
    with tracing.request_scope(request_id, label, question) as ctx:
        t0 = time.perf_counter()
        try:
            final = graph.invoke(state, config={"recursion_limit": recursion_limit})
        except Exception as exc:  # recorded, then re-raised below
            error = exc
        wall_ms = (time.perf_counter() - t0) * 1000

    roll = rollup(
        request_id,
        label,
        ctx.calls,
        total_latency_ms=wall_ms,
        n_agent_hops=len(final.get("steps", [])),
        n_critic_retries=final.get("critic_retries", 0),
    )
    if error is not None:
        roll.errors.append(f"{type(error).__name__}: {error}")
    sink = tracing.sink()
    if sink is not None:
        sink.insert_request(roll)
    if error is not None:
        raise error

    verdict = final.get("critic_verdict") or {"verdict": "invalid", "reason": "critic did not run"}
    out: dict[str, Any] = {
        "request_id": request_id,
        "question": question,
        "strategy": label,
        "answer": final.get("draft_answer", ""),
        "cited_doc_ids": final.get("cited_doc_ids", []),
        "critic_retries": final.get("critic_retries", 0),
        "capped": final.get("capped", False),
        "critic_verdict": verdict,
        "steps": final.get("steps", []),
        "usage": roll.as_dict(),
    }
    if keep_context:
        out["_context"] = ctx
        out["_retrieved"] = final.get("retrieved", [])
    return out
