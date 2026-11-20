"""Graph state schema.

`strategy` controls how `messages` is turned into model context (see
`app/strategies.py`): `full_history`, `sliding_window` (uses `window_k`), or
`rolling_summary` (uses `summary` and `summarized_upto`).
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

Strategy = Literal["full_history", "sliding_window", "rolling_summary"]
Verdict = Literal["accept", "retry_planner", "retry_retriever", "invalid"]


class Message(TypedDict, total=False):
    """One entry in the shared inter-agent transcript."""

    role: Literal["user", "assistant", "tool"]
    name: str  # node or tool that produced it
    content: str


class Passage(TypedDict):
    chunk_id: str
    doc_id: str
    text: str
    score: float


class CriticVerdict(TypedDict):
    verdict: Verdict
    reason: str


class GraphState(TypedDict, total=False):
    request_id: str
    question: str
    strategy: Strategy
    window_k: int
    summary: str  # rolling_summary: running summary of messages[:summarized_upto]
    summarized_upto: int
    plan: str
    sub_queries: list[str]
    messages: Annotated[list[Message], operator.add]
    retrieved: list[Passage]
    draft_answer: str
    cited_doc_ids: list[str]
    critic_verdict: CriticVerdict
    critic_retries: int
    capped: bool
    route: str  # next node chosen by the critic, or "end"
    steps: Annotated[list[str], operator.add]


def initial_state(
    question: str, request_id: str, strategy: Strategy = "full_history", window_k: int = 4
) -> GraphState:
    return GraphState(
        request_id=request_id,
        question=question,
        strategy=strategy,
        window_k=window_k,
        summary="",
        summarized_upto=0,
        plan="",
        sub_queries=[],
        messages=[Message(role="user", name="user", content=question)],
        retrieved=[],
        draft_answer="",
        cited_doc_ids=[],
        critic_retries=0,
        capped=False,
        route="",
        steps=[],
    )
