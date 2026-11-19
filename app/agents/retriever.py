"""Retriever: ReAct loop where the model decides when to call `search_documents`.

The loop is capped at `max_steps` tool-call steps. A step is one model turn
that requests one or more tool calls; all calls in that turn are executed.
"""

from __future__ import annotations

from typing import Callable

from app import llm
from app.state import GraphState, Message, Passage
from app.strategies import context_block
from app.tools.search import SEARCH_TOOL, SearchDocuments, format_results

SYSTEM = (
    "You are the retriever in a retrieval-augmented question-answering system. "
    "Use the search_documents tool to collect passages that answer the question. "
    "Search for each sub-query, and refine queries if results are weak. "
    "When you have enough evidence, reply with a one-line note and no tool call."
)


def build_prompt(state: GraphState) -> str:
    parts = [
        "Transcript so far:",
        context_block(state["messages"], state["strategy"]),
        "",
        f"Question: {state['question']}",
        "Sub-queries:",
        *[f"- {q}" for q in state.get("sub_queries") or [state["question"]]],
    ]
    verdict = state.get("critic_verdict")
    if verdict and verdict["verdict"] == "retry_retriever":
        parts.append(f"Critic feedback: {verdict['reason']}")
    return "\n".join(parts)


def merge_passages(existing: list[Passage], new: list[Passage]) -> list[Passage]:
    seen = {p["chunk_id"] for p in existing}
    merged = list(existing)
    for p in new:
        if p["chunk_id"] not in seen:
            seen.add(p["chunk_id"])
            merged.append(p)
    return merged


def make_retriever_node(search: SearchDocuments, max_steps: int) -> Callable[[GraphState], dict]:
    def retriever_node(state: GraphState) -> dict:
        scratch: list[llm.ChatMessage] = [{"role": "user", "content": build_prompt(state)}]
        found: list[Passage] = []
        transcript: list[Message] = []
        tool_steps = 0

        while tool_steps < max_steps:
            resp = llm.generate("retriever", SYSTEM, scratch, tools=[SEARCH_TOOL])
            if not resp.tool_calls:
                if resp.text.strip():
                    transcript.append(Message(role="assistant", name="retriever", content=resp.text.strip()))
                break
            tool_steps += 1
            scratch.append({"role": "assistant", "content": resp.text, "tool_calls": resp.tool_calls})
            for call in resp.tool_calls:
                if call.name != SEARCH_TOOL.name:
                    result_text = f"Unknown tool {call.name!r}."
                else:
                    query = str(call.args.get("query", ""))
                    passages = search(query)
                    found = merge_passages(found, passages)
                    result_text = format_results(passages)
                    transcript.append(
                        Message(role="tool", name=SEARCH_TOOL.name, content=f"query={query!r}\n{result_text}")
                    )
                scratch.append(
                    {"role": "tool", "tool_name": call.name, "tool_call_id": call.id, "content": result_text}
                )

        return {
            "retrieved": merge_passages(state.get("retrieved", []), found),
            "messages": transcript,
            "steps": ["retriever"],
        }

    return retriever_node
