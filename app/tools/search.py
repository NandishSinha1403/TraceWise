"""`search_documents` tool: wraps a VectorStore for the retriever's ReAct loop."""

from __future__ import annotations

from app.llm import ToolSpec
from app.state import Passage
from app.vectorstore import VectorStore

SEARCH_TOOL = ToolSpec(
    name="search_documents",
    description="Search the document collection. Returns the most relevant passages with their document IDs.",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Natural-language search query."}},
        "required": ["query"],
    },
)


class SearchDocuments:
    def __init__(self, store: VectorStore, top_k: int) -> None:
        self._store = store
        self._top_k = top_k

    def __call__(self, query: str) -> list[Passage]:
        query = (query or "").strip()
        if not query:
            return []
        return self._store.search(query, self._top_k)


def format_results(passages: list[Passage]) -> str:
    if not passages:
        return "No passages found."
    return "\n".join(f"<<doc_id={p['doc_id']}>> {p['text']}" for p in passages)
