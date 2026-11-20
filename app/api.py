"""FastAPI service. `python -m app.api` runs it with uvicorn."""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app import llm
from app.config import Settings, load_settings
from app.graph import build_graph, run
from app.tracing import middleware as tracing
from app.tracing.store import MetricsStore
from app.vectorstore import VectorStore, open_store

log = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


class CriticVerdictOut(BaseModel):
    verdict: str
    reason: str


class UsageOut(BaseModel):
    total_latency_ms: float
    total_cost_usd: float
    cost_complete: bool
    n_llm_calls: int
    n_embed_calls: int
    n_tool_calls: int
    n_agent_hops: int
    n_critic_retries: int
    input_tokens: int
    output_tokens: int
    thinking_tokens: int
    llm_latency_ms: float
    backoff_ms: float
    n_retried_calls: int


class AskResponse(BaseModel):
    request_id: str
    question: str
    strategy: str
    answer: str
    cited_doc_ids: list[str]
    critic_retries: int
    capped: bool
    critic_verdict: CriticVerdictOut
    steps: list[str]
    usage: UsageOut


def create_app(
    settings: Settings | None = None,
    store: VectorStore | None = None,
    metrics_sink: MetricsStore | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    llm.configure(settings)
    tracing.configure(settings, sink=metrics_sink)
    store = store if store is not None else open_store(settings)
    graph = build_graph(settings, store)

    app = FastAPI(title="TraceWise", version="0.2.0")

    @app.get("/healthz")
    def healthz() -> dict:
        return {
            "status": "ok",
            "llm_backend": settings.llm_backend,
            "strategy": settings.strategy,
            "indexed_chunks": store.count(),
        }

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> dict:  # sync: FastAPI runs it in a worker thread
        question = req.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="question must not be blank")
        if store.count() == 0:
            raise HTTPException(
                status_code=503,
                detail="vector store is empty; add documents to DOCS_DIR and run `python -m app.ingest`",
            )
        try:
            return run(graph, settings, question)
        except Exception as exc:
            log.exception("request failed")
            raise HTTPException(status_code=502, detail=f"model call failed: {type(exc).__name__}") from exc

    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    main()
