"""FastAPI service. `python -m app.api` runs it with uvicorn."""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app import llm
from app.config import Settings, load_settings
from app.graph import build_graph, run
from app.vectorstore import VectorStore, open_store


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


class CriticVerdictOut(BaseModel):
    verdict: str
    reason: str


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


def create_app(settings: Settings | None = None, store: VectorStore | None = None) -> FastAPI:
    settings = settings or load_settings()
    llm.configure(settings)
    store = store if store is not None else open_store(settings)
    graph = build_graph(settings, store)

    app = FastAPI(title="TraceWise", version="0.1.0")

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "llm_backend": settings.llm_backend, "indexed_chunks": store.count()}

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
        return run(graph, settings, question)

    return app


def main() -> None:
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    main()
