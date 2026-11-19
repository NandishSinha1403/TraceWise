"""End-to-end graph run with the fake LLM. No network."""

import pytest

from app.config import ConfigError, load_settings
from app.graph import build_graph, run
from app.ingest import chunk_text
from tests.conftest import FIXTURE_DOC_IDS


def test_graph_end_to_end_accept(make_env):
    settings, store = make_env()
    out = run(build_graph(settings, store), settings, "What colour is the lamp of the Varnholt Lighthouse?")

    assert out["steps"] == ["planner", "retriever", "answer", "critic"]
    assert out["strategy"] == "full_history"
    assert out["critic_verdict"]["verdict"] == "accept"
    assert out["critic_retries"] == 0
    assert out["capped"] is False
    assert out["answer"]
    assert out["cited_doc_ids"] == ["fixture_lighthouse.md"]
    assert set(out["cited_doc_ids"]) <= FIXTURE_DOC_IDS


def test_retriever_respects_step_cap(make_env, monkeypatch):
    """A model that always asks for another search is stopped at MAX_RETRIEVER_STEPS."""
    from app import llm
    from app.llm import LLMResponse, ToolCall

    settings, store = make_env(MAX_RETRIEVER_STEPS="3")
    fake = llm.FakeBackend(settings)
    retriever_calls = []

    class Greedy:
        def generate(self, role, system, messages, tools, json_output):
            if role == "retriever":
                retriever_calls.append(1)
                return LLMResponse(text="", tool_calls=[ToolCall("search_documents", {"query": "orchard"})])
            return fake.generate(role, system, messages, tools, json_output)

        def embed(self, texts, task):
            return fake.embed(texts, task)

    llm.set_backend(Greedy())
    out = run(build_graph(settings, store), settings, "Which apple grows at Pellam Orchard?")
    assert len(retriever_calls) == 3
    assert out["steps"] == ["planner", "retriever", "answer", "critic"]


def test_vertex_backend_requires_env():
    with pytest.raises(ConfigError) as exc:
        load_settings({"LLM_BACKEND": "vertex", "GCP_PROJECT": "p"})
    msg = str(exc.value)
    for name in ("GCP_REGION", "GEN_MODEL_ID", "EMBEDDING_MODEL_ID"):
        assert name in msg
    assert "GCP_PROJECT" not in msg


def test_chunk_text_overlap_and_bounds():
    text = " ".join(f"w{i}" for i in range(200))
    chunks = chunk_text(text, size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)
    assert chunks[0].split()[0] == "w0" and chunks[-1].split()[-1] == "w199"
