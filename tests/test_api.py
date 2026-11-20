"""/ask returns 200 and the expected JSON shape with the fake LLM."""

import uuid

from fastapi.testclient import TestClient

from app.api import create_app
from app.vectorstore import ChromaStore
from tests.conftest import fake_settings

EXPECTED_KEYS = {
    "request_id",
    "question",
    "strategy",
    "answer",
    "cited_doc_ids",
    "critic_retries",
    "capped",
    "critic_verdict",
    "steps",
    "usage",
}


def test_ask_returns_expected_shape(make_env):
    settings, store = make_env()
    client = TestClient(create_app(settings, store))
    resp = client.post("/ask", json={"question": "When is the Pellam Orchard harvest festival?"})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == EXPECTED_KEYS
    assert isinstance(body["answer"], str) and body["answer"]
    assert body["cited_doc_ids"] == ["fixture_orchard.md"]
    assert body["critic_retries"] == 0
    assert body["capped"] is False
    assert body["steps"] == ["planner", "retriever", "answer", "critic"]
    usage = body["usage"]
    assert usage["n_llm_calls"] == 5  # planner, retriever x2 (search, then stop), answer, critic
    assert usage["n_tool_calls"] == 1
    assert usage["n_agent_hops"] == 4
    assert usage["n_critic_retries"] == 0


def test_ask_rejects_blank_question(make_env):
    settings, store = make_env()
    client = TestClient(create_app(settings, store))
    assert client.post("/ask", json={"question": "   "}).status_code == 422
    assert client.post("/ask", json={}).status_code == 422


def test_ask_503_when_store_empty():
    settings = fake_settings()
    client = TestClient(create_app(settings, ChromaStore(f"test-{uuid.uuid4().hex}")))
    resp = client.post("/ask", json={"question": "anything"})
    assert resp.status_code == 503
    assert "app.ingest" in resp.json()["detail"]
