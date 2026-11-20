from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from app import llm
from app.config import load_settings
from app.ingest import ingest
from app.vectorstore import ChromaStore

FIXTURE_DOCS = Path(__file__).parent / "fixtures" / "docs"
FIXTURE_DOC_IDS = {p.name for p in FIXTURE_DOCS.glob("*.md")}


def fake_settings(**overrides: str):
    env = {"LLM_BACKEND": "fake", "DOCS_DIR": str(FIXTURE_DOCS)}
    env.update(overrides)
    return load_settings(env)


def fixture_store(settings) -> ChromaStore:
    llm.configure(settings)
    store = ChromaStore(f"test-{uuid.uuid4().hex}")  # in-memory, unique per test
    ingest(settings, store)
    return store


@pytest.fixture(autouse=True)
def _reset_tracing():
    from app.tracing import middleware

    middleware.reset()
    yield
    middleware.reset()


@pytest.fixture
def make_env():
    """Return (settings, store) for the fake backend with the fixture corpus ingested."""

    def _make(**overrides: str):
        settings = fake_settings(**overrides)
        return settings, fixture_store(settings)

    return _make
