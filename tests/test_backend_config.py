"""Backend selection, .env loading, and rate-limit retries. No network."""

import pytest
from google.genai import errors

from app import llm
from app.config import ConfigError, load_dotenv, load_settings


def test_gemini_api_requires_key_and_models():
    with pytest.raises(ConfigError) as exc:
        load_settings({"LLM_BACKEND": "gemini_api"})
    msg = str(exc.value)
    for name in ("GEMINI_API_KEY", "GEN_MODEL_ID", "EMBEDDING_MODEL_ID"):
        assert name in msg


def test_settings_repr_hides_api_key():
    s = load_settings(
        {"LLM_BACKEND": "gemini_api", "GEMINI_API_KEY": "secret-value", "GEN_MODEL_ID": "g", "EMBEDDING_MODEL_ID": "e"}
    )
    assert "secret-value" not in repr(s)
    assert "secret-value" not in str(s)


def test_load_dotenv(tmp_path):
    f = tmp_path / ".env"
    f.write_text('# comment\nA=1\nexport B="two"\nC=\nbad line\n')
    assert load_dotenv(str(f)) == {"A": "1", "B": "two", "C": ""}
    assert load_dotenv(str(tmp_path / "missing")) == {}


def _backend(monkeypatch, max_retries: str):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    s = load_settings(
        {
            "LLM_BACKEND": "gemini_api",
            "GEMINI_API_KEY": "test-key",
            "GEN_MODEL_ID": "g",
            "EMBEDDING_MODEL_ID": "e",
            "LLM_MAX_RETRIES": max_retries,
        }
    )
    return llm.GenAIBackend(s)  # constructing the client makes no network call


def _flaky(failures: list[int]):
    calls = []

    def fn():
        calls.append(1)
        if len(calls) <= len(failures):
            raise errors.APIError(failures[len(calls) - 1], {"error": {"message": "x"}})
        return "ok"

    return fn, calls


def test_retries_rate_limit_then_succeeds(monkeypatch):
    backend = _backend(monkeypatch, "3")
    fn, calls = _flaky([429, 503])
    assert backend._call(fn) == "ok"
    assert len(calls) == 3


def test_gives_up_after_max_retries(monkeypatch):
    backend = _backend(monkeypatch, "1")
    fn, calls = _flaky([429, 429, 429])
    with pytest.raises(errors.APIError):
        backend._call(fn)
    assert len(calls) == 2


def test_does_not_retry_client_errors(monkeypatch):
    backend = _backend(monkeypatch, "5")
    fn, calls = _flaky([400])
    with pytest.raises(errors.APIError):
        backend._call(fn)
    assert len(calls) == 1


def test_thinking_level_validated_and_exclusive_with_budget():
    base = {"LLM_BACKEND": "fake"}
    assert load_settings({**base, "GEN_THINKING_LEVEL": "Minimal"}).gen_thinking_level == "minimal"
    assert load_settings(base).gen_thinking_level is None
    with pytest.raises(ConfigError):
        load_settings({**base, "GEN_THINKING_LEVEL": "none"})
    with pytest.raises(ConfigError):
        load_settings({**base, "GEN_THINKING_LEVEL": "low", "GEN_THINKING_BUDGET": "0"})


def test_tool_call_thought_signature_is_sent_back():
    """Gemini 3+ returns 400 if a replayed function call lacks its thought_signature."""
    sig = b"\x01opaque-signature"
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [llm.ToolCall("search_documents", {"query": "x"}, "id1", sig)]},
        {"role": "tool", "tool_name": "search_documents", "content": "passages"},
    ]
    contents = llm.GenAIBackend._to_contents(messages)
    fc_part = contents[1].parts[0]
    assert fc_part.function_call.name == "search_documents"
    assert fc_part.thought_signature == sig
