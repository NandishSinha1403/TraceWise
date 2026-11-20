"""State strategies: what transcript each node sees."""

from app.config import ConfigError
from app.graph import build_graph, run
from app.state import Message, initial_state
from app.strategies import build_context
from app.tracing import middleware as tracing

import pytest


def _state(strategy, n, **extra):
    s = initial_state("q", "r", strategy, window_k=extra.pop("window_k", 2))
    s["messages"] = [Message(role="assistant", name=f"m{i}", content=f"c{i}") for i in range(n)]
    s.update(extra)
    return s


def test_full_history_includes_everything():
    text, upd = build_context(_state("full_history", 5), keep_recent=2)
    assert all(f"[m{i}]" in text for i in range(5)) and upd == {}


def test_sliding_window_keeps_last_k():
    text, upd = build_context(_state("sliding_window", 5, window_k=2), keep_recent=2)
    assert "[m3]" in text and "[m4]" in text and "[m2]" not in text and upd == {}


def test_rolling_summary_folds_when_tail_exceeds_threshold(make_env):
    make_env()  # configures fake llm
    # 4 messages, keep 2 -> threshold 4, not exceeded: no fold
    text, upd = build_context(_state("rolling_summary", 4), keep_recent=2)
    assert upd == {} and "[m0]" in text
    # 5 messages -> fold first 3, keep last 2
    text, upd = build_context(_state("rolling_summary", 5), keep_recent=2)
    assert upd["summarized_upto"] == 3 and upd["summary"].startswith("Fake summary")
    summary_line, *tail = text.splitlines()
    assert summary_line.startswith("[summary of earlier messages] Fake summary")
    assert tail == ["[m3] c3", "[m4] c4"]


def test_rolling_summary_uses_existing_summary_without_refolding(make_env):
    make_env()
    text, upd = build_context(_state("rolling_summary", 6, summary="S", summarized_upto=3), keep_recent=2)
    assert upd == {} and "S" in text and "[m2]" not in text and "[m5]" in text


@pytest.mark.parametrize("strategy,k", [("full_history", None), ("sliding_window", 2), ("rolling_summary", None)])
def test_graph_runs_under_each_strategy(make_env, strategy, k):
    settings, store = make_env(MAX_CRITIC_RETRIES="2", FAKE_CRITIC_VERDICT="retry_retriever", SUMMARY_KEEP_RECENT="1")
    settings = settings.with_strategy(strategy, k)
    out = run(build_graph(settings, store), settings, "What colour is the Varnholt lamp?")
    assert out["capped"] is True and out["critic_retries"] == 2
    assert out["strategy"] == ("sliding_window_k2" if strategy == "sliding_window" else strategy)
    calls = out["usage"]["n_llm_calls"]
    if strategy == "rolling_summary":
        assert calls > 1 + 3 * 4  # summarizer calls on top of node calls
    else:
        assert calls == 1 + 3 * 4  # planner + 3 x (retriever x2, answer, critic)


def test_with_strategy_rejects_unknown(make_env):
    settings, _ = make_env()
    with pytest.raises(ConfigError):
        settings.with_strategy("nope")
