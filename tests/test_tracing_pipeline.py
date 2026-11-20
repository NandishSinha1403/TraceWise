"""Middleware records per-call rows and request rollups end to end (fake backend)."""

from app.graph import build_graph, run
from app.tracing import middleware as tracing
from app.tracing.store import MetricsStore


def test_calls_and_request_written_to_sqlite(make_env, tmp_path):
    settings, store = make_env(PRICES_FILE=str(tmp_path / "none.yaml"))
    sink = MetricsStore(tmp_path / "m.sqlite")
    tracing.configure(settings, sink=sink)

    out = run(build_graph(settings, store), settings, "Which apple grows at Pellam Orchard?", request_id="req-1")

    rows = sink.query("SELECT call_type, node, request_id FROM llm_calls ORDER BY id")
    types = [r[0] for r in rows]
    assert types.count("llm") == out["usage"]["n_llm_calls"] == 5
    assert types.count("tool") == 1 and types.count("embed") == 1  # one search -> one query embedding
    assert {r[2] for r in rows} == {"req-1"}
    assert [r[1] for r in rows if r[0] == "llm"] == ["planner", "retriever", "retriever", "answer", "critic"]

    [req] = sink.query("SELECT request_id, n_llm_calls, n_agent_hops, cost_complete FROM requests")
    assert req == ("req-1", 5, 4, 0)  # no price file -> cost incomplete


def test_fake_usage_feeds_decode_tps(make_env, tmp_path):
    settings, store = make_env()
    sink = MetricsStore(tmp_path / "m.sqlite")
    tracing.configure(settings, sink=sink)
    run(build_graph(settings, store), settings, "Name the Quill Valley Railway engines.")
    for out_tokens, ttft, latency, tps in sink.query(
        "SELECT output_tokens, ttft_ms, latency_ms, decode_tps FROM llm_calls WHERE call_type='llm'"
    ):
        assert tps == out_tokens / ((latency - ttft) / 1000)


def test_failed_request_still_writes_rollup(make_env, tmp_path):
    from app import llm

    settings, store = make_env()
    sink = MetricsStore(tmp_path / "m.sqlite")
    tracing.configure(settings, sink=sink)

    class Boom(llm.FakeBackend):
        def generate(self, role, *a, **k):
            if role == "answer":
                raise RuntimeError("quota exhausted")
            return super().generate(role, *a, **k)

    llm.set_backend(Boom(settings))
    import pytest

    with pytest.raises(RuntimeError):
        run(build_graph(settings, store), settings, "q?", request_id="bad")
    [(errors,)] = sink.query("SELECT errors FROM requests WHERE request_id='bad'")
    assert "quota exhausted" in errors
    [(err,)] = sink.query("SELECT error FROM llm_calls WHERE node='answer'")
    assert "quota exhausted" in err
