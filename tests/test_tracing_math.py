"""Tracing math: decode tokens/sec, cost, rollup totals, percentiles."""

import pytest

from app.tracing.metrics import CallRecord, Price, call_cost, decode_tokens_per_sec, percentile, rollup
from app.tracing.prices import load_prices


def test_decode_tps_basic():
    # 100 tokens over (1500 - 500) ms = 100 tokens/s
    assert decode_tokens_per_sec(100, 1500.0, 500.0) == pytest.approx(100.0)


@pytest.mark.parametrize(
    "out,total,ttft",
    [(100, 500.0, 500.0), (100, 400.0, 500.0), (0, 1000.0, 100.0), (None, 1000.0, 100.0), (10, None, 1.0), (10, 5.0, None)],
)
def test_decode_tps_undefined(out, total, ttft):
    assert decode_tokens_per_sec(out, total, ttft) is None


def test_cost_includes_thinking_at_output_rate():
    price = Price(input_per_m=0.30, output_per_m=2.50)
    # 1000 in, 200 out, 300 thinking -> 1000*0.30/1e6 + 500*2.50/1e6
    assert call_cost(1000, 200, 300, price) == pytest.approx(0.0003 + 0.00125)


def test_cost_unknown_price_or_tokens():
    assert call_cost(1000, 10, 0, None) is None
    assert call_cost(None, 10, 0, Price(1.0, 2.0)) is None
    assert call_cost(1000, 10, 0, Price(1.0, None)) is None  # output tokens but no output price
    assert call_cost(1000, 0, 0, Price(1.0, None)) == pytest.approx(0.001)  # embedding-style


def _rec(call_type, cost=None, inp=0, out=0, think=0, latency=10.0, attempts=1, backoff=0.0, error=None):
    return CallRecord(
        request_id="r", strategy="s", node="n", call_type=call_type, name="m", started_at="t",
        latency_ms=latency, wall_ms=latency + backoff, input_tokens=inp, output_tokens=out,
        thinking_tokens=think, cost_usd=cost, attempts=attempts, backoff_ms=backoff, error=error,
    )


def test_rollup_totals():
    calls = [
        _rec("llm", cost=0.01, inp=100, out=10, think=5, latency=200),
        _rec("llm", cost=0.02, inp=200, out=20, think=0, latency=300, attempts=3, backoff=1500),
        _rec("embed", cost=None, latency=20),
        _rec("tool", latency=40),
        _rec("judge", cost=5.0, inp=999, out=999, latency=999),  # excluded
    ]
    r = rollup("r", "full_history", calls, total_latency_ms=2500.0, n_agent_hops=4, n_critic_retries=1)
    assert r.total_cost_usd == pytest.approx(0.03)
    assert r.cost_complete is True
    assert r.n_llm_calls == 2 and r.n_embed_calls == 1 and r.n_tool_calls == 1
    assert (r.input_tokens, r.output_tokens, r.thinking_tokens) == (300, 30, 5)
    assert r.llm_latency_ms == pytest.approx(500.0)
    assert r.backoff_ms == pytest.approx(1500.0)
    assert r.n_retried_calls == 1
    assert r.total_latency_ms == 2500.0 and r.n_agent_hops == 4 and r.n_critic_retries == 1


def test_rollup_flags_missing_llm_price():
    r = rollup("r", "s", [_rec("llm", cost=None), _rec("llm", cost=0.5)], 1.0, 4, 0)
    assert r.cost_complete is False
    assert r.total_cost_usd == pytest.approx(0.5)


def test_percentile_matches_linear_interpolation():
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert percentile(data, 50) == pytest.approx(5.5)
    assert percentile(data, 95) == pytest.approx(9.55)
    assert percentile(data, 100) == 10
    assert percentile([7.0], 95) == 7.0
    assert percentile([], 95) is None


def test_load_prices(tmp_path):
    f = tmp_path / "p.yaml"
    f.write_text(
        "unit: usd_per_1m_tokens\nmodels:\n"
        "  a: {input: 0.3, output: 2.5, verified: true}\n"
        "  b: {input: 0.15}\n"
        "  c: {input: null}\n"
    )
    t = load_prices(f)
    assert t.get("a") == Price(0.3, 2.5)
    assert t.get("b") == Price(0.15, None)
    assert t.get("c") is None and t.get(None) is None
    assert t.unverified == ("b",)
    assert load_prices(tmp_path / "missing.yaml").prices == {}
