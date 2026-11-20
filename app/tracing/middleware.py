"""Tracing middleware: wraps every model call and tool call.

`app/llm.py` opens a `model_call` scope around each backend call, and the
retriever opens a `tool_call` scope around each tool run. Each scope:
  - opens an OpenTelemetry span (exported to Jaeger when configured),
  - builds a `CallRecord` with tokens, TTFT, latency, decode tokens/sec, cost,
  - appends it to the active request's list and writes it to the SQLite sink.

Calls made outside a request scope (e.g. ingestion embeddings) get a span but
no metrics row.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterator

from opentelemetry.trace import Status, StatusCode

from app.tracing.metrics import CallRecord, call_cost, decode_tokens_per_sec
from app.tracing.otel import get_tracer, setup_tracing
from app.tracing.prices import PriceTable, load_prices
from app.tracing.store import MetricsStore

log = logging.getLogger("tracewise.metrics")


@dataclass
class Usage:
    """What a backend reports about one model call."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    thinking_tokens: int | None = None
    ttft_ms: float | None = None
    latency_ms: float | None = None  # successful attempt
    attempts: int = 1
    backoff_ms: float = 0.0


@dataclass
class RequestContext:
    request_id: str
    strategy: str
    calls: list[CallRecord] = field(default_factory=list)


@dataclass
class _State:
    prices: PriceTable = field(default_factory=lambda: PriceTable({}, ()))
    sink: MetricsStore | None = None
    log_json: bool = False


_state = _State()
_request: ContextVar[RequestContext | None] = ContextVar("tracewise_request", default=None)
_node: ContextVar[str | None] = ContextVar("tracewise_node", default=None)


def configure(settings, sink: MetricsStore | None = None) -> None:
    setup_tracing(settings.otel_exporter, settings.otel_endpoint, settings.service_name)
    _state.prices = load_prices(settings.prices_file)
    if sink is not None:
        _state.sink = sink
    elif settings.metrics_db:
        _state.sink = MetricsStore(settings.metrics_db)
    else:
        _state.sink = None
    _state.log_json = settings.log_metrics_json


def reset() -> None:
    """Back to the unconfigured state (tests)."""
    global _state
    _state = _State()


def prices() -> PriceTable:
    return _state.prices


def sink() -> MetricsStore | None:
    return _state.sink


def current_request() -> RequestContext | None:
    return _request.get()


@contextmanager
def request_scope(request_id: str, strategy: str, question: str | None = None) -> Iterator[RequestContext]:
    ctx = RequestContext(request_id=request_id, strategy=strategy)
    token = _request.set(ctx)
    with get_tracer().start_as_current_span("request") as span:
        span.set_attribute("tracewise.request_id", request_id)
        span.set_attribute("tracewise.strategy", strategy)
        if question is not None:
            span.set_attribute("tracewise.question_chars", len(question))
        try:
            yield ctx
        finally:
            _request.reset(token)


@contextmanager
def judge_scope(ctx: RequestContext) -> Iterator[RequestContext]:
    """Re-enter an existing request (used by the eval judge after the graph finished)."""
    token = _request.set(ctx)
    try:
        yield ctx
    finally:
        _request.reset(token)


@contextmanager
def node_scope(node: str) -> Iterator[None]:
    token = _node.set(node)
    with get_tracer().start_as_current_span(f"node.{node}") as span:
        span.set_attribute("tracewise.node", node)
        try:
            yield
        finally:
            _node.reset(token)


class CallHandle:
    def __init__(self) -> None:
        self.usage: Usage | None = None

    def finish(self, usage: Usage) -> None:
        self.usage = usage


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _emit(rec: CallRecord) -> None:
    ctx = _request.get()
    if ctx is None:
        return
    ctx.calls.append(rec)
    if _state.sink is not None:
        _state.sink.insert_call(rec)
    if _state.log_json:
        log.info(json.dumps({"tracewise_call": rec.as_dict()}))


def build_record(
    call_type: str,
    name: str,
    node: str | None,
    started_at: str,
    wall_ms: float,
    usage: Usage | None,
    error: str | None = None,
) -> CallRecord:
    ctx = _request.get()
    u = usage or Usage()
    latency = u.latency_ms if u.latency_ms is not None else wall_ms
    price = _state.prices.get(name) if call_type in ("llm", "embed", "judge") else None
    return CallRecord(
        request_id=ctx.request_id if ctx else None,
        strategy=ctx.strategy if ctx else None,
        node=node,
        call_type=call_type,
        name=name,
        started_at=started_at,
        latency_ms=latency,
        wall_ms=wall_ms,
        ttft_ms=u.ttft_ms,
        input_tokens=u.input_tokens,
        output_tokens=u.output_tokens,
        thinking_tokens=u.thinking_tokens,
        decode_tps=decode_tokens_per_sec(u.output_tokens, latency, u.ttft_ms),
        cost_usd=call_cost(u.input_tokens, u.output_tokens, u.thinking_tokens, price) if usage else None,
        attempts=u.attempts,
        backoff_ms=u.backoff_ms,
        error=error,
    )


def _set_span_attrs(span, rec: CallRecord) -> None:
    attrs = {
        "tracewise.call_type": rec.call_type,
        "tracewise.node": rec.node,
        "tracewise.latency_ms": rec.latency_ms,
        "tracewise.wall_ms": rec.wall_ms,
        "tracewise.ttft_ms": rec.ttft_ms,
        "tracewise.decode_tps": rec.decode_tps,
        "tracewise.cost_usd": rec.cost_usd,
        "tracewise.attempts": rec.attempts,
        "tracewise.backoff_ms": rec.backoff_ms,
        "gen_ai.request.model": rec.name if rec.call_type != "tool" else None,
        "gen_ai.usage.input_tokens": rec.input_tokens,
        "gen_ai.usage.output_tokens": rec.output_tokens,
        "tracewise.thinking_tokens": rec.thinking_tokens,
    }
    for k, v in attrs.items():
        if v is not None:
            span.set_attribute(k, v)


@contextmanager
def _scope(span_name: str, call_type: str, name: str, node: str | None) -> Iterator[CallHandle]:
    node = node or _node.get()
    handle = CallHandle()
    started_at = _now_iso()
    t0 = time.perf_counter()
    with get_tracer().start_as_current_span(span_name) as span:
        try:
            yield handle
        except Exception as exc:
            wall = (time.perf_counter() - t0) * 1000
            rec = build_record(call_type, name, node, started_at, wall, handle.usage, error=f"{type(exc).__name__}: {exc}")
            _set_span_attrs(span, rec)
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            _emit(rec)
            raise
        wall = (time.perf_counter() - t0) * 1000
        rec = build_record(call_type, name, node, started_at, wall, handle.usage)
        _set_span_attrs(span, rec)
        _emit(rec)


def model_call(call_type: str, model_id: str, node: str | None = None):
    """Scope for one generate or embed call. The caller must `handle.finish(usage)`."""
    return _scope(f"{call_type}.{model_id}", call_type, model_id, node)


def tool_call(tool_name: str, node: str | None = None):
    return _scope(f"tool.{tool_name}", "tool", tool_name, node)
