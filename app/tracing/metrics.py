"""Pure tracing math: per-call metrics, cost, and percentiles. No I/O."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Iterable

CALL_TYPES = ("llm", "embed", "tool", "judge")


@dataclass(frozen=True)
class Price:
    """USD per 1M tokens."""

    input_per_m: float
    output_per_m: float | None = None


@dataclass
class CallRecord:
    request_id: str | None
    strategy: str | None
    node: str | None
    call_type: str  # one of CALL_TYPES
    name: str  # model id for llm/embed/judge, tool name for tool
    started_at: str  # ISO-8601 UTC
    latency_ms: float  # successful attempt only
    wall_ms: float  # including retries and backoff
    ttft_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None  # visible output (candidates)
    thinking_tokens: int | None = None
    decode_tps: float | None = None
    cost_usd: float | None = None
    attempts: int = 1
    backoff_ms: float = 0.0
    error: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class RequestRollup:
    request_id: str
    strategy: str
    total_latency_ms: float
    total_cost_usd: float
    cost_complete: bool  # False if any LLM call had no price, so total is a lower bound
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
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def decode_tokens_per_sec(output_tokens: int | None, total_ms: float | None, ttft_ms: float | None) -> float | None:
    """output tokens / (total - TTFT), in tokens per second.

    Returns None when the decode window is not positive or inputs are missing
    (e.g. a single-chunk response where TTFT == total).
    """
    if output_tokens is None or total_ms is None or ttft_ms is None:
        return None
    decode_ms = total_ms - ttft_ms
    if decode_ms <= 0 or output_tokens <= 0:
        return None
    return output_tokens / (decode_ms / 1000.0)


def call_cost(
    input_tokens: int | None,
    output_tokens: int | None,
    thinking_tokens: int | None,
    price: Price | None,
) -> float | None:
    """input x input price + (output + thinking) x output price.

    Thinking tokens are billed at the output rate. Returns None when the price
    or the input token count is unknown.
    """
    if price is None or input_tokens is None:
        return None
    cost = input_tokens * price.input_per_m / 1_000_000
    billed_out = (output_tokens or 0) + (thinking_tokens or 0)
    if billed_out:
        if price.output_per_m is None:
            return None
        cost += billed_out * price.output_per_m / 1_000_000
    return cost


def rollup(
    request_id: str,
    strategy: str,
    calls: Iterable[CallRecord],
    total_latency_ms: float,
    n_agent_hops: int,
    n_critic_retries: int,
) -> RequestRollup:
    """Per-request totals. Judge calls are excluded: they are eval overhead, not request cost."""
    calls = [c for c in calls if c.call_type != "judge"]
    llm_calls = [c for c in calls if c.call_type == "llm"]
    priced = [c for c in calls if c.call_type in ("llm", "embed")]
    return RequestRollup(
        request_id=request_id,
        strategy=strategy,
        total_latency_ms=total_latency_ms,
        total_cost_usd=sum(c.cost_usd or 0.0 for c in priced),
        cost_complete=all(c.cost_usd is not None for c in llm_calls),
        n_llm_calls=len(llm_calls),
        n_embed_calls=sum(1 for c in calls if c.call_type == "embed"),
        n_tool_calls=sum(1 for c in calls if c.call_type == "tool"),
        n_agent_hops=n_agent_hops,
        n_critic_retries=n_critic_retries,
        input_tokens=sum(c.input_tokens or 0 for c in llm_calls),
        output_tokens=sum(c.output_tokens or 0 for c in llm_calls),
        thinking_tokens=sum(c.thinking_tokens or 0 for c in llm_calls),
        llm_latency_ms=sum(c.latency_ms for c in llm_calls),
        backoff_ms=sum(c.backoff_ms for c in priced),
        n_retried_calls=sum(1 for c in priced if c.attempts > 1),
        errors=[c.error for c in calls if c.error],
    )


def percentile(values: Iterable[float], p: float) -> float | None:
    """Percentile with linear interpolation between closest ranks (numpy's default)."""
    data = sorted(values)
    if not data:
        return None
    if not 0 <= p <= 100:
        raise ValueError("p must be in [0, 100]")
    pos = (len(data) - 1) * p / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return data[lo]
    return data[lo] + (data[hi] - data[lo]) * (pos - lo)
