"""Summarize saved eval runs. Every number comes from files under results/.

    python -m eval.summarize results/
    python -m eval.summarize results/ --markdown     # table for RESULTS.md

One row per run folder. Latency is wall-clock per request; "excl. backoff"
subtracts time spent waiting on free-tier rate-limit retries.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean

from app.tracing.metrics import percentile
from app.tracing.prices import load_prices

COLUMNS = [
    ("run", "run"),
    ("strategy", "strategy"),
    ("n_ok", "n ok"),
    ("n_err", "n err"),
    ("lat_mean_ms", "latency mean ms"),
    ("lat_p95_ms", "latency p95 ms"),
    ("lat_excl_mean_ms", "latency excl. backoff mean ms"),
    ("lat_excl_p95_ms", "latency excl. backoff p95 ms"),
    ("cost_mean_usd", "cost/request USD"),
    ("llm_calls_mean", "LLM calls/request"),
    ("input_tokens_mean", "input tokens/request"),
    ("decode_tps_mean", "decode tok/s mean"),
    ("correctness_mean", "judge correctness"),
    ("groundedness_mean", "judge groundedness"),
    ("n_judged", "n judged"),
]


def _f(row: dict, key: str) -> float | None:
    v = row.get(key)
    if v in (None, "", "None"):
        return None
    return float(v)


def _mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return mean(vals) if vals else None


def summarize_run(run_dir: Path) -> dict | None:
    req_path = run_dir / "requests.csv"
    if not req_path.exists():
        return None
    with req_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ok = [r for r in rows if r["ok"] == "True"]
    meta = json.loads((run_dir / "run_meta.json").read_text()) if (run_dir / "run_meta.json").exists() else {}

    lat = [_f(r, "total_latency_ms") for r in ok]
    lat = [v for v in lat if v is not None]
    lat_excl = [_f(r, "total_latency_ms") - (_f(r, "backoff_ms") or 0.0) for r in ok if _f(r, "total_latency_ms") is not None]

    tps: list[float] = []
    calls_path = run_dir / "calls.csv"
    if calls_path.exists():
        with calls_path.open(encoding="utf-8") as f:
            for c in csv.DictReader(f):
                if c["call_type"] == "llm" and c["decode_tps"] not in ("", "None"):
                    tps.append(float(c["decode_tps"]))

    judged = [r for r in ok if r.get("judge_correctness") not in (None, "", "None")]
    cost_incomplete = any(r.get("cost_complete") == "False" for r in ok)
    return {
        "run": run_dir.name,
        "strategy": meta.get("strategy_label") or (ok[0]["strategy"] if ok else ""),
        "n_ok": len(ok),
        "n_err": len(rows) - len(ok),
        "lat_mean_ms": _mean(lat),
        "lat_p95_ms": percentile(lat, 95),
        "lat_excl_mean_ms": _mean(lat_excl),
        "lat_excl_p95_ms": percentile(lat_excl, 95),
        "cost_mean_usd": _mean([_f(r, "total_cost_usd") for r in ok]),
        "llm_calls_mean": _mean([_f(r, "n_llm_calls") for r in ok]),
        "input_tokens_mean": _mean([_f(r, "input_tokens") for r in ok]),
        "decode_tps_mean": _mean(tps),
        "correctness_mean": _mean([_f(r, "judge_correctness") for r in judged]),
        "groundedness_mean": _mean([_f(r, "judge_groundedness") for r in judged]),
        "n_judged": len(judged),
        "_cost_incomplete": cost_incomplete,
        "_meta": meta,
        "_unverified_prices": load_prices(run_dir / "prices.yaml").unverified,
    }


def find_runs(root: Path, include_smoke: bool) -> list[Path]:
    runs = []
    for req in sorted(root.rglob("requests.csv")):
        run_dir = req.parent
        if not include_smoke and any(part.startswith("_") for part in run_dir.relative_to(root).parts):
            continue
        runs.append(run_dir)
    return runs


def _fmt(key: str, v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if key == "cost_mean_usd":
            return f"{v:.6f}"
        return f"{v:.2f}" if key in ("correctness_mean", "groundedness_mean", "llm_calls_mean") else f"{v:.1f}"
    return str(v)


def render(summaries: list[dict], markdown: bool) -> str:
    headers = [h for _, h in COLUMNS]
    table = [[_fmt(k, s[k]) for k, _ in COLUMNS] for s in summaries]
    if markdown:
        lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
        lines += ["| " + " | ".join(row) + " |" for row in table]
        return "\n".join(lines)
    widths = [max(len(h), *(len(r[i]) for r in table)) for i, h in enumerate(headers)]
    out = ["  ".join(h.ljust(w) for h, w in zip(headers, widths))]
    out += ["  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in table]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", nargs="?", default="results")
    p.add_argument("--markdown", action="store_true")
    p.add_argument("--include-smoke", action="store_true", help="include fake-backend runs under results/_smoke")
    args = p.parse_args(argv)

    root = Path(args.root)
    summaries = [s for s in (summarize_run(r) for r in find_runs(root, args.include_smoke)) if s]
    if not summaries:
        print(f"no runs with requests.csv under {root}", file=sys.stderr)
        return 1
    print(render(summaries, args.markdown))

    notes = []
    for s in summaries:
        m = s["_meta"]
        if m.get("llm_backend") == "fake":
            notes.append(f"{s['run']}: FAKE backend smoke run, not a real result")
        if s["_cost_incomplete"]:
            notes.append(f"{s['run']}: some LLM calls had no price; cost is a lower bound")
        if s["_unverified_prices"]:
            notes.append(f"{s['run']}: unverified prices in prices.yaml: {', '.join(s['_unverified_prices'])}")
    models = {(s["_meta"].get("gen_model_id"), s["_meta"].get("judge_model_id"), s["_meta"].get("eval_set_sha256")) for s in summaries}
    if len(models) > 1:
        notes.append("runs differ in model ids or eval set; they are not directly comparable")
    notes.append("cost = paid-tier list-price estimate; runs used the Gemini API free tier (actual spend $0); embedding cost not measured")
    print()
    for n in notes:
        print(f"note: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
