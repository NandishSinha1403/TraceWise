"""Run one state strategy over the fixed eval set and save raw outputs under results/.

    python -m eval.run_eval --strategy full_history
    python -m eval.run_eval --strategy sliding_window --k 4
    python -m eval.run_eval --strategy rolling_summary
    python -m eval.run_eval --resume results/full_history_20261120T101500Z   # continue after a quota stop

Writes results/<strategy>_<UTC timestamp>/:
    run_meta.json   settings, model ids, eval-set hash, git commit
    prices.yaml     snapshot of the price table used
    requests.csv    one row per question: answer, rollup metrics, judge scores
    calls.csv       one row per model/tool call (including judge calls)
    raw.jsonl       full per-question output, including retrieved passages

Fake-backend runs are smoke tests: they need --allow-fake and go to results/_smoke/,
which eval.summarize ignores.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from app import llm
from app.config import ConfigError, Settings, load_settings
from app.graph import build_graph, run, strategy_label
from app.tracing import middleware as tracing
from app.tracing.metrics import CallRecord, RequestRollup
from app.vectorstore import open_store
from eval.judge import judge, load_rubric

ROLLUP_FIELDS = [f for f in RequestRollup.__dataclass_fields__ if f not in ("request_id", "strategy", "errors")]
REQUEST_FIELDS = [
    "question_id", "request_id", "strategy", "ok", "error", "answer", "cited_doc_ids", "source_doc_id",
    "cited_source", "critic_retries", "capped", "critic_verdict", "steps", *ROLLUP_FIELDS,
    "judge_correctness", "judge_groundedness", "judge_rationale", "judge_error",
]
CALL_FIELDS = list(CallRecord.__dataclass_fields__)


def load_questions(path: Path) -> list[dict]:
    items = []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            missing = {"id", "question", "reference"} - set(item)
            if missing:
                raise ValueError(f"{path}:{n} missing keys {sorted(missing)}")
            items.append(item)
    ids = [i["id"] for i in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate question ids")
    return items


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        ).stdout.strip()
    except Exception:
        return None


def run_meta(settings: Settings, questions_path: Path, n_questions: int, judge_enabled: bool) -> dict:
    return {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "strategy": settings.strategy,
        "strategy_label": strategy_label(settings),
        "window_k": settings.window_k if settings.strategy == "sliding_window" else None,
        "summary_keep_recent": settings.summary_keep_recent if settings.strategy == "rolling_summary" else None,
        "llm_backend": settings.llm_backend,
        "gen_model_id": settings.gen_model_id,
        "judge_model_id": settings.judge_model_id if judge_enabled else None,
        "embedding_model_id": settings.embedding_model_id,
        "gen_temperature": settings.gen_temperature,
        "judge_temperature": settings.judge_temperature,
        "gen_thinking_budget": settings.gen_thinking_budget,
        "gen_thinking_level": settings.gen_thinking_level,
        "max_critic_retries": settings.max_critic_retries,
        "max_retriever_steps": settings.max_retriever_steps,
        "search_top_k": settings.search_top_k,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "vector_store": settings.vector_store,
        "collection": settings.collection_name,
        "eval_set": str(questions_path),
        "eval_set_sha256": _sha256(questions_path),
        "eval_set_size": n_questions,
        "git_commit": _git_commit(),
    }


class CsvAppender:
    def __init__(self, path: Path, fields: list[str]) -> None:
        new = not path.exists()
        self._f = path.open("a", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=fields, extrasaction="ignore")
        if new:
            self._w.writeheader()

    def write(self, row: dict) -> None:
        self._w.writerow(row)
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def completed_ids(run_dir: Path) -> set[str]:
    path = run_dir / "requests.csv"
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {r["question_id"] for r in csv.DictReader(f) if r["ok"] == "True"}


def _row(item: dict, out: dict | None, error: str | None, label: str) -> dict:
    row = {"question_id": item["id"], "strategy": label, "ok": error is None, "error": error,
           "source_doc_id": item.get("source_doc_id")}
    if out is not None:
        usage = out["usage"]
        row.update(
            request_id=out["request_id"],
            answer=out["answer"],
            cited_doc_ids=json.dumps(out["cited_doc_ids"]),
            cited_source=(item.get("source_doc_id") in out["cited_doc_ids"]) if item.get("source_doc_id") else None,
            critic_retries=out["critic_retries"],
            capped=out["capped"],
            critic_verdict=out["critic_verdict"]["verdict"],
            steps=" ".join(out["steps"]),
            **{k: usage[k] for k in ROLLUP_FIELDS},
        )
    return row


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strategy", choices=["full_history", "sliding_window", "rolling_summary"])
    p.add_argument("--k", type=int, help="window size for sliding_window")
    p.add_argument("--questions", default="eval/questions.jsonl")
    p.add_argument("--limit", type=int, help="only the first N questions")
    p.add_argument("--resume", help="existing run folder to continue")
    p.add_argument("--pace", type=float, default=0.0, help="seconds to sleep between questions (free-tier pacing)")
    p.add_argument("--max-consecutive-errors", type=int, default=3)
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--allow-fake", action="store_true", help="permit LLM_BACKEND=fake (smoke test only)")
    p.add_argument("--results-dir", default="results")
    args = p.parse_args(argv)

    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.resume:
        run_dir = Path(args.resume)
        meta = json.loads((run_dir / "run_meta.json").read_text())
        settings = settings.with_strategy(meta["strategy"], meta.get("window_k"))
        questions_path = Path(meta["eval_set"])
        if _sha256(questions_path) != meta["eval_set_sha256"]:
            print("error: eval set changed since this run started; start a new run", file=sys.stderr)
            return 2
        judge_enabled = meta.get("judge_model_id") is not None
    else:
        if not args.strategy:
            p.error("--strategy is required unless --resume is given")
        if args.strategy == "sliding_window" and not args.k:
            p.error("--k is required for sliding_window")
        settings = settings.with_strategy(args.strategy, args.k)
        questions_path = Path(args.questions)
        judge_enabled = not args.no_judge
        run_dir = None

    if settings.llm_backend == "fake" and not args.allow_fake:
        print("error: LLM_BACKEND=fake produces canned answers; pass --allow-fake for a smoke test", file=sys.stderr)
        return 2
    if judge_enabled and settings.llm_backend != "fake" and not settings.judge_model_id:
        print("error: JUDGE_MODEL_ID is not set (or pass --no-judge)", file=sys.stderr)
        return 2
    if not questions_path.exists():
        print(f"error: eval set not found: {questions_path}", file=sys.stderr)
        return 2

    questions = load_questions(questions_path)
    if args.limit:
        questions = questions[: args.limit]

    if run_dir is None:
        base = Path(args.results_dir) / ("_smoke" if settings.llm_backend == "fake" else "")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = base / f"{strategy_label(settings)}_{stamp}"
        run_dir.mkdir(parents=True, exist_ok=False)
        meta = run_meta(settings, questions_path, len(questions), judge_enabled)
        (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
        if Path(settings.prices_file).exists():
            shutil.copy(settings.prices_file, run_dir / "prices.yaml")

    llm.configure(settings)
    tracing.configure(settings)
    store = open_store(settings)
    if store.count() == 0:
        print("error: vector store is empty; run `python -m app.ingest` first", file=sys.stderr)
        return 2
    graph = build_graph(settings, store)
    rubric = load_rubric() if judge_enabled else ""

    done = completed_ids(run_dir)
    todo = [q for q in questions if q["id"] not in done]
    label = strategy_label(settings)
    print(f"run: {run_dir}  strategy: {label}  questions: {len(todo)} to do, {len(done)} already done")

    req_csv = CsvAppender(run_dir / "requests.csv", REQUEST_FIELDS)
    call_csv = CsvAppender(run_dir / "calls.csv", CALL_FIELDS)
    raw = (run_dir / "raw.jsonl").open("a", encoding="utf-8")
    consecutive_errors = 0
    try:
        for n, item in enumerate(todo, 1):
            out, error = None, None
            try:
                out = run(graph, settings, item["question"], keep_context=True)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            row = _row(item, out, error, label)

            if out is not None and judge_enabled:
                ctx = out["_context"]
                with tracing.judge_scope(ctx):
                    try:
                        jr = judge(item["question"], item["reference"], out["_retrieved"], out["answer"],
                                   out["cited_doc_ids"], rubric)
                        row.update(judge_correctness=jr.correctness, judge_groundedness=jr.groundedness,
                                   judge_rationale=jr.rationale, judge_error=jr.error)
                    except Exception as exc:
                        row.update(judge_error=f"{type(exc).__name__}: {exc}")

            calls = out["_context"].calls if out is not None else []
            for c in calls:
                call_csv.write(asdict(c))
            req_csv.write(row)
            raw.write(json.dumps({
                "question": item, "error": error,
                "output": {k: v for k, v in (out or {}).items() if not k.startswith("_")},
                "retrieved": (out or {}).get("_retrieved", []),
                "judge": {k: row.get(k) for k in ("judge_correctness", "judge_groundedness", "judge_rationale", "judge_error")},
            }) + "\n")
            raw.flush()

            status = "ok" if error is None else f"ERROR {error[:120]}"
            scores = f" judge={row.get('judge_correctness')}/{row.get('judge_groundedness')}" if judge_enabled and out else ""
            print(f"[{n}/{len(todo)}] {item['id']}: {status}{scores}")

            consecutive_errors = consecutive_errors + 1 if error else 0
            if consecutive_errors >= args.max_consecutive_errors:
                print(
                    f"stopping after {consecutive_errors} consecutive errors (likely the free-tier daily quota). "
                    f"Continue later with: python -m eval.run_eval --resume {run_dir}",
                    file=sys.stderr,
                )
                return 1
            if args.pace and n < len(todo):
                time.sleep(args.pace)
    finally:
        req_csv.close()
        call_csv.close()
        raw.close()

    print(f"done: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
