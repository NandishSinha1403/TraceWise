"""Eval runner, judge parsing, and summarize, with the fake backend. No network."""

import csv
import json
from pathlib import Path

import pytest

from app import llm
from app.config import load_settings
from app.ingest import ingest
from app.vectorstore import open_store
from eval import run_eval, summarize
from eval.judge import parse_scores
from tests.conftest import FIXTURE_DOCS

QUESTIONS = [
    {"id": "a", "question": "What colour is the Varnholt lamp?", "reference": "Green", "source_doc_id": "fixture_lighthouse.md"},
    {"id": "b", "question": "Which apple grows at Pellam Orchard?", "reference": "Ember Pippin", "source_doc_id": "fixture_orchard.md"},
]


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    env = {
        "LLM_BACKEND": "fake",
        "CHROMA_DIR": str(tmp_path / "chroma"),
        "DOCS_DIR": str(FIXTURE_DOCS),
        "METRICS_DB": str(tmp_path / "metrics.sqlite"),
        "PRICES_FILE": str(tmp_path / "prices.yaml"),
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    (tmp_path / "prices.yaml").write_text(
        "unit: usd_per_1m_tokens\nmodels:\n  fake-gen: {input: 1.0, output: 2.0, verified: true}\n"
    )
    settings = load_settings()
    llm.configure(settings)
    ingest(settings, open_store(settings))
    q = tmp_path / "questions.jsonl"
    q.write_text("\n".join(json.dumps(x) for x in QUESTIONS) + "\n")
    return tmp_path, q


def _run_dir(results: Path) -> Path:
    [d] = list((results / "_smoke").iterdir())
    return d


def test_refuses_fake_without_flag(fake_env):
    tmp, q = fake_env
    assert run_eval.main(["--strategy", "full_history", "--questions", str(q), "--results-dir", str(tmp / "r")]) == 2


def test_smoke_run_writes_files_and_summarizes(fake_env, capsys):
    tmp, q = fake_env
    results = tmp / "results"
    rc = run_eval.main(["--strategy", "sliding_window", "--k", "3", "--questions", str(q),
                        "--results-dir", str(results), "--allow-fake"])
    assert rc == 0
    run_dir = _run_dir(results)
    assert run_dir.name.startswith("sliding_window_k3_")
    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert meta["eval_set_size"] == 2 and meta["window_k"] == 3 and meta["llm_backend"] == "fake"
    assert (run_dir / "prices.yaml").exists()

    rows = list(csv.DictReader((run_dir / "requests.csv").open()))
    assert [r["question_id"] for r in rows] == ["a", "b"]
    assert all(r["ok"] == "True" and r["judge_correctness"] == "3" for r in rows)
    assert rows[0]["cited_source"] == "True"
    calls = list(csv.DictReader((run_dir / "calls.csv").open()))
    assert {c["call_type"] for c in calls} == {"llm", "embed", "tool", "judge"}
    assert len((run_dir / "raw.jsonl").read_text().splitlines()) == 2

    # summarize ignores _smoke by default, includes it on request
    assert summarize.main([str(results)]) == 1
    capsys.readouterr()
    assert summarize.main([str(results), "--include-smoke", "--markdown"]) == 0
    out = capsys.readouterr().out
    data_row = out.splitlines()[2]
    assert data_row.startswith(f"| {run_dir.name} | sliding_window_k3 | 2 | 0 |")
    assert "FAKE backend smoke run" in out


def test_resume_skips_completed(fake_env):
    tmp, q = fake_env
    results = tmp / "results"
    assert run_eval.main(["--strategy", "full_history", "--questions", str(q), "--results-dir", str(results),
                          "--allow-fake", "--limit", "1"]) == 0
    run_dir = _run_dir(results)
    assert run_eval.main(["--resume", str(run_dir), "--allow-fake"]) == 0
    rows = list(csv.DictReader((run_dir / "requests.csv").open()))
    # --limit applied to the first run only; meta recorded size 1, resume reads the same file (2 items)
    assert [r["question_id"] for r in rows] == ["a", "b"]


def test_resume_refuses_changed_eval_set(fake_env):
    tmp, q = fake_env
    results = tmp / "results"
    run_eval.main(["--strategy", "full_history", "--questions", str(q), "--results-dir", str(results),
                   "--allow-fake", "--limit", "1"])
    q.write_text(q.read_text() + json.dumps({"id": "c", "question": "x", "reference": "y"}) + "\n")
    assert run_eval.main(["--resume", str(_run_dir(results)), "--allow-fake"]) == 2


def test_summary_numbers_come_from_csv(tmp_path):
    run = tmp_path / "full_history_x"
    run.mkdir()
    (run / "run_meta.json").write_text(json.dumps({"strategy_label": "full_history", "llm_backend": "gemini_api"}))
    fields = ["question_id", "ok", "strategy", "total_latency_ms", "backoff_ms", "total_cost_usd", "cost_complete",
              "n_llm_calls", "input_tokens", "judge_correctness", "judge_groundedness"]
    with (run / "requests.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow(dict(question_id="1", ok="True", strategy="full_history", total_latency_ms="1000", backoff_ms="0",
                        total_cost_usd="0.01", cost_complete="True", n_llm_calls="5", input_tokens="100",
                        judge_correctness="4", judge_groundedness="5"))
        w.writerow(dict(question_id="2", ok="True", strategy="full_history", total_latency_ms="3000", backoff_ms="1000",
                        total_cost_usd="0.03", cost_complete="True", n_llm_calls="7", input_tokens="300",
                        judge_correctness="2", judge_groundedness="3"))
        w.writerow(dict(question_id="3", ok="False", strategy="full_history"))
    s = summarize.summarize_run(run)
    assert (s["n_ok"], s["n_err"]) == (2, 1)
    assert s["lat_mean_ms"] == 2000 and s["lat_p95_ms"] == pytest.approx(2900)
    assert s["lat_excl_mean_ms"] == 1500
    assert s["cost_mean_usd"] == pytest.approx(0.02)
    assert (s["correctness_mean"], s["groundedness_mean"], s["n_judged"]) == (3, 4, 2)


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"correctness": 4, "groundedness": 5, "rationale": "ok"}', (4, 5, None)),
        ('```json\n{"correctness": 1, "groundedness": 1, "rationale": ""}\n```', (1, 1, None)),
        ('{"correctness": 6, "groundedness": 5}', (None, None, "invalid correctness: 6")),
        ('{"correctness": 3.5, "groundedness": 5}', (None, None, "invalid correctness: 3.5")),
        ('{"correctness": true, "groundedness": 5}', (None, None, "invalid correctness: True")),
        ("nope", (None, None, "judge output was not valid JSON")),
    ],
)
def test_parse_judge_scores(text, expected):
    r = parse_scores(text)
    assert (r.correctness, r.groundedness, r.error) == expected
