# Instructions for Claude Code

Read `PROJECT_SPEC.md` first. It defines scope, architecture, metrics, the state-strategy experiment, and milestones.

## Ground rules
- Do not invent numbers. Every metric, cost, latency, or judge score in any doc must come from a run whose raw output is saved under `results/`.
- Do not fill in `<FILL: ...>` placeholders with guesses. Ask the user, or leave the placeholder and say so.
- Keep prices in `config/prices.yaml`, not hard-coded. Mark prices with the date they were checked.
- Use Python 3.11+, LangGraph, FastAPI, and the Vertex AI SDK (`google-cloud-aiplatform`). Use OpenTelemetry for spans.
- Pin dependencies in `pyproject.toml` or `requirements.txt`.

## Working order
1. Confirm the milestone you are on (M1 to M5 in `PROJECT_SPEC.md`).
2. Build the smallest working version of that milestone before adding more.
3. Add tests for the tracing math: decode tokens/sec, cost calculation, and rollup totals.
4. After each milestone, update `README.md` with how to run it. Do not claim results in the README until M4.

## Code layout
```
app/
  graph.py          # LangGraph definition (planner, retriever, critic)
  agents/           # node implementations
  state.py          # state schema; strategy switch lives here
  strategies.py     # full_history, sliding_window(k), rolling_summary
  tracing/
    middleware.py   # wraps LLM and tool calls, records per-call metrics
    rollup.py       # per-request totals
    otel.py         # OpenTelemetry setup -> Cloud Trace
  api.py            # FastAPI /ask
config/prices.yaml
eval/
  questions.jsonl   # fixed eval set
  judge_prompt.md   # fixed rubric
  run_eval.py       # runs one strategy over the set, writes results/
results/            # raw logs and CSVs, one folder per run
deploy/
  Dockerfile
  cloudrun.yaml     # or a deploy script
tests/
RESULTS.md          # written only from files in results/
```

## Commands to support
- `python -m app.api` runs the service locally.
- `python -m eval.run_eval --strategy full_history|sliding_window|rolling_summary --k N` runs the eval and writes `results/<strategy>_<timestamp>/`.
- `python -m eval.summarize results/` prints the mean and p95 latency, cost per request, and judge score per strategy.

## Before you say something is done
- Run the tests.
- Run the service locally and send one request.
- For deployment, confirm the Cloud Run URL responds and spans appear in Cloud Trace.
- For results, confirm `RESULTS.md` only cites numbers that `eval.summarize` prints from the saved files.
