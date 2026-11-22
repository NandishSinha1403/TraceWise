# TraceWise

A multi-agent RAG assistant (planner → retriever → answer → critic) built with LangGraph and served by FastAPI. It uses Gemini through the Gemini API free tier. A tracing layer records tokens, time to first token, decode tokens/sec, latency, and cost for every model and tool call. The experiment compares three ways of carrying state between agents: `full_history`, `sliding_window(k)`, and `rolling_summary`.

Everything runs at no cost: no billing account and no free-trial credits. See `PROJECT_SPEC.md`.

> Results are in `RESULTS.md`. It is written only from numbers that `python -m eval.summarize` prints from files in `results/`.

## Status

| Milestone | Code | Verified |
|---|---|---|
| M1 graph + `/ask` | done | tests; real `/ask` requests against `gemini-3.5-flash-lite` |
| M2 tracing | done | tests; spans in Jaeger (`request` → `node.*`, `llm.*`, `tool.*`) and SQLite rows from real runs |
| M3 deploy | Dockerfile, compose, Cloud Run script (source or `IMAGE=` deploy) | fallback: local Docker stack runs and answers; Cloud Run Starter Tier refused the deploy (see Deployment) |
| M4 experiment | runner, judge, summarize | all three strategies run on 30 questions, but histories were too short to separate them; see `RESULTS.md` |
| M5 dashboard | Grafana provisioning + dashboard | dashboard queries return data from `data/metrics.sqlite` |

The model is `gemini-3.5-flash-lite`. `gemini-2.5-flash`, named in the original spec, is deprecated and its free tier allows only 20 requests per day. The free tier for `gemini-3.5-flash-lite` allows 500 generate requests per day, which is about 80 eval questions. `eval/run_all.sh` runs all three strategies and resumes after quota stops.

## Graph

```
planner -> retriever (ReAct, search_documents tool) -> answer -> critic
critic: accept -> END
        retry_planner -> planner      (while critic_retries < MAX_CRITIC_RETRIES)
        retry_retriever -> retriever  (while critic_retries < MAX_CRITIC_RETRIES)
        retry at cap -> END with capped: true and the latest draft answer
```

Every model call goes through `app/llm.py` (`generate()` and `embed()`), which wraps it in the tracing middleware.

## Setup

Requires Python 3.11+. Built and tested on Python 3.12.4.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

Direct dependencies are pinned in `pyproject.toml`. `requirements.lock` is the full runtime set; the Docker image installs from it.

## Configuration

Settings come from environment variables, or from `.env`, which is gitignored and loaded automatically. Values set in the shell win over `.env`. `.env.example` lists the names. **API keys go only in `.env`.**
Numeric defaults are **development defaults, not tuned values**.

| Variable | Purpose | Default |
|---|---|---|
| `LLM_BACKEND` | `gemini_api` (free tier), `vertex` (needs billing), or `fake` (tests) | `fake` |
| `GEMINI_API_KEY` | AI Studio API key | none (required for `gemini_api`) |
| `GEN_MODEL_ID` | Model for planner, retriever, answer, critic, summarizer | none (required) |
| `EMBEDDING_MODEL_ID` | Embedding model | none (required) |
| `JUDGE_MODEL_ID` | Judge model for the eval | none (required for the eval) |
| `GEN_TEMPERATURE` / `JUDGE_TEMPERATURE` | Temperatures | model default when unset |
| `GEN_THINKING_BUDGET` | Gemini 2.5 thinking budget (`0` turns thinking off) | model default when unset |
| `GEN_THINKING_LEVEL` | Gemini 3+ thinking level: `minimal`, `low`, `medium`, `high` (Gemini 3+ rejects a budget of 0; set one or the other) | model default when unset |
| `LLM_MAX_RETRIES` | Retries on 429/500/503 with exponential backoff | `5` |
| `MAX_CRITIC_RETRIES` | Critic retry cap (N) | `2` |
| `MAX_RETRIEVER_STEPS` | ReAct tool-call steps per retriever run | `4` |
| `SEARCH_TOP_K` | Passages per search | `4` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Ingestion chunking, in characters | `1000` / `150` |
| `STATE_STRATEGY` | `full_history`, `sliding_window`, `rolling_summary` (for the API; the eval takes `--strategy`) | `full_history` |
| `WINDOW_K` | Messages kept by `sliding_window` | `4` |
| `SUMMARY_KEEP_RECENT` | Messages kept verbatim by `rolling_summary` | `3` |
| `METRICS_DB` | SQLite file for metrics rows (`none` disables) | `data/metrics.sqlite` (off for `fake`) |
| `PRICES_FILE` | Price table | `config/prices.yaml` |
| `OTEL_TRACES_EXPORTER` | `none`, `console`, or `otlp` (Jaeger) | `none` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP/HTTP traces endpoint | `http://localhost:4318/v1/traces` |
| `LOG_METRICS_JSON` | Also log each call as one JSON line (on in Docker) | `false` |
| `VECTOR_STORE` / `DOCS_DIR` / `CHROMA_DIR` | Store, source docs, Chroma folder | `chroma` / `data/docs` / `data/chroma` |
| `FAKE_CRITIC_VERDICT` | Fake backend only: `accept`, `retry_planner`, `retry_retriever` | `accept` |
| `HOST` / `PORT` | Server bind address | `127.0.0.1` / `8000` |

## Run the service

```bash
python -m app.ingest            # embed data/docs into Chroma (add --reset to rebuild)
python -m app.api
curl -s -X POST localhost:8000/ask -H 'content-type: application/json' -d '{"question":"..."}'
```

The response has `answer`, `cited_doc_ids`, `critic_retries`, `capped`, `critic_verdict`, `steps`, and `usage`, which holds the per-request rollup: total cost, total latency, LLM calls, agent hops, critic retries, tokens, and backoff time.

Offline, with no network:

```bash
LLM_BACKEND=fake DOCS_DIR=tests/fixtures/docs python -m app.ingest --reset
LLM_BACKEND=fake python -m app.api
```

## Tracing (M2)

Each LLM call records: model id, node, strategy, request id, input/output/thinking tokens, TTFT, latency, decode tokens/sec = output tokens / (latency − TTFT), cost, attempts, and backoff.
Tool and embedding calls record latency. Rows go to `METRICS_DB`, with tables `llm_calls` and `requests`, and spans go to OpenTelemetry.

- **Cost:** input × input price + (output + thinking) × output price, from `config/prices.yaml`. It is a paid-tier list-price estimate; actual spend on the free tier is $0. The prices were verified against Google's pricing page on 2026-11-20.
- **Embedding calls:** the Gemini API does not return token counts for them, so their cost is recorded as unknown.
- **Retries and backoff:** rate-limit retries are counted, and the time spent waiting is recorded separately, so latency can be reported with and without it.
- **Decode tokens/sec:** it is only defined when output streams in more than one chunk, and it is noisy for short outputs.

## Local stack: Jaeger + Grafana (M2/M5)

Needs Docker Desktop running.

```bash
docker compose up --build            # app :8000, Jaeger :16686, Grafana :3000
```

Or run only Jaeger and Grafana and keep the app on the host:

```bash
docker compose up jaeger grafana
OTEL_TRACES_EXPORTER=otlp python -m app.api
```

The Grafana dashboard "TraceWise: cost and tokens/sec" reads `data/metrics.sqlite`.

## Deployment (M3)

**The service runs in local Docker. It is not deployed to Cloud Run.** The no-cost Starter Tier (project region `asia-south1`) was tried on 2026-11-21 and 2026-11-22, and refused both ways:

- prebuilt image: `Deploying container images is not supported for this project.`
- source upload without a build (`gcloud beta run deploy --no-build`): `Deployments using end-user credentials are not allowed for this account.`

Cloud Build and Artifact Registry are not enabled there either. Starter Tier accepts apps published from AI Studio. `deploy/cloudrun_deploy.sh` is kept for a standard project, which needs billing and is out of scope.

Image only: `docker build -f deploy/Dockerfile -t tracewise .`

## Experiment (M4)

```bash
python -m eval.build_questions --n 30     # drafts eval/questions.draft.jsonl from data/docs
# review it by hand, then save as eval/questions.jsonl, and don't change it afterwards
python -m eval.run_eval --strategy full_history
python -m eval.run_eval --strategy sliding_window --k 4
python -m eval.run_eval --strategy rolling_summary
python -m eval.summarize results/                # add --markdown for a table
```

Each run writes `results/<strategy>_<timestamp>/` with `run_meta.json`, `prices.yaml`, `requests.csv`, `calls.csv`, and `raw.jsonl`.
If the free-tier daily quota runs out, the run stops after 3 consecutive errors. Continue later with `python -m eval.run_eval --resume <run folder>`. Use `--pace <seconds>` to slow down between questions.
The judge uses the fixed rubric in `eval/judge_prompt.md`. Judge calls are logged but excluded from per-request cost.

## Layout

```
app/          config, state, strategies, llm (single model entry point), vectorstore, ingest, graph, api
app/agents/   planner, retriever (ReAct loop), answer, critic
app/tracing/  metrics (pure math), middleware, prices, store (SQLite), otel
config/       prices.yaml
eval/         build_questions, run_eval, judge (+ judge_prompt.md), summarize
deploy/       Dockerfile, cloudrun_deploy.sh, grafana/
results/      eval run outputs
tests/        offline tests (fake backend); tests/fixtures/docs are synthetic
```
