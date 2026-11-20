# LLM Cost & Latency Tracer for a Multi-Agent RAG Assistant

## Cost constraint
The whole project runs at **no cost**: no paid usage and no Google Cloud free-trial credits unless that becomes essential (decided 2026-11-19). Every service must work without a Cloud Billing account. Cost figures in results are therefore *what the calls would cost at paid list prices* (from `config/prices.yaml`), not money actually spent.

## Goal
Build a multi-agent RAG assistant on Google Cloud and add a tracing layer that records LLM-native metrics (tokens, tokens/sec, time to first token, cost per request) for every call. Use the tracing data to compare three ways of carrying state between agents: full history, sliding window, and rolling summarization.

## Scope

### In scope
- LangGraph graph with three agents: **planner**, **retriever**, **critic** (self-reflection loop that can send work back to the planner or retriever, capped at N retries).
- ReAct-style tool use for the retriever (search tool calls).
- Gemini API free tier (AI Studio API key on a Google Cloud project, no billing) for generation, embeddings, and the LLM judge. Generation model: `gemini-3.5-flash-lite` (switched from `gemini-2.5-flash` on 2026-11-20: 2.5-flash is deprecated, reroutes to 3.5-flash-lite from 2026-12-01, and its free tier allows only 20 requests per day). Embedding model: `gemini-embedding-001`. A Vertex AI backend is kept in code for a later paid switch.
- Vector store: Chroma (local, persisted to disk; bundled with the service when deployed).
- Deployment on Cloud Run via the Google Cloud Starter Tier (no billing account) in `<FILL: region, fixed when the Starter Tier project is created>`.
- Tracing middleware: per-call and per-request metrics, OpenTelemetry spans exported to a local Jaeger (Docker). Cloud Trace is out unless billing is enabled later.
- State-strategy experiment: `full_history`, `sliding_window(k)`, `rolling_summary`.
- Eval set and LLM-judge scoring.
- A simple dashboard (Grafana in local Docker) for cost per request and tokens/sec.

### Out of scope (for now)
- Auth, multi-tenant support, frontend UI beyond a single `/ask` endpoint.
- Fine-tuning or custom models.
- Production-grade autoscaling tuning.

## Architecture

```
client -> Cloud Run, Starter Tier (FastAPI /ask)   [or local: python -m app.api]
            -> LangGraph graph
                 planner -> retriever (Chroma) -> answer -> critic -> (loop or end)
            -> Tracing middleware (wraps every LLM and tool call)
                 -> OpenTelemetry spans -> Jaeger (local Docker)
                 -> metrics row -> SQLite table
            -> Gemini API free tier (generation + embeddings)
```

## Metrics to record

Per LLM call:
- model id, node name, strategy, request id
- input tokens, output tokens
- time to first token (ms), total latency (ms), decode tokens/sec = output tokens / (total - TTFT)
- cost (USD) = input tokens x input price + output tokens x output price (price table in `config/prices.yaml`)

Per request (rollup):
- total cost, total latency, number of LLM calls, number of agent hops, number of critic-triggered retries

## State strategies (experiment)

| Strategy | How state is passed |
|---|---|
| `full_history` | All prior messages and tool outputs are included each call |
| `sliding_window(k)` | Only the last k messages are included |
| `rolling_summary` | A running summary replaces older messages; last few messages kept verbatim |

Run all three on the same eval set with the same model, temperature, and retrieval settings.

Free-tier rate limits apply. Waiting on a rate limit (HTTP 429 retries) inflates wall-clock latency, so the tracer records retries and backoff time per call, and latency comparisons exclude or report them separately.

## Eval set
- Size: 30 questions, fixed before any runs (decided 2026-11-19; drafted with `python -m eval.build_questions`, then reviewed by hand).
- Source: the user's own documents in `data/docs/` (no private content: free-tier terms apply).
- Each item: question, reference answer (or key facts), source document id.
- Size must fit within the free-tier daily request quota (check the limits shown in AI Studio for the chosen models).
- Judge: `gemini-3.5-flash-lite` at temperature 0 (same model as generation, so self-preference bias is possible; noted in `RESULTS.md`) with a fixed rubric (1-5 scale for correctness and groundedness). Store the rubric in `eval/judge_prompt.md`.

## Milestones

1. **M1 - Graph runs locally.** Planner, retriever, critic work end-to-end on a few documents. Single `/ask` endpoint.
2. **M2 - Tracing.** Per-call metrics written for every LLM and tool call; rollups per request; spans visible locally.
3. **M3 - Deploy.** Dockerized service on Cloud Run (Starter Tier); vector store populated; spans visible in Jaeger for local runs, and metrics rows recorded for deployed runs.
4. **M4 - Experiment.** Eval set built; all three strategies run; results saved as CSV with raw logs.
5. **M5 - Dashboard and write-up.** Grafana dashboard for cost and tokens/sec; short results section in `RESULTS.md` using only measured numbers.

## Definition of done
- Every reported number in `RESULTS.md` comes from a file in `results/` that can be regenerated with one command.
- The eval set size, model ids, region, and vector store are recorded in `RESULTS.md`.
- `RESULTS.md` states that costs are list-price estimates and that runs used the Gemini API free tier.
- No placeholder values remain in the README or results.
