# LLM Cost & Latency Tracer for a Multi-Agent RAG Assistant

## Goal
Build a multi-agent RAG assistant on Google Cloud and add a tracing layer that records LLM-native metrics (tokens, tokens/sec, time to first token, cost per request) for every call. Use the tracing data to compare three ways of carrying state between agents: full history, sliding window, and rolling summarization.

## Scope

### In scope
- LangGraph graph with three agents: **planner**, **retriever**, **critic** (self-reflection loop that can send work back to the planner or retriever, capped at N retries).
- ReAct-style tool use for the retriever (search tool calls).
- Vertex AI (Gemini model) for generation and for the LLM judge.
- Vector store: `<FILL: Vertex AI Vector Search | pgvector on Cloud SQL>`.
- Deployment on Cloud Run in `<FILL: region>`.
- Tracing middleware: per-call and per-request metrics, OpenTelemetry spans exported to Cloud Trace.
- State-strategy experiment: `full_history`, `sliding_window(k)`, `rolling_summary`.
- Eval set and LLM-judge scoring.
- A simple dashboard (Cloud Monitoring or Grafana) for cost per request and tokens/sec.

### Out of scope (for now)
- Auth, multi-tenant support, frontend UI beyond a single `/ask` endpoint.
- Fine-tuning or custom models.
- Production-grade autoscaling tuning.

## Architecture

```
client -> Cloud Run (FastAPI /ask)
            -> LangGraph graph
                 planner -> retriever (vector store) -> critic -> (loop or answer)
            -> Tracing middleware (wraps every LLM and tool call)
                 -> OpenTelemetry spans -> Cloud Trace
                 -> metrics row -> SQLite/BigQuery table
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

## Eval set
- Size: `<FILL: number of questions>` questions, fixed before any runs.
- Source: `<FILL: e.g., own documents / public dataset>`.
- Each item: question, reference answer (or key facts), source document id.
- Judge: `<FILL: Gemini model id>` with a fixed rubric (1-5 scale for correctness and groundedness). Store the rubric in `eval/judge_prompt.md`.

## Milestones

1. **M1 - Graph runs locally.** Planner, retriever, critic work end-to-end on a few documents. Single `/ask` endpoint.
2. **M2 - Tracing.** Per-call metrics written for every LLM and tool call; rollups per request; spans visible locally.
3. **M3 - Deploy.** Dockerized service on Cloud Run; vector store populated; Cloud Trace shows spans.
4. **M4 - Experiment.** Eval set built; all three strategies run; results saved as CSV with raw logs.
5. **M5 - Dashboard and write-up.** Dashboard for cost and tokens/sec; short results section in `RESULTS.md` using only measured numbers.

## Definition of done
- Every reported number in `RESULTS.md` comes from a file in `results/` that can be regenerated with one command.
- The eval set size, model ids, region, and vector store are recorded in `RESULTS.md`.
- No placeholder values remain in the README or results.
