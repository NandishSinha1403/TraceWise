# TraceWise

A multi-agent RAG assistant (planner → retriever → answer → critic) built with LangGraph, served by FastAPI, using Gemini on Vertex AI. Later milestones add per-call LLM tracing and a state-strategy experiment. See `PROJECT_SPEC.md`.

**Status: M1 (graph runs locally).** No tracing, deployment, eval, or dashboard yet.

> **No results are reported until M4.** Nothing in this repository is a measured cost, latency, or quality number yet.

## Graph

```
planner -> retriever (ReAct, search_documents tool) -> answer -> critic
critic: accept -> END
        retry_planner -> planner      (while critic_retries < MAX_CRITIC_RETRIES)
        retry_retriever -> retriever  (while critic_retries < MAX_CRITIC_RETRIES)
        retry at cap -> END with capped: true and the latest draft answer
```

State is in `app/state.py`. The `strategy` field is fixed to `full_history` in M1. `sliding_window` and `rolling_summary` come in M4 (`app/strategies.py`).
Every model call goes through `app/llm.py` (`generate()` and `embed()`). M2 tracing will wrap those two functions.

## Setup

Requires Python 3.11+. M1 was built and tested on Python 3.12.4.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Direct dependencies are pinned in `pyproject.toml`. `requirements.lock` has the full resolved set from the M1 install (`pip install -r requirements.lock` reproduces it).

## Configuration

All settings are environment variables (see `.env.example`). No model IDs, project IDs, or regions have defaults. The numeric defaults below are **development defaults, not tuned values**.

| Variable | Purpose | Default |
|---|---|---|
| `LLM_BACKEND` | `vertex` or `fake` | `fake` |
| `GCP_PROJECT` | Vertex AI project ID | none (required when `LLM_BACKEND=vertex`) |
| `GCP_REGION` | Vertex AI region | none (required when `LLM_BACKEND=vertex`) |
| `GEN_MODEL_ID` | Gemini model for planner, retriever, answer, critic | none (required when `LLM_BACKEND=vertex`) |
| `JUDGE_MODEL_ID` | Gemini model for the judge (read but unused in M1) | none |
| `EMBEDDING_MODEL_ID` | Embedding model for ingestion and queries | none (required when `LLM_BACKEND=vertex`) |
| `GEN_TEMPERATURE` | Generation temperature | none (model default is used when unset) |
| `MAX_CRITIC_RETRIES` | Critic retry cap (N) | `2` (development default) |
| `MAX_RETRIEVER_STEPS` | Cap on ReAct tool-call steps per retriever run | `4` (development default) |
| `SEARCH_TOP_K` | Passages returned per search | `4` (development default) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Ingestion chunking, in characters | `1000` / `150` (development defaults) |
| `VECTOR_STORE` | `chroma` in M1 | `chroma` |
| `DOCS_DIR` | Source documents folder (`.txt`, `.md`) | `data/docs` |
| `CHROMA_DIR` | Chroma persistence folder | `data/chroma` |
| `FAKE_CRITIC_VERDICT` | Fake backend only: `accept`, `retry_planner`, `retry_retriever` | `accept` |
| `HOST` / `PORT` | Server bind address | `127.0.0.1` / `8000` |

Chroma collections are named after the embedder (`tracewise-fake`, or `tracewise-<EMBEDDING_MODEL_ID>`), so fake and real vectors never mix.

## Offline smoke test (no network, no credentials)

The `fake` backend returns fixed responses for each role. It exists for tests and this smoke run only. The fixture documents in `tests/fixtures/docs/` are synthetic and are never used for results.

```bash
pytest -q

LLM_BACKEND=fake DOCS_DIR=tests/fixtures/docs python -m app.ingest --reset
LLM_BACKEND=fake python -m app.api
# in another shell:
curl -s -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"What colour is the lamp of the Varnholt Lighthouse?"}'
```

Force the critic to reject, with a retry cap of 0 (expect one critic pass and `"capped": true`):

```bash
LLM_BACKEND=fake MAX_CRITIC_RETRIES=0 FAKE_CRITIC_VERDICT=retry_retriever python -m app.api
```

Response fields: `request_id`, `question`, `strategy`, `answer`, `cited_doc_ids`, `critic_retries`, `capped`, `critic_verdict`, `steps` (the node sequence visited).

## Real-model run (Vertex AI)

1. Authenticate: `gcloud auth application-default login`. Enable the Vertex AI API in the project.
2. Put the corpus (`.txt` / `.md`) in `data/docs/`.
3. Set the variables and ingest:
   ```bash
   export LLM_BACKEND=vertex GCP_PROJECT=... GCP_REGION=... GEN_MODEL_ID=... EMBEDDING_MODEL_ID=...
   python -m app.ingest
   python -m app.api
   ```
4. Send the same `curl` request as above, with a question about your corpus.

If a required variable is missing, the service exits with an error that names it.

## Layout

```
app/        config, state, strategies, llm (single model entry point), vectorstore, ingest,
            graph, api, agents/ (planner, retriever, answer, critic), tools/search.py
config/     prices.yaml arrives in M2
eval/       M4
results/    M4 run outputs
tests/      offline tests with the fake backend; fixtures/docs are synthetic
```
