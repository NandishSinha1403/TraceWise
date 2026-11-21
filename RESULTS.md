# Results: state-strategy experiment (M4)

Every number below comes from `python -m eval.summarize results/`. The output is saved in `results/summary.md`, and running the command again on `results/` regenerates it. Raw logs per run: `results/<run>/{run_meta.json,prices.yaml,requests.csv,calls.csv,raw.jsonl}`.

## Setup

| | |
|---|---|
| Eval set | 30 questions, `eval/questions.jsonl`, sha256 `b34a408b…f06f0da` (same for all three runs) |
| Corpus | `data/docs/`: 21 Markdown files about a fictional company, 43 chunks (1000 chars, 150 overlap) |
| Generation model | `gemini-3.5-flash-lite`, temperature 0, thinking level `minimal` |
| Judge model | `gemini-3.5-flash-lite`, temperature 0, rubric in `eval/judge_prompt.md` |
| Embedding model | `gemini-embedding-001` |
| Vector store | Chroma (local), top-k 4 |
| Graph limits | critic retries ≤ 2, retriever tool steps ≤ 4 |
| Strategies | `full_history`; `sliding_window` k=4; `rolling_summary` (keeps last 3 verbatim) |
| Backend | Gemini API **free tier**; runs on 2026-11-20 and 2026-11-21 (UTC) |
| Region / deployment | Runs were local; see the README for deployment status |

## Results

| strategy | n ok | n err | latency mean ms | latency p95 ms | latency excl. backoff mean ms | latency excl. backoff p95 ms | cost/request USD | LLM calls/request | input tokens/request | judge correctness | judge groundedness | n judged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full_history | 30 | 0 | 17947.7 | 35078.3 | 15117.5 | 31306.2 | 0.002660 | 5.10 | 6930.2 | 4.97 | 5.00 | 30 |
| sliding_window_k4 | 30 | 0 | 22318.5 | 52905.3 | 18260.9 | 52905.3 | 0.002550 | 5.10 | 6586.6 | 4.97 | 5.00 | 30 |
| rolling_summary | 30 | 6 | 14495.6 | 26931.6 | 11341.0 | 17130.4 | 0.002628 | 5.03 | 6768.7 | 4.97 | 5.00 | 29 |

`decode tok/s mean` is printed by `eval.summarize` (13974.2, 9141.3, 6306.3), but it is **not a meaningful throughput figure** in these runs. Short answers arrive in one or two stream chunks, so latency − TTFT is close to zero and the ratio is inflated. Read it as a sign that the decode-rate metric needs longer outputs, not as model speed.

## What the numbers show

- **Answer quality is the same across strategies.** Mean judge correctness is 4.97 and groundedness is 5.00 for all three. On this eval set, the state strategy made no measurable difference to quality.
- **Token use and cost differ only slightly.** Input tokens per request range from 6586.6 (sliding_window) to 6930.2 (full_history), and cost per request from $0.002550 to $0.002660. Each question averaged about 5 LLM calls (5.03 to 5.10), so the critic rarely sent work back. With a history that short, the window and the summary seldom have anything to trim. **This experiment does not stress the strategies.** Longer multi-turn sessions or harder questions that trigger critic retries would be needed to separate them.
- **Latency differences are not attributable to the strategy.** All runs went through shared free-tier capacity, with 429/503 retries and variable queueing. Even with backoff excluded, the differences (mean 11.3 s to 18.3 s) are within what that noise can produce over 30 questions and one run each.

## Caveats

- **Costs are list-price estimates.** They use the paid-tier prices in `config/prices.yaml`, verified on Google's pricing page on 2026-11-20. Actual spend was $0 (free tier). Embedding cost is not included, because the API returns no token counts for embeddings.
- **Self-preference bias is possible.** The judge is the same model as the generator. With scores at the ceiling (4.97/5.00), the judge also doesn't separate the strategies.
- **The generator changed from the original spec.** The spec named `gemini-2.5-flash`. It was deprecated (requests reroute to `gemini-3.5-flash-lite` from 2026-12-01), and its free tier allowed 20 requests per day. All three runs used `gemini-3.5-flash-lite`.
- **rolling_summary spanned two quota days.** It stopped at the free-tier limit of 500 requests per day and was resumed with identical settings. Its 6 error rows are those quota failures; each of those questions was later re-run successfully. The judge call for q010 in that run also hit the quota (429), so it has 29 judged answers.
- **The corpus is fictional and small,** and the questions are mostly single-fact lookups (4 of 30 are hand-written multi-step questions). Each strategy ran once, so there are no confidence intervals.
- **Code version:** full_history started from the working tree just before commit `1fcc62b`. The other two runs record `1fcc62b`. No code changed between the runs.
