# Results: state-strategy experiment (M4)

Every number below comes from `python -m eval.summarize results/`. The output is saved in `results/summary.md`, and running the command again on `results/` regenerates it. Raw logs per run: `results/<run>/{run_meta.json,prices.yaml,requests.csv,calls.csv,raw.jsonl}`.

## Setup

| | |
|---|---|
| Eval set | 30 questions, `eval/questions.jsonl`, sha256 `b34a408bc86833a8d3cb66e81722d75409c39c532623fb6fbf6f71869d06f0da` (same for all three runs) |
| Corpus | `data/docs/`: 21 Markdown files about a fictional company, 43 chunks (1000 chars, 150 overlap) |
| Generation model | `gemini-3.5-flash-lite`, temperature 0, thinking level `minimal` |
| Judge model | `gemini-3.5-flash-lite`, temperature 0, rubric in `eval/judge_prompt.md` |
| Embedding model | `gemini-embedding-001` |
| Vector store | Chroma (local), top-k 4 |
| Graph limits | critic retries ≤ 2, retriever tool steps ≤ 4 |
| Strategies | `full_history`; `sliding_window` k=4; `rolling_summary` (keeps last 3 verbatim) |
| Backend | Gemini API **free tier**; runs on 2026-11-20 and 2026-11-21 (UTC) |
| Region / deployment | Not deployed; all runs were local. Starter Tier region would be `asia-south1`; it refused the deploy (see README) |

## Conclusion first

**The experiment ran correctly but did not test the strategies.** The critic never asked for a retry and the summarizer was never called, so every request was one short pass. `rolling_summary` never reached its trigger (more than 6 unsummarized messages) and therefore sent the same context as `full_history`. These results cannot say which strategy is better, and the differences in the table must not be read as strategy effects.

## Results

| strategy | n ok | n err | latency mean ms | latency p95 ms | latency excl. backoff mean ms | latency excl. backoff p95 ms | cost/request USD | LLM calls/request | critic retries/request | tool calls/request | summarizer calls (total) | input tokens/request | decode tok/s mean | judge correctness | judge groundedness | n judged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full_history | 30 | 0 | 17947.7 | 35078.3 | 15117.5 | 31306.2 | 0.002660 | 5.10 | 0.00 | 1.33 | 0 | 6930.2 | 13974.2 | 4.97 | 5.00 | 30 |
| sliding_window_k4 | 30 | 0 | 22318.5 | 52905.3 | 18260.9 | 52905.3 | 0.002550 | 5.10 | 0.00 | 1.23 | 0 | 6586.6 | 9141.3 | 4.97 | 5.00 | 30 |
| rolling_summary | 30 | 6 | 14495.6 | 26931.6 | 11341.0 | 17130.4 | 0.002628 | 5.03 | 0.00 | 1.30 | 0 | 6768.7 | 6306.3 | 4.97 | 5.00 | 29 |

## What the numbers show

- **`full_history` against `rolling_summary` is a noise check.** They did the same thing, so the gaps between those two rows (input tokens, cost, and especially latency) show how much varies from run to run.
- **`sliding_window_k4` may have trimmed a little.** Its input tokens are lowest, but it also made the fewest tool calls (1.23 per request), and each search adds passages to the prompt. These runs cannot separate the two causes.
- **Quality is at the ceiling for all three.** Correctness is 4.97 and groundedness 5.00, so the questions are too easy to show a quality difference.
- **`decode tok/s mean` is not usable.** Short answers arrive almost entirely with the first chunk, so latency minus TTFT is close to zero and the ratio explodes. Do not cite it as model speed.

## Caveats

- **Latency was not measured under equal conditions.** The runs were sequential, at different times, on shared free-tier capacity. Other work also ran on the same machine: a Docker build and one extra `/ask` request (sharing the API quota) during `full_history`, and an emulated amd64 Docker build during the second part of `rolling_summary`.
- **Costs are list-price estimates** from `config/prices.yaml`, checked against Google's pricing page on 2026-11-20. The runs used the Gemini API free tier (actual spend $0). Embedding cost is not included, because the API returns no token counts for embeddings.
- **Self-preference bias is possible.** The judge is the same model as the generator.
- **The eval set was written and reviewed by Claude, not by a person.** Claude also wrote the corpus. Gemini drafted the questions; Claude edited wording and references and replaced four with multi-step questions. The spec calls for review by hand, and that has not happened.
- **The corpus is small and fictional,** most questions need one document, and each strategy ran once, so there are no confidence intervals.
- **The model differs from the original spec.** `gemini-2.5-flash` is deprecated (requests reroute to `gemini-3.5-flash-lite` from 2026-12-01) and its free tier returned a limit of 20 requests per day.
- **`rolling_summary` spanned two quota days.** It stopped at the 500 requests/day limit and was resumed with the same settings. Its 6 error rows are those quota failures, and each question was later answered. The judge call for q010 also hit the quota, so 29 answers were judged.
- **Code version.** `full_history` records commit `130d288` but ran from the uncommitted tree that became `1fcc62b`; the other two record `1fcc62b`.
- **Driver log.** `results/run_all.log` from the first day was overwritten when the run was resumed. The per-run files, including the quota errors, are intact.

## To get a real comparison

Use questions that make the critic retry, or multi-turn sessions, so the transcript passes the window and summary thresholds. Randomize the strategy order, repeat each run, and keep the machine and quota otherwise idle.
