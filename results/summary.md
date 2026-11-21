| run | strategy | n ok | n err | latency mean ms | latency p95 ms | latency excl. backoff mean ms | latency excl. backoff p95 ms | cost/request USD | LLM calls/request | input tokens/request | decode tok/s mean | judge correctness | judge groundedness | n judged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full_history_20261120T180152Z | full_history | 30 | 0 | 17947.7 | 35078.3 | 15117.5 | 31306.2 | 0.002660 | 5.10 | 6930.2 | 13974.2 | 4.97 | 5.00 | 30 |
| rolling_summary_20261120T183014Z | rolling_summary | 30 | 6 | 14495.6 | 26931.6 | 11341.0 | 17130.4 | 0.002628 | 5.03 | 6768.7 | 6306.3 | 4.97 | 5.00 | 29 |
| sliding_window_k4_20261120T181430Z | sliding_window_k4 | 30 | 0 | 22318.5 | 52905.3 | 18260.9 | 52905.3 | 0.002550 | 5.10 | 6586.6 | 9141.3 | 4.97 | 5.00 | 30 |

note: cost = paid-tier list-price estimate; runs used the Gemini API free tier (actual spend $0); embedding cost not measured
