# Status (updated 2026-11-21)

Done by Claude:
- Steps 1–6 and 8: documents, free-tier limits, verified prices, eval set, all three runs, `RESULTS.md`, the local Jaeger + Grafana stack (checked), and pushes to GitHub.
- Free-tier limits found: `gemini-2.5-flash` allows 20 requests per day and is deprecated, so the project switched to `gemini-3.5-flash-lite`, which allows 500 requests per day.

Step 7 (Cloud Run): tried and refused by the Starter Tier. M3 falls back to local Docker; the exact errors are in the README.

**Left for you:**
- Read `eval/questions.jsonl`. The spec asks for a review by hand; so far only Claude has reviewed it. Changing it means re-running all three strategies.
- Decide whether to redo the experiment with harder or multi-turn questions (see `RESULTS.md`).

The original task list follows, for reference.

---

# Your tasks

The code for M1 to M5 is written and tested offline (58 tests). These are the steps only you can do. They are in order, and every step is free: never enable billing or start a free trial. If a step asks for something that needs billing, stop and tell me.

Run every command from the project folder with the virtualenv active:

```bash
cd ~/Documents/TraceWise
source .venv/bin/activate
```

**When to send me something:** after each step, if anything fails, paste the error to me. Never paste your API key.

---

## 1. Add your documents — DONE 2026-11-20

Claude wrote a fictional corpus (Brackenfield Parcel Co., 21 files in `data/docs/`), built the index (43 chunks), and answered one real `/ask` request. The content is invented on purpose, so Gemini can't answer from what it already knows. Skim the files if you want; edit them only before step 4.


1. Put `.txt` or `.md` files in `data/docs/`. Subfolders are fine.
   - **Format:** only `.txt` and `.md` are read. Convert PDFs or Word files to text first.
   - **How much:** 10 to 50 files with real content is enough for 30 questions. Pick one topic you know, such as course notes, a public handbook, or Wikipedia articles you saved.
   - **Nothing private:** the free tier's terms let Google use what you send. Check the terms in AI Studio.
2. Build the index:
   ```bash
   python -m app.ingest --reset
   ```
   It should print `ingested N chunks ...`.
3. Ask one question about your documents:
   ```bash
   python -m app.api
   # in a second terminal:
   curl -s -X POST localhost:8000/ask -H 'content-type: application/json' -d '{"question":"<a question your docs answer>"}'
   ```

## 2. Check your free-tier limits and the model (5 minutes)

1. Open https://aistudio.google.com and go to the usage and rate-limit page for your project.
2. Write down the limits for `gemini-2.5-flash` and `gemini-embedding-001`: requests per minute (RPM), tokens per minute (TPM), and requests per day (RPD). **Send me these numbers.** They decide how many days the experiment takes. One question uses about 6 to 15 model calls plus 1 judge call.
3. Check that `gemini-2.5-flash` is not marked deprecated. One third-party page claimed a deprecation date of 2026-10-16, but the model still worked on 2026-11-19. If it is deprecated, tell me before step 4, so we pick a stable replacement before any runs.

## 3. Verify the prices (5 minutes)

1. Open https://ai.google.dev/gemini-api/docs/pricing.
2. Compare the paid-tier prices for `gemini-2.5-flash` (input, and output including thinking) and `gemini-embedding-001` with `config/prices.yaml`. The current values came from third-party sites.
3. Fix any value that differs. Then set `verified: true` and `checked: <today's date>` for each model.

`eval.summarize` warns while any price is unverified.

## 4. Build the eval set (about 30 minutes, mostly reading)

1. Draft 30 questions from your documents:
   ```bash
   python -m eval.build_questions --n 30
   ```
2. Open `eval/questions.draft.jsonl` and read every line. Fix or delete questions that are vague, wrong, or answerable without your documents. Make sure each `reference` lists the key facts.
3. Save the final file as `eval/questions.jsonl`. **Do not edit it after the first run.** The runner records its hash, so all three strategies are scored on exactly the same set.

## 5. Run the experiment (spread over one or more days)

Run the three strategies one after another:

```bash
python -m eval.run_eval --strategy full_history --pace 5
python -m eval.run_eval --strategy sliding_window --k 4 --pace 5
python -m eval.run_eval --strategy rolling_summary --pace 5
```

- **If a run stops** with "stopping after 3 consecutive errors", you most likely hit the daily quota. Continue the next day with the command it prints:
  ```bash
  python -m eval.run_eval --resume results/<that run folder>
  ```
- **If you see many 429 warnings,** raise `--pace` to 10 or 20.
- **Do not change `.env` settings between the three runs.** That means the model, temperature, thinking budget, top-k and chunking. The comparison is only fair if they match.

Then print the summary:

```bash
python -m eval.summarize results/
```

**Send me the output.** I'll write `RESULTS.md` using only those numbers.

## 6. Run the local Jaeger + Grafana stack (15 minutes)

1. Start Docker Desktop and wait until it says it is running.
2. Start the stack. Step 1 must be done first, because the image copies `data/chroma`.
   ```bash
   docker compose up --build
   ```
3. Send one question to http://localhost:8000/ask, using the same `curl` as in step 1.
4. Check the traces. In Jaeger (http://localhost:16686), pick service `tracewise` and click **Find Traces**. You should see a `request` span with `node.*`, `llm.*`, and `tool.*` spans under it.
5. Check the dashboard. In Grafana (http://localhost:3000), open Dashboards → TraceWise → "TraceWise: cost and tokens/sec". The panels should show data once some requests or eval runs exist.

Tell me whether the traces and dashboard showed up, or paste what went wrong. I couldn't test this part, because Docker wasn't running.

## 7. Deploy to Cloud Run on the Starter Tier (about 30 minutes)

1. Set up the Google Cloud **Starter Tier** with your Google account: https://docs.cloud.google.com/docs/starter-tier. Don't add a payment method. **Write down the region it gives you.**
2. Install and log in to the Google Cloud CLI:
   ```bash
   brew install --cask google-cloud-sdk
   gcloud auth login
   gcloud projects list        # find your Starter Tier project ID
   ```
3. Deploy:
   ```bash
   GCP_PROJECT=<project id> GCP_REGION=<region> deploy/cloudrun_deploy.sh
   curl -s <service URL>/healthz
   ```
4. **If it fails** with an error about billing, Cloud Build, or Artifact Registry, don't enable anything. Send me the error. The Starter Tier may not allow building from source, and we have a free fallback: build the image locally and deploy it another way, or keep M3 as local Docker.

**Send me the region and the service URL** so I can fill the last spec placeholder.

## 8. Push to GitHub

Say **"push"** and I'll commit and push everything. `.env` and your metrics database are gitignored and won't be uploaded.

---

## Decisions I made (tell me if you want any changed)

| Decision | Choice | Why |
|---|---|---|
| Judge model | `gemini-2.5-flash`, temperature 0 | Known to be on the free tier. Same model as generation, so self-preference bias is possible; this will be stated in `RESULTS.md`. |
| Generation settings | temperature 0, thinking budget 0 | Repeatable runs, much less free quota used. Thinking cost 303 extra tokens on a 4-word reply in a test. |
| Eval set | 30 questions, drafted by Gemini from your docs, reviewed by you | Fits the free quota over a few days. |
| Strategy parameters | `sliding_window` k=4; `rolling_summary` keeps the last 3 messages and summarizes once more than 6 build up | Development defaults, not tuned. |
| Tracing backend | Jaeger and Grafana in local Docker, metrics in SQLite | Free. Cloud Trace and Cloud Monitoring need billing. |
| Vector store | Chroma, baked into the Docker image | Free, no extra service. |
| Cost metric | Paid-tier list-price estimate | Actual spend is $0. Embedding cost is not measured, because the API doesn't return token counts for embeddings. |
| Removed dependency | `google-cloud-aiplatform` | Nothing imported it. `google-genai` covers both the Gemini API and Vertex AI. |
