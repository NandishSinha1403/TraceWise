#!/usr/bin/env bash
# Deploy TraceWise to Cloud Run. NOT usable on a Starter Tier project: see the
# README (tried 2026-11-21/22, refused). Needs a standard project with billing.
#
#   GCP_PROJECT=my-project GCP_REGION=<region> deploy/cloudrun_deploy.sh
#
# To deploy a prebuilt public image instead of building from source:
#
#   IMAGE=docker.io/<user>/tracewise:<tag> GCP_PROJECT=... GCP_REGION=... deploy/cloudrun_deploy.sh
#
# Before running:
#   - gcloud is installed and logged in (gcloud auth login)
#   - python -m app.ingest has populated data/chroma (it is baked into the image)
#   - .env has GEMINI_API_KEY, GEN_MODEL_ID, EMBEDDING_MODEL_ID
#
# Only the files the image needs are staged and uploaded; .env is never uploaded.
# The API key is passed as a Cloud Run environment variable via a temporary
# file (not on the command line). Anyone with access to the project can read it.
set -euo pipefail

: "${GCP_PROJECT:?set GCP_PROJECT}"
: "${GCP_REGION:?set GCP_REGION}"
SERVICE="${SERVICE:-tracewise}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

[[ -d "$ROOT/data/chroma" ]] || { echo "data/chroma missing: run python -m app.ingest first" >&2; exit 1; }
[[ -f "$ROOT/.env" ]] || { echo ".env missing" >&2; exit 1; }

env_value() { grep -E "^$1=" "$ROOT/.env" | tail -1 | cut -d= -f2-; }
for v in GEMINI_API_KEY GEN_MODEL_ID EMBEDDING_MODEL_ID; do
  [[ -n "$(env_value "$v")" ]] || { echo "$v is empty in .env" >&2; exit 1; }
done

STAGE="$(mktemp -d)"
ENVFILE="$(mktemp)"
trap 'rm -rf "$STAGE" "$ENVFILE"' EXIT
chmod 600 "$ENVFILE"

if [[ -n "${IMAGE:-}" ]]; then
  DEPLOY_FROM=(--image "$IMAGE")
else
  DEPLOY_FROM=(--source "$STAGE")
fi

cp "$ROOT/deploy/Dockerfile" "$STAGE/Dockerfile"
cp "$ROOT/requirements.lock" "$ROOT/pyproject.toml" "$STAGE/"
cp -R "$ROOT/app" "$ROOT/config" "$STAGE/"
mkdir -p "$STAGE/data" && cp -R "$ROOT/data/chroma" "$STAGE/data/chroma"
find "$STAGE" -name __pycache__ -type d -prune -exec rm -rf {} +

{
  echo "LLM_BACKEND: gemini_api"
  for v in GEMINI_API_KEY GEN_MODEL_ID EMBEDDING_MODEL_ID GEN_TEMPERATURE GEN_THINKING_BUDGET GEN_THINKING_LEVEL; do
    val="$(env_value "$v" || true)"
    [[ -n "$val" ]] && printf '%s: "%s"\n' "$v" "$val"
  done
} > "$ENVFILE"

gcloud run deploy "$SERVICE" \
  --project "$GCP_PROJECT" \
  --region "$GCP_REGION" \
  "${DEPLOY_FROM[@]}" \
  --env-vars-file "$ENVFILE" \
  --max-instances 1 \
  --memory 1Gi \
  --allow-unauthenticated

URL="$(gcloud run services describe "$SERVICE" --project "$GCP_PROJECT" --region "$GCP_REGION" --format 'value(status.url)')"
echo "Service URL: $URL"
echo "Smoke test:  curl -s $URL/healthz"
