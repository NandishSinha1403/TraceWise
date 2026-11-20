# TraceWise service image.
# Build from the repo root:  docker build -f deploy/Dockerfile -t tracewise .
# The vector store is baked in: run `python -m app.ingest` locally first so data/chroma exists.
FROM python:3.12.4-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv

COPY requirements.lock .
RUN pip install -r requirements.lock

COPY pyproject.toml .
COPY app ./app
COPY config ./config
RUN pip install --no-deps .

COPY data/chroma ./data/chroma

RUN useradd --create-home --uid 10001 tracewise && chown -R tracewise /srv
USER tracewise

# Cloud Run sets PORT. Metrics go to a local SQLite file (ephemeral on Cloud Run)
# and, with LOG_METRICS_JSON, to stdout as one JSON line per call.
ENV HOST=0.0.0.0 \
    PORT=8080 \
    LOG_METRICS_JSON=true \
    METRICS_DB=/tmp/metrics.sqlite

EXPOSE 8080
CMD ["python", "-m", "app.api"]
