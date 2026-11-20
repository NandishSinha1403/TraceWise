"""Environment-driven configuration.

No model IDs, project IDs, or regions are hard-coded. Values that are needed
for a real run must come from the environment (or `.env`); if they are
missing, `load_settings()` raises `ConfigError` naming the variable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

VALID_BACKENDS = ("gemini_api", "vertex", "fake")
VALID_VECTOR_STORES = ("chroma",)
VALID_FAKE_VERDICTS = ("accept", "retry_planner", "retry_retriever")
VALID_STRATEGIES = ("full_history", "sliding_window", "rolling_summary")
VALID_THINKING_LEVELS = ("minimal", "low", "medium", "high")


class ConfigError(RuntimeError):
    """Raised when a required environment variable is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    llm_backend: str
    gemini_api_key: str | None
    gcp_project: str | None
    gcp_region: str | None
    gen_model_id: str | None
    judge_model_id: str | None
    embedding_model_id: str | None
    gen_temperature: float | None
    judge_temperature: float | None
    gen_thinking_budget: int | None
    gen_thinking_level: str | None
    llm_max_retries: int
    max_critic_retries: int
    max_retriever_steps: int
    search_top_k: int
    chunk_size: int
    chunk_overlap: int
    vector_store: str
    docs_dir: str
    chroma_dir: str
    fake_critic_verdict: str
    # State strategy (M4)
    strategy: str
    window_k: int
    summary_keep_recent: int
    # Tracing (M2)
    metrics_db: str | None
    prices_file: str
    otel_exporter: str
    otel_endpoint: str | None
    service_name: str
    log_metrics_json: bool

    @property
    def collection_name(self) -> str:
        """Chroma collection, keyed by embedder so fake and real vectors never mix."""
        if self.llm_backend == "fake":
            return "tracewise-fake"
        safe = "".join(c if c.isalnum() or c in "._-" else "-" for c in self.embedding_model_id or "")
        return f"tracewise-{safe}".strip("-._")

    def with_strategy(self, strategy: str, k: int | None = None) -> "Settings":
        if strategy not in VALID_STRATEGIES:
            raise ConfigError(f"strategy must be one of {VALID_STRATEGIES}, got {strategy!r}")
        return replace(self, strategy=strategy, window_k=k if k is not None else self.window_k)

    def __repr__(self) -> str:  # never print the API key
        key = "<set>" if self.gemini_api_key else None
        return (
            f"Settings(llm_backend={self.llm_backend!r}, gemini_api_key={key}, "
            f"gen_model_id={self.gen_model_id!r}, strategy={self.strategy!r})"
        )

    __str__ = __repr__


def _get(env: dict[str, str], name: str) -> str | None:
    value = env.get(name)
    if value is None or value.strip() == "":
        return None
    return value.strip()


def _get_int(env: dict[str, str], name: str, default: int | None, minimum: int) -> int | None:
    raw = _get(env, name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    return value


def _get_float(env: dict[str, str], name: str) -> float | None:
    raw = _get(env, name)
    try:
        return float(raw) if raw is not None else None
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def load_dotenv(path: str = ".env") -> dict[str, str]:
    """Minimal .env reader: KEY=VALUE lines, '#' comments, optional quotes."""
    values: dict[str, str] = {}
    if not os.path.isfile(path):
        return values
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip().removeprefix("export ").strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            values[key] = value
    return values


def load_settings(env: dict[str, str] | None = None) -> Settings:
    """Read settings from `env`, or from `.env` overlaid by the process environment."""
    if env is None:
        env = {**load_dotenv(), **{k: v for k, v in os.environ.items() if v != ""}}
    env = dict(env)

    backend = (_get(env, "LLM_BACKEND") or "fake").lower()
    if backend not in VALID_BACKENDS:
        raise ConfigError(f"LLM_BACKEND must be one of {VALID_BACKENDS}, got {backend!r}")

    vector_store = (_get(env, "VECTOR_STORE") or "chroma").lower()
    if vector_store not in VALID_VECTOR_STORES:
        raise ConfigError(f"VECTOR_STORE must be one of {VALID_VECTOR_STORES}, got {vector_store!r}")

    fake_verdict = (_get(env, "FAKE_CRITIC_VERDICT") or "accept").lower()
    if fake_verdict not in VALID_FAKE_VERDICTS:
        raise ConfigError(f"FAKE_CRITIC_VERDICT must be one of {VALID_FAKE_VERDICTS}, got {fake_verdict!r}")

    strategy = (_get(env, "STATE_STRATEGY") or "full_history").lower()
    if strategy not in VALID_STRATEGIES:
        raise ConfigError(f"STATE_STRATEGY must be one of {VALID_STRATEGIES}, got {strategy!r}")

    otel_exporter = (_get(env, "OTEL_TRACES_EXPORTER") or "none").lower()
    if otel_exporter not in ("none", "console", "otlp"):
        raise ConfigError(f"OTEL_TRACES_EXPORTER must be none, console, or otlp, got {otel_exporter!r}")

    thinking_level = _get(env, "GEN_THINKING_LEVEL")
    if thinking_level is not None:
        thinking_level = thinking_level.lower()
        if thinking_level not in VALID_THINKING_LEVELS:
            raise ConfigError(f"GEN_THINKING_LEVEL must be one of {VALID_THINKING_LEVELS}, got {thinking_level!r}")
        if _get(env, "GEN_THINKING_BUDGET") is not None:
            raise ConfigError("set GEN_THINKING_BUDGET (Gemini 2.5) or GEN_THINKING_LEVEL (Gemini 3+), not both")

    metrics_db = _get(env, "METRICS_DB")
    if metrics_db is None and backend != "fake":
        metrics_db = "data/metrics.sqlite"
    if metrics_db is not None and metrics_db.lower() == "none":
        metrics_db = None

    settings = Settings(
        llm_backend=backend,
        gemini_api_key=_get(env, "GEMINI_API_KEY"),
        gcp_project=_get(env, "GCP_PROJECT"),
        gcp_region=_get(env, "GCP_REGION"),
        gen_model_id=_get(env, "GEN_MODEL_ID"),
        judge_model_id=_get(env, "JUDGE_MODEL_ID"),
        embedding_model_id=_get(env, "EMBEDDING_MODEL_ID"),
        gen_temperature=_get_float(env, "GEN_TEMPERATURE"),
        judge_temperature=_get_float(env, "JUDGE_TEMPERATURE"),
        gen_thinking_budget=_get_int(env, "GEN_THINKING_BUDGET", None, 0),
        gen_thinking_level=thinking_level,
        # Development defaults, not tuned values.
        llm_max_retries=_get_int(env, "LLM_MAX_RETRIES", 5, 0),
        max_critic_retries=_get_int(env, "MAX_CRITIC_RETRIES", 2, 0),
        max_retriever_steps=_get_int(env, "MAX_RETRIEVER_STEPS", 4, 1),
        search_top_k=_get_int(env, "SEARCH_TOP_K", 4, 1),
        chunk_size=_get_int(env, "CHUNK_SIZE", 1000, 100),
        chunk_overlap=_get_int(env, "CHUNK_OVERLAP", 150, 0),
        vector_store=vector_store,
        docs_dir=_get(env, "DOCS_DIR") or "data/docs",
        chroma_dir=_get(env, "CHROMA_DIR") or "data/chroma",
        fake_critic_verdict=fake_verdict,
        strategy=strategy,
        window_k=_get_int(env, "WINDOW_K", 4, 1),
        summary_keep_recent=_get_int(env, "SUMMARY_KEEP_RECENT", 3, 0),
        metrics_db=metrics_db,
        prices_file=_get(env, "PRICES_FILE") or "config/prices.yaml",
        otel_exporter=otel_exporter,
        otel_endpoint=_get(env, "OTEL_EXPORTER_OTLP_ENDPOINT"),
        service_name=_get(env, "OTEL_SERVICE_NAME") or "tracewise",
        log_metrics_json=(_get(env, "LOG_METRICS_JSON") or "false").lower() in ("1", "true", "yes"),
    )

    if settings.chunk_overlap >= settings.chunk_size:
        raise ConfigError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    required: dict[str, str | None] = {}
    if backend == "gemini_api":
        required = {
            "GEMINI_API_KEY": settings.gemini_api_key,
            "GEN_MODEL_ID": settings.gen_model_id,
            "EMBEDDING_MODEL_ID": settings.embedding_model_id,
        }
    elif backend == "vertex":
        required = {
            "GCP_PROJECT": settings.gcp_project,
            "GCP_REGION": settings.gcp_region,
            "GEN_MODEL_ID": settings.gen_model_id,
            "EMBEDDING_MODEL_ID": settings.embedding_model_id,
        }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ConfigError(f"LLM_BACKEND={backend} requires these environment variables: " + ", ".join(missing))

    return settings
