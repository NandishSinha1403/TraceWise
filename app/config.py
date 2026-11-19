"""Environment-driven configuration.

No model IDs, project IDs, or regions are hard-coded. Values that are needed
for a real (Vertex AI) run must come from the environment; if they are missing
`load_settings()` raises `ConfigError` naming the variable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

VALID_BACKENDS = ("vertex", "fake")
VALID_VECTOR_STORES = ("chroma",)
VALID_FAKE_VERDICTS = ("accept", "retry_planner", "retry_retriever")


class ConfigError(RuntimeError):
    """Raised when a required environment variable is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    llm_backend: str
    gcp_project: str | None
    gcp_region: str | None
    gen_model_id: str | None
    judge_model_id: str | None  # read but unused in M1
    embedding_model_id: str | None
    gen_temperature: float | None
    max_critic_retries: int
    max_retriever_steps: int
    search_top_k: int
    chunk_size: int
    chunk_overlap: int
    vector_store: str
    docs_dir: str
    chroma_dir: str
    fake_critic_verdict: str

    @property
    def collection_name(self) -> str:
        """Chroma collection, keyed by embedder so fake and real vectors never mix."""
        if self.llm_backend == "fake":
            return "tracewise-fake"
        safe = "".join(c if c.isalnum() or c in "._-" else "-" for c in self.embedding_model_id or "")
        return f"tracewise-{safe}".strip("-._")


def _get(env: dict[str, str], name: str) -> str | None:
    value = env.get(name)
    if value is None or value.strip() == "":
        return None
    return value.strip()


def _get_int(env: dict[str, str], name: str, default: int, minimum: int) -> int:
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


def load_settings(env: dict[str, str] | None = None) -> Settings:
    env = dict(os.environ if env is None else env)

    backend = (_get(env, "LLM_BACKEND") or "fake").lower()
    if backend not in VALID_BACKENDS:
        raise ConfigError(f"LLM_BACKEND must be one of {VALID_BACKENDS}, got {backend!r}")

    vector_store = (_get(env, "VECTOR_STORE") or "chroma").lower()
    if vector_store not in VALID_VECTOR_STORES:
        raise ConfigError(f"VECTOR_STORE must be one of {VALID_VECTOR_STORES} in M1, got {vector_store!r}")

    fake_verdict = (_get(env, "FAKE_CRITIC_VERDICT") or "accept").lower()
    if fake_verdict not in VALID_FAKE_VERDICTS:
        raise ConfigError(f"FAKE_CRITIC_VERDICT must be one of {VALID_FAKE_VERDICTS}, got {fake_verdict!r}")

    temp_raw = _get(env, "GEN_TEMPERATURE")
    try:
        gen_temperature = float(temp_raw) if temp_raw is not None else None
    except ValueError as exc:
        raise ConfigError(f"GEN_TEMPERATURE must be a number, got {temp_raw!r}") from exc

    settings = Settings(
        llm_backend=backend,
        gcp_project=_get(env, "GCP_PROJECT"),
        gcp_region=_get(env, "GCP_REGION"),
        gen_model_id=_get(env, "GEN_MODEL_ID"),
        judge_model_id=_get(env, "JUDGE_MODEL_ID"),
        embedding_model_id=_get(env, "EMBEDDING_MODEL_ID"),
        gen_temperature=gen_temperature,
        # Development defaults, not tuned values.
        max_critic_retries=_get_int(env, "MAX_CRITIC_RETRIES", 2, 0),
        max_retriever_steps=_get_int(env, "MAX_RETRIEVER_STEPS", 4, 1),
        search_top_k=_get_int(env, "SEARCH_TOP_K", 4, 1),
        chunk_size=_get_int(env, "CHUNK_SIZE", 1000, 100),
        chunk_overlap=_get_int(env, "CHUNK_OVERLAP", 150, 0),
        vector_store=vector_store,
        docs_dir=_get(env, "DOCS_DIR") or "data/docs",
        chroma_dir=_get(env, "CHROMA_DIR") or "data/chroma",
        fake_critic_verdict=fake_verdict,
    )

    if settings.chunk_overlap >= settings.chunk_size:
        raise ConfigError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    if backend == "vertex":
        required = {
            "GCP_PROJECT": settings.gcp_project,
            "GCP_REGION": settings.gcp_region,
            "GEN_MODEL_ID": settings.gen_model_id,
            "EMBEDDING_MODEL_ID": settings.embedding_model_id,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ConfigError(
                "LLM_BACKEND=vertex requires these environment variables: " + ", ".join(missing)
            )

    return settings
