"""Single entry point for every model call.

All generation goes through `generate()` and all embedding goes through
`embed()`. Both run inside a tracing scope (`app/tracing/middleware.py`), so
every model call is measured. No other module talks to a model API directly.

Backends (both real ones use the `google-genai` client):
- `gemini_api`: Gemini Developer API with an AI Studio API key. Free tier, no
  billing account. This is the default real backend for this project.
- `vertex`: Gemini on Vertex AI (needs a billing-enabled project). Kept so a
  later switch is a config change only.
- `fake`: deterministic canned responses per role, for tests and the offline
  smoke run only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, TypeVar

from app.config import Settings
from app.tracing import middleware as tracing
from app.tracing.middleware import Usage

log = logging.getLogger(__name__)

T = TypeVar("T")

Role = str  # "planner" | "retriever" | "answer" | "critic" | "summarizer" | "judge"

# Chat message format used inside a single node's call (e.g. the ReAct scratch):
#   {"role": "user", "content": str}
#   {"role": "assistant", "content": str, "tool_calls": [ToolCall, ...]}
#   {"role": "tool", "tool_name": str, "tool_call_id": str | None, "content": str}
ChatMessage = dict[str, Any]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any]
    id: str | None = None
    # Gemini 3+ attaches an opaque signature to function calls; it must be sent
    # back unchanged with that call in later turns, or the API returns 400.
    thought_signature: bytes | None = field(default=None, repr=False, compare=False)


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    raw: Any = None


@dataclass(frozen=True)
class GenOptions:
    model_id: str
    temperature: float | None
    thinking_budget: int | None
    thinking_level: str | None = None


class Backend(Protocol):
    def generate(
        self,
        role: Role,
        system: str,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None,
        json_output: bool,
        options: GenOptions,
    ) -> LLMResponse: ...

    def embed(self, texts: list[str], task: str) -> tuple[list[list[float]], Usage]: ...


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

_backend: Backend | None = None
_settings: Settings | None = None


def get_backend(settings: Settings) -> Backend:
    if settings.llm_backend == "fake":
        return FakeBackend(settings)
    if settings.llm_backend in ("gemini_api", "vertex"):
        return GenAIBackend(settings)
    raise ValueError(f"unknown LLM backend {settings.llm_backend!r}")


def configure(settings: Settings, backend: Backend | None = None) -> None:
    global _backend, _settings
    _settings = settings
    _backend = backend or get_backend(settings)


def set_backend(backend: Backend) -> None:
    global _backend
    _backend = backend


def _require() -> tuple[Backend, Settings]:
    if _backend is None or _settings is None:
        raise RuntimeError("app.llm is not configured; call app.llm.configure(settings) first")
    return _backend, _settings


def _model_name(settings: Settings, kind: str) -> str:
    if settings.llm_backend == "fake":
        return f"fake-{kind}"
    model = {"gen": settings.gen_model_id, "judge": settings.judge_model_id, "embed": settings.embedding_model_id}[kind]
    if model is None:
        env = {"gen": "GEN_MODEL_ID", "judge": "JUDGE_MODEL_ID", "embed": "EMBEDDING_MODEL_ID"}[kind]
        raise RuntimeError(f"{env} is not set")
    return model


def generate(
    role: Role,
    system: str,
    messages: list[ChatMessage],
    tools: list[ToolSpec] | None = None,
    json_output: bool = False,
) -> LLMResponse:
    """The only function that calls a generative model. `role="judge"` uses the judge model."""
    backend, settings = _require()
    is_judge = role == "judge"
    options = GenOptions(
        model_id=_model_name(settings, "judge" if is_judge else "gen"),
        temperature=settings.judge_temperature if is_judge else settings.gen_temperature,
        thinking_budget=settings.gen_thinking_budget,
        thinking_level=settings.gen_thinking_level,
    )
    with tracing.model_call("judge" if is_judge else "llm", options.model_id, node=role) as handle:
        resp = backend.generate(role, system, messages, tools, json_output, options)
        handle.finish(resp.usage)
    return resp


def embed(texts: list[str], task: str) -> list[list[float]]:
    """The only function that calls an embedding model. `task` is "document" or "query"."""
    if task not in ("document", "query"):
        raise ValueError(f"task must be 'document' or 'query', got {task!r}")
    if not texts:
        return []
    backend, settings = _require()
    with tracing.model_call("embed", _model_name(settings, "embed")) as handle:
        vectors, usage = backend.embed(texts, task)
        handle.finish(usage)
    return vectors


def parse_json(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from model text, tolerating ```json fences."""
    cleaned = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            value = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


# --------------------------------------------------------------------------- #
# google-genai backend (Gemini Developer API or Vertex AI)
# --------------------------------------------------------------------------- #

_EMBED_BATCH = 16  # conservative request batch size
_RETRY_CODES = (429, 500, 503)  # rate limited / transient server errors
_BACKOFF_BASE_S = 2.0
_BACKOFF_MAX_S = 60.0


class GenAIBackend:
    def __init__(self, settings: Settings) -> None:
        from google import genai

        self._settings = settings
        if settings.llm_backend == "gemini_api":
            self._client = genai.Client(api_key=settings.gemini_api_key)
        else:
            self._client = genai.Client(vertexai=True, project=settings.gcp_project, location=settings.gcp_region)

    def _call(self, fn: Callable[..., T], **kwargs) -> T:
        return self._with_retries(lambda: fn(**kwargs))[0]

    def _with_retries(self, attempt_fn: Callable[[], T]) -> tuple[T, int, float]:
        """Run `attempt_fn`, retrying rate-limit and transient errors with backoff.

        Returns (result, attempts, backoff_ms). Free-tier quotas are low, so
        429s are expected; the backoff is reported so it can be separated from
        model latency.
        """
        from google.genai import errors

        attempt = 0
        backoff_ms = 0.0
        while True:
            try:
                return attempt_fn(), attempt + 1, backoff_ms
            except errors.APIError as exc:
                if exc.code not in _RETRY_CODES or attempt >= self._settings.llm_max_retries:
                    raise
                delay = min(_BACKOFF_MAX_S, _BACKOFF_BASE_S * 2**attempt) * (0.5 + random.random() / 2)
                attempt += 1
                log.warning("model call failed with %s; retry %d in %.1fs", exc.code, attempt, delay)
                time.sleep(delay)
                backoff_ms += delay * 1000

    def generate(self, role, system, messages, tools, json_output, options: GenOptions) -> LLMResponse:
        from google.genai import types

        config_kwargs: dict[str, Any] = {
            "system_instruction": system,
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
        }
        if options.temperature is not None:
            config_kwargs["temperature"] = options.temperature
        if options.thinking_budget is not None:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=options.thinking_budget)
        elif options.thinking_level is not None:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=options.thinking_level)
        if tools:
            config_kwargs["tools"] = [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=t.name, description=t.description, parameters_json_schema=t.parameters
                        )
                        for t in tools
                    ]
                )
            ]
        if json_output:
            config_kwargs["response_mime_type"] = "application/json"
        config = types.GenerateContentConfig(**config_kwargs)
        contents = self._to_contents(messages)

        def attempt() -> tuple[str, list[ToolCall], Any, float | None, float]:
            t0 = time.perf_counter()
            ttft_ms: float | None = None
            text_parts: list[str] = []
            tool_calls: list[ToolCall] = []
            usage_md = None
            for chunk in self._client.models.generate_content_stream(
                model=options.model_id, contents=contents, config=config
            ):
                got_content = False
                candidates = chunk.candidates or []
                parts = (candidates[0].content.parts if candidates and candidates[0].content else None) or []
                for part in parts:
                    if part.function_call is not None:
                        fc = part.function_call
                        tool_calls.append(
                            ToolCall(name=fc.name, args=dict(fc.args or {}), id=fc.id, thought_signature=part.thought_signature)
                        )
                        got_content = True
                    elif part.text and not part.thought:
                        text_parts.append(part.text)
                        got_content = True
                if got_content and ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t0) * 1000
                if chunk.usage_metadata is not None:
                    usage_md = chunk.usage_metadata
            total_ms = (time.perf_counter() - t0) * 1000
            return "".join(text_parts), tool_calls, usage_md, ttft_ms, total_ms

        (text, tool_calls, usage_md, ttft_ms, total_ms), attempts, backoff_ms = self._with_retries(attempt)
        usage = Usage(ttft_ms=ttft_ms, latency_ms=total_ms, attempts=attempts, backoff_ms=backoff_ms)
        if usage_md is not None:
            usage.input_tokens = (usage_md.prompt_token_count or 0) + (usage_md.tool_use_prompt_token_count or 0)
            usage.output_tokens = usage_md.candidates_token_count or 0
            usage.thinking_tokens = usage_md.thoughts_token_count or 0
        return LLMResponse(text=text, tool_calls=tool_calls, usage=usage)

    @staticmethod
    def _to_contents(messages: list[ChatMessage]) -> list[Any]:
        from google.genai import types

        contents: list[types.Content] = []
        pending_tool_parts: list[types.Part] = []

        def flush_tools() -> None:
            if pending_tool_parts:
                contents.append(types.Content(role="user", parts=list(pending_tool_parts)))
                pending_tool_parts.clear()

        for m in messages:
            role = m["role"]
            if role == "tool":
                pending_tool_parts.append(
                    types.Part.from_function_response(name=m["tool_name"], response={"result": m["content"]})
                )
                continue
            flush_tools()
            if role == "user":
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=m["content"])]))
            elif role == "assistant":
                parts: list[types.Part] = []
                if m.get("content"):
                    parts.append(types.Part.from_text(text=m["content"]))
                for tc in m.get("tool_calls", []):
                    parts.append(
                        types.Part(
                            function_call=types.FunctionCall(name=tc.name, args=tc.args, id=tc.id),
                            thought_signature=tc.thought_signature,
                        )
                    )
                contents.append(types.Content(role="model", parts=parts))
            else:
                raise ValueError(f"unknown message role {role!r}")
        flush_tools()
        return contents

    def embed(self, texts: list[str], task: str) -> tuple[list[list[float]], Usage]:
        """Embedding token counts are not returned by the Gemini Developer API, so they stay None."""
        from google.genai import types

        task_type = "RETRIEVAL_DOCUMENT" if task == "document" else "RETRIEVAL_QUERY"
        vectors: list[list[float]] = []
        attempts_total, backoff_total = 0, 0.0
        t0 = time.perf_counter()
        for i in range(0, len(texts), _EMBED_BATCH):
            batch = texts[i : i + _EMBED_BATCH]
            resp, attempts, backoff = self._with_retries(
                lambda: self._client.models.embed_content(
                    model=self._settings.embedding_model_id,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type=task_type),
                )
            )
            attempts_total += attempts
            backoff_total += backoff
            vectors.extend(list(e.values) for e in resp.embeddings)
        total_ms = (time.perf_counter() - t0) * 1000 - backoff_total
        n_batches = math.ceil(len(texts) / _EMBED_BATCH)
        return vectors, Usage(
            latency_ms=total_ms, attempts=attempts_total - n_batches + 1, backoff_ms=backoff_total
        )


# --------------------------------------------------------------------------- #
# Fake backend (tests and offline smoke run only)
# --------------------------------------------------------------------------- #

_FAKE_DIM = 256
_TOKEN_RE = re.compile(r"[a-z0-9]+")
PASSAGE_RE = re.compile(r"^<<doc_id=(?P<doc>[^>]+)>> (?P<text>.*)$", re.MULTILINE)


def _last_user_text(messages: list[ChatMessage]) -> str:
    for m in reversed(messages):
        if m["role"] == "user":
            return m["content"]
    return ""


def _first_user_text(messages: list[ChatMessage]) -> str:
    for m in messages:
        if m["role"] == "user":
            return m["content"]
    return ""


def _question(text: str) -> str:
    match = re.search(r"^Question: (.*)$", text, re.MULTILINE)
    return match.group(1).strip() if match else text.strip()


def _fake_usage(system: str, messages: list[ChatMessage], output: str) -> Usage:
    """Deterministic, made-up counts so the tracing pipeline can be tested. Never reported."""
    n_in = len(system.split()) + sum(len(str(m.get("content", "")).split()) for m in messages)
    n_out = max(1, len(output.split()))
    return Usage(input_tokens=n_in, output_tokens=n_out, thinking_tokens=0, ttft_ms=10.0, latency_ms=10.0 + n_out)


class FakeBackend:
    """Deterministic responses keyed on role. No network, no credentials."""

    def __init__(self, settings: Settings) -> None:
        self._critic_verdict = settings.fake_critic_verdict

    def generate(self, role, system, messages, tools, json_output, options) -> LLMResponse:
        resp = self._respond(role, messages)
        out = resp.text or " ".join(f"{c.name}({json.dumps(c.args)})" for c in resp.tool_calls)
        resp.usage = _fake_usage(system, messages, out)
        return resp

    def _respond(self, role, messages) -> LLMResponse:
        if role == "planner":
            q = _question(_last_user_text(messages))
            return LLMResponse(text=json.dumps({"plan": f"Find passages that answer: {q}", "sub_queries": [q]}))

        if role == "retriever":
            prompt = _first_user_text(messages)
            # Last occurrence: the transcript may also contain the planner's sub-query list.
            block = prompt.rsplit("Sub-queries:", 1)[1] if "Sub-queries:" in prompt else ""
            sub_queries = [ln[2:].strip() for ln in block.splitlines() if ln.startswith("- ")]
            done = sum(1 for m in messages if m["role"] == "tool")
            if done < len(sub_queries):
                return LLMResponse(
                    text="",
                    tool_calls=[ToolCall(name="search_documents", args={"query": sub_queries[done]}, id=f"fake-{done}")],
                )
            return LLMResponse(text="Retrieval complete.")

        if role == "answer":
            passages = PASSAGE_RE.findall(_last_user_text(messages))
            if not passages:
                return LLMResponse(text=json.dumps({"answer": "No supporting passages were retrieved.", "cited_doc_ids": []}))
            doc, text = passages[0]
            sentence = text.split(". ")[0].rstrip(".") + "."
            return LLMResponse(text=json.dumps({"answer": f"{sentence} [{doc}]", "cited_doc_ids": [doc]}))

        if role == "critic":
            reason = "fake critic: accepted" if self._critic_verdict == "accept" else "fake critic: forced rejection"
            return LLMResponse(text=json.dumps({"verdict": self._critic_verdict, "reason": reason}))

        if role == "summarizer":
            words = _last_user_text(messages).split()
            return LLMResponse(text="Fake summary: " + " ".join(words[-30:]))

        if role == "judge":
            return LLMResponse(text=json.dumps({"correctness": 3, "groundedness": 3, "rationale": "fake judge"}))

        raise ValueError(f"fake backend has no canned response for role {role!r}")

    def embed(self, texts: list[str], task: str) -> tuple[list[list[float]], Usage]:
        return [_hash_embed(t) for t in texts], Usage(latency_ms=1.0)


def _hash_embed(text: str) -> list[float]:
    vec = [0.0] * _FAKE_DIM
    for tok in _TOKEN_RE.findall(text.lower()):
        h = int.from_bytes(hashlib.sha256(tok.encode()).digest()[:4], "big")
        vec[h % _FAKE_DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]
