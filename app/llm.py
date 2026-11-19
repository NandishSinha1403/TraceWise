"""Single entry point for every model call.

All generation goes through `generate()` and all embedding goes through
`embed()`. M2 will wrap these two functions with the tracing middleware, so no
other module should talk to Vertex AI directly.

Backends:
- `vertex`: Gemini on Vertex AI via the `google-genai` client in Vertex mode
  (installed as a dependency of `google-cloud-aiplatform`).
- `fake`: deterministic canned responses per role, for tests and the offline
  smoke run only.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.config import Settings

Role = str  # "planner" | "retriever" | "answer" | "critic"

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


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None


class Backend(Protocol):
    def generate(
        self,
        role: Role,
        system: str,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None,
        json_output: bool,
    ) -> LLMResponse: ...

    def embed(self, texts: list[str], task: str) -> list[list[float]]: ...


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

_backend: Backend | None = None


def get_backend(settings: Settings) -> Backend:
    if settings.llm_backend == "fake":
        return FakeBackend(settings)
    if settings.llm_backend == "vertex":
        return VertexBackend(settings)
    raise ValueError(f"unknown LLM backend {settings.llm_backend!r}")


def configure(settings: Settings) -> None:
    global _backend
    _backend = get_backend(settings)


def set_backend(backend: Backend) -> None:
    global _backend
    _backend = backend


def _require_backend() -> Backend:
    if _backend is None:
        raise RuntimeError("app.llm is not configured; call app.llm.configure(settings) first")
    return _backend


def generate(
    role: Role,
    system: str,
    messages: list[ChatMessage],
    tools: list[ToolSpec] | None = None,
    json_output: bool = False,
) -> LLMResponse:
    """The only function that calls a generative model."""
    return _require_backend().generate(role, system, messages, tools, json_output)


def embed(texts: list[str], task: str) -> list[list[float]]:
    """The only function that calls an embedding model. `task` is "document" or "query"."""
    if task not in ("document", "query"):
        raise ValueError(f"task must be 'document' or 'query', got {task!r}")
    if not texts:
        return []
    return _require_backend().embed(texts, task)


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
# Vertex AI backend
# --------------------------------------------------------------------------- #

_EMBED_BATCH = 16  # conservative request batch size


class VertexBackend:
    def __init__(self, settings: Settings) -> None:
        from google import genai

        self._settings = settings
        self._client = genai.Client(
            vertexai=True, project=settings.gcp_project, location=settings.gcp_region
        )

    def generate(self, role, system, messages, tools, json_output) -> LLMResponse:
        from google.genai import types

        config_kwargs: dict[str, Any] = {
            "system_instruction": system,
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
        }
        if self._settings.gen_temperature is not None:
            config_kwargs["temperature"] = self._settings.gen_temperature
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

        response = self._client.models.generate_content(
            model=self._settings.gen_model_id,
            contents=self._to_contents(messages),
            config=types.GenerateContentConfig(**config_kwargs),
        )

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        candidates = response.candidates or []
        parts = (candidates[0].content.parts if candidates and candidates[0].content else None) or []
        for part in parts:
            if part.function_call is not None:
                fc = part.function_call
                tool_calls.append(ToolCall(name=fc.name, args=dict(fc.args or {}), id=fc.id))
            elif part.text:
                text_parts.append(part.text)
        return LLMResponse(text="".join(text_parts), tool_calls=tool_calls, raw=response)

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
                    parts.append(types.Part(function_call=types.FunctionCall(name=tc.name, args=tc.args, id=tc.id)))
                contents.append(types.Content(role="model", parts=parts))
            else:
                raise ValueError(f"unknown message role {role!r}")
        flush_tools()
        return contents

    def embed(self, texts: list[str], task: str) -> list[list[float]]:
        from google.genai import types

        task_type = "RETRIEVAL_DOCUMENT" if task == "document" else "RETRIEVAL_QUERY"
        vectors: list[list[float]] = []
        for i in range(0, len(texts), _EMBED_BATCH):
            batch = texts[i : i + _EMBED_BATCH]
            resp = self._client.models.embed_content(
                model=self._settings.embedding_model_id,
                contents=batch,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            vectors.extend(list(e.values) for e in resp.embeddings)
        return vectors


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


class FakeBackend:
    """Deterministic responses keyed on role. No network, no credentials."""

    def __init__(self, settings: Settings) -> None:
        self._critic_verdict = settings.fake_critic_verdict

    def generate(self, role, system, messages, tools, json_output) -> LLMResponse:
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

        raise ValueError(f"fake backend has no canned response for role {role!r}")

    def embed(self, texts: list[str], task: str) -> list[list[float]]:
        return [_hash_embed(t) for t in texts]


def _hash_embed(text: str) -> list[float]:
    vec = [0.0] * _FAKE_DIM
    for tok in _TOKEN_RE.findall(text.lower()):
        h = int.from_bytes(hashlib.sha256(tok.encode()).digest()[:4], "big")
        vec[h % _FAKE_DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]
