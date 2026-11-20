"""LLM judge: scores an answer for correctness and groundedness with the fixed rubric."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app import llm
from app.tools.search import format_results

RUBRIC_PATH = Path(__file__).with_name("judge_prompt.md")


@dataclass
class JudgeResult:
    correctness: int | None
    groundedness: int | None
    rationale: str
    error: str | None = None


def load_rubric(path: Path = RUBRIC_PATH) -> str:
    return path.read_text(encoding="utf-8")


def build_prompt(question: str, reference: str, passages: list[dict], answer: str, cited: list[str]) -> str:
    return "\n".join(
        [
            f"QUESTION: {question}",
            f"REFERENCE: {reference}",
            "PASSAGES:",
            format_results(passages),
            f"ANSWER: {answer}",
            f"CITED doc_ids: {cited}",
        ]
    )


def parse_scores(text: str) -> JudgeResult:
    data = llm.parse_json(text)
    if data is None:
        return JudgeResult(None, None, "", error="judge output was not valid JSON")
    scores = {}
    for key in ("correctness", "groundedness"):
        value = data.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value or not 1 <= value <= 5:
            return JudgeResult(None, None, str(data.get("rationale", "")), error=f"invalid {key}: {value!r}")
        scores[key] = int(value)
    return JudgeResult(scores["correctness"], scores["groundedness"], str(data.get("rationale", "")).strip())


def judge(question: str, reference: str, passages: list[dict], answer: str, cited: list[str], rubric: str) -> JudgeResult:
    resp = llm.generate(
        "judge",
        rubric,
        [{"role": "user", "content": build_prompt(question, reference, passages, answer, cited)}],
        json_output=True,
    )
    return parse_scores(resp.text)
