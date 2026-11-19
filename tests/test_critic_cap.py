"""The critic retry cap is enforced."""

import pytest

from app.agents.critic import decide, parse_verdict
from app.graph import build_graph, run

Q = "Name the steam engines of the Quill Valley Railway."


def test_cap_zero_stops_after_one_critic_pass(make_env):
    settings, store = make_env(MAX_CRITIC_RETRIES="0", FAKE_CRITIC_VERDICT="retry_retriever")
    out = run(build_graph(settings, store), settings, Q)
    assert out["steps"] == ["planner", "retriever", "answer", "critic"]
    assert out["capped"] is True
    assert out["critic_retries"] == 0
    assert out["answer"]  # best available answer is still returned


def test_retry_retriever_loops_until_cap(make_env):
    settings, store = make_env(MAX_CRITIC_RETRIES="2", FAKE_CRITIC_VERDICT="retry_retriever")
    out = run(build_graph(settings, store), settings, Q)
    assert out["steps"] == ["planner"] + ["retriever", "answer", "critic"] * 3
    assert out["critic_retries"] == 2
    assert out["capped"] is True


def test_retry_planner_routes_to_planner(make_env):
    settings, store = make_env(MAX_CRITIC_RETRIES="1", FAKE_CRITIC_VERDICT="retry_planner")
    out = run(build_graph(settings, store), settings, Q)
    assert out["steps"] == ["planner", "retriever", "answer", "critic"] * 2
    assert out["critic_retries"] == 1
    assert out["capped"] is True


@pytest.mark.parametrize(
    "verdict,retries,cap,expected",
    [
        ("accept", 0, 2, ("end", 0, False)),
        ("retry_planner", 0, 2, ("planner", 1, False)),
        ("retry_retriever", 1, 2, ("retriever", 2, False)),
        ("retry_retriever", 2, 2, ("end", 2, True)),
        ("retry_planner", 0, 0, ("end", 0, True)),
        ("invalid", 0, 2, ("end", 0, False)),
    ],
)
def test_decide(verdict, retries, cap, expected):
    assert decide({"verdict": verdict, "reason": ""}, retries, cap) == expected


def test_parse_verdict_handles_bad_output():
    assert parse_verdict("not json")["verdict"] == "invalid"
    assert parse_verdict('{"verdict": "maybe"}')["verdict"] == "invalid"
    assert parse_verdict('```json\n{"verdict": "accept", "reason": "ok"}\n```') == {"verdict": "accept", "reason": "ok"}
