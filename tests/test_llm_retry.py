import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))
from run_eval import evaluate, run_questions, summarize, tally, token_totals  # noqa: E402
from src.llm_retry import DailyLimitError, InfraError, is_rate_limit, with_backoff  # noqa: E402


class RateLimited(Exception):
    status_code = 429


def flaky(failures: int, exc: Exception):
    calls = {"n": 0}

    def call() -> str:
        calls["n"] += 1
        if calls["n"] <= failures:
            raise exc
        return "ok"

    return call, calls


def test_is_rate_limit():
    assert is_rate_limit(RateLimited())
    assert is_rate_limit(Exception("Error code: 429 - Rate limit reached"))
    assert is_rate_limit(Exception("RESOURCE_EXHAUSTED"))
    assert not is_rate_limit(ValueError("bad sql"))


def test_retries_with_exponential_backoff_then_succeeds():
    call, calls = flaky(3, RateLimited())
    waits: list[float] = []
    assert with_backoff(call, sleep=waits.append) == "ok"
    assert waits == [2, 4, 8] and calls["n"] == 4


def test_raises_infra_error_when_retries_exhausted():
    call, calls = flaky(99, RateLimited())
    waits: list[float] = []
    with pytest.raises(InfraError):
        with_backoff(call, sleep=waits.append)
    assert waits == [2, 4, 8] and calls["n"] == 4


def test_other_errors_are_not_retried():
    call, calls = flaky(1, ValueError("boom"))
    with pytest.raises(ValueError):
        with_backoff(call, sleep=lambda s: None)
    assert calls["n"] == 1


def raise_infra(question: str) -> str:
    raise InfraError("rate limited")


def test_evaluate_marks_infra_error_and_tally_excludes_it():
    q = {"id": "x", "difficulty": "easy", "question": "q", "gold_sql": "SELECT 1", "check": {"type": "scalar"}}
    infra = evaluate(q, raise_infra)
    ok = evaluate(q, lambda question: "SELECT 1")
    assert infra["infra_error"] and not infra["passed"]
    assert tally([infra, ok]) == {"score": 1, "total": 1, "infra_errors": 1}


def test_summary_keeps_infra_errors_out_of_flips():
    def report(passed: bool, infra: bool) -> dict:
        r = {"id": "x", "difficulty": "easy", "passed": passed, "infra_error": infra}
        return {"answerer": "a", **tally([r]), "by_difficulty": {"easy": tally([r])}, "results": [r]}

    s = summarize([report(True, False), report(False, True), report(True, False)])
    assert s["flipped"] == {} and s["infra_errors"] == {"x": 1}
    assert s["overall"]["mean_pct"] == 100.0 and s["overall"]["infra_errors"] == 1


DAILY = Exception(
    "Error code: 429 - Rate limit reached for model `openai/gpt-oss-120b` on tokens per day (TPD): Limit 200000"
)


def test_daily_limit_raises_without_backoff():
    call, calls = flaky(99, DAILY)
    waits: list[float] = []
    with pytest.raises(DailyLimitError):
        with_backoff(call, sleep=waits.append)
    assert waits == [] and calls["n"] == 1


def test_daily_limit_stops_run_and_marks_it_incomplete():
    questions = [
        {"id": f"q{i}", "difficulty": "easy", "question": f"q{i}", "gold_sql": "SELECT 1", "check": {"type": "scalar"}}
        for i in range(4)
    ]
    asked: list[str] = []

    def answer(question: str) -> str:
        asked.append(question)
        if question == "q1":
            raise DailyLimitError("tokens per day")
        return "SELECT 1"

    results, complete = run_questions(questions, answer)
    assert not complete and asked == ["q0", "q1"] and len(results) == 2
    assert results[1]["daily_limit"] and results[1]["infra_error"]


def test_summary_excludes_incomplete_runs():
    def report(passed: bool, complete: bool) -> dict:
        r = {"id": "x", "difficulty": "easy", "passed": passed, "infra_error": False}
        return {"answerer": "a", "complete": complete, **tally([r]), "by_difficulty": {"easy": tally([r])}, "results": [r]}

    s = summarize([report(True, True), report(False, False), report(True, True)])
    assert s["runs"] == 2 and s["incomplete_runs"] == [2]
    assert s["overall"]["mean_pct"] == 100.0 and s["flipped"] == {}
    assert summarize([report(False, False)]) == {"runs": 0, "incomplete_runs": [1]}


def test_token_usage_is_logged_per_question():
    from src import llm_usage

    def answer(question: str) -> str:
        llm_usage.record(120, 30)
        return "SELECT 1"

    q = {"id": "x", "difficulty": "easy", "question": "q", "gold_sql": "SELECT 1", "check": {"type": "scalar"}}
    r = evaluate(q, answer)
    assert r["tokens"] == {"prompt": 120, "completion": 30}
    assert evaluate(q, lambda question: "SELECT 1")["tokens"] is None
    assert token_totals([r, r])["mean_prompt"] == 120
