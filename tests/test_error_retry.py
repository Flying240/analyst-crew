import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))
from run_eval import evaluate  # noqa: E402
from src import llm_usage  # noqa: E402
from src.agents import baseline  # noqa: E402

BAD = "SELECT nope FROM orders"
GOOD = "SELECT 1 FROM (SELECT 1 FROM orders LIMIT 1)"
Q = {"id": "x", "difficulty": "easy", "question": "q", "gold_sql": GOOD, "check": {"type": "scalar"}}


@pytest.fixture
def scripted(monkeypatch):
    """Make baseline.ask return the given SQL strings in order and record each prompt."""
    prompts: list[str] = []

    def install(*replies: str) -> list[str]:
        queue = list(replies)

        def ask(prompt: str) -> str:
            prompts.append(prompt)
            llm_usage.record(100, 10)
            return queue.pop(0)

        monkeypatch.setattr(baseline, "ask", ask)
        return prompts

    return install


def test_default_makes_one_call_even_if_sql_fails(scripted):
    prompts = scripted(BAD)
    r = evaluate(Q, baseline.answer)
    assert len(prompts) == 1
    assert not r["passed"] and r["retries_used"] == 0 and r["from_retry"] is False


def test_retry_sends_exact_error_and_failed_sql(scripted):
    prompts = scripted(BAD, GOOD)
    r = evaluate(Q, lambda question: baseline.answer(question, retries=2))
    assert len(prompts) == 2
    error = baseline.run_sql(BAD).error
    assert error and error in prompts[1] and BAD in prompts[1] and "q" in prompts[1]
    assert r["passed"] and r["retries_used"] == 1 and r["from_retry"] is True
    assert r["tokens"] == {"prompt": 200, "completion": 20}


def test_no_retry_when_first_sql_runs(scripted):
    prompts = scripted(GOOD)
    r = evaluate(Q, lambda question: baseline.answer(question, retries=2))
    assert len(prompts) == 1 and r["retries_used"] == 0 and r["from_retry"] is False


def test_retries_stop_at_max(scripted):
    prompts = scripted(BAD, BAD, BAD)
    r = evaluate(Q, lambda question: baseline.answer(question, retries=2))
    assert len(prompts) == 3
    assert not r["passed"] and r["retries_used"] == 2 and r["from_retry"] is True


def test_answerer_without_retry_info_records_zero():
    r = evaluate(Q, lambda question: GOOD)
    assert r["retries_used"] == 0 and r["from_retry"] is False


PROSE = "You are using the Olist e-commerce DuckDB dataset, which includes the tables: orders, sellers."
NO_TABLE = "SELECT 'Olist e-commerce DuckDB' AS dataset"


def test_extract_sql_rejects_prose():
    from src import llm

    with pytest.raises(llm.NotSqlError, match="Return only a single SQL query, no prose"):
        llm.extract_sql(PROSE)
    assert llm.extract_sql("```sql\nSELECT 1;\n```") == "SELECT 1"


def test_prose_reply_is_retried_with_clear_error(monkeypatch):
    from src import llm

    replies = [PROSE, GOOD]
    prompts: list[str] = []
    monkeypatch.setattr(llm, "complete", lambda p, m=None: prompts.append(p) or replies.pop(0))
    monkeypatch.setattr(baseline, "ask", lambda p: llm.ask(p))
    attempts: list[dict] = []
    sql = baseline.answer_from_prompt("Q", retries=1, attempts=attempts)
    assert sql == GOOD
    assert attempts[0] == {"sql": PROSE, "error": "Return only a single SQL query, no prose"}
    assert "Return only a single SQL query, no prose" in prompts[1]


def test_sql_without_a_table_is_retried(scripted):
    prompts = scripted(NO_TABLE, GOOD)
    attempts: list[dict] = []
    assert baseline.answer_from_prompt("Q", retries=1, attempts=attempts) == GOOD
    assert "at least one table" in attempts[0]["error"] and "at least one table" in prompts[1]
