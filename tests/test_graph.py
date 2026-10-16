import pytest

from src import graph, llm
from src.agents import baseline, viz
from src.agents.duckdb_tips import DUCKDB_TIPS
from src.config import DATA_MODEL, PLANNER_MODEL, REPORTER_MODEL

BAD = "SELECT nope FROM orders"
GOOD = "SELECT 1 AS x"
MONTHS = "SELECT '2017-01' AS month, 1 AS n UNION ALL SELECT '2017-02', 2 ORDER BY 1"


@pytest.fixture
def fake_llm(monkeypatch):
    """Script llm replies: planner/reporter via complete, SQL via ask. Records (model, prompt)."""
    calls: list[tuple[str | None, str]] = []

    def install(*sql_replies: str) -> list[tuple[str | None, str]]:
        queue = list(sql_replies)

        def complete(prompt: str, model: str | None = None) -> str:
            calls.append((model, prompt))
            return "a report" if model == REPORTER_MODEL and "Answer:" in prompt else "- a plan"

        def ask(prompt: str, model: str | None = None) -> str:
            calls.append((model, prompt))
            return queue.pop(0)

        monkeypatch.setattr(llm, "complete", complete)
        monkeypatch.setattr(llm, "ask", ask)
        return calls

    return install


def test_eval_answer_skips_viz_and_reporter(fake_llm):
    calls = fake_llm(GOOD)
    assert graph.answer("q") == GOOD
    assert [m for m, _ in calls] == [PLANNER_MODEL, DATA_MODEL]
    assert "- a plan" in calls[1][1] and DUCKDB_TIPS in calls[1][1]


def test_planner_prompt_forbids_extra_filters(fake_llm):
    calls = fake_llm(GOOD)
    graph.answer("q")
    assert "only filters stated in the question or the project rules; never add others" in calls[0][1]


def test_retry_fills_attempts(fake_llm):
    calls = fake_llm(BAD, GOOD)
    state = graph.run("q", retries=2, skip_report=True)
    assert state["sql"] == GOOD
    assert [a["sql"] for a in state["attempts"]] == [BAD, GOOD]
    assert state["attempts"][0]["error"] and state["attempts"][1]["error"] is None
    assert state["retries_used"] == 1 and state["critic_feedback"] is None
    assert "failed with this database error" in calls[2][1]


def test_no_retries_by_default(fake_llm):
    fake_llm(BAD)
    state = graph.run("q", skip_report=True)
    assert state["sql"] == BAD and len(state["attempts"]) == 1 and state["error"]


def test_full_run_with_chart(fake_llm):
    calls = fake_llm(MONTHS)
    state = graph.run("q")
    assert state["chart"] is not None and state["report"] == "a report"
    assert calls[-1][0] == REPORTER_MODEL and "2017-01 | 1" in calls[-1][1]


def test_full_run_scalar_has_no_chart(fake_llm):
    fake_llm(GOOD)
    state = graph.run("q")
    assert "chart" not in state and state["report"] == "a report"


@pytest.mark.parametrize("columns, rows, kind", [
    (["m", "n"], [("2017-01", 1), ("2017-02", 2)], "line"),
    (["state", "n"], [("SP", 5), ("RJ", 3)], "bar"),
    (["x"], [(1,)], None),
    (["a", "b"], [("SP", "x"), ("RJ", "y")], None),
])
def test_chart_kind(columns, rows, kind):
    assert viz.chart_kind(columns, rows) == kind


def test_answer_with_tips_adds_tips(monkeypatch):
    prompts: list[str] = []
    monkeypatch.setattr(baseline, "ask", lambda p: prompts.append(p) or GOOD)
    baseline.answer_with_tips("q")
    baseline.answer("q")
    assert DUCKDB_TIPS in prompts[0] and DUCKDB_TIPS not in prompts[1]


def test_tips_have_no_date_trunc():
    assert "date_trunc" not in DUCKDB_TIPS and "strftime(ts, '%Y-%m')" in DUCKDB_TIPS


def test_planner_on_120b_reporter_on_20b():
    assert PLANNER_MODEL == "openai/gpt-oss-120b" and REPORTER_MODEL == "openai/gpt-oss-20b"


def test_planner_prompt_has_rules_and_bans_sql_functions(fake_llm):
    calls = fake_llm(GOOD)
    graph.answer("q")
    prompt = calls[0][1]
    assert baseline.RULES in prompt and "Never name SQL functions" in prompt


def test_data_prompt_puts_question_over_plan(fake_llm):
    calls = fake_llm(GOOD)
    graph.answer("q")
    assert "The plan is guidance. If it conflicts with the question, follow the question." in calls[1][1]


def test_eval_result_saves_plan_and_attempts(fake_llm):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))
    from run_eval import evaluate

    fake_llm(BAD, GOOD)
    q = {"id": "x", "question": "q", "gold_sql": "SELECT 1", "check": {"type": "scalar"}}
    r = evaluate(q, lambda question: graph.answer(question, retries=2))
    assert r["passed"] and r["trace"]["plan"] == "- a plan"
    assert [a["sql"] for a in r["trace"]["attempts"]] == [BAD, GOOD] and r["trace"]["attempts"][0]["error"]


def test_baseline_eval_result_has_no_trace(monkeypatch):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))
    from run_eval import evaluate

    monkeypatch.setattr(baseline, "ask", lambda p: "SELECT 1")
    r = evaluate({"id": "x", "question": "q", "gold_sql": "SELECT 1", "check": {"type": "scalar"}}, baseline.answer)
    assert "trace" not in r
