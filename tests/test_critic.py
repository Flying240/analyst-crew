import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))
from run_eval import answerer_slug, critic_totals, evaluate  # noqa: E402
from src import graph, llm  # noqa: E402
from src.agents import critic  # noqa: E402
from src.config import CRITIC_MODEL  # noqa: E402

FIRST = "SELECT 1 AS x"
SECOND = "SELECT 2 AS x"
THIRD = "SELECT 3 AS x"
BAD = "SELECT nope FROM orders"
OK = json.dumps({"verdict": "ok", "feedback": ""})


def revise(text: str = "wrong grain") -> str:
    return json.dumps({"verdict": "revise", "feedback": text})


@pytest.fixture
def fake_llm(monkeypatch):
    """Script SQL replies (llm.ask) and Critic replies (llm.complete on CRITIC_MODEL with a JSON prompt)."""
    calls: list[tuple[str, str | None, str]] = []

    def install(sql_replies: list[str], critic_replies: list[str]) -> list[tuple[str, str | None, str]]:
        sqls, verdicts = list(sql_replies), list(critic_replies)

        def complete(prompt: str, model: str | None = None) -> str:
            if prompt.endswith("JSON:"):
                calls.append(("critic", model, prompt))
                return verdicts.pop(0)
            calls.append(("planner", model, prompt))
            return "- a plan"

        def ask(prompt: str, model: str | None = None) -> str:
            calls.append(("data", model, prompt))
            return sqls.pop(0)

        monkeypatch.setattr(llm, "complete", complete)
        monkeypatch.setattr(llm, "ask", ask)
        return calls

    return install


def kinds(calls):
    return [k for k, _, _ in calls]


def test_critic_off_never_reviews(fake_llm):
    calls = fake_llm([FIRST], [])
    state = graph.run("q", skip_report=True)
    assert kinds(calls) == ["planner", "data"] and state["returned"] == "no_critic"


def test_ok_on_first_round(fake_llm):
    calls = fake_llm([FIRST], [OK])
    state = graph.run("q", skip_report=True, critic=True)
    assert kinds(calls) == ["planner", "data", "critic"] and calls[2][1] == CRITIC_MODEL
    assert state["sql"] == FIRST and state["returned"] == "ok" and len(state["critic_rounds"]) == 1


def test_critic_prompt_has_schema_rules_and_result(fake_llm):
    calls = fake_llm([FIRST], [OK])
    graph.run("q", skip_report=True, critic=True)
    prompt = calls[2][2]
    assert "product_category_name" in prompt and "customer_unique_id" in prompt
    assert FIRST in prompt and "x\n1" in prompt and "1 row(s)." in prompt


def test_revise_then_ok(fake_llm):
    calls = fake_llm([FIRST, SECOND], [revise("count per order"), OK])
    state = graph.run("q", skip_report=True, critic=True)
    rewrite = calls[3][2]
    assert "count per order" in rewrite and FIRST in rewrite
    assert state["sql"] == SECOND and state["returned"] == "ok"
    assert [r["verdict"] for r in state["critic_rounds"]] == ["revise", "ok"]


def test_three_flags_fall_back_to_first_ok_sql(fake_llm):
    calls = fake_llm([FIRST, SECOND, THIRD], [revise(), revise(), revise()])
    state = graph.run("q", skip_report=True, critic=True)
    assert kinds(calls).count("critic") == 3 and kinds(calls).count("data") == 3
    assert state["sql"] == FIRST and state["returned"] == "first_ok_fallback" and state["rows"] == [(1,)]


def test_same_sql_after_feedback_stops_as_data_disagreed(fake_llm):
    calls = fake_llm([FIRST, "  select 1   AS x\n"], [revise()])
    state = graph.run("q", skip_report=True, critic=True)
    assert kinds(calls).count("critic") == 1
    assert state["returned"] == "data_disagreed" and state["sql"] == FIRST


def test_failed_rewrite_falls_back_to_first_ok_sql(fake_llm):
    fake_llm([FIRST, BAD], [revise()])
    state = graph.run("q", skip_report=True, critic=True)
    assert state["sql"] == FIRST and state["returned"] == "first_ok_fallback" and state["error"] is None


def test_unparseable_reply_counts_as_ok(fake_llm):
    fake_llm([FIRST], ["looks fine to me"])
    state = graph.run("q", skip_report=True, critic=True)
    assert state["returned"] == "ok" and state["critic_rounds"][0]["parse_error"]


def test_result_facts():
    assert critic.result_facts(["x"], []) == "The result is empty (0 rows)."
    assert "NULL in every row: b" in critic.result_facts(["a", "b"], [(1, None), (2, None)])


def test_eval_labels_false_alarm_and_caught(fake_llm):
    # gold is SELECT 1: round 1 flags a correct SQL (false alarm), round 2 flags a wrong one (caught)
    fake_llm([FIRST, SECOND, THIRD], [revise(), revise(), OK])
    q = {"id": "x", "question": "q", "gold_sql": "SELECT 1", "check": {"type": "scalar"}}
    r = evaluate(q, lambda question: graph.answer(question, critic=True))
    outcomes = [rnd["outcome"] for rnd in r["trace"]["critic_rounds"]]
    assert outcomes == ["false_alarm", "caught", "missed"] and not r["passed"]
    totals = critic_totals([r])
    assert totals["false_alarm"] == 1 and totals["caught"] == 1 and totals["missed"] == 1
    assert totals["returned"] == {"ok": 1}


def test_critic_results_get_own_filename():
    assert answerer_slug("src.graph:answer+critic") == "graph-critic"
    assert answerer_slug("src.graph:answer") == "graph"


def test_critic_prompt_has_no_plan_and_asks_for_checklist():
    prompt = critic.build_prompt("How many?", "SELECT 1", ["n"], [(1,)])
    assert "# Plan" not in prompt and '"checklist"' in prompt


def test_checklist_is_parsed():
    reply = json.dumps({"checklist": [{"requirement": "count products", "sql": "COUNT(*)", "status": "match"}],
                        "verdict": "ok", "feedback": ""})
    verdict, _, checklist, parse_error = critic.parse(reply)
    assert verdict == "ok" and checklist[0]["status"] == "match" and not parse_error
