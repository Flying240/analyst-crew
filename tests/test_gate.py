import json

import pytest

from src import graph, llm
from src.agents import planner
from src.agents.gate import precheck
from src.config import PLANNER_MODEL, REPORTER_MODEL


@pytest.mark.parametrize("question", [
    "", "   ", "revenue", "What's the weather in Sao Paulo?", "write me a poem about sellers",
    "Write a Python function to sort a list", "Who is Elon Musk?",
])
def test_precheck_rejects(question):
    assert precheck(question)


@pytest.mark.parametrize("question", [
    "Who is the top seller by revenue?", "top states", "Which zip code prefix has the most customers?",
    "How many orders?", "What is the profit margin per category?",
])
def test_precheck_passes_everything_else(question):
    assert precheck(question) is None


def test_parse_gate_reads_json():
    reply = 'sure {"answerable": "partly", "reason": "no costs", "plan": "- sum revenue"}'
    assert planner.parse_gate(reply) == ("partly", "no costs", "- sum revenue")


@pytest.mark.parametrize("reply", ["- just a plan", '{"answerable": "maybe"}', "{broken"])
def test_parse_gate_falls_back_to_answerable(reply):
    assert planner.parse_gate(reply) == ("yes", "", reply.strip())


def test_gate_prompt_only_when_on():
    assert "JSON only" not in planner.build_prompt("q") and planner.build_prompt("q").endswith("Plan:")
    assert "JSON only" in planner.build_prompt("q", gate=True)


def gate_llm(monkeypatch, answerable: str, sql: str = "SELECT 1 AS x FROM (SELECT 1 FROM orders LIMIT 1)") -> list:
    calls = []

    def complete(prompt, model=None, **_):
        calls.append((model, prompt))
        if model == PLANNER_MODEL:
            return json.dumps({"answerable": answerable, "reason": "no cost data", "plan": "- a plan"})
        return "a report"

    monkeypatch.setattr(llm, "complete", complete)
    monkeypatch.setattr(llm, "ask", lambda p, m=None: calls.append((m, p)) or sql)
    return calls


def test_gate_no_stops_before_data(monkeypatch):
    calls = gate_llm(monkeypatch, "no")
    state = graph.run("profit?", gate=True)
    assert len(calls) == 1 and state["returned"] == "unanswerable" and "sql" not in state


def test_gate_partly_continues_with_caveat(monkeypatch):
    calls = gate_llm(monkeypatch, "partly")
    state = graph.run("q", gate=True)
    assert state["sql"] == "SELECT 1 AS x FROM (SELECT 1 FROM orders LIMIT 1)" and state["plan"] == "- a plan"
    reporter_prompt = [p for m, p in calls if m == REPORTER_MODEL][0]
    assert "only partly answers" in reporter_prompt and "no cost data" in reporter_prompt


def test_gate_on_by_default(monkeypatch):
    calls = gate_llm(monkeypatch, "no")
    state = graph.run("q", skip_report=True)
    assert "JSON only" in calls[0][1] and len(calls) == 1 and state["answerable"] == "no"


def test_gate_can_be_turned_off(monkeypatch):
    calls = gate_llm(monkeypatch, "no")
    graph.run("q", skip_report=True, gate=False)
    assert "JSON only" not in calls[0][1] and len(calls) == 2


def test_eval_answer_with_gate_returns_empty_sql_when_rejected(monkeypatch):
    gate_llm(monkeypatch, "no")
    assert graph.answer("profit?", gate=True) == ""


def test_slug_marks_gate():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))
    from run_eval import answerer_slug

    assert answerer_slug("src.graph:answer+gate+critic", 2) == "graph-gate-critic-r2"


def test_parse_gate_accepts_meta():
    assert planner.parse_gate('{"answerable": "meta", "reason": "about the dataset", "plan": ""}')[0] == "meta"


def test_gate_meta_answers_from_description_without_data_agent(monkeypatch):
    calls = gate_llm(monkeypatch, "meta")
    state = graph.run("Which dataset am I using?", gate=True)
    assert len(calls) == 1 and "sql" not in state and state["returned"] == "meta"
    assert "Olist" in state["report"] and "`order_items`" in state["report"]


def test_gate_eval_scores_meta_cases():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))
    from gate_eval import is_correct, load_cases

    assert sum(c["expect"] == "meta" for c in load_cases()) == 3
    assert is_correct("meta", "meta") and not is_correct("meta", "yes")
    assert is_correct("yes", "partly") and not is_correct("yes", "meta")


def test_partly_keeps_suggestion_and_reporter_does_not_repeat_caveat(monkeypatch):
    from src.agents import reporter

    def complete(prompt, model=None, **_):
        if model == PLANNER_MODEL:
            return json.dumps({"answerable": "partly", "reason": "no costs", "plan": "- p", "suggestion": "Revenue per category?"})
        return "a report"

    monkeypatch.setattr(llm, "complete", complete)
    update = planner.plan({"question": "profit?", "use_gate": True})
    assert update["suggestion"] == "Revenue per category?"
    prompt = reporter.build_prompt({"question": "q", "columns": ["n"], "rows": [(1,)], **update})
    assert "do not repeat it" in prompt and "State this caveat" not in prompt


def test_suggestion_only_for_partly(monkeypatch):
    monkeypatch.setattr(llm, "complete", lambda p, m=None: json.dumps({"answerable": "yes", "reason": "", "plan": "- p", "suggestion": "x"}))
    assert "suggestion" not in planner.plan({"question": "q", "use_gate": True})
    assert planner.parse_suggestion("not json") == ""
