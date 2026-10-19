import pytest

from evals.run_eval import build_report, merge_reports
from src import graph, llm, usage_ledger

IDS = ["q1", "q2", "q3"]
META = {"retries": 2, "models": {"planner": "a", "data": "a", "critic": "b"}, "code_version": "abc123"}


def result(qid: str, passed: bool = True) -> dict:
    return {"id": qid, "difficulty": "easy", "split": "dev", "passed": passed, "infra_error": False,
            "tokens": {"prompt": 10, "completion": 1}}


def part(ids: list[str], answerer: str = "src.graph:answer+gate+critic", **meta) -> dict:
    return build_report([result(i, i != "q2") for i in ids], answerer, True, {**META, **meta})


def test_merge_combines_parts_in_question_order():
    merged = merge_reports([part(["q3", "q2"]), part(["q1"])], IDS)
    assert [r["id"] for r in merged["results"]] == IDS
    assert merged["merged"] and merged["complete"] and merged["parts"] == 2
    assert (merged["score"], merged["total"]) == (2, 3) and merged["tokens"]["prompt"] == 30
    assert merged["code_version"] == "abc123" and merged["answerer"] == "src.graph:answer+gate+critic"


@pytest.mark.parametrize("parts, message", [
    ([part(["q1", "q2"]), part(["q2", "q3"])], "q2 is in part 1 and part 2"),
    ([part(["q1"]), part(["q2"])], "missing ['q3']"),
    ([part(["q1"]), part(["q2", "q3"], answerer="src.graph:answer+critic")], "differ in answerer"),
    ([part(["q1"]), part(["q2", "q3"], retries=0)], "differ in retries"),
    ([part(["q1"]), part(["q2", "q3"], models={"planner": "a", "data": "a", "critic": "c"})], "differ in models"),
    ([part(["q1"]), part(["q2", "q3"], code_version="def456")], "differ in code_version"),
    ([part(["q1", "q2", "q3"])], "at least two"),
])
def test_merge_refuses(parts, message):
    with pytest.raises(ValueError, match=message.replace("[", r"\[").replace("]", r"\]")):
        merge_reports(parts, IDS)


def test_merge_refuses_incomplete_and_old_parts():
    incomplete = {**part(["q2", "q3"]), "complete": False}
    with pytest.raises(ValueError, match="incomplete"):
        merge_reports([part(["q1"]), incomplete], IDS)
    old = part(["q2", "q3"])
    del old["code_version"]
    with pytest.raises(ValueError, match="no 'code_version'"):
        merge_reports([part(["q1"]), old], IDS)


def test_eval_runs_ignore_the_local_budget_stop(monkeypatch):
    """Only the app checks the 90% budget; graph.answer (used by evals) runs even with the ledger over budget."""
    monkeypatch.setattr(usage_ledger, "tokens_today", lambda *a, **k: 10**9)
    monkeypatch.setattr(llm, "complete", lambda prompt, model=None: "- a plan")
    monkeypatch.setattr(llm, "ask", lambda prompt, model=None: "SELECT 1 AS x FROM (SELECT 1 FROM orders LIMIT 1)")
    assert graph.answer("q", gate=False).startswith("SELECT 1")
