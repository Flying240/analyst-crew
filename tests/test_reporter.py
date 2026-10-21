from src import llm
from src.agents import reporter
from src.config import REPORTER_MODEL

H04 = {
    "question": "Which product category had the largest revenue decrease in the month with the biggest drop?",
    "plan": "- Find the month with the biggest month-over-month revenue drop.\n- Compare each category with the month before.",
    "sql": "SELECT worst_month, category, decline FROM ranked LIMIT 1",
    "columns": ["worst_month", "category", "decline"],
    "rows": [("2017-12", "bed_bath_table", 45585.09)],
    "error": None,
}


def stub(monkeypatch) -> list[tuple[str | None, str]]:
    calls: list[tuple[str | None, str]] = []
    monkeypatch.setattr(llm, "complete", lambda p, m=None: calls.append((m, p)) or " The answer. ")
    return calls


def test_prompt_has_question_plan_final_sql_and_result(monkeypatch):
    calls = stub(monkeypatch)
    assert reporter.report(H04) == {"report": "The answer."}
    (model, prompt), = calls
    assert model == REPORTER_MODEL and prompt.endswith("Answer:")
    for part in ("# Question\n" + H04["question"], "# Plan\n" + H04["plan"], "# Final SQL\n" + H04["sql"],
                 "# Query result\nworst_month | category | decline\n2017-12 | bed_bath_table | 45585.09"):
        assert part in prompt
    assert "computed answer from the database. Never judge whether it is sufficient" in prompt
    assert "state the answer directly using the values in the result" in prompt
    assert "Only say data is missing if the result is empty or the query failed." in prompt


def test_empty_result_and_error_are_stated(monkeypatch):
    calls = stub(monkeypatch)
    reporter.report({**H04, "rows": []})
    reporter.report({**H04, "rows": [], "error": "Binder Error: no column x"})
    assert "# Query result\nThe query returned no rows." in calls[0][1]
    assert "# Query result\nThe query failed: Binder Error: no column x" in calls[1][1]
