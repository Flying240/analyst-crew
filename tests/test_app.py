"""Streamlit AppTest smoke tests: cached examples render with no LLM calls."""
import pytest
from streamlit.testing.v1 import AppTest

from src import llm


@pytest.fixture(autouse=True)
def no_llm(monkeypatch, tmp_path):
    from src import run_cache

    monkeypatch.setattr(run_cache, "RUNS_DIR", tmp_path)

    def fail(*a, **k):
        raise AssertionError("LLM called")

    monkeypatch.setattr(llm, "complete", fail)
    monkeypatch.setattr(llm, "ask", fail)


def ask_page() -> AppTest:
    at = AppTest.from_file("../app_pages/ask.py", default_timeout=30)
    return at.run()


def test_page_loads():
    at = ask_page()
    assert not at.exception and at.toggle[0].label == "Critic loop"


def test_example_click_renders_cached_trace_and_result():
    at = ask_page()
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert any("cached from" in m.value for m in at.markdown)
    assert at.dataframe, "result table shown"


def test_off_topic_question_is_rejected_without_llm():
    at = ask_page()
    at.text_area[0].input("What's the weather in Sao Paulo?")
    next(b for b in at.button if b.label == "Ask").click().run()
    assert not at.exception and at.warning


def test_live_run_streams_cards_saves_and_then_hits_cache(monkeypatch, tmp_path):
    import json

    from src import run_cache
    from src.config import PLANNER_MODEL

    calls = []

    def complete(prompt, model=None):
        calls.append(model)
        if model == PLANNER_MODEL:
            return json.dumps({"answerable": "yes", "reason": "orders table", "plan": "- count orders"})
        return "There are N orders."

    monkeypatch.setattr(llm, "complete", complete)
    monkeypatch.setattr(llm, "ask", lambda p, m=None: calls.append(m) or "SELECT COUNT(*) AS n FROM orders")
    at = ask_page()
    at.text_area[0].input("How many orders are there in total?")
    next(b for b in at.button if b.label == "Ask").click().run()
    assert not at.exception
    assert any("live run" in m.value for m in at.markdown)
    assert any("There are N orders." in m.value for m in at.markdown)
    assert len(list(tmp_path.glob("*.json"))) == 1 and len(calls) == 3
    next(b for b in at.button if b.label == "Ask").click().run()
    assert len(calls) == 3 and any("cached from" in m.value for m in at.markdown)


@pytest.mark.parametrize("page", ["eval_dashboard", "schema_explorer"])
def test_other_pages_render(page):
    at = AppTest.from_file(f"../app_pages/{page}.py", default_timeout=30).run()
    assert not at.exception and at.dataframe
