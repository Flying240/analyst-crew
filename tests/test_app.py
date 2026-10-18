"""Streamlit AppTest smoke tests: cached examples render with no LLM calls."""
import pytest
from streamlit.testing.v1 import AppTest

from src import llm


@pytest.fixture(autouse=True)
def no_llm(monkeypatch, tmp_path):
    from src import run_cache, usage_ledger

    monkeypatch.setattr(run_cache, "RUNS_DIR", tmp_path)
    ledger_root = tmp_path / "ledger"  # empty ledger, so the real daily budget never blocks live runs
    real_path = usage_ledger.ledger_path
    monkeypatch.setattr(usage_ledger, "ledger_path", lambda day, root=None: real_path(day, ledger_root))

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


def test_about_page_renders_via_navigation():
    at = AppTest.from_file("../app.py", default_timeout=30).run()
    assert not at.exception
    at.switch_page("app_pages/about.py").run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Why this app", "How it works", "How well it works", "Limits & FAQ"]
    assert any(s.value == "Why not just ask a general chatbot?" for s in at.subheader)
    assert any("m19_avg_freight_per_order" in m.value for m in at.markdown)  # computed from evals/summaries
    assert any("already gets about 94% of questions right" in m.value for m in at.markdown)
    assert len(at.get("graphviz_chart")) == 1 and len(at.expander) == 7  # pipeline diagram, 7 FAQ answers


@pytest.mark.parametrize("page", ["app_pages/about.py", "app_pages/schema_explorer.py", "app_pages/eval_dashboard.py"])
def test_footer_on_every_page(page):
    from src.ui.footer import footer_text

    at = AppTest.from_file("../app.py", default_timeout=30).run()
    assert any(c.value == footer_text() for c in at.caption)  # Ask (default page)
    at.switch_page(page).run()
    assert not at.exception and any(c.value == footer_text() for c in at.caption)


def test_sidebar_has_no_models_or_about():
    at = ask_page()
    assert not any(s.value in ("Models",) for s in at.sidebar.subheader) and not at.sidebar.expander
    assert any("since tracking started" in c.value for c in at.sidebar.caption)


def test_partly_shows_caveat_once_and_suggestion(monkeypatch):
    import json

    from src.config import PLANNER_MODEL

    caveat = "There is no cost data, so profit can't be computed."
    suggestion = "What is the revenue per product category?"

    def complete(prompt, model=None):
        if model == PLANNER_MODEL:
            return json.dumps({"answerable": "partly", "reason": caveat, "plan": "- sum revenue", "suggestion": suggestion})
        return "Revenue by category is shown above."

    monkeypatch.setattr(llm, "complete", complete)
    monkeypatch.setattr(llm, "ask", lambda p, m=None: "SELECT COUNT(*) AS n FROM orders")
    at = ask_page()
    at.text_area[0].input("What is the profit per category?")
    next(b for b in at.button if b.label == "Ask").click().run()
    assert not at.exception
    assert [i.value for i in at.info].count(caveat) == 1
    assert any(b.label == f"Try instead: {suggestion}" for b in at.button)


def test_schema_explorer_shows_overview_er_keys_traps_and_stats():
    at = AppTest.from_file("../app_pages/schema_explorer.py", default_timeout=30).run()
    assert not at.exception
    overview, all_columns, columns = (d.value for d in at.dataframe[:3])
    assert len(overview) == 9 and {"rows", "columns", "description"} <= set(overview.columns)
    assert len(all_columns) == 9 and "customer_unique_id" in " ".join(all_columns["columns"])
    assert {"type", "% NULL", "distinct", "examples"} <= set(columns.columns)
    assert len(at.get("graphviz_chart")) == 1
    at.selectbox[0].select("customers").run()
    assert any("customer_unique_id" in w.value for w in at.warning)
    assert any("`orders.customer_id` → `customers.customer_id`" in m.value for m in at.markdown)
    assert len(at.dataframe) == 4  # overview, all columns, column stats, sample rows


def test_empty_state_offers_three_questions():
    at = ask_page()
    assert any("crew of AI agents" in m.value for m in at.markdown)
    starters = [b for b in at.button if b.key and b.key.startswith("start_")]
    assert len(starters) == 3
    starters[0].click().run()
    assert not at.exception and any("cached from" in m.value for m in at.markdown)


def test_sidebar_token_usage_first_and_examples_by_difficulty():
    at = ask_page()
    assert at.sidebar.subheader[0].value == "Token usage today"
    labels = [m.value for m in at.sidebar.markdown]
    assert labels.index("**Easy**") < labels.index("**Medium**") < labels.index("**Hard**")
    assert at.sidebar.selectbox[0].label.endswith("easy questions")


def test_record_order_answer_summary_sql_chart_table_trace():
    at = ask_page()
    at.text_area[0].input("Show monthly revenue for the complete months (Jan 2017 to Aug 2018).")  # cached, has a chart
    next(b for b in at.button if b.label == "Ask").click().run()
    assert not at.exception
    assert [s.value for s in at.main.subheader] == ["Answer", "Result"]
    labels = [e.label for e in at.main.expander]
    assert labels[0] == "Final SQL" and labels[1] == "How the agents got here"  # agent cards are nested after it
    trace = at.main.expander[1]
    assert trace.proto.expanded and all("**" in e.label for e in trace.expander)
    assert any(c.value.startswith("Data (SQL) · ") and "tokens in the eval" in c.value for c in at.main.caption)
    assert len(at.get("download_button")) == 1 and len(at.get("plotly_chart")) == 1
