import pandas as pd
import pytest

from src.agents import viz
from src.ui import formatting, schema_info
from src.ui.trace_view import summary_line


@pytest.mark.parametrize("column", ["revenue", "total_revenue", "avg_price", "freight_value", "payment_value", "avg_order_value"])
def test_money_columns(column):
    assert formatting.is_money(column)


@pytest.mark.parametrize("column", ["order_count", "review_score", "late_pct", "payment_installments", "n", "month"])
def test_not_money_columns(column):
    assert not formatting.is_money(column)


def test_number_formats():
    df = pd.DataFrame({"category": ["a"], "revenue": [1234.5], "orders": [12345], "avg_days": [3.14159]})
    formats = {c: formatting.number_format(c, df[c]) for c in df.columns}
    assert formats == {"category": None, "revenue": "R$ %,.2f", "orders": "%,d", "avg_days": "%,.2f"}
    assert set(formatting.column_config(df)) == {"revenue", "orders", "avg_days"}


def test_number_text():
    assert formatting.number(1234567) == "1,234,567"
    assert formatting.number(1234.5, money=True) == "R$ 1,234.50"


def test_chart_axes_use_separators_and_reais():
    money = viz.make_chart({"columns": ["month", "revenue"], "rows": [("2017-01", 1000.0), ("2017-02", 2000.0)]})["chart"]
    assert money.layout.yaxis.tickprefix == "R$ " and money.layout.yaxis.tickformat == ",.2f"
    count = viz.make_chart({"columns": ["state", "orders"], "rows": [("SP", 40000), ("RJ", 12000)]})["chart"]
    assert count.layout.yaxis.tickformat == "," and not count.layout.yaxis.tickprefix


def test_summary_line():
    steps = [
        {"node": "planner", "calls": [{"prompt": 1000, "completion": 200}]},
        {"node": "data", "calls": [{"prompt": 2000, "completion": 10}]},
        {"node": "reporter", "calls": []},
    ]
    assert summary_line(steps, 4.25, "no_critic") == "Planner → Data (SQL) → Reporter · 3,210 tokens · 4.2s · Critic off"
    eval_steps = [{"node": "data", "calls": []}]
    assert summary_line(eval_steps, None, None, {"prompt": 1500, "completion": 34}) == "Data (SQL) · 1,534 tokens in the eval"


def test_er_diagram_uses_light_boxes_dark_text():
    dot = schema_info.er_dot()
    assert f'fillcolor="{schema_info.BOX_FILL}"' in dot and f'fontcolor="{schema_info.BOX_TEXT}"' in dot
    assert 'edge [color="' in dot
    # boxes hold only name, rows and PK/FK lines
    label = next(line for line in dot.splitlines() if line.startswith("  customers [label="))
    assert label.count(r"\n") == 2 and "customer_city" not in label
