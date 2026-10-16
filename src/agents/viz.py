"""Viz agent: picks a Plotly chart from the result shape (no LLM)."""
import re
from typing import Any

from src.state import CrewState

MONTH_RE = re.compile(r"^\d{4}-\d{2}$")


def chart_kind(columns: list[str], rows: list[tuple[Any, ...]]) -> str | None:
    """'line' for YYYY-MM x values, 'bar' for categories, None for scalars or unchartable results."""
    if len(rows) <= 1 or len(columns) < 2:
        return None
    if not all(isinstance(r[1], (int, float)) for r in rows):
        return None
    if all(isinstance(r[0], str) and MONTH_RE.match(r[0]) for r in rows):
        return "line"
    return "bar"


def make_chart(state: CrewState) -> CrewState:
    columns, rows = state.get("columns", []), state.get("rows", [])
    kind = chart_kind(columns, rows)
    if kind is None:
        return {"chart": None}
    import plotly.express as px

    data = {columns[0]: [r[0] for r in rows], columns[1]: [r[1] for r in rows]}
    draw = px.line if kind == "line" else px.bar
    return {"chart": draw(data, x=columns[0], y=columns[1])}
