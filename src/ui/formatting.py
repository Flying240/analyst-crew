"""Number formatting shared by tables, metrics and charts: thousands separators, money as R$ with 2 decimals."""
import re
from typing import Any

import pandas as pd

MONEY_FORMAT = "R$ %,.2f"
MONEY_RE = re.compile(r"revenue|price|freight|payment_value|value|amount|spend|sales|gmv|aov|ticket|cost", re.I)
NOT_MONEY_RE = re.compile(r"count|share|pct|percent|ratio|rate|score|qty|days|hours|_n$|^n_|num|installments", re.I)


def is_money(column: str) -> bool:
    return bool(MONEY_RE.search(column)) and not NOT_MONEY_RE.search(column)


def number_format(column: str, values: pd.Series) -> str | None:
    """printf format for a numeric column, or None for non-numeric ones."""
    if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
        return None
    if is_money(column):
        return MONEY_FORMAT
    return "%,d" if pd.api.types.is_integer_dtype(values) else "%,.2f"


def column_config(df: pd.DataFrame) -> dict[str, Any]:
    """st.dataframe column_config: separators on every numeric column, R$ on money columns."""
    import streamlit as st

    return {
        c: st.column_config.NumberColumn(format=f)
        for c in df.columns if (f := number_format(str(c), df[c])) is not None
    }


def number(value: float | int, money: bool = False) -> str:
    """Text for metrics and captions: 12,345 / 1,234.57 / R$ 1,234.57."""
    if money:
        return f"R$ {value:,.2f}"
    return f"{value:,}" if isinstance(value, int) else f"{value:,.2f}"


def style_chart(fig: Any, y_column: str) -> Any:
    """Plotly axis and hover formatting to match the tables."""
    money = is_money(y_column)
    fig.update_yaxes(tickformat=",.2f" if money else ",", tickprefix="R$ " if money else "")
    fig.update_traces(hovertemplate="%{x}<br>" + ("R$ %{y:,.2f}" if money else "%{y:,}") + "<extra></extra>")
    return fig
