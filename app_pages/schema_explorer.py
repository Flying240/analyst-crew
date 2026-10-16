"""Schema explorer: tables, columns with types and descriptions, row counts and sample rows (read-only)."""
import re

import pandas as pd
import streamlit as st

from src.config import SCHEMA_PATH
from src.tools.sql_tool import run_sql


@st.cache_data(ttl=3600)
def tables() -> list[str]:
    r = run_sql("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' ORDER BY table_name")
    return [row[0] for row in r.rows]


@st.cache_data(ttl=3600)
def columns(table: str) -> pd.DataFrame:
    r = run_sql(
        "SELECT column_name, data_type FROM information_schema.columns "
        f"WHERE table_schema = 'main' AND table_name = '{table}' ORDER BY ordinal_position"
    )
    return pd.DataFrame(r.rows, columns=["column", "type"])


@st.cache_data(ttl=3600)
def row_count(table: str) -> int:
    return run_sql(f'SELECT COUNT(*) FROM "{table}"').rows[0][0]


@st.cache_data(ttl=3600)
def sample(table: str) -> pd.DataFrame:
    r = run_sql(f'SELECT * FROM "{table}" LIMIT 10')
    return pd.DataFrame(r.rows, columns=r.columns)


@st.cache_data(ttl=3600)
def schema_section(table: str) -> str:
    """The table's section of schema.md, if any."""
    text = SCHEMA_PATH.read_text(encoding="utf-8")
    m = re.search(rf"^#+ .*\b{re.escape(table)}\b.*$(.*?)(?=^#+ |\Z)", text, re.MULTILINE | re.DOTALL)
    return m.group(1).strip() if m else ""


names = tables()
if not names:
    st.error("The DuckDB database isn't available. Build it with `python scripts/build_db.py`.")
    st.stop()

table = st.selectbox("Table", names)
st.metric("Rows", f"{row_count(table):,}", border=True, width="content")
st.subheader("Columns")
st.dataframe(columns(table), hide_index=True)
st.subheader("Sample rows")
st.dataframe(sample(table), hide_index=True)
section = schema_section(table)
if section:
    with st.expander("From schema.md", icon=":material/description:"):
        st.markdown(section)
