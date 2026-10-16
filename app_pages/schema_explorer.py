"""Schema explorer: table overview, ER diagram, keys, known traps, column stats and sample rows (read-only).

Keys and traps come from schema_meta.json; column stats are precomputed in schema_stats.json
(scripts/export_schema_stats.py), so only the sample rows query the database.
"""
import re

import pandas as pd
import streamlit as st

from src.config import SCHEMA_PATH
from src.tools.sql_tool import run_sql
from src.ui import schema_info


@st.cache_data(ttl=3600)
def sample(table: str) -> pd.DataFrame | None:
    r = run_sql(f'SELECT * FROM "{table}" LIMIT 10')
    return pd.DataFrame(r.rows, columns=r.columns) if r.ok else None


@st.cache_data(ttl=3600)
def schema_section(table: str) -> str:
    """The table's section of schema.md, if any."""
    text = SCHEMA_PATH.read_text(encoding="utf-8")
    m = re.search(rf"^#+ .*\b{re.escape(table)}\b.*$(.*?)(?=^#+ |\Z)", text, re.MULTILINE | re.DOTALL)
    return m.group(1).strip() if m else ""


st.subheader("Tables")
st.dataframe(pd.DataFrame(schema_info.overview()), hide_index=True, column_config={
    "rows": st.column_config.NumberColumn(format="localized"),
    "description": st.column_config.TextColumn(width="large"),
})

st.subheader("How the tables join")
st.caption("Arrows point from the referenced key (one) to the table that holds the foreign key (many). "
           "Dashed: zip-prefix links to geolocation, which has many rows per prefix. "
           "Keys are defined in schema_meta.json, because the CSV-loaded DuckDB has none declared.")
st.graphviz_chart(schema_info.er_dot())

st.subheader("Table details")
table = st.selectbox("Table", sorted(schema_info.meta()))
info, stats = schema_info.meta()[table], schema_info.stats()[table]
st.markdown(info["description"])
with st.container(horizontal=True):
    st.metric("Rows", f"{stats['rows']:,}", border=True)
    st.metric("Columns", len(stats["columns"]), border=True)
    st.metric("Primary key", ", ".join(info["primary_key"]) or "none", border=True)

relations = schema_info.links(table)
if relations:
    st.markdown("**Relationships**")
    st.markdown("\n".join(
        f"- `{child}` → `{parent}`" + (" (approximate: zip prefix, many-to-many)" if soft else "")
        for child, parent, soft in relations
    ))
if info["traps"]:
    st.markdown("**Known traps**")
    for trap in info["traps"]:
        st.warning(trap, icon=":material/warning:")

st.markdown("**Columns**")
st.dataframe(pd.DataFrame(schema_info.column_table(table)), hide_index=True, column_config={
    "% NULL": st.column_config.NumberColumn(format="%.2f%%"),
    "distinct": st.column_config.NumberColumn(format="localized"),
})
st.caption("Examples are the most frequent values. Stats are precomputed by scripts/export_schema_stats.py.")

st.markdown("**Sample rows**")
rows = sample(table)
if rows is None:
    st.info("The database isn't available, so sample rows can't be shown.")
else:
    st.dataframe(rows, hide_index=True)
section = schema_section(table)
if section:
    with st.expander("From schema.md", icon=":material/description:"):
        st.markdown(section)
