"""About: the dataset, how the agents work, which models they use, and the eval setup."""
import pandas as pd
import streamlit as st

from src.config import LLM_PROVIDER, PROMPT_SCHEMA, REASONING_EFFORT
from src.tools.sql_tool import run_sql
from src.ui.trace_view import AGENTS, model_for

TABLES = {
    "orders": "one row per order: status and purchase, approval, carrier, delivery and estimated delivery timestamps",
    "order_items": "one row per item in an order: product, seller, price and freight (revenue)",
    "payments": "payment type, installments and value per order",
    "reviews": "1-5 review score and comment text (mostly Portuguese) per order",
    "customers": "city, state and zip code prefix; customer_unique_id identifies a person across orders",
    "sellers": "city, state and zip code prefix per seller",
    "products": "category (Portuguese), weight, size and photo count; no product names",
    "category_translation": "Portuguese category name to English",
    "geolocation": "latitude and longitude per zip code prefix",
}
STEPS = {
    "planner": "Checks the question can be answered from the data, then writes a short plan of the logic (no SQL).",
    "data": "Writes DuckDB SQL from the plan and runs it read-only; on a database error it retries with the error.",
    "critic": "Optional: checks the SQL against the question as a checklist and sends it back with feedback (max 3 rounds).",
    "viz": "Picks a chart from the result shape: line for months, bar for categories, none for a single value.",
    "reporter": "Writes a 2-4 sentence plain-English answer from the result.",
}
WHY = {
    "planner": "Big model: understanding the question and the schema is the hard part.",
    "data": "Big model: writing correct SQL needs the strongest reasoning.",
    "critic": "Big model: it has to spot subtle logic mistakes in the SQL.",
    "viz": "Rules, no LLM: the chart type follows from the result shape, so an LLM adds cost and nothing else.",
    "reporter": "Small model: turning a small table into a few sentences is easy, so it's faster and cheaper.",
}

PIPELINE = """
digraph {
  rankdir=LR; node [shape=box, style=rounded, fontname="sans-serif"]; edge [fontname="sans-serif", fontsize=10];
  Q [label="Question", shape=plaintext];
  Q -> Planner -> Data;
  Planner -> Stop [label="not answerable"]; Stop [label="Friendly message", shape=plaintext];
  Data -> Critic [label="if on"];
  Critic -> Data [label="revise", style=dashed];
  Critic -> Viz [label="ok"]; Data -> Viz;
  Viz -> Reporter -> A; A [label="Answer", shape=plaintext];
}
"""


@st.cache_data(ttl=3600)
def row_counts() -> dict[str, int]:
    return {t: (r.rows[0][0] if (r := run_sql(f'SELECT COUNT(*) FROM "{t}"')).ok else None) for t in TABLES}


st.subheader("About the data")
st.markdown(
    "The [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) "
    "holds about 100k orders placed on the Olist marketplace in Brazil from **September 2016 to October 2018**, "
    "with their items, payments, reviews, customers, sellers and products. Customer and seller data is anonymized. "
    "2016 and September-October 2018 are incomplete months, so trend questions leave them out."
)
counts = row_counts()
st.dataframe(
    pd.DataFrame([{"table": t, "rows": counts.get(t), "covers": d} for t, d in TABLES.items()]),
    hide_index=True, column_config={"rows": st.column_config.NumberColumn(format="%,d")},
)
st.caption(
    "Data: Olist, published on Kaggle under the CC BY-NC-SA 4.0 license (attribution, non-commercial, share-alike). "
    "This app is a non-commercial demo and is not affiliated with Olist."
)

st.subheader("How it works")
st.graphviz_chart(PIPELINE)
for node, text in STEPS.items():
    st.markdown(f"{AGENTS[node][1]} **{AGENTS[node][0]}:** {text}")

st.subheader("Models")
st.dataframe(
    pd.DataFrame([{"agent": AGENTS[n][0], "model": model_for(n), "why": WHY[n]} for n in AGENTS]),
    hide_index=True,
)
if LLM_PROVIDER == "gemini":
    st.caption("With Gemini, every agent uses the same model (GEMINI_MODEL).")

st.subheader("Eval setup")
st.markdown(
    "Every score uses one frozen setup, so runs stay comparable:\n"
    f"- **Schema:** compact schema in the prompts (now `{PROMPT_SCHEMA}`)\n"
    f"- **Reasoning effort:** low (now `{REASONING_EFFORT}`)\n"
    "- **Temperature:** 0\n"
    "- **Questions:** 50 hand-checked questions with gold SQL (dev and held-out splits), scored on the result rows"
)
st.page_link("app_pages/eval_dashboard.py", label="Open the Eval dashboard", icon=":material/monitoring:")
