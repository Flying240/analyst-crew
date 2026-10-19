"""About: why the app exists, how it works, how well it works (from committed eval summaries), limits and FAQ."""
import pandas as pd
import streamlit as st

from src.config import LLM_PROVIDER
from src.tools.sql_tool import run_sql
from src.ui import eval_story
from src.ui.schema_info import BOX_FILL, BOX_TEXT, LINE
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
    "planner": "Checks that the data can answer the question, then writes a short plan of the logic.",
    "data": "Writes the SQL from the plan and runs it on a read-only copy of the database. If the database returns an error, it fixes the query and tries again.",
    "critic": "Optional. Compares the SQL with the question point by point and sends it back if something doesn't match (at most 3 rounds).",
    "viz": "Picks a chart from the shape of the result: a line for months, bars for categories, none for a single number.",
    "reporter": "Writes a short plain-English answer using only the numbers in the result.",
}
WHY = {
    "planner": "Larger model: understanding the question and the data is the hard part.",
    "data": "Larger model: correct SQL needs the strongest reasoning.",
    "critic": "A different model family from the agents it checks, so it is less likely to repeat their mistakes; it caught as many wrong queries as the larger model in tests.",
    "viz": "Rules, no AI model: the chart type follows from the result's shape.",
    "reporter": "Smaller model: a few sentences from a small table is easy, so it's faster and cheaper.",
}

# Light boxes with dark text and mid-gray lines read on both light and dark themes (same colors as the ER diagram).
PIPELINE = f"""
digraph {{
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="{BOX_FILL}", color="{LINE}", fontcolor="{BOX_TEXT}", fontname="Helvetica", fontsize=11];
  edge [color="{LINE}", fontcolor="{LINE}", fontname="Helvetica", fontsize=10];
  Q [label="Your question"]; A [label="Answer, SQL,\\nchart and table"];
  Stop [label="Explains why it\\ncan't answer"]; Meta [label="Describes\\nthe dataset"];
  Q -> Planner -> Data;
  Planner -> Stop [label="not in the data"];
  Planner -> Meta [label="about the data"];
  Data -> Critic [label="if on"];
  Critic -> Data [label="revise", style=dashed];
  Critic -> Viz [label="ok"]; Data -> Viz;
  Viz -> Reporter -> A;
}}
"""


@st.cache_data(ttl=3600)
def row_counts() -> dict[str, int | None]:
    return {t: (r.rows[0][0] if (r := run_sql(f'SELECT COUNT(*) FROM "{t}"')).ok else None) for t in TABLES}


@st.cache_data(ttl=600)
def summaries() -> list[dict]:
    return eval_story.load_summaries()


def why_tab() -> None:
    st.markdown(
        "This app answers questions about an online store's orders, deliveries, reviews, customers and sellers. "
        "You ask in plain English, and it gives you the answer, the chart and the exact query behind it."
    )
    st.subheader("Why not just ask a general chatbot?")
    st.markdown(
        "- **A chatbot doesn't have your data.** It can only guess from what it saw during training.\n"
        "- **Uploading business data is a privacy risk.** Here the data stays in a database the app controls. "
        "The AI model sees the question and the table layout, plus the small result table for the written answer.\n"
        "- **Chatbots estimate numbers, and can be confidently wrong.** Here every number comes from a SQL query on the "
        "real data, and the app shows you that query so you can check it.\n"
        "- **The data is too large to paste.** This dataset has about 1.5 million rows across 9 tables.\n"
        "- **It can't change anything.** The database is opened read-only, and only a single read query is allowed."
    )
    st.subheader("The data")
    st.markdown(
        "The [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) "
        "holds about 100k orders placed on the Olist marketplace in Brazil from **September 2016 to October 2018**. "
        "Customer and seller data is anonymized. 2016 and September–October 2018 are incomplete, so trend questions leave them out."
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


def how_tab() -> None:
    st.markdown("A small team of AI agents works through each question, each with one job.")
    st.graphviz_chart(PIPELINE, alt="Pipeline: question to Planner, Data (with optional Critic loop), Viz and Reporter, then the answer")
    for node, text in STEPS.items():
        st.markdown(f"{AGENTS[node][1]} **{AGENTS[node][0]}:** {text}")
    st.subheader("Models, and why")
    st.dataframe(
        pd.DataFrame([{"agent": AGENTS[n][0], "model": model_for(n), "why": WHY[n]} for n in AGENTS]),
        hide_index=True,
    )
    if LLM_PROVIDER == "gemini":
        st.caption("With Gemini, every agent uses the same model.")


def score_tab() -> None:
    runs = summaries()
    updated = eval_story.last_updated(runs)
    st.markdown(
        "The app is tested on 50 hand-checked questions (easy, medium and hard), each with a known correct answer. "
        "A question passes only if the returned rows match. Some questions are kept aside and never used while "
        "tuning prompts (held-out), to check that improvements are general."
    )
    st.subheader("Each step, measured")
    st.dataframe(
        pd.DataFrame(eval_story.ladder(runs)), hide_index=True,
        column_config={
            "mean score %": st.column_config.NumberColumn(format="%.1f%%"),
            "tokens / question": st.column_config.NumberColumn(format="%,d"),
            "what it adds": st.column_config.TextColumn(width="large"),
        },
    )
    st.caption(f"Mean over complete 50-question runs. Last updated: {updated or 'no runs yet'}. "
               "Rows fill in as full runs are added to evals/summaries/.")
    st.page_link("app_pages/eval_dashboard.py", label="See every run on the Eval dashboard", icon=":material/monitoring:")

    st.subheader("Why several agents, if one already scores well?")
    retry = next(r for r in eval_story.ladder(runs) if r["step"] == "+ retry")["mean score %"]
    retry_text = f"about {retry:.0f}% of questions" if retry is not None else "most questions"
    st.markdown(
        f"A single model call with an error retry already gets {retry_text} right. The extra agents are not mainly "
        "about that score. The Planner checks whether a question can be answered at all, and plans the logic first. "
        "The Critic can catch SQL that runs fine but answers the wrong thing, which an error retry never sees. "
        "The Viz and Reporter agents turn the result into a chart and a plain answer. Each of these is measured "
        "separately, and the table above will show if they help, hurt, or just cost more tokens."
    )

    st.subheader("Questions that failed in every run")
    failed = eval_story.always_failed(runs)
    if not failed:
        st.markdown("None: every question passed in at least one full run.")
    texts = eval_story.question_text()
    for qid in failed:
        st.markdown(f"**{texts.get(qid, qid)}** (`{qid}`)")
        st.markdown(eval_story.FAILURE_NOTES.get(qid, "Not analysed yet."))
    if failed:
        st.markdown(
            "**Why not add a rule for each failing question?** It would teach the test: the score would go up "
            "without the app getting better at new questions. Instead, the Critic checks the *grain* of every answer "
            "(per order, per item or per customer), which is the general mistake behind these failures. Its effect "
            "is measured on the held-out questions."
        )


FAQ = [
    ("Can it change or delete data?",
     "No. The database is opened read-only, and the app accepts only a single SELECT query. Writes, multiple "
     "statements and admin commands are rejected before anything runs. Queries also stop after a time limit and "
     "return at most a fixed number of rows."),
    ("How do I know an answer is right?",
     "Every answer shows the exact SQL, the result table and a chart, so you can check the logic yourself. "
     "\"How the agents got here\" shows the plan, every query tried and any Critic feedback. The test scores on "
     "this page show how often it is right on questions with known answers."),
    ("What can't it answer?",
     "Anything that isn't in the data: product names, customer age or gender, costs or profit, marketing channels, "
     "or anything after 2018. It tells you when that's the case instead of guessing."),
    ("Why does an answer take 10–15 seconds?",
     "Several AI calls run one after another (plan, SQL, sometimes a review, then the written answer), and each "
     "takes a few seconds. The database query itself is fast. Example questions load instantly because they are cached."),
    ("Does it remember my earlier questions?",
     "No. Each question is answered on its own. Recent answers are cached so the same question doesn't use the AI again."),
    ("Could it work on other data?",
     "Yes, in principle. The agents get the table layout and a few business rules (for example, what counts as "
     "revenue) as text. Another database would need its own layout description, rules and test questions."),
    ("Which AI models does it use, and why?",
     "Open-weight models served by Groq: a larger one for understanding the question and writing SQL, "
     "a model from a different family for reviewing that SQL, and a smaller, faster one for writing the answer. Charts are picked by fixed rules, with no model."),
]


def faq_tab() -> None:
    for question, answer in FAQ:
        with st.expander(question):
            st.markdown(answer)


why, how, score, faq = st.tabs(["Why this app", "How it works", "How well it works", "Limits & FAQ"])
with why:
    why_tab()
with how:
    how_tab()
with score:
    score_tab()
with faq:
    faq_tab()
