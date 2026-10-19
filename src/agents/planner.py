"""Planner: turns the question into a short plan of the logic (no SQL).

With the question gate on, it also decides whether the dataset can answer the question and replies with JSON
{answerable: yes|partly|no|meta, reason, plan}.
"""
import json
import re

from src import llm
from src.agents.baseline import RULES, load_schema
from src.agents.gate import DATASET_LIMITS, DATASET_SUMMARY
from src.config import PLANNER_GATE, PLANNER_MODEL
from src.state import CrewState
from src.tools.sql_tool import table_names

INSTRUCTIONS = """\
Write a short plan (at most 6 bullet points) describing only the logic needed to answer the question:
- what to count, sum or average, and at what grain (e.g. per order, per item, per customer), stated in the question's own terms
- which tables and columns hold that data
- which filters apply: only filters stated in the question or the project rules; never add others
- grouping and ordering, if any
Use the project rules for definitions such as revenue, customers and late delivery.
Never name SQL functions or write SQL."""

GATE = f"""\
First decide whether this database can answer the question.
The tables are listed in the schema above. {DATASET_LIMITS}
- "yes": the tables above hold the data the question needs.
- "partly": part of the question can be answered, or only with a stated approximation; say what is missing.
- "no": the question needs data the database does not have, or is not about this data.
- "meta": the question is about the dataset itself: what data or tables it has, its schema, or what can be asked.
  Use an empty plan.
Prefer "yes" when the question is a normal analysis of orders, revenue, deliveries, reviews, customers, sellers,
products, payments or locations, even if it is phrased casually.
For "partly", also suggest a closely related question that this database can answer fully.
Reply with JSON only:
{{"answerable": "yes" | "partly" | "no" | "meta", "reason": "<one sentence>", "plan": "<the bullet-point plan; empty if no>",
"suggestion": "<for partly: a related fully answerable question; otherwise empty>"}}"""

ANSWERABLE = ("yes", "partly", "no", "meta")


def build_prompt(question: str, gate: bool = False) -> str:
    task = f"{INSTRUCTIONS}\n\n{GATE}" if gate else INSTRUCTIONS
    return (
        "You are planning an analysis of the Olist e-commerce DuckDB database.\n\n"
        f"# Schema\n{load_schema()}\n\n# Project rules\n{RULES}\n\n"
        f"# Task\n{task}\n\n# Question\n{question}\n\n{'JSON' if gate else 'Plan'}:"
    )


def parse_gate(text: str) -> tuple[str, str, str]:
    """Return (answerable, reason, plan). Unparseable replies count as answerable, with the raw text as the plan."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict) or data.get("answerable") not in ANSWERABLE:
        return "yes", "", text.strip()
    return data["answerable"], str(data.get("reason") or ""), str(data.get("plan") or "").strip()


def parse_suggestion(text: str) -> str:
    """The gate reply's suggested related question, or "" if missing or unparseable."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        return ""
    return str(data.get("suggestion") or "").strip() if isinstance(data, dict) else ""


def plan(state: CrewState) -> CrewState:
    gate = state.get("use_gate", PLANNER_GATE)
    reply = llm.complete(build_prompt(state["question"], gate), PLANNER_MODEL)
    if not gate:
        return {"plan": reply.strip()}
    answerable, reason, plan_text = parse_gate(reply)
    if answerable == "meta":
        return {"plan": "", "answerable": "meta", "gate_reason": reason, "report": meta_answer()}
    update: CrewState = {"plan": plan_text, "answerable": answerable, "gate_reason": reason}
    if answerable == "partly" and (suggestion := parse_suggestion(reply)):
        update["suggestion"] = suggestion
    return update


def meta_answer() -> str:
    """Answer to questions about the dataset itself, from the fixed summary and table list (no Data agent)."""
    tables = ", ".join(f"`{t}`" for t in sorted(table_names()))
    return f"{DATASET_SUMMARY}\n\nTables in the Olist DuckDB database: {tables}."
