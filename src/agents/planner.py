"""Planner: turns the question into a short plan of the logic (no SQL).

With the question gate on, it also decides whether the dataset can answer the question and replies with JSON
{answerable: yes|partly|no, reason, plan}.
"""
import json
import re

from src import llm
from src.agents.baseline import RULES, load_schema
from src.agents.gate import DATASET_SUMMARY
from src.config import PLANNER_GATE, PLANNER_MODEL
from src.state import CrewState

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
# What the dataset covers
{DATASET_SUMMARY}
- "yes": the tables above hold the data the question needs.
- "partly": part of the question can be answered, or only with a stated approximation; say what is missing.
- "no": the question needs data the database does not have, or is not about this data.
Prefer "yes" when the question is a normal analysis of orders, revenue, deliveries, reviews, customers, sellers,
products, payments or locations, even if it is phrased casually.
Reply with JSON only:
{{"answerable": "yes" | "partly" | "no", "reason": "<one sentence>", "plan": "<the bullet-point plan; empty if no>"}}"""

ANSWERABLE = ("yes", "partly", "no")


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


def plan(state: CrewState) -> CrewState:
    gate = state.get("use_gate", PLANNER_GATE)
    reply = llm.complete(build_prompt(state["question"], gate), PLANNER_MODEL)
    if not gate:
        return {"plan": reply.strip()}
    answerable, reason, plan_text = parse_gate(reply)
    return {"plan": plan_text, "answerable": answerable, "gate_reason": reason}
