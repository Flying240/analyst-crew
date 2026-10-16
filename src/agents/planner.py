"""Planner: turns the question into a short plan of the logic (no SQL)."""
from src import llm
from src.agents.baseline import RULES, load_schema
from src.config import PLANNER_MODEL
from src.state import CrewState

INSTRUCTIONS = """\
Write a short plan (at most 6 bullet points) describing only the logic needed to answer the question:
- what to count, sum or average, and at what grain (e.g. per order, per item, per customer), stated in the question's own terms
- which tables and columns hold that data
- which filters apply: only filters stated in the question or the project rules; never add others
- grouping and ordering, if any
Use the project rules for definitions such as revenue, customers and late delivery.
Never name SQL functions or write SQL."""


def build_prompt(question: str) -> str:
    return (
        "You are planning an analysis of the Olist e-commerce DuckDB database.\n\n"
        f"# Schema\n{load_schema()}\n\n# Project rules\n{RULES}\n\n"
        f"# Task\n{INSTRUCTIONS}\n\n# Question\n{question}\n\nPlan:"
    )


def plan(state: CrewState) -> CrewState:
    return {"plan": llm.complete(build_prompt(state["question"]), PLANNER_MODEL).strip()}
