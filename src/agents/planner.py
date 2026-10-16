"""Planner: turns the question into a short analysis plan (no SQL)."""
from src import llm
from src.agents.baseline import RULES, load_schema
from src.config import PLANNER_MODEL
from src.state import CrewState

INSTRUCTIONS = """\
Write a short plan (at most 6 bullet points) for answering the question with one SQL query:
metrics and how to compute them, tables and joins, filters, grouping, ordering, and whether a chart would help.
Only use filters stated in the question or the project rules; never add others.
Do not write SQL."""


def build_prompt(question: str) -> str:
    return (
        "You are planning an analysis of the Olist e-commerce DuckDB database.\n\n"
        f"# Schema\n{load_schema()}\n\n# Project rules\n{RULES}\n\n"
        f"# Task\n{INSTRUCTIONS}\n\n# Question\n{question}\n\nPlan:"
    )


def plan(state: CrewState) -> CrewState:
    return {"plan": llm.complete(build_prompt(state["question"]), PLANNER_MODEL).strip()}
