"""Shared state passed between agents in the LangGraph crew."""
from typing import Any, TypedDict


class Attempt(TypedDict):
    sql: str
    error: str | None


class CrewState(TypedDict, total=False):
    question: str
    max_retries: int
    skip_report: bool  # evals only need the SQL
    plan: str
    sql: str
    columns: list[str]
    rows: list[tuple[Any, ...]]
    error: str | None
    attempts: list[Attempt]
    retries_used: int
    critic_feedback: str | None  # unused until the Critic loop (roadmap step 3)
    chart: Any  # plotly Figure or None
    report: str
