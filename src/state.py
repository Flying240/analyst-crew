"""Shared state passed between agents in the LangGraph crew."""
from typing import Any, TypedDict


class Attempt(TypedDict):
    sql: str
    error: str | None


class CriticRound(TypedDict, total=False):
    round: int
    sql: str
    verdict: str  # "ok" | "revise"
    feedback: str
    checklist: list[dict]  # [{requirement, sql, status: match|mismatch}]
    parse_error: bool
    attempts: list[Attempt]  # the Data agent's tries (SQL + error) that produced this SQL


class CrewState(TypedDict, total=False):
    question: str
    max_retries: int
    skip_report: bool  # evals only need the SQL
    plan: str
    sql: str
    columns: list[str]
    rows: list[tuple[Any, ...]]
    error: str | None
    attempts: list[Attempt]  # every Data try across all rounds
    round_attempts: list[Attempt]  # the latest Data run's tries
    retries_used: int
    use_critic: bool
    critic_feedback: str | None  # latest Critic feedback, sent back to the Data agent
    critic_rounds: list[CriticRound]
    first_ok_sql: str | None  # first SQL that ran; returned if the Critic never accepts a rewrite
    data_runs: int
    returned: str  # "no_critic" | "ok" | "first_ok_fallback" | "data_disagreed" | "data_error"
    chart: Any  # plotly Figure or None
    report: str
