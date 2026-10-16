"""Data agent: writes and runs the SQL, with the baseline's error-feedback retry and Critic feedback."""
import re

from src import llm
from src.agents.baseline import RULES, answer_from_prompt, load_schema
from src.agents.duckdb_tips import DUCKDB_TIPS
from src.config import DATA_MODEL
from src.state import Attempt, CrewState
from src.tools.sql_tool import run_sql


def build_prompt(question: str, plan: str, previous_sql: str | None = None, feedback: str | None = None) -> str:
    review = ""
    if previous_sql and feedback:
        review = (
            f"# Reviewer feedback on your previous SQL\nPrevious SQL:\n{previous_sql}\n\nFeedback:\n{feedback}\n\n"
            "Fix the SQL to address it if the feedback is correct for the question.\n\n"
        )
    return (
        "You are a data analyst writing SQL for the Olist e-commerce DuckDB database.\n\n"
        f"# Schema\n{load_schema()}\n\n# Rules\n{RULES}\n\n# DuckDB tips\n{DUCKDB_TIPS}\n\n"
        "# Plan\nThe plan is guidance. If it conflicts with the question, follow the question.\n"
        f"{plan}\n\n{review}# Question\n{question}\n\nSQL:"
    )


def ask(prompt: str) -> str:
    return llm.ask(prompt, DATA_MODEL)


def normalize(sql: str) -> str:
    return re.sub(r"\s+", " ", sql).strip().lower()


def write_sql(state: CrewState) -> CrewState:
    previous = state.get("sql") if state.get("critic_feedback") else None
    attempts: list[Attempt] = []
    prompt = build_prompt(state["question"], state.get("plan", ""), previous, state.get("critic_feedback"))
    sql = answer_from_prompt(prompt, state.get("max_retries", 0), attempts, ask_fn=ask)
    error = attempts[-1]["error"]
    update: CrewState = {
        "attempts": state.get("attempts", []) + attempts,
        "round_attempts": attempts,
        "data_runs": state.get("data_runs", 0) + 1,
        "retries_used": state.get("retries_used", 0) + len(attempts) - 1,
    }
    if previous is not None and normalize(sql) == normalize(previous):
        return {**update, "returned": "data_disagreed"}
    if previous is not None and error is not None:
        return {**update, **use_sql(state["first_ok_sql"]), "returned": "first_ok_fallback"}
    first_ok = state.get("first_ok_sql") or (sql if error is None else None)
    return {**update, **use_sql(sql, error), "first_ok_sql": first_ok}


def use_sql(sql: str, error: str | None = None) -> CrewState:
    """State fields for returning `sql`, with its result rows if it ran."""
    result = run_sql(sql) if error is None else None
    return {
        "sql": sql,
        "error": error if result is None else result.error,
        "columns": result.columns if result else [],
        "rows": result.rows if result else [],
    }
