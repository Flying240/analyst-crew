"""Data agent: writes and runs the SQL, with the baseline's error-feedback retry."""
from src import llm
from src.agents.baseline import RULES, answer_from_prompt, load_schema
from src.agents.duckdb_tips import DUCKDB_TIPS
from src.config import DATA_MODEL
from src.state import Attempt, CrewState
from src.tools.sql_tool import run_sql


def build_prompt(question: str, plan: str) -> str:
    return (
        "You are a data analyst writing SQL for the Olist e-commerce DuckDB database.\n\n"
        f"# Schema\n{load_schema()}\n\n# Rules\n{RULES}\n\n# DuckDB tips\n{DUCKDB_TIPS}\n\n"
        "# Plan\nThe plan is guidance. If it conflicts with the question, follow the question.\n"
        f"{plan}\n\n# Question\n{question}\n\nSQL:"
    )


def ask(prompt: str) -> str:
    return llm.ask(prompt, DATA_MODEL)


def write_sql(state: CrewState) -> CrewState:
    attempts: list[Attempt] = []
    prompt = build_prompt(state["question"], state.get("plan", ""))
    sql = answer_from_prompt(prompt, state.get("max_retries", 0), attempts, ask_fn=ask)
    last = attempts[-1]
    result = run_sql(sql) if last["error"] is None else None
    return {
        "sql": sql,
        "attempts": attempts,
        "retries_used": len(attempts) - 1,
        "error": last["error"],
        "columns": result.columns if result else [],
        "rows": result.rows if result else [],
    }
