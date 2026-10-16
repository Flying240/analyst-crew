"""Single-agent baseline: one LLM call turns a question into DuckDB SQL."""
from typing import Callable

from src import llm, retry_info
from src.agents.duckdb_tips import DUCKDB_TIPS
from src.config import PROMPT_SCHEMA, SCHEMA_COMPACT_PATH, SCHEMA_PATH
from src.tools.sql_tool import check_sql, run_sql

AskFn = Callable[[str], str]

RULES = """\
- Write a single read-only DuckDB SELECT (or WITH) query. Never INSERT/UPDATE/DELETE/DROP/ALTER.
- Revenue = price + freight_value from order_items, delivered orders only (order_status = 'delivered').
- Count customers with customer_unique_id, not customer_id.
- Exclude 2016 and Sept-Oct 2018 from trend analysis (incomplete months).
- Label months as text 'YYYY-MM' using strftime(ts, '%Y-%m'); label years as integers.
- For questions asking for one value, return exactly one row with one numeric column (text labels OK).
- Return only the SQL, no explanation, no markdown fences."""


def load_schema() -> str:
    path = SCHEMA_PATH if PROMPT_SCHEMA == "full" else SCHEMA_COMPACT_PATH
    return path.read_text(encoding="utf-8")


def build_prompt(question: str, tips: bool = False) -> str:
    extra = f"\n\n# DuckDB tips\n{DUCKDB_TIPS}" if tips else ""
    return (
        "You are a data analyst writing SQL for the Olist e-commerce DuckDB database.\n\n"
        f"# Schema\n{load_schema()}\n\n# Rules\n{RULES}{extra}\n\n# Question\n{question}\n\nSQL:"
    )


def build_retry_prompt(prompt: str, failed_sql: str, error: str) -> str:
    return (
        f"{prompt}\n{failed_sql}\n\n"
        f"# That SQL failed with this database error\n{error}\n\n"
        "Fix the SQL so it runs and answers the question. Follow the same rules.\n\nSQL:"
    )


def ask(prompt: str) -> str:
    return llm.ask(prompt)


def try_ask(ask_fn: AskFn, prompt: str) -> tuple[str, str | None]:
    """(sql, None), or (the model's prose, error) when the reply holds no SQL query."""
    try:
        return ask_fn(prompt), None
    except llm.NotSqlError as exc:
        return exc.text, str(exc)


def answer_from_prompt(
    prompt: str, retries: int, attempts: list[dict] | None = None, ask_fn: AskFn | None = None
) -> str:
    """Return SQL; on a database error, feed the error back and retry up to `retries` times.

    If `attempts` is given, every SQL tried is appended with its error (None if it ran).
    """
    ask_fn = ask_fn or ask
    sql, error = try_ask(ask_fn, prompt)
    used = 0
    while True:
        if error is None and (retries or attempts is not None):
            error = check_sql(sql) or run_sql(sql).error
        if attempts is not None:
            attempts.append({"sql": sql, "error": error})
        if error is None or used >= retries:
            break
        used += 1
        sql, error = try_ask(ask_fn, build_retry_prompt(prompt, sql, error))
    retry_info.record(used)
    return sql


def answer(question: str, retries: int = 0) -> str:
    return answer_from_prompt(build_prompt(question), retries)


def answer_with_tips(question: str, retries: int = 0) -> str:
    """Baseline + DuckDB tips, to measure the tips apart from the multi-agent design."""
    return answer_from_prompt(build_prompt(question, tips=True), retries)
