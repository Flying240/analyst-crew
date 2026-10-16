"""Critic: reviews SQL that ran and either accepts it or sends specific feedback to the Data agent."""
import json
import re
from typing import Any

from src import llm
from src.agents.baseline import RULES
from src.config import CRITIC_MODEL, SCHEMA_COMPACT_PATH
from src.state import CrewState, CriticRound

SHOW_ROWS = 10

CHECKS = """\
1. Does every filter and the grain (per order / per item / per customer) match what the question asks?
2. Were the project rules followed (customer_unique_id for customers, the revenue definition, the late definition)?
3. Is the result suspicious: empty, NULL, or implausibly large from join fan-out?
Only flag a concrete problem you can point to in the SQL or the result; if unsure, answer ok.
Reply with JSON only: {"verdict": "ok" | "revise", "feedback": "<what is wrong and how to fix it; empty if ok>"}"""


def result_facts(columns: list[str], rows: list[tuple[Any, ...]]) -> str:
    """Facts computed in Python so the Critic doesn't have to infer them."""
    if not rows:
        return "The result is empty (0 rows)."
    null_cols = [c for i, c in enumerate(columns) if all(r[i] is None for r in rows)]
    facts = [f"{len(rows)} row(s)."]
    if null_cols:
        facts.append(f"Columns that are NULL in every row: {', '.join(null_cols)}.")
    return " ".join(facts)


def build_prompt(state: CrewState) -> str:
    columns, rows = state.get("columns", []), state.get("rows", [])
    shown = "\n".join([" | ".join(columns)] + [" | ".join(map(str, r)) for r in rows[:SHOW_ROWS]])
    return (
        "You are reviewing SQL written for the Olist e-commerce DuckDB database.\n\n"
        f"# Schema\n{SCHEMA_COMPACT_PATH.read_text(encoding='utf-8')}\n\n# Project rules\n{RULES}\n\n"
        f"# Question\n{state['question']}\n\n# Plan\n{state.get('plan', '')}\n\n"
        f"# SQL\n{state['sql']}\n\n# Result (first {SHOW_ROWS} rows)\n{shown}\n\n"
        f"# Result facts\n{result_facts(columns, rows)}\n\n# Checks\n{CHECKS}\n\nJSON:"
    )


def parse(text: str) -> tuple[str, str, bool]:
    """Return (verdict, feedback, parse_error); anything unparseable counts as ok."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict) or data.get("verdict") not in ("ok", "revise"):
        return "ok", "", True
    return data["verdict"], str(data.get("feedback") or ""), False


def review(state: CrewState) -> CrewState:
    rounds = state.get("critic_rounds", [])
    verdict, feedback, parse_error = parse(llm.complete(build_prompt(state), CRITIC_MODEL))
    entry: CriticRound = {
        "round": len(rounds) + 1, "sql": state["sql"], "verdict": verdict, "feedback": feedback,
        "parse_error": parse_error, "attempts": state.get("round_attempts", []),
    }
    return {"critic_rounds": rounds + [entry], "critic_feedback": feedback if verdict == "revise" else None}
