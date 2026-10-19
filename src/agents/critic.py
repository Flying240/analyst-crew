"""Critic: reviews SQL that ran and either accepts it or sends specific feedback to the Data agent."""
import json
import re
from typing import Any

from src import llm
from src.agents.baseline import RULES
from src.config import CRITIC_MAX_TOKENS, CRITIC_MODEL, CRITIC_REASONING_EFFORT, SCHEMA_COMPACT_PATH
from src.state import CrewState, CriticRound

SHOW_ROWS = 10

CHECKS = """Check the SQL against the question as a checklist:
1. Before reading the SQL, list each requirement from the question's own words: what is counted, summed or
   averaged; at what grain (per order, per item, per customer, per category); which filters; which definitions
   (project rules, which column and value a term in the question refers to). Do not restate the SQL's logic
   as a requirement.
2. For each requirement, quote the SQL part that implements it ("missing" if none) and mark it match or mismatch.
3. Also check for things the question did not ask for: an unrequested filter or a join that repeats rows
   and inflates sums or counts is a mismatch. Check how NULL or missing values are treated: a join or
   NOT IN / IS NULL test can count, drop or include NULLs differently from what the question means.
The verdict is "revise" only if at least one item is a concrete mismatch; otherwise "ok".
Keep it short: at most 6 checklist items, each requirement in a few words, the SQL part as a short
fragment (not the whole query), and feedback of at most 3 sentences.
Reply with JSON only:
{"checklist": [{"requirement": "...", "sql": "...", "status": "match" | "mismatch"}],
 "verdict": "ok" | "revise",
 "feedback": "<each mismatch and how to fix it; empty if ok>"}"""


def result_facts(columns: list[str], rows: list[tuple[Any, ...]]) -> str:
    """Facts computed in Python so the Critic doesn't have to infer them."""
    if not rows:
        return "The result is empty (0 rows)."
    null_cols = [c for i, c in enumerate(columns) if all(r[i] is None for r in rows)]
    facts = [f"{len(rows)} row(s)."]
    if null_cols:
        facts.append(f"Columns that are NULL in every row: {', '.join(null_cols)}.")
    return " ".join(facts)


def build_prompt(question: str, sql: str, columns: list[str], rows: list[tuple[Any, ...]]) -> str:
    """The Critic judges the SQL against the question, schema and rules only (no plan)."""
    shown = "\n".join([" | ".join(columns)] + [" | ".join(map(str, r)) for r in rows[:SHOW_ROWS]])
    return (
        "You are reviewing SQL written for the Olist e-commerce DuckDB database.\n\n"
        f"# Schema\n{SCHEMA_COMPACT_PATH.read_text(encoding='utf-8')}\n\n# Project rules\n{RULES}\n\n"
        f"# Question\n{question}\n\n# SQL\n{sql}\n\n# Result (first {SHOW_ROWS} rows)\n{shown}\n\n"
        f"# Result facts\n{result_facts(columns, rows)}\n\n# Checks\n{CHECKS}\n\nJSON:"
    )


def parse(text: str) -> tuple[str, str, list[dict], bool]:
    """Return (verdict, feedback, checklist, parse_error); anything unparseable counts as ok."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict) or data.get("verdict") not in ("ok", "revise"):
        return "ok", "", [], True
    checklist = data.get("checklist") if isinstance(data.get("checklist"), list) else []
    return data["verdict"], str(data.get("feedback") or ""), checklist, False


def review_sql(question: str, sql: str, columns: list[str], rows: list[tuple[Any, ...]]) -> dict:
    """One Critic call on a (question, SQL, result) triple; used by the graph and evals/critic_eval.py."""
    verdict, feedback, checklist, parse_error = parse(
        llm.complete(
            build_prompt(question, sql, columns, rows), CRITIC_MODEL,
            max_tokens=CRITIC_MAX_TOKENS, reasoning_effort=CRITIC_REASONING_EFFORT,
        )
    )
    return {"verdict": verdict, "feedback": feedback, "checklist": checklist, "parse_error": parse_error}


def review(state: CrewState) -> CrewState:
    rounds = state.get("critic_rounds", [])
    review = review_sql(state["question"], state["sql"], state.get("columns", []), state.get("rows", []))
    entry: CriticRound = {
        "round": len(rounds) + 1, "sql": state["sql"], **review, "attempts": state.get("round_attempts", []),
    }
    feedback = review["feedback"] if review["verdict"] == "revise" else None
    return {"critic_rounds": rounds + [entry], "critic_feedback": feedback}
