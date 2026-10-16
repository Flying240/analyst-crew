"""About page numbers, read from the committed eval summaries (evals/summaries/), never hard-coded."""
import json
from pathlib import Path
from statistics import mean
from typing import Any

from src.config import ROOT

SUMMARIES_DIR = ROOT / "evals" / "summaries"
QUESTIONS_PATH = ROOT / "evals" / "questions.json"
FULL = 50

# (rung label, what was added, test for a summary)
LADDER = [
    ("Baseline", "One LLM call writes the SQL.",
     lambda r: r["answerer"] == "src.agents.baseline:answer" and r.get("official") == "baseline_50q_runs"),
    ("+ retry", "If the SQL fails, the database error goes back to the model (up to 2 retries).",
     lambda r: r["answerer"] == "src.agents.baseline:answer" and r.get("official") == "retry_50q_runs"),
    ("+ DuckDB tips", "Adds short DuckDB syntax tips to the prompt.",
     lambda r: r["answerer"] == "src.agents.baseline:answer_with_tips"),
    ("Multi-agent graph", "Planner, Data, Viz and Reporter agents.",
     lambda r: r["answerer"] == "src.graph:answer"),
    ("Graph + Critic", "A Critic reviews SQL that runs but may answer the wrong thing.",
     lambda r: r["answerer"] == "src.graph:answer+critic"),
]

# Known failures, explained in general terms (no question-specific fixes in the prompts).
FAILURE_NOTES = {
    "m19_avg_freight_per_order": (
        "The question asks for the average freight per order, but the model averages over items. An order with "
        "3 items counts 3 times, so the number comes out too low. The SQL runs fine, so the error retry never kicks in."
    ),
    "m11_null_category_products": (
        "\"Products without a category\" means the category is empty (NULL). The model instead looks for categories "
        "that have no English translation, which is a different set of products."
    ),
}


def load_summaries(root: Path = SUMMARIES_DIR) -> list[dict[str, Any]]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.glob("*.json"))]


def full_runs(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Complete runs over all 50 questions, excluding the gold-SQL oracle."""
    return [r for r in summaries if r["answerer"] != "oracle" and r["total"] == FULL and r.get("complete")]


def ladder(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per rung: mean score over its full runs, range, run count and tokens; None scores if not measured yet."""
    runs = full_runs(summaries)
    rows = []
    for label, added, match in LADDER:
        mine = [r for r in runs if match(r)]
        scores = [100 * r["score"] / r["total"] for r in mine]
        tokens = [(r.get("tokens") or {}).get("mean_prompt", 0) + (r.get("tokens") or {}).get("mean_completion", 0) for r in mine]
        rows.append({
            "step": label, "what it adds": added, "runs": len(mine),
            "mean score %": round(mean(scores), 1) if mine else None,
            "range": f"{min(r['score'] for r in mine)}–{max(r['score'] for r in mine)} / {FULL}" if mine else "not measured on all 50 yet",
            "tokens / question": round(mean(tokens)) if mine else None,
        })
    return rows


def last_updated(summaries: list[dict[str, Any]]) -> str | None:
    dates = [r["date"] for r in full_runs(summaries)]
    return max(dates) if dates else None


def always_failed(summaries: list[dict[str, Any]]) -> list[str]:
    """Question ids that failed in every full run (infra errors don't count as failures)."""
    runs = [r for r in full_runs(summaries) if any(match(r) for _, _, match in LADDER)]
    if not runs:
        return []
    failed = [{q["id"] for q in r["questions"] if not q["passed"] and not q.get("infra_error")} for r in runs]
    return sorted(set.intersection(*failed))


def question_text() -> dict[str, str]:
    raw = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    return {q["id"]: q["question"] for q in (raw if isinstance(raw, list) else raw["questions"])}
