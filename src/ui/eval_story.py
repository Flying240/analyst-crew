"""About page numbers, read from the committed eval summaries (evals/summaries/), never hard-coded."""
import json
from pathlib import Path
from statistics import mean
from typing import Any

from src.config import ROOT

SUMMARIES_DIR = ROOT / "evals" / "summaries"
QUESTIONS_PATH = ROOT / "evals" / "questions.json"
FULL = 50

# (rung label, what was added, test for a summary, score from round-1 SQL)
# The graph row is scored on the official graph + critic runs' round-1 SQL (before any Critic rewrite).
GRAPH_RUNS = "graph_critic_50q_runs"
LADDER = [
    ("Baseline", "One LLM call writes the SQL.",
     lambda r: r["answerer"] == "src.agents.baseline:answer" and r.get("official") == "baseline_50q_runs", False),
    ("+ retry", "If the SQL fails, the database error goes back to the model (up to 2 retries).",
     lambda r: r["answerer"] == "src.agents.baseline:answer" and r.get("official") == "retry_50q_runs", False),
    ("+ DuckDB tips", "Adds short DuckDB syntax tips to the prompt.",
     lambda r: r["answerer"] == "src.agents.baseline:answer_with_tips", False),
    ("Multi-agent graph", "Planner, Data, Viz and Reporter agents.",
     lambda r: r.get("official") == GRAPH_RUNS and bool(r.get("round1")), True),
    ("Graph + Critic", "A Critic reviews SQL that runs but may answer the wrong thing.",
     lambda r: r.get("official") == GRAPH_RUNS, False),
]
FINAL_STEP = "Graph + Critic"  # the setup the app runs

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
    "m10_untranslated_categories": (
        "Some products have no category at all (NULL). The model finds untranslated categories with a left join to "
        "the translation table, and the empty category has no translation either, so it is counted as one more "
        "untranslated category and the count is 1 too high. The Critic checks the join and the filter, finds both "
        "match the question, and passes it. Known limitation: the Critic doesn't yet look at what NULL values do "
        "to a count."
    ),
}


def load_summaries(root: Path = SUMMARIES_DIR) -> list[dict[str, Any]]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.glob("*.json"))]


def full_runs(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Complete runs over all 50 questions, excluding the gold-SQL oracle."""
    return [r for r in summaries if r["answerer"] != "oracle" and r["total"] == FULL and r.get("complete")]


def tokens_per_question(run: dict[str, Any], round1: bool) -> float:
    """Mean tokens per question; for a round-1 row, without the Critic model's calls."""
    t = run.get("tokens") or {}
    by_model = run.get("tokens_by_model")
    if round1 and by_model and t.get("questions"):
        kept = [m for name, m in by_model.items() if name != run.get("critic_model")]
        return sum(m["prompt"] + m["completion"] for m in kept) / t["questions"]
    return t.get("mean_prompt", 0) + t.get("mean_completion", 0)


def ladder(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per rung: mean score over its full runs, range, run count and tokens; None scores if not measured yet."""
    runs = full_runs(summaries)
    rows = []
    for label, added, match, round1 in LADDER:
        mine = [r for r in runs if match(r)]
        points = [r["round1"]["score"] if round1 else r["score"] for r in mine]
        scores = [100 * p / r["total"] for p, r in zip(points, mine)]
        tokens = [tokens_per_question(r, round1) for r in mine]
        rows.append({
            "step": label, "what it adds": added, "runs": len(mine),
            "mean score %": round(mean(scores), 1) if mine else None,
            "range": f"{min(points)}–{max(points)} / {FULL}" if mine else "not measured on all 50 yet",
            "tokens / question": round(mean(tokens)) if mine else None,
        })
    return rows


def last_updated(summaries: list[dict[str, Any]]) -> str | None:
    dates = [r["date"] for r in full_runs(summaries)]
    return max(dates) if dates else None


def always_failed(summaries: list[dict[str, Any]], step: str = FINAL_STEP) -> list[str]:
    """Question ids that failed in every full run of one ladder step (infra errors don't count as failures)."""
    rung = next(r for r in LADDER if r[0] == step)
    match, round1 = rung[2], rung[3]
    runs = [r for r in full_runs(summaries) if match(r)]
    if not runs:
        return []
    key = "round1_passed" if round1 else "passed"
    failed = [{q["id"] for q in r["questions"] if not q[key] and not q.get("infra_error")} for r in runs]
    return sorted(set.intersection(*failed))


def question_text() -> dict[str, str]:
    raw = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    return {q["id"]: q["question"] for q in (raw if isinstance(raw, list) else raw["questions"])}
