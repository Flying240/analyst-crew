"""Eval harness: run an answerer over evals/questions.json and score it against gold SQL.

An answerer is any function `answer(question: str) -> str` that returns SQL.
Usage:
    python evals/run_eval.py                                  # oracle (returns gold SQL)
    python evals/run_eval.py --answerer src.agents.baseline:answer
    python evals/run_eval.py --answerer src.agents.baseline:answer --runs 3
"""
import argparse
import importlib
import json
import math
import re
import sys
import time
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.llm_retry import InfraError  # noqa: E402
from src.tools.sql_tool import SqlResult, run_sql  # noqa: E402

QUESTIONS_PATH = ROOT / "evals" / "questions.json"
RESULTS_DIR = ROOT / "evals" / "results"

Answerer = Callable[[str], str]
NUMERIC = (int, float, Decimal)


def load_answerer(spec: str, questions: list[dict]) -> Answerer:
    if spec == "oracle":
        gold = {q["question"]: q["gold_sql"] for q in questions}
        return lambda question: gold[question]
    module, _, func = spec.partition(":")
    return getattr(importlib.import_module(module), func or "answer")


def is_number(v: Any) -> bool:
    return isinstance(v, NUMERIC) and not isinstance(v, bool)


def values_match(a: Any, b: Any, tol: float) -> bool:
    if is_number(a) and is_number(b):
        return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)
    return str(a) == str(b)


def rows_match(got: list[tuple], gold: list[tuple], tol: float, ordered: bool) -> bool:
    if len(got) != len(gold):
        return False
    if not ordered:
        got, gold = sorted(got, key=str), sorted(gold, key=str)
    return all(
        len(g) == len(e) and all(values_match(x, y, tol) for x, y in zip(g, e))
        for g, e in zip(got, gold)
    )


def column(rows: list[tuple], i: int) -> list[Any]:
    return [r[i] for r in rows]


def column_matches(got_col: list[Any], gold_col: list[Any], tol: float, ordered: bool) -> bool:
    if not ordered:
        got_col, gold_col = sorted(got_col, key=str), sorted(gold_col, key=str)
    return len(got_col) == len(gold_col) and all(values_match(a, b, tol) for a, b in zip(got_col, gold_col))


def rows_match_mapped(got: list[tuple], gold: list[tuple], tol: float, ordered: bool) -> bool:
    """True if some one-to-one mapping of gold columns onto got columns makes the rows match.

    Column order and extra got columns are ignored. Backtracks, so an early
    choice between equal-valued columns can't block a valid mapping.
    """
    width, n_gold = len(got[0]), len(gold[0])
    candidates = [
        [i for i in range(width) if column_matches(column(got, i), column(gold, g), tol, ordered)]
        for g in range(n_gold)
    ]

    def search(g: int, mapping: list[int]) -> bool:
        if g == n_gold:
            projected = [tuple(r[i] for i in mapping) for r in got]
            return rows_match(projected, gold, tol, ordered)
        return any(search(g + 1, mapping + [i]) for i in candidates[g] if i not in mapping)

    return search(0, [])


def score(got: SqlResult, gold: SqlResult, check: dict) -> bool:
    if not got.ok or not gold.ok or not got.rows or not gold.rows:
        return False
    tol = float(check.get("tolerance", 0))
    if check["type"] == "scalar":
        # Exactly one row with exactly one numeric value; text labels are allowed.
        numbers = [v for v in got.rows[0] if is_number(v)]
        return len(got.rows) == 1 and len(numbers) == 1 and values_match(numbers[0], gold.rows[0][0], tol)
    return rows_match_mapped(got.rows, gold.rows, tol, check.get("ordered", True))


def evaluate(q: dict, answer: Answerer) -> dict:
    start = time.perf_counter()
    infra_error = False
    try:
        sql = answer(q["question"])
        got = run_sql(sql)
    except InfraError as exc:
        sql, got, infra_error = "", SqlResult(error=f"infra_error: {exc}"), True
    except Exception as exc:
        sql, got = "", SqlResult(error=f"answerer failed: {exc}")
    gold = run_sql(q["gold_sql"])
    return {
        "id": q["id"],
        "difficulty": q.get("difficulty", "unknown"),
        "split": q.get("split", "unknown"),
        "passed": not infra_error and score(got, gold, q["check"]),
        "infra_error": infra_error,
        "sql": sql,
        "error": got.error or gold.error,
        "seconds": round(time.perf_counter() - start, 2),
    }


def tally(results: list[dict]) -> dict[str, int]:
    """Score over scored questions; infra errors are counted apart and excluded from total."""
    infra = sum(r.get("infra_error", False) for r in results)
    return {"score": sum(r["passed"] for r in results), "total": len(results) - infra, "infra_errors": infra}


def group_tally(results: list[dict], key: str) -> dict[str, dict[str, int]]:
    groups: dict[str, list[dict]] = {}
    for r in results:
        groups.setdefault(r.get(key, "unknown"), []).append(r)
    return {name: tally(rs) for name, rs in groups.items()}


def by_difficulty(results: list[dict]) -> dict[str, dict[str, int]]:
    return group_tally(results, "difficulty")


def by_split(results: list[dict]) -> dict[str, dict[str, int]]:
    """Dev questions are for prompt tuning; heldout scores show whether gains generalize."""
    return group_tally(results, "split")


def fmt(b: dict[str, int]) -> str:
    pct = f"{b['score'] / b['total']:.0%}" if b["total"] else "n/a"
    infra = f"  +{b['infra_errors']} infra_error (not scored)" if b["infra_errors"] else ""
    return f"{b['score']}/{b['total']} ({pct}){infra}"


def answerer_slug(answerer: str) -> str:
    """Short filename-safe name for an answerer, e.g. src.agents.baseline:answer -> baseline."""
    module = answerer.split(":")[0]
    return re.sub(r"[^a-z0-9]+", "-", module.split(".")[-1].lower()).strip("-") or "unknown"


def run_once(questions: list[dict], answer: Answerer, answerer: str) -> dict:
    results = [evaluate(q, answer) for q in questions]
    for r in results:
        status = "INFRA" if r["infra_error"] else "PASS" if r["passed"] else "FAIL"
        print(f"{status:<5} {r['difficulty']:<7} {r['id']:<30} {r['seconds']:>6}s  {r['error'] or ''}")
    overall = tally(results)
    levels = by_difficulty(results)
    splits = by_split(results)
    print()
    for name, b in [*levels.items(), *splits.items()]:
        print(f"{name:<8} {fmt(b)}")
    print(f"\nScore: {fmt(overall)}  answerer={answerer}")
    report = {"answerer": answerer, **overall, "by_difficulty": levels, "by_split": splits, "results": results}
    save(report, answerer_slug(answerer))
    return report


def save(report: dict, name: str) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{name}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {out}")


def spread(buckets: list[dict]) -> dict:
    """Mean and range of each run's pass rate; rates, because infra errors can shrink a run's total.

    A run with nothing scored (all infra errors) is left out rather than counted as 0%.
    """
    rates = [b["score"] / b["total"] for b in buckets if b["total"]] or [0.0]
    return {
        "mean_pct": round(100 * sum(rates) / len(rates), 1),
        "min_pct": round(100 * min(rates), 1),
        "max_pct": round(100 * max(rates), 1),
        "scores": [f"{b['score']}/{b['total']}" for b in buckets],
        "infra_errors": sum(b["infra_errors"] for b in buckets),
    }


def summarize(reports: list[dict]) -> dict:
    """Mean and min-max score across runs, plus questions whose pass/fail differs between runs.

    Infra-error runs of a question are left out of its pass/fail history and listed separately.
    """
    levels = reports[0]["by_difficulty"]
    outcomes: dict[str, list[bool]] = {}
    infra: dict[str, int] = {}
    for rep in reports:
        for r in rep["results"]:
            if r.get("infra_error"):
                infra[r["id"]] = infra.get(r["id"], 0) + 1
            else:
                outcomes.setdefault(r["id"], []).append(r["passed"])
    return {
        "answerer": reports[0]["answerer"],
        "runs": len(reports),
        "overall": spread(reports),
        "by_difficulty": {lv: spread([r["by_difficulty"][lv] for r in reports]) for lv in levels},
        "by_split": {sp: spread([r["by_split"][sp] for r in reports]) for sp in reports[0].get("by_split", {})},
        "flipped": {qid: runs for qid, runs in outcomes.items() if len(set(runs)) > 1},
        "always_failed": [qid for qid, runs in outcomes.items() if not any(runs)],
        "infra_errors": infra,
    }


def print_summary(s: dict) -> None:
    print(f"\n=== Summary over {s['runs']} runs ===")
    for name, b in [("overall", s["overall"]), *s["by_difficulty"].items(), *s["by_split"].items()]:
        infra = f"  ({b['infra_errors']} infra_error, not scored)" if b["infra_errors"] else ""
        print(f"{name:<8} mean {b['mean_pct']}%  range {b['min_pct']}-{b['max_pct']}%  runs {', '.join(b['scores'])}{infra}")
    print("Flipped between runs:")
    for qid, runs in s["flipped"].items():
        print(f"  {qid:<30} {' '.join('P' if p else 'F' for p in runs)}")
    if not s["flipped"]:
        print("  (none)")
    print(f"Failed every run: {', '.join(s['always_failed']) or '(none)'}")
    infra = ", ".join(f"{qid} x{n}" for qid, n in s["infra_errors"].items())
    print(f"Infra errors (not scored): {infra or '(none)'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answerer", default="oracle", help="'oracle' or module:function")
    parser.add_argument("--runs", type=int, default=1, help="repeat the eval N times and summarize")
    args = parser.parse_args()

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    answer = load_answerer(args.answerer, questions)
    reports = []
    for i in range(args.runs):
        if args.runs > 1:
            print(f"\n=== Run {i + 1}/{args.runs} ===")
        reports.append(run_once(questions, answer, args.answerer))
    if args.runs > 1:
        summary = summarize(reports)
        print_summary(summary)
        save(summary, f"{answerer_slug(args.answerer)}_summary{args.runs}")


if __name__ == "__main__":
    main()
