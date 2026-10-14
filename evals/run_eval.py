"""Eval harness: run an answerer over evals/questions.json and score it against gold SQL.

An answerer is any function `answer(question: str) -> str` that returns SQL.
Usage:
    python evals/run_eval.py                                  # oracle (returns gold SQL)
    python evals/run_eval.py --answerer src.agents.baseline:answer
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
    try:
        sql = answer(q["question"])
        got = run_sql(sql)
    except Exception as exc:
        sql, got = "", SqlResult(error=f"answerer failed: {exc}")
    gold = run_sql(q["gold_sql"])
    return {
        "id": q["id"],
        "difficulty": q.get("difficulty", "unknown"),
        "passed": score(got, gold, q["check"]),
        "sql": sql,
        "error": got.error or gold.error,
        "seconds": round(time.perf_counter() - start, 2),
    }


def by_difficulty(results: list[dict]) -> dict[str, dict[str, int]]:
    scores: dict[str, dict[str, int]] = {}
    for r in results:
        bucket = scores.setdefault(r["difficulty"], {"score": 0, "total": 0})
        bucket["score"] += r["passed"]
        bucket["total"] += 1
    return scores


def answerer_slug(answerer: str) -> str:
    """Short filename-safe name for an answerer, e.g. src.agents.baseline:answer -> baseline."""
    module = answerer.split(":")[0]
    return re.sub(r"[^a-z0-9]+", "-", module.split(".")[-1].lower()).strip("-") or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answerer", default="oracle", help="'oracle' or module:function")
    args = parser.parse_args()

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    answer = load_answerer(args.answerer, questions)
    results = [evaluate(q, answer) for q in questions]

    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"{status}  {r['difficulty']:<7} {r['id']:<30} {r['seconds']:>6}s  {r['error'] or ''}")
    passed = sum(r["passed"] for r in results)
    levels = by_difficulty(results)
    print()
    for level, b in levels.items():
        print(f"{level:<8} {b['score']}/{b['total']} ({b['score'] / b['total']:.0%})")
    print(f"\nScore: {passed}/{len(results)} ({passed / len(results):.0%})  answerer={args.answerer}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{answerer_slug(args.answerer)}.json"
    out.write_text(json.dumps({"answerer": args.answerer, "score": passed, "total": len(results), "by_difficulty": levels, "results": results}, indent=2), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
