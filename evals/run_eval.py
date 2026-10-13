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
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.tools.sql_tool import SqlResult, run_sql  # noqa: E402

QUESTIONS_PATH = ROOT / "evals" / "questions.json"
RESULTS_DIR = ROOT / "evals" / "results"

Answerer = Callable[[str], str]


def load_answerer(spec: str, questions: list[dict]) -> Answerer:
    if spec == "oracle":
        gold = {q["question"]: q["gold_sql"] for q in questions}
        return lambda question: gold[question]
    module, _, func = spec.partition(":")
    return getattr(importlib.import_module(module), func or "answer")


def values_match(a: Any, b: Any, tol: float) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
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


def map_columns(got: list[tuple], gold: list[tuple], tol: float, ordered: bool) -> list[int] | None:
    """Map each gold column to a distinct got column with matching values (order/extras ignored)."""
    width = len(got[0]) if got else 0
    used: set[int] = set()
    mapping: list[int] = []
    for g in range(len(gold[0]) if gold else 0):
        gold_col = column(gold, g)
        match = next(
            (i for i in range(width) if i not in used and column_matches(column(got, i), gold_col, tol, ordered)),
            None,
        )
        if match is None:
            return None
        used.add(match)
        mapping.append(match)
    return mapping


def score(got: SqlResult, gold: SqlResult, check: dict) -> bool:
    if not got.ok or not gold.ok or not got.rows or not gold.rows:
        return False
    tol = float(check.get("tolerance", 0))
    if check["type"] == "scalar":
        # Pass if any cell in the first row equals the gold value.
        return any(values_match(v, gold.rows[0][0], tol) for v in got.rows[0])
    ordered = check.get("ordered", True)
    mapping = map_columns(got.rows, gold.rows, tol, ordered)
    if mapping is None:
        return False
    projected = [tuple(r[i] for i in mapping) for r in got.rows]
    return rows_match(projected, gold.rows, tol, ordered)


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
        "passed": score(got, gold, q["check"]),
        "sql": sql,
        "error": got.error or gold.error,
        "seconds": round(time.perf_counter() - start, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answerer", default="oracle", help="'oracle' or module:function")
    args = parser.parse_args()

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    answer = load_answerer(args.answerer, questions)
    results = [evaluate(q, answer) for q in questions]

    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"{status}  {r['id']:<26} {r['seconds']:>6}s  {r['error'] or ''}")
    passed = sum(r["passed"] for r in results)
    print(f"\nScore: {passed}/{len(results)} ({passed / len(results):.0%})  answerer={args.answerer}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps({"answerer": args.answerer, "score": passed, "total": len(results), "results": results}, indent=2), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
