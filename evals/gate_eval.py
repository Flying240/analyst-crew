"""Question-gate eval: does the Planner (gate on) reject unanswerable questions without rejecting answerable ones?

Each case runs the free pre-check, then one Planner call. Cases are in evals/gate_cases.json.
Usage:
    python evals/gate_eval.py --check     # offline: pre-check only, no LLM calls
    python evals/gate_eval.py             # 12 Planner calls
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evals.run_eval import RESULTS_DIR  # noqa: E402
from src import llm_usage  # noqa: E402
from src.agents import planner  # noqa: E402
from src.agents.gate import precheck  # noqa: E402

CASES_PATH = ROOT / "evals" / "gate_cases.json"


def load_cases() -> list[dict]:
    raw = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    return (
        [{**c, "expect": "no"} for c in raw["unanswerable"]] + [{**c, "expect": "yes"} for c in raw["answerable"]]
        + [{**c, "expect": "meta"} for c in raw.get("meta", [])]
    )


def is_correct(expect: str, answerable: str) -> bool:
    """no -> rejected; meta -> answered from the description; yes -> sent on to the Data agent (yes or partly)."""
    return answerable in ("yes", "partly") if expect == "yes" else answerable == expect


def judge(case: dict, offline: bool = False) -> dict:
    """answerable is "no" if the pre-check rejects; otherwise the Planner's verdict (offline: skipped)."""
    start = time.perf_counter()
    llm_usage.take()
    rejected_by = None
    reason = precheck(case["question"])
    if reason:
        answerable, rejected_by = "no", "precheck"
    elif offline:
        answerable = None
    else:
        update = planner.plan({"question": case["question"], "use_gate": True})
        answerable, reason = update["answerable"], update["gate_reason"]
        rejected_by = "planner" if answerable == "no" else None
    return {
        "id": case["id"], "expect": case["expect"], "answerable": answerable, "reason": reason,
        "rejected_by": rejected_by, "correct": None if answerable is None else is_correct(case["expect"], answerable),
        "tokens": llm_usage.take(), "seconds": round(time.perf_counter() - start, 2),
    }


def summarize(results: list[dict]) -> dict:
    judged = [r for r in results if r["correct"] is not None]
    wrong_rejections = [r["id"] for r in judged if r["expect"] == "yes" and r["answerable"] == "no"]
    missed = [r["id"] for r in judged if r["expect"] == "no" and r["answerable"] != "no"]
    n_yes = sum(r["expect"] == "yes" for r in judged)
    n_no = sum(r["expect"] == "no" for r in judged)
    return {
        "judged": len(judged), "correct": sum(r["correct"] for r in judged),
        "wrong_rejections": f"{len(wrong_rejections)}/{n_yes}", "wrong_rejection_ids": wrong_rejections,
        "missed_rejections": f"{len(missed)}/{n_no}", "missed_rejection_ids": missed,
        "partly": [r["id"] for r in judged if r["answerable"] == "partly"],
        "meta_correct": f"{sum(r['correct'] for r in judged if r['expect'] == 'meta')}/{sum(r['expect'] == 'meta' for r in judged)}",
        "tokens": sum((r["tokens"] or {}).get("prompt", 0) + (r["tokens"] or {}).get("completion", 0) for r in judged),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="pre-check only; no LLM calls")
    args = parser.parse_args()
    results = []
    for case in load_cases():
        r = judge(case, offline=args.check)
        mark = "-" if r["correct"] is None else "OK " if r["correct"] else "BAD"
        print(f"{mark:<4} expect {r['expect']:<4} got {str(r['answerable']):<7} {r['id']:<24} {r['reason'] or ''}")
        results.append(r)
    summary = summarize(results)
    print(f"\n{json.dumps(summary, indent=2)}")
    if not args.check:
        RESULTS_DIR.mkdir(exist_ok=True)
        out = RESULTS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_gate-eval.json"
        out.write_text(json.dumps({"summary": summary, "results": results}, indent=2), encoding="utf-8")
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
