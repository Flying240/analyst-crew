"""Direct Critic eval: feed the Critic (question, SQL, result) triples, with no Planner or Data agent.

Cases are in evals/critic_cases.json: known-wrong SQL (should be flagged) and gold SQL (should pass).
Usage:
    python evals/critic_eval.py --check            # offline: confirm wrong SQL fails gold and gold passes
    python evals/critic_eval.py                    # dev cases (use these while tuning the Critic prompt)
    python evals/critic_eval.py --set heldout      # held-out wrong cases; run once, at the end
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evals.run_eval import QUESTIONS_PATH, RESULTS_DIR, score  # noqa: E402
from src import llm_usage  # noqa: E402
from src.agents.critic import review_sql  # noqa: E402
from src.tools.sql_tool import run_sql  # noqa: E402

CASES_PATH = ROOT / "evals" / "critic_cases.json"


def load_cases(which: str) -> list[dict]:
    """Each case: id, question, sql, expect ("revise" for wrong SQL, "ok" for gold), mistake."""
    questions = {q["id"]: q for q in json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))}
    raw = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    cases = [{**c, "expect": "revise"} for c in raw[f"{which}_wrong"]]
    cases += [{"id": i, "sql": questions[i]["gold_sql"], "expect": "ok", "mistake": "gold"} for i in raw.get(f"{which}_correct", [])]
    for c in cases:
        q = questions[c["id"]]
        if q["split"] != which:
            raise ValueError(f"{c['id']} is a {q['split']} question, not {which}")
        c["question"], c["gold_sql"], c["check"] = q["question"], q["gold_sql"], q["check"]
    return cases


def check_cases(cases: list[dict]) -> bool:
    """Wrong SQL must run and fail the gold check; gold SQL must pass."""
    ok = True
    for c in cases:
        got = run_sql(c["sql"])
        passed = score(got, run_sql(c["gold_sql"]), c["check"])
        good = got.ok and passed == (c["expect"] == "ok")
        ok &= good
        print(f"{'OK ' if good else 'BAD'} {c['expect']:6} {c['id']}  {got.error or ''}")
    return ok


def run_case(c: dict) -> dict:
    llm_usage.take()
    got = run_sql(c["sql"])
    review = review_sql(c["question"], c["sql"], got.columns, got.rows)
    return {
        "id": c["id"], "expect": c["expect"], "mistake": c["mistake"], **review,
        "correct_verdict": review["verdict"] == c["expect"], "tokens": llm_usage.take(), "sql": c["sql"],
    }


def rate(results: list[dict], expect: str) -> tuple[int, int]:
    group = [r for r in results if r["expect"] == expect]
    return sum(r["verdict"] == "revise" for r in group), len(group)


def print_case(r: dict) -> None:
    mark = "PASS" if r["correct_verdict"] else "FAIL"
    print(f"\n[{mark}] {r['id']} expect={r['expect']} verdict={r['verdict']} ({r['mistake']})")
    for item in r["checklist"]:
        print(f"   {item.get('status', '?'):8} {item.get('requirement', '')}  <-  {item.get('sql', '')}")
    if r["feedback"]:
        print(f"   feedback: {r['feedback']}")
    if r["parse_error"]:
        print("   (unparseable reply, counted as ok)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--set", choices=["dev", "heldout"], default="dev")
    p.add_argument("--check", action="store_true", help="validate the cases offline, no LLM calls")
    args = p.parse_args()
    cases = load_cases(args.set)
    if args.check:
        sys.exit(0 if check_cases(cases) else 1)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # model text can hold non-cp1252 characters
    results = [run_case(c) for c in cases]
    caught, wrong = rate(results, "revise")
    alarms, right = rate(results, "ok")
    summary = {"set": args.set, "caught": caught, "wrong": wrong, "false_alarms": alarms, "correct": right,
               "parse_errors": sum(r["parse_error"] for r in results),
               "tokens": sum(sum((r["tokens"] or {}).values()) for r in results)}
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_critic-eval-{args.set}.json"
    path.write_text(json.dumps({**summary, "results": results}, indent=2, default=str), encoding="utf-8")
    for r in results:
        print_case(r)
    print(f"\ncatch rate {caught}/{wrong}" + (f", false alarm rate {alarms}/{right}" if right else ""))
    print(f"saved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
