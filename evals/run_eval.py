"""Eval harness: run an answerer over evals/questions.json and score it against gold SQL.

An answerer is any function `answer(question: str) -> str` that returns SQL.
Usage:
    python evals/run_eval.py                                  # oracle (returns gold SQL)
    python evals/run_eval.py --answerer src.agents.baseline:answer
    python evals/run_eval.py --answerer src.agents.baseline:answer --runs 3
    python evals/run_eval.py --answerer src.agents.baseline:answer --include evals/results/<earlier run>.json
    python evals/run_eval.py --answerer src.agents.baseline:answer --retries 2   # feed DB errors back, retry up to 2x

A run stopped by a daily quota is saved as *_incomplete.json and left out of the summary.
"""
import argparse
import functools
import importlib
import json
import math
import re
import subprocess
import sys
import time
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import CRITIC_MODEL, DATA_MODEL, PLANNER_GATE, PLANNER_MODEL  # noqa: E402
from src import llm_usage, retry_info, trace_info  # noqa: E402
from src.llm_retry import DailyLimitError, InfraError  # noqa: E402
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
    infra_error = daily_limit = False
    llm_usage.take()
    retry_info.take()
    trace_info.take()
    try:
        sql = answer(q["question"])
        got = run_sql(sql)
    except InfraError as exc:
        sql, got, infra_error = "", SqlResult(error=f"infra_error: {exc}"), True
        daily_limit = isinstance(exc, DailyLimitError)
    except Exception as exc:
        sql, got = "", SqlResult(error=f"answerer failed: {exc}")
    gold = run_sql(q["gold_sql"])
    retries_used = retry_info.take()
    trace = trace_info.take()
    result = {
        "id": q["id"],
        "difficulty": q.get("difficulty", "unknown"),
        "split": q.get("split", "unknown"),
        "passed": not infra_error and score(got, gold, q["check"]),
        "infra_error": infra_error,
        "daily_limit": daily_limit,
        "tokens": llm_usage.take(),
        "retries_used": retries_used,
        "from_retry": retries_used > 0,
        "sql": sql,
        "error": got.error or gold.error,
        "seconds": round(time.perf_counter() - start, 2),
    }
    if trace is not None:
        label_critic_rounds(trace, gold, q["check"])
        result["trace"] = trace
        if trace.get("returned", "no_critic") != "no_critic" and not infra_error:
            result["round1_passed"] = round1_passed(trace, result["passed"])
    return result


def label_critic_rounds(trace: dict, gold: SqlResult, check: dict) -> None:
    """Score each Critic round's SQL against gold and label the verdict (false_alarm = flagged a correct SQL)."""
    for rnd in trace.get("critic_rounds", []):
        rnd["correct"] = score(run_sql(rnd["sql"]), gold, check)
        flagged = rnd["verdict"] == "revise"
        rnd["outcome"] = {
            (True, True): "false_alarm", (True, False): "caught",
            (False, True): "confirmed", (False, False): "missed",
        }[(flagged, rnd["correct"])]


def round1_passed(trace: dict, final_passed: bool) -> bool:
    """Whether the Data agent's first SQL (before any Critic rewrite) was correct: the plain "graph" score."""
    rounds = trace.get("critic_rounds", [])
    return rounds[0]["correct"] if rounds else final_passed  # no rounds: the SQL errored, so first == final


def round1_tally(results: list[dict]) -> dict[str, int] | None:
    """Score round-1 SQL in a --critic run, so one run gives both the graph and graph + critic rows."""
    if not any("round1_passed" in r for r in results):
        return None
    infra = sum(r.get("infra_error", False) for r in results)
    return {"score": sum(r.get("round1_passed", False) for r in results), "total": len(results) - infra, "infra_errors": infra}


CRITIC_OUTCOMES = ("caught", "false_alarm", "missed", "confirmed")


def critic_totals(results: list[dict]) -> dict:
    """Count Critic round outcomes and how the returned SQL was chosen."""
    rounds = [r for res in results for r in (res.get("trace") or {}).get("critic_rounds", [])]
    returned: dict[str, int] = {}
    for res in results:
        if res.get("trace") and "returned" in res["trace"]:
            how = res["trace"]["returned"]
            returned[how] = returned.get(how, 0) + 1
    return {
        "rounds": len(rounds),
        **{o: sum(r["outcome"] == o for r in rounds) for o in CRITIC_OUTCOMES},
        "parse_errors": sum(bool(r.get("parse_error")) for r in rounds),
        "returned": returned,
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


def answerer_slug(answerer: str, retries: int = 0) -> str:
    """Filename-safe name for an answerer and its flags.

    src.agents.baseline:answer_with_tips, retries 2 -> baseline-answer_with_tips-r2;
    src.graph:answer+critic, retries 2 -> graph-critic-r2 (the default function `answer` is left out).
    """
    flags = [f for f in ("gate", "critic") if f"+{f}" in answerer]
    spec = answerer.split("+")[0]
    module, _, func = spec.partition(":")
    clean = lambda t: re.sub(r"[^a-z0-9_]+", "-", t.lower()).strip("-")
    parts = [clean(module.split(".")[-1]) or "unknown"]
    if func and func != "answer":
        parts.append(clean(func))
    parts += flags
    if retries:
        parts.append(f"r{retries}")
    return "-".join(parts)


def token_totals(results: list[dict]) -> dict:
    used = [r["tokens"] for r in results if r.get("tokens")]
    prompt, completion = sum(u["prompt"] for u in used), sum(u["completion"] for u in used)
    n = len(used)
    return {
        "questions": n,
        "prompt": prompt,
        "completion": completion,
        "mean_prompt": round(prompt / n, 1) if n else 0,
        "mean_completion": round(completion / n, 1) if n else 0,
    }


def run_questions(questions: list[dict], answer: Answerer) -> tuple[list[dict], bool]:
    """Evaluate questions in order; stop early if the provider's daily quota runs out."""
    results = []
    for q in questions:
        r = evaluate(q, answer)
        status = "INFRA" if r["infra_error"] else "PASS" if r["passed"] else "FAIL"
        print(f"{status:<5} {r['difficulty']:<7} {r['id']:<30} {r['seconds']:>6}s  {r['error'] or ''}")
        results.append(r)
        if r["daily_limit"]:
            print(f"\nDaily quota exhausted at {r['id']}: stopping this run ({len(results)}/{len(questions)} asked).")
            return results, False
    return results, True


def code_version() -> str:
    """Git commit of the code that ran, with "-dirty" if there were uncommitted changes."""
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    commit = git("rev-parse", "--short", "HEAD") or "unknown"
    return commit + ("-dirty" if git("status", "--porcelain", "--untracked-files=no") else "")


def run_meta(retries: int = 0) -> dict:
    """What a run's score depends on besides the answerer name; merged parts must agree on all of it."""
    return {
        "retries": retries,
        "models": {"planner": PLANNER_MODEL, "data": DATA_MODEL, "critic": CRITIC_MODEL},
        "code_version": code_version(),
    }


def build_report(results: list[dict], answerer: str, complete: bool, meta: dict | None = None) -> dict:
    return {
        "answerer": answerer, "complete": complete, **(meta or {}), **tally(results),
        "by_difficulty": by_difficulty(results), "by_split": by_split(results), "tokens": token_totals(results),
        "critic": critic_totals(results), "round1": round1_tally(results), "results": results,
    }


def run_once(
    questions: list[dict], answer: Answerer, answerer: str, slug: str | None = None, meta: dict | None = None,
) -> dict:
    results, complete = run_questions(questions, answer)
    report = build_report(results, answerer, complete, meta)
    tokens, critic, round1 = report["tokens"], report["critic"], report["round1"]
    print()
    for name, b in [*report["by_difficulty"].items(), *report["by_split"].items()]:
        print(f"{name:<8} {fmt(b)}")
    if tokens["questions"]:
        print(f"tokens   prompt {tokens['mean_prompt']}/q, completion {tokens['mean_completion']}/q, total {tokens['prompt'] + tokens['completion']}")
    if critic["rounds"]:
        print("critic   " + ", ".join(f"{k} {critic[k]}" for k in ("rounds", *CRITIC_OUTCOMES, "parse_errors")) + f"; returned {critic['returned']}")
    if round1:
        print(f"round-1  {fmt(round1)}  (first SQL, before Critic rewrites = the plain graph row)")
    print(f"\nScore: {fmt(report)}  answerer={answerer}{'' if complete else '  [INCOMPLETE]'}")
    save(report, (slug or answerer_slug(answerer)) + ("" if complete else "_incomplete"))
    return report


MERGE_KEYS = ("answerer", "retries", "models", "code_version")


def merge_reports(parts: list[dict], all_ids: list[str]) -> dict:
    """One complete run from part runs with disjoint --ids that together cover every question.

    Raises ValueError if a part is incomplete, ids overlap or are missing, or the parts differ in
    answerer (with flags), retries, models or code version.
    """
    if len(parts) < 2:
        raise ValueError("--merge needs at least two part files")
    for key in MERGE_KEYS:
        values = [p.get(key) for p in parts]
        if any(v is None for v in values):
            raise ValueError(f"a part file has no '{key}' (saved before merging was supported)")
        if any(v != values[0] for v in values):
            raise ValueError(f"parts differ in {key}: {values}")
    if not all(p.get("complete", True) for p in parts):
        raise ValueError("a part run is incomplete (stopped by a daily quota); rerun its ids first")
    seen: dict[str, int] = {}
    for i, p in enumerate(parts):
        for r in p["results"]:
            if r["id"] in seen:
                raise ValueError(f"{r['id']} is in part {seen[r['id']] + 1} and part {i + 1}")
            seen[r["id"]] = i
    missing = [q for q in all_ids if q not in seen]
    extra = [q for q in seen if q not in set(all_ids)]
    if missing or extra:
        raise ValueError(f"parts must cover exactly the {len(all_ids)} questions; missing {missing}, unknown {extra}")
    order = {q: i for i, q in enumerate(all_ids)}
    results = sorted((r for p in parts for r in p["results"]), key=lambda r: order[r["id"]])
    meta = {k: parts[0][k] for k in MERGE_KEYS if k != "answerer"}
    return {**build_report(results, parts[0]["answerer"], True, meta), "merged": True, "parts": len(parts)}


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

    Incomplete runs (stopped by a daily quota) are excluded from everything but the
    `incomplete_runs` count. Infra-error runs of a question are left out of its pass/fail
    history and listed separately.
    """
    incomplete = [i + 1 for i, r in enumerate(reports) if not r.get("complete", True)]
    reports = [r for r in reports if r.get("complete", True)]
    if not reports:
        return {"runs": 0, "incomplete_runs": incomplete}
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
        "incomplete_runs": incomplete,
        "tokens": [r.get("tokens") for r in reports],
        "overall": spread(reports),
        "by_difficulty": {lv: spread([r["by_difficulty"][lv] for r in reports]) for lv in levels},
        "by_split": {sp: spread([r["by_split"][sp] for r in reports]) for sp in reports[0].get("by_split", {})},
        "flipped": {qid: runs for qid, runs in outcomes.items() if len(set(runs)) > 1},
        "always_failed": [qid for qid, runs in outcomes.items() if not any(runs)],
        "infra_errors": infra,
        "round1": spread([r["round1"] for r in reports]) if all(r.get("round1") for r in reports) else None,
        "critic": {o: sum(r.get("critic", {}).get(o, 0) for r in reports) for o in ("rounds", *CRITIC_OUTCOMES, "parse_errors")},
    }


def print_summary(s: dict) -> None:
    skipped = f" (excluded incomplete runs: {s['incomplete_runs']})" if s["incomplete_runs"] else ""
    print(f"\n=== Summary over {s['runs']} complete runs{skipped} ===")
    if not s["runs"]:
        return
    for name, b in [("overall", s["overall"]), *s["by_difficulty"].items(), *s["by_split"].items()]:
        infra = f"  ({b['infra_errors']} infra_error, not scored)" if b["infra_errors"] else ""
        print(f"{name:<8} mean {b['mean_pct']}%  range {b['min_pct']}-{b['max_pct']}%  runs {', '.join(b['scores'])}{infra}")
    if s.get("round1"):
        b = s["round1"]
        print(f"round-1  mean {b['mean_pct']}%  range {b['min_pct']}-{b['max_pct']}%  runs {', '.join(b['scores'])}  (graph row, same runs)")
    print("Flipped between runs:")
    for qid, runs in s["flipped"].items():
        print(f"  {qid:<30} {' '.join('P' if p else 'F' for p in runs)}")
    if not s["flipped"]:
        print("  (none)")
    print(f"Failed every run: {', '.join(s['always_failed']) or '(none)'}")
    infra = ", ".join(f"{qid} x{n}" for qid, n in s["infra_errors"].items())
    print(f"Infra errors (not scored): {infra or '(none)'}")
    for i, t in enumerate(s["tokens"], 1):
        if t and t["questions"]:
            print(f"Run {i} tokens: prompt {t['mean_prompt']}/q, completion {t['mean_completion']}/q")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answerer", default="oracle", help="'oracle' or module:function")
    parser.add_argument("--runs", type=int, default=1, help="repeat the eval N times and summarize")
    parser.add_argument(
        "--include", nargs="*", default=[], metavar="RESULT_JSON",
        help="saved run files to summarize together with this invocation's runs (e.g. runs from earlier days)",
    )
    parser.add_argument(
        "--retries", type=int, default=0,
        help="on a database error, send the error back to the answerer and retry up to N times (answerer must accept retries=)",
    )
    parser.add_argument(
        "--critic", action="store_true",
        help="turn on the Critic loop (answerer must accept critic=); results are labelled '<answerer>+critic'",
    )
    parser.add_argument(
        "--gate", action="store_true",
        help="turn on the Planner question gate (answerer must accept gate=); results are labelled '<answerer>+gate'",
    )
    parser.add_argument(
        "--no-gate", action="store_true",
        help="turn the gate off for src.graph:answer, which has it on by default (PLANNER_GATE=1)",
    )
    parser.add_argument(
        "--merge", nargs="*", default=[], metavar="RESULT_JSON",
        help="combine part runs with disjoint --ids covering all questions into one complete run; no LLM calls",
    )
    parser.add_argument("--ids", nargs="*", default=[], metavar="ID", help="only ask these question ids")
    args = parser.parse_args()

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    if args.merge:
        parts = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.merge]
        try:
            report = merge_reports(parts, [q["id"] for q in questions])
        except ValueError as exc:
            sys.exit(f"Cannot merge: {exc}")
        print(f"Merged {len(parts)} parts: {fmt(report)}  answerer={report['answerer']}")
        save(report, answerer_slug(report["answerer"], report["retries"]) + "_merged")
        return
    if args.ids:
        questions = [q for q in questions if q["id"] in set(args.ids)]
    answer = load_answerer(args.answerer, questions)
    if args.retries:
        answer = functools.partial(answer, retries=args.retries)
    if args.no_gate:
        answer = functools.partial(answer, gate=False)
    elif args.gate or (PLANNER_GATE and args.answerer.startswith("src.graph:")):
        answer = functools.partial(answer, gate=True)
        args.answerer += "+gate"
    if args.critic:
        answer = functools.partial(answer, critic=True)
        args.answerer += "+critic"
    slug = answerer_slug(args.answerer, args.retries)
    reports = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.include]
    for i in range(args.runs):
        if args.runs > 1:
            print(f"\n=== Run {i + 1}/{args.runs} ===")
        reports.append(run_once(questions, answer, args.answerer, slug, run_meta(args.retries)))
        if not reports[-1]["complete"]:
            print("Skipping remaining runs: the daily quota will not reset within this invocation.")
            break
    if len(reports) > 1:
        summary = summarize(reports)
        print_summary(summary)
        save(summary, f"{slug}_summary{len(reports)}")


if __name__ == "__main__":
    main()
