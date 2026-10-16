"""Build the committed examples/ folder from saved eval results (zero-token examples for the app).

For each of the 50 questions, use the newest passing graph trace (plan, attempts, Critic rounds); if there is
none, the newest passing SQL from a complete 50-question baseline run. Questions with neither are skipped.
Eval runs skip the Reporter, so examples have no report; the app offers to generate one.
Usage:
    python scripts/build_example_cache.py
Re-run after the full graph + critic runs to replace SQL-only examples with full traces.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evals.run_eval import QUESTIONS_PATH, RESULTS_DIR  # noqa: E402
from src.config import EXAMPLES_DIR  # noqa: E402
from src.run_cache import write_json  # noqa: E402

FILE_RE = re.compile(r"^(\d{8})_(\d{6})_(.+)\.json$")


def result_files() -> list[tuple[str, str, dict]]:
    """(date YYYY-MM-DD, answerer slug, report) for saved runs, newest first; summaries and incomplete runs skipped."""
    out = []
    for path in sorted(RESULTS_DIR.glob("*.json"), reverse=True):
        m = FILE_RE.match(path.name)
        if not m or "summary" in m.group(3) or m.group(3).endswith("_incomplete"):
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        if "results" in report:
            d = m.group(1)
            out.append((f"{d[:4]}-{d[4:6]}-{d[6:]}", m.group(3), report))
    return out


def pick(qid: str, files: list[tuple[str, str, dict]]) -> tuple[str, str, dict] | None:
    """Newest passing graph trace, else newest passing SQL from a complete 50-question baseline run."""
    for want_trace in (True, False):
        for date, slug, report in files:
            if not want_trace and (not slug.startswith("baseline") or report.get("total") != 50):
                continue
            r = next((x for x in report["results"] if x["id"] == qid), None)
            if r and r.get("passed") and r.get("sql") and (("trace" in r) == want_trace):
                return date, slug, r
    return None


def example(q: dict, date: str, slug: str, r: dict) -> dict:
    trace = r.get("trace", {})
    nodes = [{"node": "eval", "seconds": r.get("seconds"), "calls": []}]
    return {
        "id": q["id"], "question": q["question"], "source": "eval", "eval_run": slug, "created_at": date,
        "settings": {"critic": "critic" in slug, "retries": int(m.group(1)) if (m := re.search(r"-r(\d+)", slug)) else None},
        "eval_tokens": r.get("tokens"), "sql": r["sql"], "error": None,
        "plan": trace.get("plan"), "attempts": trace.get("attempts", []), "critic_rounds": trace.get("critic_rounds", []),
        "returned": trace.get("returned"), "nodes": nodes,
    }


def main() -> None:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    files = result_files()
    EXAMPLES_DIR.mkdir(exist_ok=True)
    traced = sql_only = 0
    skipped = []
    for q in questions:
        found = pick(q["id"], files)
        if not found:
            skipped.append(q["id"])
            continue
        ex = example(q, *found)
        traced += bool(ex["plan"])
        sql_only += not ex["plan"]
        write_json(EXAMPLES_DIR / f"{q['id']}.json", ex)
    print(f"Wrote {traced + sql_only} examples to {EXAMPLES_DIR} ({traced} with graph traces, {sql_only} SQL only)")
    if skipped:
        print(f"Skipped (no passing run): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
