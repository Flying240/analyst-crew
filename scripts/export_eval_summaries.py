"""Write compact, committed eval summaries to evals/summaries/ (evals/results/ is gitignored).

One file per run: score, breakdowns, tokens and per-question pass/fail (no SQL), so the app's
Eval dashboard works when deployed. Runs listed in evals/*_50q_runs.json are marked official.
Usage:
    python scripts/export_eval_summaries.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evals.run_eval import RESULTS_DIR  # noqa: E402
from src.run_cache import write_json  # noqa: E402

SUMMARIES_DIR = ROOT / "evals" / "summaries"
FILE_RE = re.compile(r"^(\d{8})_(\d{6})_(.+)\.json$")
QUESTION_FIELDS = ("id", "difficulty", "split", "passed", "round1_passed", "infra_error", "seconds", "tokens", "retries_used")


def official_files() -> dict[str, tuple[str, dict]]:
    """Result file name -> (official run set it belongs to, its entry in that set)."""
    out = {}
    for path in ROOT.glob("evals/*_50q_runs.json"):
        runs = json.loads(path.read_text(encoding="utf-8"))
        for r in runs.get("runs", []):
            out[Path(r["file"]).name] = (path.stem, r)
    return out


def summarize(path: Path, report: dict, official: dict[str, tuple[str, dict]]) -> dict:
    m = FILE_RE.match(path.name)
    d, t = m.group(1), m.group(2)
    run_set, entry = official.get(path.name, (None, {}))
    return {
        "file": path.name, "run": m.group(3), "date": f"{d[:4]}-{d[4:6]}-{d[6:]}", "time": f"{t[:2]}:{t[2:4]}",
        "official": run_set, "answerer": report.get("answerer"), "complete": report.get("complete", True),
        "score": report["score"], "total": report["total"], "infra_errors": report.get("infra_errors", 0),
        "by_difficulty": report.get("by_difficulty"), "by_split": report.get("by_split"), "tokens": report.get("tokens"),
        "critic": report.get("critic"), "round1": report.get("round1"), "tokens_by_model": entry.get("tokens_by_model"),
        "critic_model": (report.get("models") or {}).get("critic") if (report.get("critic") or {}).get("rounds") else None,
        "questions": [{k: r.get(k) for k in QUESTION_FIELDS} for r in report["results"]],
    }


def main() -> None:
    official = official_files()
    SUMMARIES_DIR.mkdir(exist_ok=True)
    n = 0
    for path in sorted(RESULTS_DIR.glob("*.json")):
        if not FILE_RE.match(path.name):
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        if "results" not in report or "score" not in report:
            continue  # multi-run summaries, critic/gate evals
        write_json(SUMMARIES_DIR / path.name, summarize(path, report, official))
        n += 1
    print(f"Wrote {n} run summaries to {SUMMARIES_DIR}")


if __name__ == "__main__":
    main()
