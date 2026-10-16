"""Cache of full run traces: live runs in .cache/runs (keyed by question + settings), eval examples in examples/.

Rows and charts are not stored; they are rebuilt by re-running the SQL read-only (no LLM call).
"""
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import CACHE_DIR, EXAMPLES_DIR, LLM_PROVIDER, PROMPT_SCHEMA
from src.config import CRITIC_MODEL, DATA_MODEL, PLANNER_MODEL, REPORTER_MODEL

RUNS_DIR = CACHE_DIR / "runs"
TRACE_FIELDS = (
    "sql", "error", "plan", "answerable", "gate_reason", "attempts", "critic_rounds", "returned", "report",
)


def normalize(question: str) -> str:
    return re.sub(r"\s+", " ", question).strip().lower().rstrip("?.! ").strip()


def settings(critic: bool, retries: int, gate: bool) -> dict[str, Any]:
    return {
        "critic": critic, "retries": retries, "gate": gate, "provider": LLM_PROVIDER, "schema": PROMPT_SCHEMA,
        "models": {"planner": PLANNER_MODEL, "data": DATA_MODEL, "critic": CRITIC_MODEL, "reporter": REPORTER_MODEL},
    }


def key(question: str, run_settings: dict[str, Any]) -> str:
    raw = json.dumps({"q": normalize(question), "s": run_settings}, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def make_record(
    question: str, run_settings: dict[str, Any], state: dict[str, Any], nodes: list[dict[str, Any]], source: str = "live",
) -> dict[str, Any]:
    """JSON-safe trace: the state's trace fields plus per-node seconds and LLM calls."""
    return {
        "question": question, "settings": run_settings, "source": source,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **{f: state[f] for f in TRACE_FIELDS if f in state}, "nodes": nodes,
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def save(record: dict[str, Any], root: Path = RUNS_DIR) -> Path:
    path = root / f"{key(record['question'], record['settings'])}.json"
    write_json(path, record)
    return path


def read(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_live(question: str, run_settings: dict[str, Any], root: Path = RUNS_DIR) -> dict[str, Any] | None:
    return read(root / f"{key(question, run_settings)}.json")


def load_examples(root: Path = EXAMPLES_DIR) -> list[dict[str, Any]]:
    return [r for p in sorted(root.glob("*.json")) if (r := read(p))]


def find_example(question: str, root: Path = EXAMPLES_DIR) -> dict[str, Any] | None:
    """Eval examples match on the question alone, whatever settings they ran with."""
    q = normalize(question)
    return next((r for r in load_examples(root) if normalize(r["question"]) == q), None)


def lookup(question: str, run_settings: dict[str, Any]) -> dict[str, Any] | None:
    return load_live(question, run_settings) or find_example(question)


def total_tokens(record: dict[str, Any]) -> int:
    return sum(c["prompt"] + c["completion"] for n in record.get("nodes", []) for c in n.get("calls", []))
