"""Local per-model token ledger for the current UTC day, plus the latest Groq rate-limit headers.

Groq returns requests-per-day and tokens-per-minute headers but no tokens-per-day header,
so daily tokens are summed locally from each response's usage (an estimate).
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from src.config import CACHE_DIR

RATE_HEADERS = (
    "x-ratelimit-limit-requests", "x-ratelimit-remaining-requests", "x-ratelimit-reset-requests",
    "x-ratelimit-limit-tokens", "x-ratelimit-remaining-tokens", "x-ratelimit-reset-tokens",
)

Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def today(clock: Clock = utc_now) -> str:
    return clock().strftime("%Y-%m-%d")


def ledger_path(day: str, root: Path = CACHE_DIR) -> Path:
    return root / f"usage_{day}.json"


def load(clock: Clock = utc_now, root: Path = CACHE_DIR) -> dict[str, Any]:
    """Today's ledger: {"models": {model: {prompt, completion, requests}}, "headers": {model: {...}}}."""
    path = ledger_path(today(clock), root)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"models": {}, "headers": {}}


def save(data: dict[str, Any], clock: Clock = utc_now, root: Path = CACHE_DIR) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = ledger_path(today(clock), root)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def rate_headers(headers: Mapping[str, str] | None) -> dict[str, str]:
    """Keep only Groq's x-ratelimit-* headers."""
    if not headers:
        return {}
    return {h: headers[h] for h in RATE_HEADERS if headers.get(h) is not None}


def add(
    model: str, prompt: int, completion: int, headers: Mapping[str, str] | None = None,
    clock: Clock = utc_now, root: Path = CACHE_DIR,
) -> None:
    data = load(clock, root)
    m = data["models"].setdefault(model, {"prompt": 0, "completion": 0, "requests": 0})
    m["prompt"] += int(prompt or 0)
    m["completion"] += int(completion or 0)
    m["requests"] += 1
    found = rate_headers(headers)
    if found:
        data["headers"][model] = {**found, "seen_at": clock().isoformat(timespec="seconds")}
    save(data, clock, root)


def tokens_today(model: str, clock: Clock = utc_now, root: Path = CACHE_DIR) -> int:
    m = load(clock, root)["models"].get(model, {})
    return m.get("prompt", 0) + m.get("completion", 0)
