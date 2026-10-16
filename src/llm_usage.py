"""Token usage of the LLM calls for the current question (summed over retries), so the eval can log it without changing answerer signatures.

Each call is also kept separately (model, tokens) so the app can show usage per agent.
"""
from typing import Any

_last: dict[str, int] | None = None
_calls: list[dict[str, Any]] = []


def record(prompt_tokens: int, completion_tokens: int, model: str | None = None) -> None:
    global _last
    prompt, completion = int(prompt_tokens or 0), int(completion_tokens or 0)
    prev = _last or {"prompt": 0, "completion": 0}
    _last = {"prompt": prev["prompt"] + prompt, "completion": prev["completion"] + completion}
    _calls.append({"model": model, "prompt": prompt, "completion": completion})


def take() -> dict[str, int] | None:
    """Return the usage recorded since the last take(), and clear it."""
    global _last
    usage, _last = _last, None
    return usage


def take_calls() -> list[dict[str, Any]]:
    """Return the per-call usage recorded since the last take_calls(), and clear it."""
    calls = list(_calls)
    _calls.clear()
    return calls
