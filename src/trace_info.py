"""Agent trace (e.g. the graph's plan and SQL attempts) for the most recent answer, so the eval can save it."""
from typing import Any

_last: dict[str, Any] | None = None


def record(trace: dict[str, Any]) -> None:
    global _last
    _last = trace


def take() -> dict[str, Any] | None:
    """Return the trace recorded since the last take() (None if none), and clear it."""
    global _last
    trace, _last = _last, None
    return trace
