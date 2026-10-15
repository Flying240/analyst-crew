"""Error-feedback retries used for the most recent answer, so the eval can log them without changing answerer signatures."""

_last: int | None = None


def record(retries_used: int) -> None:
    global _last
    _last = retries_used


def take() -> int:
    """Return the retries recorded since the last take() (0 if none), and clear it."""
    global _last
    used, _last = _last or 0, None
    return used
