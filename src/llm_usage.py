"""Token usage of the LLM calls for the current question (summed over retries), so the eval can log it without changing answerer signatures."""

_last: dict[str, int] | None = None


def record(prompt_tokens: int, completion_tokens: int) -> None:
    global _last
    prev = _last or {"prompt": 0, "completion": 0}
    _last = {"prompt": prev["prompt"] + int(prompt_tokens or 0), "completion": prev["completion"] + int(completion_tokens or 0)}


def take() -> dict[str, int] | None:
    """Return the usage recorded since the last take(), and clear it."""
    global _last
    usage, _last = _last, None
    return usage
