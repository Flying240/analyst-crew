"""Token usage of the most recent LLM call, so the eval can log it without changing answerer signatures."""

_last: dict[str, int] | None = None


def record(prompt_tokens: int, completion_tokens: int) -> None:
    global _last
    _last = {"prompt": int(prompt_tokens or 0), "completion": int(completion_tokens or 0)}


def take() -> dict[str, int] | None:
    """Return the usage recorded since the last take(), and clear it."""
    global _last
    usage, _last = _last, None
    return usage
