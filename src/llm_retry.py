"""Retry LLM calls on rate limits; surface persistent failures as infrastructure errors."""
import time
from typing import Callable, TypeVar

T = TypeVar("T")
BACKOFF_SECONDS = (2, 4, 8)


class InfraError(Exception):
    """The LLM provider failed for reasons unrelated to the answer (e.g. rate limits)."""


def is_rate_limit(exc: Exception) -> bool:
    if getattr(exc, "status_code", None) == 429 or getattr(exc, "code", None) == 429:
        return True
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "resource_exhausted" in text


def with_backoff(
    call: Callable[[], T],
    delays: tuple[float, ...] = BACKOFF_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run `call`, retrying rate-limit errors after each delay; raise InfraError if all fail."""
    for delay in (*delays, None):
        try:
            return call()
        except Exception as exc:
            if not is_rate_limit(exc):
                raise
            if delay is None:
                raise InfraError(f"rate limited after {len(delays) + 1} attempts: {exc}") from exc
            sleep(delay)
    raise AssertionError("unreachable")
