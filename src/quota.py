"""Limits for live (LLM) runs in the app: questions per session per hour, and a daily token budget per model.

Cached runs and examples are never limited.
"""
import os
import time
from dataclasses import dataclass

from src import usage_ledger
from src.config import APP_DAILY_TOKEN_BUDGET, BUDGET_STOP_FRACTION, DAILY_TOKEN_BUDGET

LIVE_PER_HOUR = int(os.getenv("APP_LIVE_PER_HOUR", "5"))
WINDOW_SECONDS = 3600
# On the deployed app (APP_MODE=deployed) visitors get the smaller budget, so the eval budget stays untouched.
DEPLOYED = os.getenv("APP_MODE", "local").lower() == "deployed"


def daily_budget() -> int:
    return APP_DAILY_TOKEN_BUDGET if DEPLOYED else DAILY_TOKEN_BUDGET


def session_wait(started: list[float], now: float | None = None, limit: int = LIVE_PER_HOUR) -> float:
    """Seconds until another live question is allowed (0 if allowed now), over a rolling hour."""
    now = time.time() if now is None else now
    recent = sorted(t for t in started if now - t < WINDOW_SECONDS)
    if len(recent) < limit:
        return 0.0
    return WINDOW_SECONDS - (now - recent[-limit])


def live_left(started: list[float], now: float | None = None, limit: int = LIVE_PER_HOUR) -> int:
    now = time.time() if now is None else now
    return max(0, limit - sum(now - t < WINDOW_SECONDS for t in started))


@dataclass
class Budget:
    model: str
    used: int
    budget: int

    @property
    def fraction(self) -> float:
        return min(1.0, self.used / self.budget) if self.budget else 1.0

    @property
    def blocked(self) -> bool:
        return self.used >= BUDGET_STOP_FRACTION * self.budget


def budget(model: str, clock: usage_ledger.Clock = usage_ledger.utc_now) -> Budget:
    return Budget(model, usage_ledger.tokens_today(model, clock), daily_budget())


def blocked_models(models: list[str]) -> list[Budget]:
    return [b for b in (budget(m) for m in dict.fromkeys(models)) if b.blocked]


def seconds_to_utc_midnight(clock: usage_ledger.Clock = usage_ledger.utc_now) -> int:
    now = clock()
    return 86400 - (now.hour * 3600 + now.minute * 60 + now.second)
