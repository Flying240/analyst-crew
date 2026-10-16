from datetime import datetime, timezone

from src import quota


def test_session_limit_rolling_hour():
    starts = [0, 100, 200, 300, 400]
    assert quota.session_wait(starts[:4], now=500) == 0
    assert quota.session_wait(starts, now=500) == 3600 - 500
    assert quota.session_wait(starts, now=3601) == 0
    assert quota.live_left(starts, now=500) == 0 and quota.live_left(starts, now=3650) == 1


def test_budget_blocks_at_ninety_percent():
    assert not quota.Budget("m", 179_999, 200_000).blocked
    assert quota.Budget("m", 180_000, 200_000).blocked


def test_seconds_to_utc_midnight():
    clock = lambda: datetime(2026, 10, 16, 23, 0, 0, tzinfo=timezone.utc)
    assert quota.seconds_to_utc_midnight(clock) == 3600
