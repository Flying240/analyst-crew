from datetime import datetime, timezone
from types import SimpleNamespace

from src import llm, llm_usage, usage_ledger
from src.llm_retry import InfraError, with_backoff


def at(hour: int, day: int = 16):
    return lambda: datetime(2026, 10, day, hour, 0, tzinfo=timezone.utc)


def test_ledger_sums_per_model_and_resets_at_utc_midnight(tmp_path):
    usage_ledger.add("m1", 100, 20, clock=at(10), root=tmp_path)
    usage_ledger.add("m1", 50, 5, clock=at(23), root=tmp_path)
    usage_ledger.add("m2", 7, 3, clock=at(23), root=tmp_path)
    assert usage_ledger.tokens_today("m1", clock=at(23), root=tmp_path) == 175
    assert usage_ledger.load(clock=at(23), root=tmp_path)["models"]["m1"]["requests"] == 2
    assert usage_ledger.tokens_today("m1", clock=at(0, day=17), root=tmp_path) == 0


def test_ledger_keeps_only_rate_limit_headers(tmp_path):
    headers = {"x-ratelimit-remaining-requests": "990", "x-ratelimit-limit-tokens": "8000", "date": "x"}
    usage_ledger.add("m1", 1, 1, headers, clock=at(10), root=tmp_path)
    saved = usage_ledger.load(clock=at(10), root=tmp_path)["headers"]["m1"]
    assert saved["x-ratelimit-remaining-requests"] == "990" and "date" not in saved


def test_call_groq_reads_headers_from_the_same_call(monkeypatch, tmp_path):
    monkeypatch.setattr(usage_ledger, "CACHE_DIR", tmp_path)
    added = []
    monkeypatch.setattr(usage_ledger, "add", lambda *a, **k: added.append(a))
    resp = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
        choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))],
    )
    creates = []

    class Raw:
        def create(self, **kw):
            creates.append(kw)
            return SimpleNamespace(headers={"x-ratelimit-remaining-tokens": "7000"}, parse=lambda: resp)

    class FakeGroq:
        def __init__(self, api_key):
            self.chat = SimpleNamespace(completions=SimpleNamespace(with_raw_response=Raw()))

    import groq
    monkeypatch.setattr(groq, "Groq", FakeGroq)
    monkeypatch.setenv("GROQ_API_KEY", "k")
    llm_usage.take(), llm_usage.take_calls()
    assert llm.call_groq("p", "some-model") == "hi"
    assert len(creates) == 1
    assert added == [("some-model", 10, 2, {"x-ratelimit-remaining-tokens": "7000"})]
    assert llm_usage.take_calls() == [{"model": "some-model", "prompt": 10, "completion": 2}]


def test_infra_error_carries_retry_after():
    class Limited(Exception):
        status_code = 429
        response = SimpleNamespace(headers={"retry-after": "12"})

    def call():
        raise Limited("rate limit")

    try:
        with_backoff(call, delays=(), sleep=lambda s: None)
    except InfraError as exc:
        assert exc.retry_after == 12.0
    else:
        raise AssertionError("expected InfraError")
