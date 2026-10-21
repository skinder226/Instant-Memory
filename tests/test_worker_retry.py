"""Regression tests for provider retry behavior without live API calls."""
import httpx
from openai import APIStatusError

from memori import worker


def _status_error(status_code: int, headers: dict[str, str] | None = None) -> APIStatusError:
    request = httpx.Request("POST", "https://provider.invalid/v1/chat/completions")
    response = httpx.Response(status_code, request=request, headers=headers or {})
    return APIStatusError("temporary provider failure", response=response, body=None)


def test_retry_honors_numeric_retry_after(monkeypatch):
    waits = []
    monkeypatch.setattr(worker.time, "sleep", waits.append)

    class Runnable:
        calls = 0

        def invoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                raise _status_error(429, {"Retry-After": "2.5"})
            return "ok"

    runnable = Runnable()
    assert worker.invoke_with_retry(runnable, []) == "ok"
    assert runnable.calls == 2
    assert waits == [2.5]


def test_retry_uses_bounded_backoff_when_retry_after_is_invalid(monkeypatch):
    waits = []
    monkeypatch.setattr(worker.time, "sleep", waits.append)
    monkeypatch.setattr(worker.random, "uniform", lambda low, high: 0.0)

    class Runnable:
        calls = 0

        def invoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                raise _status_error(503, {"Retry-After": "not-a-number"})
            return "ok"

    assert worker.invoke_with_retry(Runnable(), []) == "ok"
    assert waits == [1]


def test_retry_does_not_sleep_past_worker_budget(monkeypatch):
    waits = []
    monkeypatch.setattr(worker.time, "sleep", waits.append)

    class Runnable:
        def invoke(self, messages):
            raise _status_error(429, {"Retry-After": "120"})

    import pytest

    with pytest.raises(APIStatusError):
        worker.invoke_with_retry(Runnable(), [])
    assert waits == []
