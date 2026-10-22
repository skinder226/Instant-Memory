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


def test_pinecone_batch_rejects_missing_embedding_vectors(monkeypatch):
    """Never mark a memory synchronized when the embedding batch is incomplete."""
    errors = []
    completed = []
    upserts = []

    class FakeEmbeddings:
        def embed_documents(self, contents):
            return []  # Provider bug: one input, zero vectors.

    class FakeIndex:
        def upsert(self, **kwargs):
            upserts.append(kwargs)

    monkeypatch.setattr(worker, "embeddings", FakeEmbeddings())
    monkeypatch.setattr(worker, "index", FakeIndex())
    monkeypatch.setattr(
        worker, "set_memory_error_or_attempts",
        lambda user_id, mongo_id, provider, error: errors.append(
            (user_id, str(mongo_id), provider, str(error))
        ),
    )
    monkeypatch.setattr(
        worker, "set_memory_status",
        lambda user_id, mongo_id, provider, status: completed.append(
            (user_id, str(mongo_id), provider, status)
        ),
    )

    memory = {
        "_id": "memory-1",
        "user_id": "test-user",
        "type": "profile",
        "content": "The user works at Example Corp.",
        "temporal": {"is_current": True},
    }
    worker._process_pinecone("test-user", [(memory, [])])

    assert len(errors) == 1
    assert errors[0][0:3] == ("test-user", "memory-1", "pinecone")
    assert "0 vectors for 1 memories" in errors[0][3]
    assert not completed
    assert not upserts
