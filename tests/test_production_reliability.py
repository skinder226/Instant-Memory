"""Tenant-isolation and failure-boundary regression tests; no live services required."""
from memori import api


def test_current_memory_cache_is_scoped_by_user(monkeypatch):
    """A cache hit for one tenant must never satisfy another tenant's lookup."""
    api._USER_MEMORY_CACHE.clear()
    calls = []

    class FakeCursor:
        def __init__(self, user_id):
            self.user_id = user_id

        def sort(self, *args, **kwargs):
            return self

        def __iter__(self):
            return iter([{
                "content": f"Private memory for {self.user_id}",
                "type": "fact",
                "temporal": {"is_current": True},
                "entities": [],
            }])

    class FakeCollection:
        def find(self, query, projection):
            calls.append(query["user_id"])
            return FakeCursor(query["user_id"])

    class FakeDB:
        def __getitem__(self, name):
            assert name == "memories"
            return FakeCollection()

    import memori.mongodb_connection as mongo
    monkeypatch.setattr(mongo, "db", FakeDB())

    try:
        first = api._get_current_user_memories("tenant-a")
        second = api._get_current_user_memories("tenant-b")
        first_again = api._get_current_user_memories("tenant-a")

        assert first[0]["content"] == "Private memory for tenant-a"
        assert second[0]["content"] == "Private memory for tenant-b"
        assert first_again[0]["content"] == "Private memory for tenant-a"
        assert calls == ["tenant-a", "tenant-b"]
    finally:
        api._USER_MEMORY_CACHE.clear()


def test_cache_query_excludes_explicitly_historical_memories(monkeypatch):
    api._USER_MEMORY_CACHE.clear()
    captured = {}

    class FakeCursor:
        def sort(self, *args, **kwargs):
            return self

        def __iter__(self):
            return iter([])

    class FakeCollection:
        def find(self, query, projection):
            captured["query"] = query
            return FakeCursor()

    class FakeDB:
        def __getitem__(self, name):
            return FakeCollection()

    import memori.mongodb_connection as mongo
    monkeypatch.setattr(mongo, "db", FakeDB())
    try:
        assert api._get_current_user_memories("tenant-a") == []
        assert captured["query"] == {
            "user_id": "tenant-a",
            "temporal.is_current": {"$ne": False},
        }
    finally:
        api._USER_MEMORY_CACHE.clear()


def test_background_task_submission_failure_does_not_affect_fast_paths(monkeypatch):
    """Fast paths must not submit background jobs; they remain deterministic under queue issues."""
    monkeypatch.setattr(
        api, "_retrieve_workplace_memories_fast",
        lambda user_id: ["The user works at Example Corp."],
    )

    class BrokenExecutor:
        def submit(self, *args, **kwargs):
            raise AssertionError("Fast-path retrieval must not submit a background job")

    monkeypatch.setattr(api, "_background_executor", BrokenExecutor())
    result = api.memori("tenant-a", "Where do I work?", source="user")
    assert result["retrieval"] == ["The user works at Example Corp."]
    assert result["memory_save_scheduled"] is False
