import re

import pytest

from memori import agentic_memory


class _FakeCursor:
    def __init__(self, documents):
        self.documents = list(documents)

    def sort(self, spec):
        if isinstance(spec, str):
            spec = [(spec, 1)]
        for key, direction in reversed(spec):
            self.documents.sort(key=lambda item: item.get(key) or "", reverse=direction < 0)
        return self

    def limit(self, count):
        self.documents = self.documents[:count]
        return self

    def __iter__(self):
        return iter(self.documents)


class _FakeCollection:
    def __init__(self):
        self.documents = []

    def create_index(self, *args, **kwargs):
        return kwargs.get("name", "fake_index")

    def update_one(self, query, update, upsert=False):
        existing = next((d for d in self.documents if all(d.get(k) == v for k, v in query.items())), None)
        if existing is None and upsert:
            existing = dict(update.get("$setOnInsert", {}))
            existing.update(update.get("$set", {}))
            self.documents.append(existing)
        elif existing is not None:
            existing.update(update.get("$set", {}))
            for key, value in update.get("$setOnInsert", {}).items():
                existing.setdefault(key, value)

    @staticmethod
    def _matches(document, query):
        for key, expected in query.items():
            if key == "$or":
                if not any(_FakeCollection._matches(document, option) for option in expected):
                    return False
                continue
            actual = document.get(key)
            if isinstance(expected, dict) and "$in" in expected:
                if actual not in expected["$in"]:
                    return False
            elif isinstance(expected, dict) and "$regex" in expected:
                if actual is None or not re.search(
                    expected["$regex"], str(actual),
                    re.IGNORECASE if expected.get("$options") == "i" else 0,
                ):
                    return False
            elif actual != expected:
                return False
        return True

    def find_one(self, query, projection=None):
        doc = next((d for d in self.documents if self._matches(d, query)), None)
        if doc is None:
            return None
        if projection and projection.get("_id") == 0:
            return {k: v for k, v in doc.items() if k != "_id"}
        return dict(doc)

    def find(self, query, projection=None):
        docs = [
            dict(d) for d in self.documents
            if self._matches(d, query)
        ]
        if projection and projection.get("_id") == 0:
            docs = [{k: v for k, v in d.items() if k != "_id"} for d in docs]
        return _FakeCursor(docs)


@pytest.fixture
def fake_db(monkeypatch):
    events = _FakeCollection()
    tasks = _FakeCollection()
    monkeypatch.setattr(agentic_memory, "_collections", lambda: (events, tasks))
    return events, tasks


def test_tool_call_and_result_can_be_stored_and_retrieved(fake_db):
    record = agentic_memory.record_agent_event(
        "user-1", "session-1", "turn-1", "assistant",
        event_type="tool_call", tool_name="search_web", tool_call_id="call-1",
        tool_arguments={"query": "FastAPI"},
        event_id="event-1",
    )
    agentic_memory.record_agent_event(
        "user-1", "session-1", "turn-1", "tool",
        event_type="tool_result", tool_name="search_web", tool_call_id="call-1",
        tool_result={"summary": "Found release notes"},
        event_id="event-2",
    )

    history = agentic_memory.retrieve_agent_history(
        "user-1", session_id="session-1"
    )
    assert record["event_id"] == "event-1"
    assert [event["event_type"] for event in history] == ["tool_call", "tool_result"]
    assert history[0]["tool_call_id"] == history[1]["tool_call_id"] == "call-1"


def test_event_id_makes_retries_idempotent(fake_db):
    for _ in range(2):
        agentic_memory.record_agent_event(
            "user-1", "session-1", "turn-1", "user", "hello",
            event_id="stable-event-id",
        )
    assert len(fake_db[0].documents) == 1


def test_history_cannot_be_read_without_session_or_task_scope():
    with pytest.raises(ValueError, match="session_id or task_id is required"):
        agentic_memory.retrieve_agent_history("user-1")


def test_task_state_can_be_saved_and_resumed(fake_db):
    saved = agentic_memory.save_agent_task(
        "user-1",
        "task-1",
        goal="Investigate a failed deployment",
        status="paused",
        current_step="Check service logs",
        completed_steps=["Check health endpoint"],
        pending_steps=["Inspect logs", "Retry deployment"],
    )
    loaded = agentic_memory.get_agent_task("user-1", "task-1")
    resumable = agentic_memory.list_resumable_agent_tasks("user-1")

    assert saved["status"] == "paused"
    assert loaded["current_step"] == "Check service logs"
    assert loaded["pending_steps"] == ["Inspect logs", "Retry deployment"]
    assert [task["task_id"] for task in resumable] == ["task-1"]


def test_task_state_is_tenant_scoped(fake_db):
    agentic_memory.save_agent_task(
        "user-1", "same-task-id", goal="User one's task", status="paused"
    )
    assert agentic_memory.get_agent_task("user-2", "same-task-id") is None


def test_invalid_event_role_is_rejected_before_database_access(monkeypatch):
    monkeypatch.setattr(
        agentic_memory, "_collections",
        lambda: (_ for _ in ()).throw(AssertionError("database should not be accessed")),
    )
    with pytest.raises(ValueError, match="role must be one of"):
        agentic_memory.record_agent_event(
            "user-1", "session-1", "turn-1", "browser-extension"
        )



def test_cross_session_episodic_search_is_user_scoped(fake_db):
    agentic_memory.record_agent_event(
        "user-1", "session-old", "turn-1", "tool",
        event_type="tool_result", content="Deployment failed after timeout",
        event_id="event-old",
    )
    agentic_memory.record_agent_event(
        "user-2", "session-other", "turn-1", "tool",
        event_type="tool_result", content="Deployment failed for another user",
        event_id="event-private",
    )

    results = agentic_memory.search_agent_events("user-1", "deployment failed")

    assert len(results) == 1
    assert results[0]["event_id"] == "event-old"
