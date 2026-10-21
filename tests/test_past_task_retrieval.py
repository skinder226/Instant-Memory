"""Regression tests for direct retrospective task retrieval."""
from memori import api


def test_past_task_question_is_detected():
    assert api._requires_past_task_lookup("What website did I ask you to create last time?")
    assert api._requires_past_task_lookup("Which task did I work on previously?")
    assert not api._requires_past_task_lookup("What project am I working on?")
    assert not api._requires_past_task_lookup("What is photosynthesis?")


def test_past_task_question_uses_task_search_without_scheduling_save(monkeypatch):
    expected = ["Past task — Original request: Create a mini website; Task goal: Create a mini website"]
    calls = []

    def fake_retrieve(user_id, query):
        calls.append((user_id, query))
        return expected

    def unexpected_submit(*args, **kwargs):
        raise AssertionError("Retrospective lookup must not schedule extraction of the question")

    monkeypatch.setattr(api, "_retrieve_past_tasks_fast", fake_retrieve)
    monkeypatch.setattr(api._background_executor, "submit", unexpected_submit)

    result = api.memori(
        "tenant-a",
        "What website did I ask you to create last time?",
        source="user",
    )

    assert calls == [("tenant-a", "What website did I ask you to create last time?")]
    assert result["retrieval"] == expected
    assert result["memory_save_scheduled"] is False


def test_past_task_retrieval_formats_only_available_task_fields(monkeypatch):
    monkeypatch.setattr(
        "memori.agentic_memory.search_agent_tasks",
        lambda user_id, query, limit: [
            {
                "user_id": user_id,
                "task_id": "task-1",
                "goal": "Create a mini website",
                "context": {"last_user_request": "Create a mini website"},
                "result_summary": "Static landing page completed",
            }
        ],
    )

    result = api._retrieve_past_tasks_fast(
        "tenant-a", "What website did I ask you to create last time?"
    )
    assert len(result) == 1
    assert "Task goal: Create a mini website" in result[0]
    assert "Result: Static landing page completed" in result[0]
    assert "task-1" not in result[0]
