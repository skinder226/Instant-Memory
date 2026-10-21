"""Production regression tests for latency, retrieval intent, and task history.

These are deterministic unit tests: they do not require live MongoDB, Pinecone,
Neo4j, or an LLM provider. Run with: uv run pytest tests/test_production.py -v
"""
from datetime import datetime, timezone

from memori import api
from memori import agentic_memory


def test_greeting_skips_retrieval_and_background_llm(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A greeting must not start memory retrieval or LLM work")

    monkeypatch.setattr(api._background_executor, "submit", forbidden)
    monkeypatch.setattr(api, "_retrieve_memories_fast", forbidden)

    result = api.memori("tenant-a", "hey", source="user")

    assert result["retrieval"] == []
    assert result["memory_save_scheduled"] is False


def test_thanks_and_goodbye_skip_memory_pipeline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Small talk must bypass the memory pipeline")

    monkeypatch.setattr(api._background_executor, "submit", forbidden)
    monkeypatch.setattr(api, "_retrieve_memories_fast", forbidden)

    for message in ("Thanks!", "good morning", "bye"):
        result = api.memori("tenant-a", message, source="user")
        assert result["retrieval"] == []
        assert result["memory_save_scheduled"] is False


def test_generic_knowledge_question_skips_memory_retrieval(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Generic knowledge questions should not invoke memory retrieval")

    monkeypatch.setattr(api._background_executor, "submit", forbidden)
    monkeypatch.setattr(api, "_retrieve_memories_fast", forbidden)

    result = api.memori("tenant-a", "What is photosynthesis?", source="user")

    assert result["retrieval"] == []
    assert result["memory_save_scheduled"] is False


def test_personal_project_question_is_not_misclassified_as_generic(monkeypatch):
    expected = ["The user is building Neo."]
    monkeypatch.setattr(api, "_retrieve_project_memories_fast", lambda user_id, **kwargs: expected)
    monkeypatch.setattr(api._background_executor, "submit",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            AssertionError("Direct memory lookup should not route through an LLM")))

    result = api.memori("tenant-a", "What is my current project?", source="user")

    assert result["retrieval"] == expected
    assert result["memory_save_scheduled"] is False


def test_personal_workplace_question_is_not_misclassified_as_generic(monkeypatch):
    expected = ["The user works at Microsoft."]
    monkeypatch.setattr(api, "_retrieve_workplace_memories_fast", lambda user_id: expected)

    result = api.memori("tenant-a", "Where do I work?", source="user")

    assert result["retrieval"] == expected
    assert result["memory_save_scheduled"] is False


def test_personal_memory_intent_overrides_generic_question_shortcut():
    for question in (
        "What was my project before Orion?",
        "What do you remember about me?",
        "What website did I ask you to create last time?",
        "Who is my friend?",
    ):
        assert not api._is_low_value_chat_turn(question), question


def test_task_search_prefers_matching_original_creation_request(monkeypatch):
    tasks = [
        {
            "user_id": "tenant-a",
            "task_id": "current-chat",
            "goal": "Interactive Streamlit agent session",
            "status": "running",
            "created_at": datetime(2026, 10, 21, 10, tzinfo=timezone.utc),
            "context": {"last_user_request": "What website did I ask you to create last time?"},
            "result_summary": "Current chat session",
        },
        {
            "user_id": "tenant-a",
            "task_id": "prior-site",
            "goal": "Create a mini website",
            "status": "completed",
            "created_at": datetime(2026, 10, 20, 10, tzinfo=timezone.utc),
            "context": {"last_user_request": "OK create a mini website"},
            "result_summary": "Static landing page with HTML and CSS",
        },
    ]
    requested_users = []

    def fake_list(user_id, limit=100, **kwargs):
        requested_users.append(user_id)
        return tasks

    monkeypatch.setattr(agentic_memory, "list_agent_tasks", fake_list)

    ranked = agentic_memory.search_agent_tasks(
        "tenant-a", "What website did I ask you to create last time?"
    )

    assert requested_users == ["tenant-a"]
    assert ranked
    assert ranked[0]["task_id"] == "prior-site"
    assert all(item["task_id"] != "current-chat" for item in ranked)


def test_task_search_does_not_cross_user_boundary(monkeypatch):
    def fake_list(user_id, limit=100, **kwargs):
        assert user_id == "tenant-a"
        return [{
            "user_id": "tenant-a",
            "task_id": "tenant-a-site",
            "goal": "Create a mini website",
            "status": "completed",
            "context": {"last_user_request": "Create a mini website"},
        }]

    monkeypatch.setattr(agentic_memory, "list_agent_tasks", fake_list)
    result = agentic_memory.search_agent_tasks(
        "tenant-a", "What website did I ask you to create last time?"
    )

    assert result
    assert all(item["user_id"] == "tenant-a" for item in result)


def test_fast_path_helpers_do_not_call_external_llm():
    assert api._is_low_value_chat_turn("hey")
    assert api._is_low_value_chat_turn("What is photosynthesis?")
    assert not api._is_low_value_chat_turn("Where do I work?")
