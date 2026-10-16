"""Integration smoke test for Instant-Memory's episodic and procedural APIs.

Run from the repository root:
    uv run python testing_agentic_memory.py

This writes temporary records to your configured MongoDB and removes only those
records in a finally block. Set MEMORI_TEST_USER_ID to choose a test tenant.
"""
from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

if not os.getenv("mongodb_URI"):
    print("ERROR: mongodb_URI is missing. Configure it in your .env file.")
    sys.exit(1)

from memori import (  # noqa: E402
    get_agent_task,
    list_resumable_agent_tasks,
    record_agent_event,
    retrieve_agent_history,
    save_agent_task,
    search_agent_events,
)
from memori.mongodb_connection import db  # noqa: E402


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS  {message}")


def main() -> int:
    user_id = os.getenv("MEMORI_TEST_USER_ID", "agentic_memory_test_user")
    run_id = uuid.uuid4().hex[:12]
    session_id = f"smoke-session-{run_id}"
    turn_id = f"smoke-turn-{run_id}"
    task_id = f"smoke-task-{run_id}"
    event_ids = [f"smoke-event-{run_id}-user", f"smoke-event-{run_id}-tool"]
    unique_phrase = f"agentic-memory-smoke-{run_id}"
    events = db["agent_events"]
    tasks = db["agent_tasks"]

    print("\nInstant-Memory agentic API integration test")
    print(f"Test user: {user_id}")
    print("Uses the configured MongoDB; temporary test records are cleaned up.\n")

    try:
        # 1. Episodic memory: store a user message and a tool result.
        record_agent_event(
            user_id, session_id, turn_id, "user",
            content=f"Please investigate {unique_phrase}",
            event_type="message",
            event_id=event_ids[0],
        )
        record_agent_event(
            user_id, session_id, turn_id, "tool",
            content=f"Tool result: {unique_phrase} completed successfully",
            event_type="tool_result",
            tool_name="smoke_test_tool",
            tool_call_id=f"call-{run_id}",
            tool_result={"status": "ok", "test_run": run_id},
            event_id=event_ids[1],
        )
        check(
            events.count_documents({"user_id": user_id, "event_id": {"$in": event_ids}}) == 2,
            "record_agent_event stores user and tool events",
        )

        # 2. Retry the same event ID and ensure it does not create a duplicate.
        record_agent_event(
            user_id, session_id, turn_id, "user",
            content="This retry must not create a second record",
            event_id=event_ids[0],
        )
        check(
            events.count_documents({"user_id": user_id, "event_id": event_ids[0]}) == 1,
            "event_id retries are idempotent",
        )

        # 3. Retrieve chronological session history.
        history = retrieve_agent_history(user_id, session_id=session_id)
        check(
            len(history) == 2 and [item["event_id"] for item in history] == event_ids,
            "retrieve_agent_history returns the session events in order",
        )

        # 4. Search events across sessions using the unique phrase.
        matches = search_agent_events(user_id, unique_phrase)
        check(
            any(item.get("event_id") == event_ids[1] for item in matches),
            "search_agent_events finds the saved tool result",
        )

        # 5. Save and retrieve resumable procedural/task state.
        saved = save_agent_task(
            user_id,
            task_id,
            goal=f"Smoke test task {run_id}",
            status="paused",
            session_id=session_id,
            current_step="Verify integration test",
            completed_steps=["Create test session"],
            pending_steps=["Resume and complete test task"],
            tool_history=[{"tool": "smoke_test_tool", "status": "ok"}],
            context={"test_run": run_id},
        )
        loaded = get_agent_task(user_id, task_id)
        resumable = list_resumable_agent_tasks(user_id)
        check(saved.get("status") == "paused", "save_agent_task stores paused task state")
        check(
            loaded is not None and loaded.get("current_step") == "Verify integration test",
            "get_agent_task retrieves saved task state",
        )
        check(
            any(item.get("task_id") == task_id for item in resumable),
            "list_resumable_agent_tasks includes the paused task",
        )

        print("\nALL CHECKS PASSED")
        print("The agentic APIs successfully wrote to and read from MongoDB.")
        print("Note: this smoke test does not test automatic tool interception or LLM integration.")
        return 0

    except Exception as exc:
        print(f"\nTEST FAILED: {type(exc).__name__}: {exc}")
        return 1

    finally:
        # Delete only documents created by this run, never the user's other data.
        try:
            events.delete_many({"user_id": user_id, "event_id": {"$in": event_ids}})
            tasks.delete_one({"user_id": user_id, "task_id": task_id})
            print("Cleaned up temporary smoke-test records.")
        except Exception as cleanup_exc:
            print(f"WARNING: automatic cleanup failed: {cleanup_exc}")
            print(f"Clean manually using user_id={user_id!r}, task_id={task_id!r}.")


if __name__ == "__main__":
    raise SystemExit(main())
