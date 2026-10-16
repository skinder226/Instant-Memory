"""Persistent episodic and procedural memory for tool-using agents.

This module is intentionally separate from semantic memory extraction:
- agent_events stores append-only conversation/tool events for replay and audit.
- agent_tasks stores the latest resumable state of an agent task.
All reads and writes are scoped by user_id. Callers must pass authenticated user IDs.
Redact credentials and other secrets from tool arguments/results before persisting them.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping

from pymongo import ASCENDING, DESCENDING

logger = logging.getLogger(__name__)

_EVENT_ROLES = {"user", "assistant", "tool", "system"}
_EVENT_TYPES = {
    "message",
    "tool_call",
    "tool_result",
    "task_started",
    "task_step",
    "task_completed",
    "task_failed",
    "task_paused",
}
_TASK_STATUSES = {"pending", "running", "paused", "completed", "failed", "cancelled"}
_MAX_TEXT_CHARS = 50_000
_MAX_JSON_CHARS = 50_000


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _required_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _bounded_text(value: str | None, name: str, maximum: int = _MAX_TEXT_CHARS) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string or None.")
    if len(value) > maximum:
        raise ValueError(f"{name} exceeds the {maximum}-character limit.")
    return value


def _json_size(value: Any, name: str) -> None:
    try:
        encoded = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be JSON-serializable.") from exc
    if len(encoded) > _MAX_JSON_CHARS:
        raise ValueError(f"{name} exceeds the {_MAX_JSON_CHARS}-character limit.")


def _collections():
    # Import lazily so importing memori does not force a database connection.
    from .mongodb_connection import db
    return db["agent_events"], db["agent_tasks"]


def ensure_agent_memory_indexes() -> None:
    """Create indexes used by agent event and task retrieval."""
    events, tasks = _collections()
    events.create_index(
        [("user_id", ASCENDING), ("event_id", ASCENDING)],
        unique=True,
        name="agent_event_user_event_unique",
    )
    events.create_index(
        [("user_id", ASCENDING), ("session_id", ASCENDING), ("created_at", DESCENDING)],
        name="agent_event_user_session_time",
    )
    events.create_index(
        [("user_id", ASCENDING), ("task_id", ASCENDING), ("created_at", DESCENDING)],
        name="agent_event_user_task_time",
        sparse=True,
    )
    tasks.create_index(
        [("user_id", ASCENDING), ("task_id", ASCENDING)],
        unique=True,
        name="agent_task_user_task_unique",
    )
    tasks.create_index(
        [("user_id", ASCENDING), ("status", ASCENDING), ("updated_at", DESCENDING)],
        name="agent_task_user_status_updated",
    )


def record_agent_event(
    user_id: str,
    session_id: str,
    turn_id: str,
    role: str,
    content: str = "",
    *,
    event_type: str = "message",
    task_id: str | None = None,
    tool_name: str | None = None,
    tool_call_id: str | None = None,
    tool_arguments: Mapping[str, Any] | list[Any] | None = None,
    tool_result: Any = None,
    metadata: Mapping[str, Any] | None = None,
    event_id: str | None = None,
) -> dict[str, Any]:
    """Append a structured event, returning its event_id and timestamp.

    Supply the same event_id again to make retries idempotent. Tool arguments
    and results should be redacted by the host application before this call.
    """
    user_id = _required_text(user_id, "user_id")
    session_id = _required_text(session_id, "session_id")
    turn_id = _required_text(turn_id, "turn_id")
    role = _required_text(role, "role").lower()
    event_type = _required_text(event_type, "event_type").lower()
    content = _bounded_text(content, "content") or ""
    if role not in _EVENT_ROLES:
        raise ValueError(f"role must be one of {sorted(_EVENT_ROLES)}.")
    if event_type not in _EVENT_TYPES:
        raise ValueError(f"event_type must be one of {sorted(_EVENT_TYPES)}.")
    if tool_arguments is not None:
        _json_size(tool_arguments, "tool_arguments")
    if tool_result is not None:
        _json_size(tool_result, "tool_result")
    if metadata is not None:
        _json_size(metadata, "metadata")

    events, _ = _collections()
    ensure_agent_memory_indexes()
    event_id = _required_text(event_id, "event_id") if event_id else str(uuid.uuid4())
    document = {
        "user_id": user_id,
        "event_id": event_id,
        "session_id": session_id,
        "turn_id": turn_id,
        "task_id": task_id,
        "role": role,
        "event_type": event_type,
        "content": content,
        "tool_name": tool_name,
        "tool_call_id": tool_call_id,
        "tool_arguments": dict(tool_arguments) if isinstance(tool_arguments, Mapping) else tool_arguments,
        "tool_result": tool_result,
        "metadata": dict(metadata or {}),
        "created_at": _now(),
    }
    # Idempotent retries must not duplicate a previously accepted event.
    events.update_one(
        {"user_id": user_id, "event_id": event_id},
        {"$setOnInsert": document},
        upsert=True,
    )
    saved = events.find_one(
        {"user_id": user_id, "event_id": event_id},
        {"_id": 0, "event_id": 1, "created_at": 1},
    )
    return saved or {"event_id": event_id, "created_at": document["created_at"]}


def retrieve_agent_history(
    user_id: str,
    *,
    session_id: str | None = None,
    task_id: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return recent events for a user/session/task in chronological order."""
    user_id = _required_text(user_id, "user_id")
    if session_id is not None and task_id is not None:
        raise ValueError("Provide session_id or task_id, not both.")
    if session_id is None and task_id is None:
        raise ValueError("session_id or task_id is required to scope history.")
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 500:
        raise ValueError("limit must be an integer between 1 and 500.")

    events, _ = _collections()
    query: dict[str, Any] = {"user_id": user_id}
    if session_id is not None:
        query["session_id"] = _required_text(session_id, "session_id")
    if task_id is not None:
        query["task_id"] = _required_text(task_id, "task_id")
    documents = list(
        events.find(query, {"_id": 0})
        .sort([("created_at", DESCENDING), ("event_id", DESCENDING)])
        .limit(limit)
    )
    documents.reverse()
    return documents


def save_agent_task(
    user_id: str,
    task_id: str,
    *,
    goal: str,
    status: str = "pending",
    session_id: str | None = None,
    current_step: str | None = None,
    completed_steps: list[Any] | None = None,
    pending_steps: list[Any] | None = None,
    tool_history: list[Any] | None = None,
    result_summary: str | None = None,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create or replace a user's resumable task state."""
    user_id = _required_text(user_id, "user_id")
    task_id = _required_text(task_id, "task_id")
    goal = _bounded_text(_required_text(goal, "goal"), "goal") or ""
    status = _required_text(status, "status").lower()
    if status not in _TASK_STATUSES:
        raise ValueError(f"status must be one of {sorted(_TASK_STATUSES)}.")
    for name, value in (
        ("completed_steps", completed_steps),
        ("pending_steps", pending_steps),
        ("tool_history", tool_history),
        ("context", context),
    ):
        if value is not None:
            _json_size(value, name)
    if result_summary is not None:
        _bounded_text(result_summary, "result_summary")

    _, tasks = _collections()
    ensure_agent_memory_indexes()
    now = _now()
    document = {
        "user_id": user_id,
        "task_id": task_id,
        "session_id": session_id,
        "goal": goal,
        "status": status,
        "current_step": current_step,
        "completed_steps": completed_steps or [],
        "pending_steps": pending_steps or [],
        "tool_history": tool_history or [],
        "result_summary": result_summary,
        "context": dict(context or {}),
        "updated_at": now,
    }
    tasks.update_one(
        {"user_id": user_id, "task_id": task_id},
        {"$set": document, "$setOnInsert": {"created_at": now}},
        upsert=True,
    )
    return tasks.find_one({"user_id": user_id, "task_id": task_id}, {"_id": 0}) or document


def get_agent_task(user_id: str, task_id: str) -> dict[str, Any] | None:
    """Retrieve one task state, scoped to the owning user."""
    user_id = _required_text(user_id, "user_id")
    task_id = _required_text(task_id, "task_id")
    _, tasks = _collections()
    return tasks.find_one({"user_id": user_id, "task_id": task_id}, {"_id": 0})


def list_resumable_agent_tasks(user_id: str, limit: int = 50) -> list[dict[str, Any]]:
    """List paused, pending, or running tasks for a user."""
    user_id = _required_text(user_id, "user_id")
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 200:
        raise ValueError("limit must be an integer between 1 and 200.")
    _, tasks = _collections()
    return list(
        tasks.find(
            {"user_id": user_id, "status": {"$in": ["pending", "running", "paused"]}},
            {"_id": 0},
        )
        .sort("updated_at", DESCENDING)
        .limit(limit)
    )
