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
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping

from pymongo import ASCENDING, DESCENDING

logger = logging.getLogger(__name__)
_indexes_ready = False
_indexes_lock = threading.Lock()

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
    """Create indexes once per process, using a thread-safe lazy initializer."""
    global _indexes_ready
    if _indexes_ready:
        return
    with _indexes_lock:
        if _indexes_ready:
            return
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
        tasks.create_index(
            [("user_id", ASCENDING), ("created_at", DESCENDING)],
            name="agent_task_user_created",
        )
        _indexes_ready = True


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



def search_agent_events(user_id: str, query: str, limit: int = 20) -> list[dict[str, Any]]:
    """Search prior agent events across a user's sessions using lexical matching.

    This is a lightweight episodic lookup, not semantic/vector search. For best
    results, keep concise summaries in event content and task summaries.
    """
    import re

    user_id = _required_text(user_id, "user_id")
    query = _required_text(query, "query")
    if len(query) > 300:
        raise ValueError("query must be 300 characters or fewer.")
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
        raise ValueError("limit must be an integer between 1 and 100.")

    events, _ = _collections()
    pattern = re.escape(query)
    documents = list(
        events.find(
            {
                "user_id": user_id,
                "$or": [
                    {"content": {"$regex": pattern, "$options": "i"}},
                    {"tool_name": {"$regex": pattern, "$options": "i"}},
                    {"task_id": {"$regex": pattern, "$options": "i"}},
                ],
            },
            {"_id": 0},
        )
        .sort("created_at", DESCENDING)
        .limit(limit)
    )
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


def find_latest_active_agent_task(user_id: str, session_id: str) -> dict[str, Any] | None:
    """Find the most recently updated active task in one user's conversation session.

    This lets applications recreate AgenticMemoryRuntime between turns without
    accidentally creating a fresh task each time. The caller must keep session_id
    stable for the lifetime of the conversation.
    """
    user_id = _required_text(user_id, "user_id")
    session_id = _required_text(session_id, "session_id")
    _, tasks = _collections()
    return tasks.find_one(
        {
            "user_id": user_id,
            "session_id": session_id,
            "status": {"$in": ["pending", "running", "paused", "completed", "failed"]},
        },
        {"_id": 0},
        sort=[("updated_at", DESCENDING), ("created_at", DESCENDING)],
    )


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



def search_agent_tasks(user_id: str, query: str, limit: int = 50) -> list[dict[str, Any]]:
    """Rank a user's task history by the original request and goal.

    All statuses are included. For retrospective creation questions, prefer
    tasks that actually represent website/app creation requests and rank by
    original request/goal relevance before using creation time as a tie-breaker.
    Activity timestamps are deliberately not used for recency: a running task
    can be updated by the current question and otherwise outrank the task sought.
    """
    import re

    user_id = _required_text(user_id, "user_id")
    query = _required_text(query, "query")
    if len(query) > 500:
        raise ValueError("query must be 500 characters or fewer.")
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
        raise ValueError("limit must be an integer between 1 and 100.")

    # Historical ranking needs only compact text fields. Avoid transferring
    # potentially large tool_history/completed_steps payloads for every task.
    tasks = list_agent_tasks(
        user_id,
        limit=500,
        projection={
            "_id": 0,
            "user_id": 1,
            "task_id": 1,
            "goal": 1,
            "status": 1,
            "created_at": 1,
            "result_summary": 1,
            "context.last_user_request": 1,
            "context.created_from": 1,
        },
        sort_by_created=True,
    )
    tokens = [
        token.casefold()
        for token in re.findall(r"[a-zA-Z0-9_-]+", query)
        if len(token) > 2
    ]
    stop_words = {
        "what", "which", "when", "where", "who", "how", "did", "does",
        "the", "you", "your", "for", "and", "was", "were", "have", "has",
        "last", "time", "about", "from", "with", "that", "this", "ask",
        "asked", "tell", "type", "please", "me",
    }
    terms = [token for token in tokens if token not in stop_words]

    retrospective_creation = bool(
        re.search(r"\b(last time|previously|before|earlier|previous|last)\b", query, re.I)
        and re.search(r"\b(website|web site|site|landing page|web app|website)\b", query, re.I)
        and re.search(r"\b(create|build|make|design|develop|asked|request)\b", query, re.I)
    )

    def task_text(item: dict[str, Any]) -> tuple[str, str, str]:
        context = item.get("context") or {}
        request = str(context.get("last_user_request") or "")
        goal = str(item.get("goal") or "")
        summary = str(item.get("result_summary") or "")
        return request, goal, summary

    def is_creation_task(item: dict[str, Any]) -> bool:
        request, goal, summary = task_text(item)
        target_text = f"{goal} {request} {summary}".casefold()
        goal_text = goal.casefold()
        has_web_target = bool(re.search(
            r"\b(website|web site|websites|site|landing page|web app|web application)\b",
            target_text,
        ))
        has_creation_action = bool(re.search(
            r"\b(create|created|build|built|make|made|design|designed|develop|developed|landing page)\b",
            goal_text,
        ) or re.search(
            r"^\s*(create|build|make|design|develop|please create|please build)\b",
            request,
            re.I,
        ))
        # Do not treat the current retrospective question itself as the old
        # creation request, even though it is stored in last_user_request.
        request_is_meta_question = bool(re.search(
            r"\b(what type|what kind|did i ask|last time|previously|which website)\b",
            request,
            re.I,
        ))
        if request_is_meta_question and not re.search(
            r"^\s*(create|build|make|design|develop|please create|please build)\b",
            request,
            re.I,
        ):
            return False
        return has_web_target and has_creation_action

    scored: list[tuple[int, str, dict[str, Any]]] = []
    for item in tasks:
        request, goal, summary = task_text(item)
        request_is_meta_question = bool(re.search(
            r"\b(what type|what kind|did i ask|last time|previously|which website)\b",
            request,
            re.I,
        ))
        request_text = request.casefold() if not request_is_meta_question else ""
        goal_text = goal.casefold()
        summary_text = summary.casefold()
        context = item.get("context") or {}
        created_from = str(context.get("created_from") or "").casefold()

        if retrospective_creation and not is_creation_task(item):
            continue

        score = 0
        for term in terms:
            if term in request_text:
                score += 5
            if term in goal_text:
                score += 4
            if term in summary_text:
                score += 1
            if term in created_from:
                score += 1

        if retrospective_creation and is_creation_task(item):
            score += 8
        if score:
            created_at = item.get("created_at")
            # ISO strings and datetimes sort deterministically as strings.
            created_key = created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at or "")
            scored.append((score, created_key, item))

    scored.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
    return [item for _, _, item in scored[:limit]]


def list_agent_tasks(
    user_id: str,
    limit: int = 100,
    *,
    projection: Mapping[str, int] | None = None,
    sort_by_created: bool = False,
) -> list[dict[str, Any]]:
    """List tasks for one user, including completed/failed tasks.

    Optional projection and creation-time sorting support compact historical
    search without transferring potentially large tool and step histories.
    """
    user_id = _required_text(user_id, "user_id")
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 500:
        raise ValueError("limit must be an integer between 1 and 500.")
    if projection is not None and not isinstance(projection, Mapping):
        raise TypeError("projection must be a mapping or None.")
    _, tasks = _collections()
    cursor = tasks.find({"user_id": user_id}, projection or {"_id": 0})
    if sort_by_created:
        cursor = cursor.sort([("created_at", DESCENDING), ("task_id", DESCENDING)])
    else:
        cursor = cursor.sort([("updated_at", DESCENDING), ("created_at", DESCENDING)])
    return list(cursor.limit(limit))
