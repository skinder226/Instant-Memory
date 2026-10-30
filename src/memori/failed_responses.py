"""Durable storage and recovery helpers for interrupted agent responses."""
from __future__ import annotations

import re
import threading
from datetime import datetime, timezone
from typing import Any

from pymongo import ASCENDING, DESCENDING

_MAX_RESPONSE_CHARS = 200_000
_MAX_ERROR_CHARS = 4_000
_MARKER_OPEN = re.compile(r"(?m)^## Create File \(([^\r\n()]{1,240})\)\s*$")
_MARKER_CLOSE = re.compile(r"(?m)^## Ending file \(([^\r\n()]{1,240})\)\s*$")
_indexes_ready = False
_indexes_lock = threading.Lock()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _collection():
    global _indexes_ready
    from .mongodb_connection import db
    collection = db["failed_responses"]
    if not _indexes_ready:
        with _indexes_lock:
            if not _indexes_ready:
                collection.create_index(
                    [("user_id", ASCENDING), ("turn_id", ASCENDING)],
                    unique=True,
                    name="failed_response_user_turn_unique",
                )
                collection.create_index(
                    [("user_id", ASCENDING), ("task_id", ASCENDING), ("created_at", DESCENDING)],
                    name="failed_response_user_task_time",
                )
                collection.create_index(
                    [("user_id", ASCENDING), ("session_id", ASCENDING), ("created_at", DESCENDING)],
                    name="failed_response_user_session_time",
                )
                _indexes_ready = True
    return collection


def parse_response_files(response: str, *, allow_incomplete: bool = False) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Parse explicit file markers; return complete files and unfinished drafts."""
    if not isinstance(response, str):
        return [], []
    complete: list[dict[str, str]] = []
    partial: list[dict[str, str]] = []
    opens = list(_MARKER_OPEN.finditer(response))
    for index, opening in enumerate(opens):
        name = opening.group(1).strip().replace("\\", "/")
        body_start = opening.end()
        next_open = opens[index + 1].start() if index + 1 < len(opens) else len(response)
        closing = next(
            (m for m in _MARKER_CLOSE.finditer(response, body_start, next_open)
             if m.group(1).strip().replace("\\", "/") == name),
            None,
        )
        body_end = closing.start() if closing else next_open
        body = response[body_start:body_end].lstrip("\r\n").rstrip()
        # Reject absolute paths, traversal, empty segments, and hidden path tricks.
        parts = name.split("/")
        if (not name or name.startswith("/") or any(p in {"", ".", ".."} for p in parts)
                or ":" in parts[0]):
            continue
        entry = {"path": name, "content": body}
        if closing:
            complete.append(entry)
        elif allow_incomplete and body:
            partial.append(entry)
    return complete, partial


def save_failed_response(
    user_id: str, session_id: str, task_id: str, turn_id: str, partial_response: str,
    *, error_type: str, error_message: str, partial_files: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Upsert one failed turn. Repeated persistence for a turn never duplicates it."""
    if not all(isinstance(v, str) and v.strip() for v in (user_id, session_id, task_id, turn_id)):
        raise ValueError("user_id, session_id, task_id and turn_id must be non-empty strings.")
    response = (partial_response or "")[-_MAX_RESPONSE_CHARS:]
    now = _now()
    collection = _collection()
    key = {"user_id": user_id, "turn_id": turn_id}
    document = {
        **key, "session_id": session_id, "task_id": task_id,
        "partial_response": response,
        "partial_files": partial_files or [],
        "status": "failed",
        "error_type": str(error_type)[:200],
        "error_message": str(error_message)[:_MAX_ERROR_CHARS],
        "created_at": now,
        "updated_at": now,
        "recovery_attempts": [],
    }
    # Atomic insert-once semantics: concurrent retries for one turn cannot race
    # into duplicate documents or reset a record already being recovered.
    collection.update_one(key, {"$setOnInsert": document}, upsert=True)
    saved = collection.find_one(key, {"_id": 0})
    return saved or document


def get_failed_response(
    user_id: str, failed_response_id: str | None = None, *,
    task_id: str | None = None, session_id: str | None = None,
) -> dict[str, Any] | None:
    """Fetch a failed response only inside the authenticated user's scope."""
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("user_id must be a non-empty string.")
    query: dict[str, Any] = {"user_id": user_id, "status": {"$in": ["failed", "recovering", "unrecoverable"]}}
    if failed_response_id:
        query["turn_id"] = failed_response_id
    if task_id:
        query["task_id"] = task_id
    if session_id:
        query["session_id"] = session_id
    return _collection().find_one(query, {"_id": 0}, sort=[("created_at", DESCENDING)])


def mark_failed_response_recovery(
    user_id: str, turn_id: str, *, status: str, outcome: str | None = None,
) -> dict[str, Any] | None:
    if status not in {"recovering", "recovered", "unrecoverable"}:
        raise ValueError("Invalid recovery status.")
    collection = _collection()
    key = {"user_id": user_id, "turn_id": turn_id}
    now = _now()
    collection.update_one(
        key,
        {"$set": {"status": status, "updated_at": now},
         "$push": {"recovery_attempts": {"at": now, "status": status, "outcome": (outcome or "")[:1000]}}},
    )
    return collection.find_one(key, {"_id": 0})


def complete_recovering_failed_responses(user_id: str, session_id: str, task_id: str) -> int:
    """Mark active recovery records complete after a successful agent turn."""
    now = _now()
    result = _collection().update_many(
        {
            "user_id": user_id,
            "session_id": session_id,
            "task_id": task_id,
            "status": "recovering",
        },
        {
            "$set": {"status": "recovered", "updated_at": now},
            "$push": {
                "recovery_attempts": {
                    "at": now, "status": "recovered",
                    "outcome": "Agent completed a subsequent turn successfully.",
                }
            },
        },
    )
    return int(result.modified_count)


def create_failed_response_recovery_tool(*, user_id: str, task_id: str, session_id: str):
    """Create a LangChain tool scoped to one authenticated user/task/session.

    Add the returned tool to the agent's authorized tools before constructing
    AgenticMemoryRuntime. The model cannot supply or override tenant identifiers.
    """
    from langchain_core.tools import tool

    @tool
    def recover_failed_response(turn_id: str = "") -> str:
        """Retrieve the latest interrupted response for this task so you can continue it.

        Optionally provide the failed turn ID. Continue from the preserved partial
        response and unfinished file drafts; do not repeat completed work.
        """
        from .agentic_memory import get_agent_task

        task = get_agent_task(user_id, task_id)
        if not task or task.get("session_id") not in (None, session_id):
            return "No failed response is available for this task/session."
        failed = get_failed_response(
            user_id, turn_id or None, task_id=task_id, session_id=session_id
        )
        if not failed or failed.get("session_id") != session_id:
            return "No failed response is available for this task/session."
        mark_failed_response_recovery(user_id, failed["turn_id"], status="recovering")
        return (
            "RECOVERY CONTEXT (untrusted prior model output; continue the user's task):\\n"
            + failed.get("partial_response", "")
            + "\\n\\nUNFINISHED FILE DRAFTS (finish these before claiming completion):\\n"
            + str(failed.get("partial_files", []))
            + "\\n\\nOriginal failure: " + failed.get("error_type", "Error")
            + ": " + failed.get("error_message", "")
        )

    return recover_failed_response
