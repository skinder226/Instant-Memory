"""Durable storage for oversized user, assistant, and interrupted messages.

Stored in the same configured MongoDB database as the rest of Instant-Memory.
Paths are opaque references, not filesystem paths.
"""
from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock
from typing import Any
import re

_indexes_ready = False
_indexes_lock = Lock()

MAX_MESSAGE_BYTES = 1_000_000
_LONG_MESSAGE_THRESHOLD = 8_000


def _collection():
    global _indexes_ready
    from .mongodb_connection import db
    collection = db["agent_message_files"]
    if not _indexes_ready:
        with _indexes_lock:
            if not _indexes_ready:
                collection.create_index(
                    [("user_id", 1), ("session_id", 1), ("path", 1)],
                    unique=True,
                    name="agent_message_file_user_session_path_unique",
                )
                collection.create_index(
                    [("user_id", 1), ("session_id", 1), ("turn_id", 1)],
                    name="agent_message_file_user_session_turn",
                )
                _indexes_ready = True
    return collection


def _validate_scope(user_id: str, session_id: str) -> tuple[str, str]:
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("user_id must be a non-empty string.")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be a non-empty string.")
    return user_id.strip(), session_id.strip()


def save_agent_message_file(
    user_id: str,
    session_id: str,
    task_id: str,
    turn_id: str,
    role: str,
    content: str,
) -> dict[str, Any]:
    """Persist one full message and return its stable conversation-scoped path."""
    user_id, session_id = _validate_scope(user_id, session_id)
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("task_id must be a non-empty string.")
    if not isinstance(turn_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", turn_id):
        raise ValueError("turn_id must contain only letters, numbers, underscores, or hyphens.")
    if role not in {"user", "assistant", "assistant_partial"}:
        raise ValueError("role must be user, assistant, or assistant_partial.")
    if not isinstance(content, str):
        raise TypeError("content must be a string.")
    size = len(content.encode("utf-8"))
    if size > MAX_MESSAGE_BYTES:
        raise ValueError(f"message exceeds {MAX_MESSAGE_BYTES} bytes.")
    path = f"messages/{turn_id}-{role}.txt"
    now = datetime.now(timezone.utc)
    collection = _collection()
    collection.update_one(
        {"user_id": user_id, "session_id": session_id, "path": path},
        {"$set": {
            "user_id": user_id,
            "session_id": session_id,
            "task_id": task_id,
            "turn_id": turn_id,
            "role": role,
            "path": path,
            "content": content,
            "bytes": size,
            "line_count": len(content.splitlines()),
            "updated_at": now,
        }, "$setOnInsert": {"created_at": now}},
        upsert=True,
    )
    return {"path": path, "bytes": size, "line_count": len(content.splitlines()), "role": role}


def read_agent_message_file(
    user_id: str,
    session_id: str,
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> dict[str, Any]:
    """Read a saved message, optionally selecting a 1-based inclusive line range."""
    user_id, session_id = _validate_scope(user_id, session_id)
    if not isinstance(path, str) or not path.startswith("messages/") or ".." in path.split("/"):
        raise ValueError("path must be a message-file path returned by Instant-Memory.")
    document = _collection().find_one(
        {"user_id": user_id, "session_id": session_id, "path": path},
        {"_id": 0, "path": 1, "role": 1, "turn_id": 1, "content": 1, "line_count": 1, "bytes": 1},
    )
    if document is None:
        raise FileNotFoundError(path)
    lines = document["content"].splitlines(keepends=True)
    start = 1 if start_line is None else start_line
    end = len(lines) if end_line is None else end_line
    if not isinstance(start, int) or not isinstance(end, int) or start < 1 or end < start:
        raise ValueError("line range must be valid and 1-based.")
    if end > len(lines):
        raise ValueError(f"line range exceeds message length ({len(lines)} lines).")
    return {
        **{key: document[key] for key in ("path", "role", "turn_id", "bytes", "line_count")},
        "start_line": start,
        "end_line": end,
        "content": "".join(lines[start - 1:end]),
    }


def list_agent_message_files(user_id: str, session_id: str, limit: int = 20) -> list[dict[str, Any]]:
    """List recent saved message paths for exactly one user and conversation."""
    user_id, session_id = _validate_scope(user_id, session_id)
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
        raise ValueError("limit must be an integer between 1 and 100.")
    cursor = _collection().find(
        {"user_id": user_id, "session_id": session_id},
        {"_id": 0, "path": 1, "role": 1, "turn_id": 1, "bytes": 1, "line_count": 1, "updated_at": 1},
    ).sort("updated_at", -1).limit(limit)
    return list(cursor)


def delete_agent_message_file(user_id: str, session_id: str, path: str) -> bool:
    """Delete a saved message only within the supplied user's conversation."""
    user_id, session_id = _validate_scope(user_id, session_id)
    if not isinstance(path, str) or not path.startswith("messages/") or ".." in path.split("/"):
        raise ValueError("path must be a message-file path returned by Instant-Memory.")
    result = _collection().delete_one(
        {"user_id": user_id, "session_id": session_id, "path": path}
    )
    return result.deleted_count == 1
