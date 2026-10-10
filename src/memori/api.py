"""Public, single-call API for Memori memory ingestion and retrieval."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _source_role(source: str) -> str:
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source must be a non-empty role such as 'user', 'assistant', or 'tool'.")

    role = source.strip().lower()
    aliases = {"ai": "assistant", "bot": "assistant", "human": "user"}
    role = aliases.get(role, role)
    if role not in {"user", "assistant", "tool", "system", "other"}:
        role = "other"
    return role


def _all_memory_collections() -> list[str]:
    return [
        "profile", "fact", "preference", "semantic", "episodic",
        "procedural", "goal", "event", "task", "project", "temporal",
    ]


def _save_memory_and_outbox(db: Any, memory_data: dict, user_id: str) -> tuple[str, str]:
    incoming_id = memory_data.get("id")
    placeholders = memory_data.pop("placeholders", []) or []
    memory_data["user_id"] = user_id

    previous = []
    if incoming_id:
        for memory_type in _all_memory_collections():
            previous.extend(db[memory_type].find({
                "user_id": user_id,
                "id": incoming_id,
                "temporal.is_current": True,
            }))

    now = datetime.now(timezone.utc).isoformat()
    temporal = memory_data.setdefault("temporal", {}) or {}
    temporal["valid_from"] = temporal.get("valid_from") or now
    temporal["valid_until"] = None
    temporal["is_current"] = True
    memory_data["temporal"] = temporal

    if incoming_id:
        stable_id = str(incoming_id)
        for old in previous:
            db[old["type"]].update_one(
                {"_id": old["_id"], "user_id": user_id},
                {"$set": {"temporal.is_current": False, "temporal.valid_until": now}},
            )
        memory_data["id"] = stable_id
    else:
        stable_id = ""

    result = db[memory_data["type"]].insert_one(memory_data)
    mongo_id = str(result.inserted_id)
    if not stable_id:
        stable_id = mongo_id
    db[memory_data["type"]].update_one(
        {"_id": result.inserted_id, "user_id": user_id},
        {"$set": {"id": stable_id}},
    )

    placeholder_docs = []
    for placeholder in placeholders:
        item = dict(placeholder)
        item["statuses"] = {
            "mongo_status": "pending",
            "pinecone_status": "pending",
            "neo4j_status": "pending",
        }
        placeholder_docs.append(item)

    db["outbox"].insert_one({
        "event_id": mongo_id,
        "event_type": "MEMORY_UPDATED" if previous else "MEMORY_CREATED",
        "memory_type": memory_data["type"],
        "memory_id": stable_id,
        "mongo_id": mongo_id,
        "user_id": user_id,
        "supersedes_memory_ids": [str(old["_id"]) for old in previous],
        "placeholders": placeholder_docs,
        "pinecone": {"status": "pending", "attempts": 0, "last_error": None},
        "neo4j": {"status": "pending", "attempts": 0, "last_error": None},
        "created_at": now,
        "updated_at": now,
    })
    return stable_id, mongo_id


def memori(user_id: str, user_input: str, source: str = "user") -> dict:
    """Process one message.

    Args:
        user_id: Stable ID of the person/conversation owner.
        user_input: The message text to process.
        source: Message author, e.g. 'user', 'assistant'/'ai', 'tool',
            'system', or another role. Stored on extracted memories as
            source_role. Assistant messages are not treated as user-authored.

    Returns:
        A small result dictionary describing retrieval and memory writes.
    """
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("user_id must be a non-empty string.")
    if not isinstance(user_input, str) or not user_input.strip():
        raise ValueError("user_input must be a non-empty string.")

    user_id = user_id.strip()
    user_input = user_input.strip()
    role = _source_role(source)

    # Heavy providers and connections are imported only when the API is called.
    from .fast_context import get_context_fast, build_extractor_prompt
    from .mongodb_connection import db
    from .prompts import system_message, memory_decision_prompt
    from .llms import memory_extractor, MemoryDecisionGate
    from .worker import save_pending_memory, invoke_with_retry
    from .retival_worker import retrieve_memory
    from .update_placeholders import (
        update_mongo_placeholders, update_neo4j_placeholders,
        update_pinecone_placeholders,
    )

    result: dict[str, Any] = {
        "user_id": user_id,
        "source": role,
        "retrieval": None,
        "memories_saved": 0,
        "placeholders_resolved": 0,
    }

    # Retrieval answers questions from the user only; assistant/tool/system
    # messages may still be recorded with their original source role.
    decision = None
    if role == "user":
        decision = invoke_with_retry(
            MemoryDecisionGate,
            [
                {"role": "system", "content": memory_decision_prompt},
                {"role": "user", "content": user_input},
            ],
        )
        if decision.needs_retrieval:
            result["retrieval"] = retrieve_memory(user_query=user_input, user_id=user_id)

    if role == "user" and not decision.needs_saving:
        return result

    saving_needs_context = True if role != "user" else decision.saving_needs_context
    context, pending_placeholders = [], []
    if saving_needs_context:
        context_data = get_context_fast(user_id)
        context = context_data["context"]
        pending_placeholders = context_data["pending_placeholders"]

    current_time = datetime.now(timezone.utc).isoformat()
    prompt = build_extractor_prompt(
        system_message, current_time, user_id, context, pending_placeholders
    )
    prompt += (
        f"\n\nCALLER-PROVIDED MESSAGE SOURCE: {role}. "
        "Set source_role on every extracted memory to this exact role. "
        "Do not silently relabel assistant/tool/system/other messages as user messages. "
        "Extract only durable information actually supported by this message."
    )

    response = invoke_with_retry(
        memory_extractor,
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input},
        ],
    )

    for memory in response.memories:
        memory_data = memory.model_dump()
        memory_data["source_role"] = role
        _save_memory_and_outbox(db, memory_data, user_id)
    if response.memories:
        save_pending_memory(user_id)
    result["memories_saved"] = len(response.memories)

    if response.placeholder_updates:
        now = datetime.now(timezone.utc).isoformat()
        for update in response.placeholder_updates:
            fp = update.filled_placeholders
            db.outbox.update_many(
                {
                    "user_id": user_id,
                    "placeholders": {"$elemMatch": {
                        "name": fp.name, "statuses.mongo_status": "pending"
                    }},
                },
                {"$set": {
                    "placeholders.$[p].value": fp.value,
                    "updated_at": now,
                }},
                array_filters=[{
                    "p.name": fp.name, "p.statuses.mongo_status": "pending"
                }],
            )

        pending = list(db.outbox.find({
            "user_id": user_id,
            "placeholders": {"$elemMatch": {
                "statuses.mongo_status": {"$in": ["pending", "error"]}
            }},
        }))
        docs = update_mongo_placeholders(pending, response.placeholder_updates)
        if docs:
            update_neo4j_placeholders(user_id, response.placeholder_updates, docs)
            update_pinecone_placeholders(user_id, response.placeholder_updates, docs)
    result["placeholders_resolved"] = len(response.placeholder_updates)
    return result
