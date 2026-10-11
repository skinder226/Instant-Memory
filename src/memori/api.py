"""Public API: retrieve synchronously and save memories in background."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from pymongo import UpdateOne
from .mongodb_connection import ensure_indexes

logger = logging.getLogger(__name__)
_background_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="memori-ingest")
_routing_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="memori-routing")


def _source_role(source: str) -> str:
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source must be a non-empty message role.")
    role = source.strip().lower()
    role = {"ai": "assistant", "bot": "assistant", "human": "user"}.get(role, role)
    return role if role in {"user", "assistant", "tool", "system", "other"} else "other"


def _all_memory_collections() -> list[str]:
    return ["profile", "fact", "preference", "semantic", "episodic", "procedural",
            "goal", "event", "task", "project", "temporal"]


def _save_memories_and_outbox(db: Any, memories_data: list[dict], user_id: str) -> list[tuple[str, str]]:
    """Persist one extraction response using batched MongoDB operations."""
    if not memories_data:
        return []
    ensure_indexes()

    now = datetime.now(timezone.utc).isoformat()
    prepared = []
    incoming_ids = set()
    for raw in memories_data:
        memory = dict(raw)
        placeholders = memory.pop("placeholders", []) or []
        memory["user_id"] = user_id
        incoming_id = memory.get("id")
        if incoming_id:
            incoming_ids.add(str(incoming_id))
        temporal = dict(memory.get("temporal") or {})
        temporal["valid_from"] = temporal.get("valid_from") or now
        temporal["valid_until"] = None
        temporal["is_current"] = True
        memory["temporal"] = temporal
        prepared.append({
            "memory": memory,
            "incoming_id": str(incoming_id) if incoming_id else None,
            "placeholders": placeholders,
        })

    # Query each memory collection once per extraction response, rather than
    # querying all 11 collections separately for every updated memory.
    previous_by_id = {}
    if incoming_ids:
        for memory_type in _all_memory_collections():
            cursor = db[memory_type].find({
                "user_id": user_id,
                "id": {"$in": list(incoming_ids)},
                "temporal.is_current": True,
            })
            for old in cursor:
                previous_by_id.setdefault(str(old.get("id")), []).append(old)

    previous_updates = {}
    for old_docs in previous_by_id.values():
        for old in old_docs:
            previous_updates.setdefault(old["type"], []).append(
                UpdateOne(
                    {"_id": old["_id"], "user_id": user_id},
                    {"$set": {"temporal.is_current": False,
                              "temporal.valid_until": now}},
                )
            )
    for memory_type, operations in previous_updates.items():
        if operations:
            db[memory_type].bulk_write(operations, ordered=False)

    memory_docs_by_type = {}
    outbox_docs = []
    output_ids = []
    for item in prepared:
        memory = item["memory"]
        stable_id = item["incoming_id"]
        mongo_oid = ObjectId()
        if not stable_id:
            stable_id = str(mongo_oid)
        memory["_id"] = mongo_oid
        memory["id"] = stable_id

        previous = previous_by_id.get(stable_id, [])
        memory_docs_by_type.setdefault(memory["type"], []).append(memory)

        placeholder_docs = []
        for placeholder in item["placeholders"]:
            p = dict(placeholder)
            p["statuses"] = {
                "mongo_status": "pending",
                "pinecone_status": "pending",
                "neo4j_status": "pending",
            }
            placeholder_docs.append(p)

        mongo_id = str(mongo_oid)
        outbox_docs.append({
            "event_id": mongo_id,
            "event_type": "MEMORY_UPDATED" if previous else "MEMORY_CREATED",
            "memory_type": memory["type"],
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
        output_ids.append((stable_id, mongo_id))

    # Set both _id and public id before insert; no follow-up update_one needed.
    for memory_type, documents in memory_docs_by_type.items():
        if documents:
            db[memory_type].insert_many(documents, ordered=True)
    if outbox_docs:
        db["outbox"].insert_many(outbox_docs, ordered=True)

    return output_ids


def _save_memory_and_outbox(db: Any, memory_data: dict, user_id: str) -> tuple[str, str]:
    """Backward-compatible single-memory wrapper."""
    return _save_memories_and_outbox(db, [memory_data], user_id)[0]


def _save_message_background(user_id: str, user_input: str, role: str,
                             saving_needs_context: bool) -> None:
    """Extract and persist memories without delaying the retrieval response."""
    try:
        from .fast_context import get_context_fast, build_extractor_prompt
        from .mongodb_connection import db
        from .prompts import system_message
        from .llms import memory_extractor
        from .worker import save_pending_memory, invoke_with_retry
        from .update_placeholders import (
            update_mongo_placeholders, update_neo4j_placeholders,
            update_pinecone_placeholders,
        )

        context, pending_placeholders = [], []
        if saving_needs_context:
            context_data = get_context_fast(user_id)
            context = context_data["context"]
            pending_placeholders = context_data["pending_placeholders"]

        prompt = build_extractor_prompt(
            system_message, datetime.now(timezone.utc).isoformat(),
            user_id, context, pending_placeholders,
        )
        prompt += (
            f"\n\nCALLER-PROVIDED MESSAGE SOURCE: {role}. "
            "Set source_role on every extracted memory to this exact role. "
            "Do not relabel assistant/tool/system/other messages as user messages. "
            "Extract only durable information supported by this message."
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input},
        ]
        try:
            response = invoke_with_retry(memory_extractor, messages)
        except Exception as first_error:
            # Groq may reject requests that exceed the configured token budget.
            # Retry once without historical context, preserving the extraction
            # instructions and current message. Do not retry unrelated errors.
            status_code = getattr(first_error, "status_code", None)
            error_text = str(first_error).lower()
            is_oversized = status_code == 413 or (
                "tokens per minute" in error_text
                or "request too large" in error_text
                or "reduce your message size" in error_text
            )
            if not is_oversized or not context:
                raise

            logger.warning(
                "Extractor request too large for user_id=%s; retrying without memory context",
                user_id,
            )
            compact_prompt = build_extractor_prompt(
                system_message,
                datetime.now(timezone.utc).isoformat(),
                user_id,
                [],
                pending_placeholders[:20],
            )
            compact_prompt += (
                f"\n\nCALLER-PROVIDED MESSAGE SOURCE: {role}. "
                "Set source_role on every extracted memory to this exact role. "
                "Do not relabel assistant/tool/system/other messages as user messages. "
                "Extract only durable information supported by this message."
            )
            response = invoke_with_retry(memory_extractor, [
                {"role": "system", "content": compact_prompt},
                {"role": "user", "content": user_input},
            ])

        if response.memories:
            memory_data = []
            for memory in response.memories:
                item = memory.model_dump()
                item["source_role"] = role
                memory_data.append(item)
            _save_memories_and_outbox(db, memory_data, user_id)

        if response.memories:
            save_pending_memory(user_id)

        if response.placeholder_updates:
            now = datetime.now(timezone.utc).isoformat()
            for update in response.placeholder_updates:
                fp = update.filled_placeholders
                db.outbox.update_many(
                    {"user_id": user_id, "placeholders": {"$elemMatch": {
                        "name": fp.name, "statuses.mongo_status": "pending",
                    }}},
                    {"$set": {"placeholders.$[p].value": fp.value, "updated_at": now}},
                    array_filters=[{"p.name": fp.name, "p.statuses.mongo_status": "pending"}],
                )
            pending = list(db.outbox.find({
                "user_id": user_id, "placeholders": {"$elemMatch": {
                    "statuses.mongo_status": {"$in": ["pending", "error"]},
                }},
            }))
            docs = update_mongo_placeholders(pending, response.placeholder_updates)
            if docs:
                update_neo4j_placeholders(user_id, response.placeholder_updates, docs)
                update_pinecone_placeholders(user_id, response.placeholder_updates, docs)

        logger.info("Background memory processing finished for user_id=%s source=%s",
                    user_id, role)
    except Exception:
        logger.exception("Background memory processing failed for user_id=%s source=%s",
                         user_id, role)


def _log_background_future(future):
    try:
        future.result()
    except Exception:
        logger.debug("Unused parallel query-converter call failed", exc_info=True)


def memori(user_id: str, user_input: str, source: str = "user") -> dict:
    """Return retrieved memories synchronously; schedule extraction and saving asynchronously.

    Your application LLM should use result['retrieval'] to generate its answer.
    """
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("user_id must be a non-empty string.")
    if not isinstance(user_input, str) or not user_input.strip():
        raise ValueError("user_input must be a non-empty string.")

    user_id, user_input = user_id.strip(), user_input.strip()
    role = _source_role(source)

    result: dict[str, Any] = {
        "user_id": user_id,
        "source": role,
        "retrieval": [],
        "memory_save_scheduled": False,
    }

    if role == "user":
        # Non-user sources never import or execute the retrieval path.
        from .prompts import memory_decision_prompt, query_converter_prompt
        from .llms import MemoryDecisionGate, query_converter
        from .worker import invoke_with_retry
        from .retival_worker import retrieve_memory

        # These two independent classifications run concurrently. The retrieval
        # planner still runs after query conversion because it needs each
        # converted query, but the gate no longer adds a full serial LLM wait.
        gate_future = _routing_executor.submit(
            invoke_with_retry,
            MemoryDecisionGate,
            [{"role": "system", "content": memory_decision_prompt},
             {"role": "user", "content": user_input}],
        )
        converter_future = _routing_executor.submit(
            invoke_with_retry,
            query_converter,
            [{"role": "system", "content": query_converter_prompt},
             {"role": "user", "content": user_input}],
        )
        decision = gate_future.result()

        # Retrieval must finish before the caller's LLM can use its results.
        if decision.needs_retrieval:
            converter_response = converter_future.result()
            result["retrieval"] = retrieve_memory(
                user_query=user_input,
                user_id=user_id,
                converter_response=converter_response,
            )
        else:
            # If the converter already started, consume/log its exception in
            # the background rather than blocking a non-retrieval request.
            if not converter_future.cancel():
                converter_future.add_done_callback(_log_background_future)
        if not decision.needs_saving:
            return result
        saving_needs_context = bool(decision.saving_needs_context)
    elif role == "assistant":
        # Assistant-generated answers are not source facts. Do not send them
        # through extraction or save them back as user memories.
        return result
    else:
        saving_needs_context = True

    # Do not wait for this future. Extraction, MongoDB writes, and placeholder
    # updates happen off the request path; vector/graph sync is queued separately.
    _background_executor.submit(
        _save_message_background, user_id, user_input, role, saving_needs_context,
    )
    result["memory_save_scheduled"] = True
    return result
