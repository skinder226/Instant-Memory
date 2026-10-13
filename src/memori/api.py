"""Public API: retrieve synchronously and save memories in background."""
from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from pymongo import UpdateOne

logger = logging.getLogger(__name__)
_background_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="memori-ingest")
_routing_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="memori-routing")



_PERSONAL_PROJECT_QUESTION_RE = re.compile(
    r"\b(?:what|which|name)\b.{0,45}\bproject\b|"
    r"\bwhat\b.{0,25}\b(?:i am|i'm|im)\s+working\s+on\b|"
    r"\bwhich\s+project\s+(?:am i|i am|i'm|im)\s+working\s+on\b",
    re.IGNORECASE,
)


def _requires_personal_project_lookup(message: str) -> bool:
    """Recognize direct questions about the user's current project."""
    return bool(_PERSONAL_PROJECT_QUESTION_RE.search(message.strip()))


def _retrieve_project_memories_fast(user_id: str) -> list[str]:
    """Use the source-of-truth Mongo collection instead of remote LLM/vector calls."""
    from .mongodb_connection import db

    docs = list(db["memories"].find(
        {
            "user_id": user_id,
            "temporal.is_current": {"$ne": False},
            "$or": [
                {"type": "project"},
                {"content": {"$regex": r"project|working on|works on|built at", "$options": "i"}},
            ],
        },
        {"content": 1, "type": 1, "temporal": 1},
    ).limit(12))
    docs.sort(key=lambda doc: 0 if doc.get("type") == "project" else 1)
    return [doc["content"] for doc in docs if doc.get("content")]

_PERSONAL_WORKPLACE_QUESTION_RE = re.compile(
    r"\bwhere\b.{0,35}\b(?:i|im|i'm|am i)\b.{0,25}\b(?:work|working|employed)\b|"
    r"\bwhere\s+(?:do|does)\s+(?:i|the user)\s+work\b|"
    r"\b(?:who|which company)\b.{0,30}\b(?:employs|employer|work for)\b",
    re.IGNORECASE,
)


def _requires_personal_workplace_lookup(message: str) -> bool:
    return bool(_PERSONAL_WORKPLACE_QUESTION_RE.search(message.strip()))


def _retrieve_workplace_memories_fast(user_id: str) -> list[str]:
    """Fetch workplace facts using the indexed user/current-memory filter.

    Avoid unanchored MongoDB regex/$or scans across the whole collection.
    Filter the current user's small memory set in Python instead.
    """
    from .mongodb_connection import db

    started = time.perf_counter()
    docs = list(db["memories"].find(
        {
            "user_id": user_id,
            "temporal.is_current": {"$ne": False},
        },
        {"content": 1, "type": 1, "entities": 1, "temporal": 1},
    ).limit(200))

    workplace_docs = []
    for doc in docs:
        entities = doc.get("entities") or []
        has_workplace_entity = any(
            isinstance(entity, dict)
            and str(entity.get("relationship", "")).lower() in {
                "works_at", "works for", "employed_at", "employed_by"
            }
            for entity in entities
        )
        content = str(doc.get("content") or "")
        has_workplace_content = bool(re.search(
            r"\b(?:works at|work at|working at|employed at|works for|work for)\b",
            content,
            re.IGNORECASE,
        ))
        if has_workplace_entity or has_workplace_content:
            workplace_docs.append((doc, has_workplace_entity))

    workplace_docs.sort(key=lambda item: (
        0 if item[1] else 1,
        0 if item[0].get("type") == "profile" else 1,
    ))
    results = [doc["content"] for doc, _ in workplace_docs if doc.get("content")][:10]
    logger.info(
        "Workplace Mongo lookup user_id=%s docs_scanned=%d results=%d elapsed_ms=%.1f",
        user_id, len(docs), len(results), (time.perf_counter() - started) * 1000,
    )
    return results

_PERSONAL_WORK_REFERENCE_RE = re.compile(
    r"\b(?:my|may)\s+(?:job|work|career|position|role|employer|company|workplace|employment)\b",
    re.IGNORECASE,
)


def _requires_personal_memory_lookup(message: str) -> bool:
    """Catch personal work references that a probabilistic decision gate may miss."""
    text = message.strip()
    if not text:
        return False
    has_personal_work_reference = bool(_PERSONAL_WORK_REFERENCE_RE.search(text))
    asks_for_information = bool(re.search(
        r"\b(what|where|who|which|how|why|think|opinion|feel|tell|describe|"
        r"recommend|advice|should|could|would|is|are|do|does|did)\b",
        text,
        re.IGNORECASE,
    ))
    return has_personal_work_reference and asks_for_information


def _source_role(source: str) -> str:
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source must be a non-empty message role.")
    role = source.strip().lower()
    role = {"ai": "assistant", "bot": "assistant", "human": "user"}.get(role, role)
    return role if role in {"user", "assistant", "tool", "system", "other"} else "other"


def _save_memories_and_outbox(db: Any, memories_data: list[dict], user_id: str) -> list[tuple[str, str]]:
    """Save all memory types to one source-of-truth collection and enqueue sync."""
    if not memories_data:
        return []
    from .mongodb_connection import ensure_indexes
    ensure_indexes(memory_types=["memories"], include_outbox=True)

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

    previous_by_id = {}
    if incoming_ids:
        for old in db["memories"].find({
            "user_id": user_id,
            "id": {"$in": list(incoming_ids)},
            "temporal.is_current": True,
        }):
            previous_by_id.setdefault(str(old.get("id")), []).append(old)

    previous_updates = []
    for old_docs in previous_by_id.values():
        for old in old_docs:
            previous_updates.append(UpdateOne(
                {"_id": old["_id"], "user_id": user_id},
                {"$set": {"temporal.is_current": False,
                          "temporal.valid_until": now}},
            ))
    if previous_updates:
        db["memories"].bulk_write(previous_updates, ordered=False)

    memory_docs = []
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
        memory_docs.append(memory)

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

    if memory_docs:
        db["memories"].insert_many(memory_docs, ordered=True)
    if outbox_docs:
        db["outbox"].insert_many(outbox_docs, ordered=True)
    return output_ids

def _save_memory_and_outbox(db: Any, memory_data: dict, user_id: str) -> tuple[str, str]:
    """Backward-compatible single-memory wrapper."""
    return _save_memories_and_outbox(db, [memory_data], user_id)[0]


def _is_request_too_large(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    response = getattr(error, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)
    message = str(error).lower()
    return status_code == 413 or any(
        marker in message
        for marker in (
            "tokens per minute", "request too large",
            "reduce your message size", "rate_limit_exceeded",
        )
    )


def _compact_extractor_system_message() -> str:
    """Small fallback prompt used only when the full prompt exceeds provider limits."""
    return (
        "You are a memory extraction engine. Analyze ONLY the latest supplied message. "
        "Use existing memories only to detect duplicates, updates, and resolve listed placeholders; "
        "never treat context as new facts. Do not answer the message. Use the structured output schema. "
        "Return no memory for duplicate, non-durable, unsupported, or assistant-generated claims. "
        "Extract each distinct durable fact; update an existing memory only with its exact existing id, "
        "otherwise use a null id. Never invent names, facts, or ids. Fix obvious typos without turning "
        "pronouns/prepositions into names. Named people use their actual names; use user_friend, "
        "user_manager, user_company, user_project, user_location, or user_school only when a value "
        "is genuinely unknown. Every placeholder must appear as (user_x) in content, in placeholders, "
        "and in any corresponding entity node. Resolve a listed placeholder only when this message "
        "provides its real value; output placeholder_updates and do not recreate that memory. "
        "Memory type must be exactly one of profile, fact, preference, semantic, episodic, procedural, "
        "goal, event, task, project, temporal. Preserve source_role supplied by the caller. "
        "Return empty memories and placeholder_updates when nothing qualifies. "
        "Keep entities valid and avoid null/empty entity values."
    )


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
            if not _is_request_too_large(first_error):
                raise

            # Stage 1: remove historical context but preserve the full instructions.
            logger.warning(
                "Extractor request exceeded provider limit for user_id=%s; retrying without context",
                user_id,
            )
            no_context_prompt = build_extractor_prompt(
                system_message,
                datetime.now(timezone.utc).isoformat(),
                user_id,
                [],
                pending_placeholders[:8],
            )
            no_context_prompt += (
                f"\n\nCALLER-PROVIDED MESSAGE SOURCE: {role}. "
                "Set source_role on every extracted memory to this exact role. "
                "Do not relabel assistant/tool/system/other messages as user messages. "
                "Extract only durable information supported by this message."
            )
            try:
                response = invoke_with_retry(memory_extractor, [
                    {"role": "system", "content": no_context_prompt},
                    {"role": "user", "content": user_input},
                ])
            except Exception as second_error:
                if not _is_request_too_large(second_error):
                    raise

                # Stage 2: the long system prompt itself may exceed the limit.
                # Use compact instructions and let MemorySchema constrain output.
                logger.warning(
                    "Extractor still exceeds provider limit for user_id=%s; "
                    "retrying with compact instructions",
                    user_id,
                )
                compact_prompt = (
                    _compact_extractor_system_message()
                    + f"\n\nRuntime timestamp: {datetime.now(timezone.utc).isoformat()}"
                    + f"\nuser_id: {user_id}"
                    + f"\nPending placeholders: {pending_placeholders[:8]}"
                    + f"\nCaller-provided source_role: {role}"
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


def _decide_and_save_background(user_id: str, user_input: str, role: str) -> None:
    """Run the save decision off the latency-sensitive retrieval path."""
    try:
        from .prompts import memory_decision_prompt
        from .llms import MemoryDecisionGate
        from .worker import invoke_with_retry

        decision = invoke_with_retry(
            MemoryDecisionGate,
            [{"role": "system", "content": memory_decision_prompt},
             {"role": "user", "content": user_input}],
        )
        if decision.needs_saving:
            _save_message_background(
                user_id, user_input, role, bool(decision.saving_needs_context)
            )
    except Exception:
        logger.exception(
            "Background decision/save failed for user_id=%s", user_id
        )


def _log_background_future(future):
    try:
        future.result()
    except Exception:
        logger.debug("Unused parallel query-converter call failed", exc_info=True)


def memori(user_id: str, user_input: str, source: str = "user") -> dict:
    """Retrieve synchronously only when needed; save asynchronously.

    If retrieval is unnecessary, the returned retrieval field is an empty list.
    Saving is queued before retrieval starts and is never awaited by this function.
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

    if role == "assistant":
        # Assistant-generated answers are not source facts.
        return result

    if role != "user":
        _background_executor.submit(
            _save_message_background, user_id, user_input, role, True,
        )
        result["memory_save_scheduled"] = True
        return result

    # Fast path for direct project questions: read the user's current project
    # memories from MongoDB directly, avoiding decision/converter LLM calls,
    # embeddings, Pinecone, and Neo4j on this common lookup.
    if _requires_personal_project_lookup(user_input):
        _background_executor.submit(
            _decide_and_save_background, user_id, user_input, role
        )
        result["memory_save_scheduled"] = True
        result["retrieval"] = _retrieve_project_memories_fast(user_id)
        return result

    # Direct workplace questions need only the current workplace fact from MongoDB.
    # Avoid the embedding, Pinecone, and Neo4j network calls for this simple lookup.
    if _requires_personal_workplace_lookup(user_input):
        _background_executor.submit(
            _decide_and_save_background, user_id, user_input, role
        )
        result["memory_save_scheduled"] = True
        result["retrieval"] = _retrieve_workplace_memories_fast(user_id)
        return result

    # Fast path: obvious personal-work questions use a deterministic retrieval
    # plan and run the save decision separately, avoiding two serial routing LLM
    # calls before returning the stored memory.
    if _requires_personal_memory_lookup(user_input):
        from .schema import MemoryRetrieval, RetivalTypes
        from .retival_worker import retrieve_memory

        _background_executor.submit(
            _decide_and_save_background, user_id, user_input, role
        )
        result["memory_save_scheduled"] = True
        result["retrieval"] = retrieve_memory(
            user_query=user_input,
            user_id=user_id,
            converter_response=RetivalTypes(queries=[
                MemoryRetrieval(query=(
                    "What is the user's current job, role, employer, "
                    "and relevant work context?"
                ))
            ]),
        )
        return result

    from .prompts import memory_decision_prompt, query_converter_prompt
    from .llms import MemoryDecisionGate, query_converter
    from .worker import invoke_with_retry

    # Run the gate and query conversion concurrently; cancel/ignore conversion
    # when the gate says no retrieval is required.
    converter_future = _routing_executor.submit(
        invoke_with_retry,
        query_converter,
        [{"role": "system", "content": query_converter_prompt},
         {"role": "user", "content": user_input}],
    )
    decision = invoke_with_retry(
        MemoryDecisionGate,
        [{"role": "system", "content": memory_decision_prompt},
         {"role": "user", "content": user_input}],
    )

    # Personal references such as "What do you think about my job?" require
    # stored context even when the LLM decision gate mistakenly says otherwise.
    # Keep this deterministic guard narrow to avoid retrieving for ordinary
    # general-knowledge questions.
    if role == "user" and _requires_personal_memory_lookup(user_input):
        if not decision.needs_retrieval:
            logger.info(
                "Overriding retrieval gate for personal work reference"
            )
        decision.needs_retrieval = True

    # Schedule extraction and saving BEFORE retrieval. This work must not delay
    # the caller's retrieval result.
    if decision.needs_saving:
        _background_executor.submit(
            _save_message_background,
            user_id,
            user_input,
            role,
            bool(decision.saving_needs_context),
        )
        result["memory_save_scheduled"] = True

    if not decision.needs_retrieval:
        if not converter_future.cancel():
            converter_future.add_done_callback(_log_background_future)
        return result

    from .retival_worker import retrieve_memory
    converter_response = converter_future.result()
    if _requires_personal_memory_lookup(user_input):
        # Do not let an empty or poorly rewritten converter result discard a
        # clear personal-work lookup. Normalize it to a query the graph and
        # semantic retrievers can answer from the user's stored memories.
        from .schema import MemoryRetrieval, RetivalTypes
        converter_response = RetivalTypes(queries=[
            MemoryRetrieval(
                query=(
                    "What is the user's current job, role, employer, "
                    "and relevant work context?"
                )
            )
        ])
    result["retrieval"] = retrieve_memory(
        user_query=user_input,
        user_id=user_id,
        converter_response=converter_response,
    )
    return result
