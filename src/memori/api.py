"""Public API: retrieve synchronously and save memories in background."""
from __future__ import annotations

import logging
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from pymongo import UpdateOne
from rapidfuzz.distance import Levenshtein

logger = logging.getLogger(__name__)
_background_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="memori-ingest")
_routing_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="memori-routing")



_PERSONAL_PROJECT_QUESTION_RE = re.compile(
    r"\b(?:what|which|name)\b.{0,60}\bproject\b|"
    r"\b(?:what|which)\b.{0,40}\b(?:project|app|system|product)\b.{0,50}\b(?:i|we)\s+(?:am|are|'m|'re)?\s*(?:working|building|developing|making)\b|"
    r"\b(?:what|which)\b.{0,35}\b(?:i|we)\s+(?:am|are|'m|'re)?\s*(?:working|building|developing)\s+on\b|"
    r"\b(?:which|what)\s+project\s+(?:am i|i am|i'm|im|are we|we are|we're)\s+working\s+on\b",
    re.IGNORECASE,
)


def _requires_personal_project_lookup(message: str) -> bool:
    """Recognize direct questions about the user's current project."""
    return bool(_PERSONAL_PROJECT_QUESTION_RE.search(message.strip()))


def _retrieve_project_memories_fast(user_id: str) -> list[str]:
    """Retrieve current project memories from MongoDB without routing LLM calls."""
    from .mongodb_connection import db

    collection = db["memories"]
    base_filter = {
        "user_id": user_id,
        "temporal.is_current": {"$ne": False},
    }
    projection = {"content": 1, "type": 1, "temporal": 1}
    started = time.perf_counter()

    # Keep the common query index-friendly; don't combine it with a regex OR.
    query_started = time.perf_counter()
    docs = list(collection.find(
        {**base_filter, "type": "project"},
        projection,
    ).sort("temporal.valid_from", -1).limit(12))
    structured_ms = (time.perf_counter() - query_started) * 1000

    # Legacy data may have project facts stored under another type.
    text_ms = 0.0
    if not docs:
        query_started = time.perf_counter()
        docs = list(collection.find(
            {
                **base_filter,
                "content": {
                    "$regex": r"\b(?:project|working on|works on|building|built at)\b",
                    "$options": "i",
                },
            },
            projection,
        ).sort("temporal.valid_from", -1).limit(12))
        text_ms = (time.perf_counter() - query_started) * 1000

    results = [doc["content"] for doc in docs if doc.get("content")]
    logger.info(
        "✅ RETRIEVAL COMPLETE | user=%s | route=project_lookup | structured=%.1f ms | "
        "fallback=%.1f ms | memories=%d | total=%.1f ms | selected=%s",
        user_id, structured_ms, text_ms, len(results),
        (time.perf_counter() - started) * 1000,
        "structured" if structured_ms and not text_ms else "text_fallback",
    )
    return results

_PERSONAL_WORKPLACE_QUESTION_RE = re.compile(
    r"\bwhere\b.{0,45}\b(?:i|im|i'm|am i|we|we're|are we)\b.{0,30}\b(?:work\w{0,5}|wokring|wroking|employ\w{0,5})\b|"
    r"\bwhere\s+(?:do|does|am|are)\s+(?:i|we|the user)\b.{0,20}\b(?:work\w{0,5}|employ\w{0,5})\b|"
    r"\b(?:who|which company)\b.{0,30}\b(?:employ\w{0,5}|employer|work\s+for)\b",
    re.IGNORECASE,
)


def _requires_personal_workplace_lookup(message: str) -> bool:
    return bool(_PERSONAL_WORKPLACE_QUESTION_RE.search(message.strip()))


def _retrieve_workplace_memories_fast(user_id: str) -> list[str]:
    """Retrieve current workplace facts with selective MongoDB queries.

    First use the explicit entity relationship, then fall back to workplace
    wording in the user's own current memories. MongoDB remains the source of
    truth; no LLM, embedding, Pinecone, or Neo4j call is on this path.
    """
    from .mongodb_connection import db

    collection = db["memories"]
    projection = {"content": 1, "type": 1, "entities": 1, "temporal": 1}
    base_filter = {
        "user_id": user_id,
        "temporal.is_current": {"$ne": False},
    }
    started = time.perf_counter()

    # Best path: match the structured relationship directly. This is backed
    # by the compound index added for entities.relationship.
    query_started = time.perf_counter()
    docs = list(collection.find(
        {
            **base_filter,
            "entities.relationship": {
                "$in": ["works_at", "works for", "employed_at", "employed_by"]
            },
        },
        projection,
    ).limit(20))
    structured_ms = (time.perf_counter() - query_started) * 1000

    # Fallback: support older memories that contain the fact only in text.
    # Do not limit an unrelated batch before filtering it, which could omit
    # the correct workplace fact when a user's memory store grows.
    text_ms = 0.0
    if not docs:
        query_started = time.perf_counter()
        docs = list(collection.find(
            {
                **base_filter,
                "content": {
                    "$regex": r"\b(?:works at|work at|working at|employed at|works for|work for)\b",
                    "$options": "i",
                },
            },
            projection,
        ).limit(20))
        text_ms = (time.perf_counter() - query_started) * 1000

    docs.sort(key=lambda doc: (
        0 if any(
            isinstance(entity, dict)
            and str(entity.get("relationship", "")).lower()
            in {"works_at", "works for", "employed_at", "employed_by"}
            for entity in (doc.get("entities") or [])
        ) else 1,
        0 if doc.get("type") == "profile" else 1,
    ))
    results = [doc["content"] for doc in docs if doc.get("content")][:10]

    logger.info(
        "✅ RETRIEVAL COMPLETE | user=%s | route=workplace_lookup | structured=%.1f ms | "
        "fallback=%.1f ms | memories=%d | total=%.1f ms",
        user_id, structured_ms, text_ms, len(results),
        (time.perf_counter() - started) * 1000,
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




_KNOWS_USER_QUESTION_RE = re.compile(
    r"\bwho\b.{0,35}\b(?:knows?|know)\s+me\b|"
    r"\bwho\b.{0,35}\b(?:is aware of me|knows about me)\b",
    re.IGNORECASE,
)


def _requires_knows_user_lookup(message: str) -> bool:
    """Recognize relationship questions, including common misspellings like 'knwos'."""
    text = message.strip().lower()
    if _KNOWS_USER_QUESTION_RE.search(text):
        return True

    # Keep this narrow: it must look like a "who ... [knows] me" question,
    # and only fuzzy-match the action word so unrelated questions don't route here.
    tokens = re.findall(r"[a-z]+", text)
    if not tokens or tokens[0] != "who" or "me" not in tokens:
        return False
    for token in tokens[1:]:
        if 3 <= len(token) <= 8 and Levenshtein.distance(token, "knows") <= 2:
            return True
    return False


def _retrieve_people_who_know_user(user_id: str) -> list[str]:
    """Retrieve current relationship memories about people connected to the user."""
    from .mongodb_connection import db

    collection = db["memories"]
    base = {"user_id": user_id, "temporal.is_current": {"$ne": False}}
    projection = {"content": 1, "entities": 1, "temporal": 1}
    relations = ["knows", "works_with", "has_friend", "friend_of", "has_colleague"]
    docs = list(collection.find(
        {**base, "entities.relationship": {"$in": relations}},
        projection,
    ).sort("temporal.valid_from", -1).limit(30))
    if not docs:
        docs = list(collection.find(
            {**base, "content": {"$regex": r"\b(?:knows|friend|works with|colleague)\b", "$options": "i"}},
            projection,
        ).sort("temporal.valid_from", -1).limit(30))

    results, seen = [], set()
    for doc in docs:
        content = str(doc.get("content") or "").strip()
        if content and content not in seen:
            results.append(content)
            seen.add(content)
    logger.info("🧠 RETRIEVAL | user=%s | route=relationship_lookup | matched_memories=%d", user_id, len(results))
    return results[:10]


_RETRIEVAL_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but",
    "by", "did", "do", "does", "for", "from", "had", "has", "have", "he",
    "her", "here", "hers", "him", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "me", "my", "of", "on", "or", "our", "ours",
    "she", "so", "that", "the", "their", "them", "there", "these", "they",
    "this", "those", "to", "us", "was", "we", "were", "what", "when",
    "where", "which", "who", "why", "will", "with", "you", "your",
}


_USER_MEMORY_CACHE: dict[str, tuple[float, list[dict]]] = {}
_USER_MEMORY_CACHE_LOCK = threading.Lock()
_USER_MEMORY_CACHE_TTL_SECONDS = 3.0


def _get_current_user_memories(user_id: str) -> list[dict]:
    """Reuse a short-lived per-user snapshot to avoid a Mongo round-trip per turn."""
    now = time.monotonic()
    with _USER_MEMORY_CACHE_LOCK:
        cached = _USER_MEMORY_CACHE.get(user_id)
        if cached and now - cached[0] < _USER_MEMORY_CACHE_TTL_SECONDS:
            return cached[1]

    from .mongodb_connection import db
    docs = list(db["memories"].find(
        {"user_id": user_id, "temporal.is_current": {"$ne": False}},
        {"_id": 0, "content": 1, "type": 1, "temporal": 1, "entities": 1},
    ).sort("temporal.valid_from", -1))

    with _USER_MEMORY_CACHE_LOCK:
        _USER_MEMORY_CACHE[user_id] = (time.monotonic(), docs)
    return docs


def _retrieve_memories_fast(user_id: str, query: str, limit: int = 8) -> list[str]:
    """Fast local lexical ranking over a cached snapshot of all current user memories."""
    started = time.perf_counter()
    tokens = [
        token for token in re.findall(r"[a-z0-9_+#.-]{2,}", query.lower())
        if token not in _RETRIEVAL_STOPWORDS
    ]
    if not tokens:
        return []

    docs = _get_current_user_memories(user_id)
    query_terms = set(tokens)
    scored = []
    for doc in docs:
        content = str(doc.get("content") or "")
        if not content:
            continue
        content_lc = content.lower()
        words = set(re.findall(r"[a-z0-9_+#.-]{2,}", content_lc))
        overlap = query_terms & words
        if not overlap:
            continue

        score = sum(2.0 if len(term) >= 6 else 1.0 for term in overlap)
        score += 3.0 * sum(1 for term in tokens if term in words and len(term) >= 5)
        # Strongly prefer exact phrase matches, but don't require them.
        normalized_query = " ".join(tokens)
        if normalized_query in content_lc:
            score += 10.0
        # Prefer structured relationship matches for entity/action questions.
        entities = doc.get("entities") or []
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            relation = str(entity.get("relationship") or "").lower()
            source_name = str((entity.get("source") or {}).get("name") or "").lower()
            target_name = str((entity.get("target") or {}).get("name") or "").lower()
            if any(term in relation for term in query_terms):
                score += 3.0
            if any(term in source_name or term in target_name for term in query_terms):
                score += 2.0
        if doc.get("type") in {"profile", "project", "preference"}:
            score += 0.25
        scored.append((score, str(doc.get("temporal", {}).get("valid_from") or ""), content))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    results, seen = [], set()
    for _, _, content in scored:
        if content not in seen:
            results.append(content)
            seen.add(content)
        if len(results) >= limit:
            break

    logger.debug(
        "RETRIEVAL_TIMING user_id=%s path=mongo_cached_lexical total_ms=%.2f "
        "scanned=%d matched=%d result_count=%d",
        user_id, (time.perf_counter() - started) * 1000,
        len(docs), len(scored), len(results),
    )
    return results

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

        router_started = time.perf_counter()
        logger.info("🤖 ROUTER START | user=%s | source=%s | message=%r",
                    user_id, role, user_input[:300])
        decision = invoke_with_retry(
            MemoryDecisionGate,
            [{"role": "system", "content": memory_decision_prompt},
             {"role": "user", "content": user_input}],
        )
        logger.info(
            "🤖 ROUTER DONE | user=%s | duration=%.0f ms | save=%s | needs_context=%s | retrieval_intent=%s",
            user_id, (time.perf_counter() - router_started) * 1000,
            decision.needs_saving, decision.saving_needs_context, decision.needs_retrieval,
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
        fast_started = time.perf_counter()
        logger.info("RETRIEVAL_PATH user_id=%s path=project_fast", user_id)
        # A direct lookup question is not new source information. Avoid an
        # unnecessary provider call just to decide whether this question is savable.
        result["retrieval"] = _retrieve_project_memories_fast(user_id)
        logger.info(
            "RETRIEVAL_TIMING user_id=%s path=project_fast total_ms=%.2f result_count=%d",
            user_id, (time.perf_counter() - fast_started) * 1000,
            len(result["retrieval"]),
        )
        return result

    # Direct workplace questions need only the current workplace fact from MongoDB.
    # Avoid the embedding, Pinecone, and Neo4j network calls for this simple lookup.
    if _requires_personal_workplace_lookup(user_input):
        fast_started = time.perf_counter()
        logger.info("RETRIEVAL_PATH user_id=%s path=workplace_fast", user_id)
        # Avoid an unnecessary provider request for a direct fact lookup.
        result["retrieval"] = _retrieve_workplace_memories_fast(user_id)
        logger.info(
            "RETRIEVAL_TIMING user_id=%s path=workplace_fast total_ms=%.2f result_count=%d",
            user_id, (time.perf_counter() - fast_started) * 1000,
            len(result["retrieval"]),
        )
        return result

    # Relationship questions need relation-aware lookup, not keyword overlap.
    if _requires_knows_user_lookup(user_input):
        fast_started = time.perf_counter()
        result["retrieval"] = _retrieve_people_who_know_user(user_id)
        logger.info(
            "✅ RETRIEVAL COMPLETE | user=%s | route=relationship_lookup | elapsed=%.2f ms | memories=%d",
            user_id, (time.perf_counter() - fast_started) * 1000, len(result["retrieval"]),
        )
        return result

    # Latency-first path: all user queries retrieve from MongoDB directly.
    # Memory extraction/save decisions run in the background, never on the response path.
    fast_started = time.perf_counter()
    _background_executor.submit(
        _decide_and_save_background, user_id, user_input, role
    )
    result["memory_save_scheduled"] = True

    if _requires_personal_project_lookup(user_input):
        logger.info("RETRIEVAL_PATH user_id=%s path=project_fast", user_id)
        result["retrieval"] = _retrieve_project_memories_fast(user_id)
        path = "project_fast"
    elif _requires_personal_workplace_lookup(user_input):
        logger.info("RETRIEVAL_PATH user_id=%s path=workplace_fast", user_id)
        result["retrieval"] = _retrieve_workplace_memories_fast(user_id)
        path = "workplace_fast"
    else:
        result["retrieval"] = _retrieve_memories_fast(user_id, user_input)
        path = "mongo_lexical"

    logger.info(
        "✅ RETRIEVAL COMPLETE | user=%s | route=%s | elapsed=%.2f ms | memories=%d",
        user_id, path, (time.perf_counter() - fast_started) * 1000,
        len(result["retrieval"]),
    )
    return result

