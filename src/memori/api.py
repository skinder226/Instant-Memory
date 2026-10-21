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


def _requires_historical_project_lookup(message: str) -> bool:
    """Recognize questions about prior or former projects."""
    return bool(re.search(r"\b(?:before|previously|previous|former|formerly|used to|historical|history)\b", message, re.I)
                and re.search(r"\b(?:project|test project|app|system)\b", message, re.I))


def _retrieve_project_memories_fast(user_id: str, include_history: bool = False, query: str = "") -> list[str]:
    """Retrieve relevant project facts, including the predecessor of a named current project."""
    from .mongodb_connection import db

    collection = db["memories"]
    projection = {"content": 1, "type": 1, "temporal": 1, "entities": 1, "id": 1}
    started = time.perf_counter()

    if include_history:
        # For questions such as "What was my project before Orion?", first
        # resolve Orion and follow its explicit outbox supersession link.
        before_match = re.search(r"\bbefore\s+([A-Z][\w.-]*)\b", query, re.IGNORECASE)
        anchor = before_match.group(1) if before_match else None
        if anchor:
            current_docs = list(collection.find({
                "user_id": user_id,
                "temporal.is_current": {"$ne": False},
                "$or": [
                    {"entities.target.name": {"$regex": "^" + re.escape(anchor) + "$", "$options": "i"}},
                    {"content": {"$regex": r"\b" + re.escape(anchor) + r"\b", "$options": "i"}},
                ],
            }, projection).limit(10))
            current_docs = [
                doc for doc in current_docs
                if any(
                    isinstance(entity, dict)
                    and str((entity.get("target") or {}).get("name") or "").casefold() == anchor.casefold()
                    for entity in (doc.get("entities") or [])
                )
            ]
            superseded_ids = []
            for current in current_docs:
                event = db["outbox"].find_one(
                    {"user_id": user_id, "mongo_id": str(current.get("_id"))},
                    {"supersedes_memory_ids": 1},
                )
                if event:
                    superseded_ids.extend(str(value) for value in (event.get("supersedes_memory_ids") or []))
            object_ids = [ObjectId(value) for value in dict.fromkeys(superseded_ids) if ObjectId.is_valid(value)]
            if object_ids:
                linked_docs = list(collection.find({
                    "user_id": user_id, "_id": {"$in": object_ids},
                }, projection).sort("temporal.valid_from", -1))
                linked_results = [doc["content"] for doc in linked_docs if doc.get("content")]
                if linked_results:
                    logger.info(
                        "RETRIEVAL_TIMING user_id=%s path=project_predecessor total_ms=%.2f result_count=%d anchor=%s",
                        user_id, (time.perf_counter() - started) * 1000, len(linked_results), anchor,
                    )
                    return linked_results[:5]

        # Fallback for legacy records without an outbox supersession link:
        # only consider historical project memories, not every project ever stored.
        historical_docs = list(collection.find({
            "user_id": user_id,
            "type": "project",
            "temporal.is_current": False,
        }, projection).sort("temporal.valid_until", -1).limit(30))
        if anchor:
            historical_docs = [
                doc for doc in historical_docs
                if not any(
                    isinstance(entity, dict)
                    and str((entity.get("target") or {}).get("name") or "").casefold() == anchor.casefold()
                    for entity in (doc.get("entities") or [])
                )
            ]
        # The most recently superseded project is the best fallback candidate
        # when old records predate explicit supersedes links.
        historical_docs.sort(
            key=lambda doc: str((doc.get("temporal") or {}).get("valid_until") or
                                (doc.get("temporal") or {}).get("valid_from") or ""),
            reverse=True,
        )
        if historical_docs:
            results = [doc["content"] for doc in historical_docs if doc.get("content")]
            logger.info(
                "RETRIEVAL_TIMING user_id=%s path=project_history_fallback total_ms=%.2f result_count=%d",
                user_id, (time.perf_counter() - started) * 1000, len(results),
            )
            return results[:5]

    # Reuse the current-user snapshot instead of issuing two MongoDB queries.
    docs = [
        doc for doc in _get_current_user_memories(user_id)
        if doc.get("type") == "project"
        or re.search(
            r"\b(?:project|working on|works on|building|built at|test project)\b",
            str(doc.get("content") or ""),
            re.IGNORECASE,
        )
    ]
    docs.sort(
        key=lambda doc: str((doc.get("temporal") or {}).get("valid_from") or ""),
        reverse=True,
    )
    results = [doc["content"] for doc in docs if doc.get("content")]
    logger.info(
        "RETRIEVAL_TIMING user_id=%s path=project_lookup total_ms=%.2f history=%s result_count=%d",
        user_id, (time.perf_counter() - started) * 1000, include_history, len(results),
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
    """Retrieve current workplace facts from the shared tenant-scoped snapshot."""
    started = time.perf_counter()
    docs = _get_current_user_memories(user_id)
    relations = {"works_at", "works for", "employed_at", "employed_by"}
    matched = []
    for doc in docs:
        entities = doc.get("entities") or []
        structured = any(
            isinstance(entity, dict)
            and str(entity.get("relationship") or "").strip().lower() in relations
            for entity in entities
        )
        content = str(doc.get("content") or "")
        text_match = bool(re.search(
            r"\b(?:works at|work at|working at|employed at|works for|work for)\b",
            content,
            re.IGNORECASE,
        ))
        if structured or text_match:
            matched.append((
                0 if structured else 1,
                0 if doc.get("type") == "profile" else 1,
                str((doc.get("temporal") or {}).get("valid_from") or ""),
                content,
            ))
    matched.sort(key=lambda item: item[2], reverse=True)
    matched.sort(key=lambda item: (item[0], item[1]))
    results, seen = [], set()
    for _, _, _, content in matched:
        if content and content not in seen:
            results.append(content)
            seen.add(content)
        if len(results) >= 10:
            break
    logger.info(
        "RETRIEVAL_TIMING user_id=%s path=workplace_snapshot total_ms=%.2f scanned=%d result_count=%d",
        user_id, (time.perf_counter() - started) * 1000, len(docs), len(results),
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
    """Retrieve relationship memories from the shared short-lived user snapshot.

    Avoid a relationship-specific Mongo query with a potentially expensive
    server-side sort. The lexical path already loads the same current-user
    snapshot; reusing it makes this lookup local after the first cache miss.
    """
    started = time.perf_counter()
    docs = _get_current_user_memories(user_id)
    snapshot_ms = (time.perf_counter() - started) * 1000

    relations = {"knows", "works_with", "has_friend", "friend_of", "has_colleague"}
    matched = []
    for doc in docs:
        entities = doc.get("entities") or []
        has_relationship = any(
            isinstance(entity, dict)
            and str(entity.get("relationship") or "").strip().lower() in relations
            for entity in entities
        )
        content = str(doc.get("content") or "").strip()
        has_text_match = bool(content) and bool(
            re.search(r"\b(?:knows|friend|works with|colleague)\b", content, re.IGNORECASE)
        )
        if has_relationship or has_text_match:
            matched.append((
                str((doc.get("temporal") or {}).get("valid_from") or ""),
                content,
            ))

    # Newest facts first, but sort locally instead of forcing MongoDB to sort
    # the relationship query before returning any rows.
    matched.sort(key=lambda item: item[0], reverse=True)
    results, seen = [], set()
    for _, content in matched:
        if content and content not in seen:
            results.append(content)
            seen.add(content)

    logger.info(
        "🧠 RETRIEVAL | user=%s | route=relationship_lookup | snapshot=%.2f ms | "
        "scanned=%d | matched=%d",
        user_id, snapshot_ms, len(docs), len(results),
    )
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
        # Bound process memory when a shared service sees many tenants.
        while len(_USER_MEMORY_CACHE) > 256:
            _USER_MEMORY_CACHE.pop(next(iter(_USER_MEMORY_CACHE)))
    return docs


def _retrieve_memories_fast(user_id: str, query: str, limit: int = 8) -> list[str]:
    """Rank current memories using both prose and their structured graph entities."""
    started = time.perf_counter()
    tokens = [
        token for token in re.findall(r"[a-z0-9_+#.-]{2,}", query.lower())
        if token not in _RETRIEVAL_STOPWORDS
    ]
    if not tokens:
        return []

    # Expand common question concepts into graph relationship vocabulary.
    # This is generic retrieval normalization, not a one-question special case.
    relation_synonyms = {
        "sibling": {"sibling", "siblings", "sibling_of", "brother", "sister"},
        "siblings": {"sibling", "siblings", "sibling_of", "brother", "sister"},
        "brother": {"sibling", "siblings", "sibling_of", "brother"},
        "sister": {"sibling", "siblings", "sibling_of", "sister"},
        "friend": {"friend", "friends", "has_friend", "friend_of", "knows"},
        "friends": {"friend", "friends", "has_friend", "friend_of", "knows"},
        "know": {"know", "knows", "known_by", "friend_of", "has_friend"},
        "knows": {"know", "knows", "known_by", "friend_of", "has_friend"},
        "work": {"work", "works_at", "works_with", "employed_at", "employed_by"},
        "working": {"work", "works_at", "works_with", "employed_at", "employed_by"},
        "parent": {"parent", "mother", "father", "has_parent", "child_of"},
        "parents": {"parent", "mother", "father", "has_parent", "child_of"},
        "child": {"child", "son", "daughter", "has_child"},
        "children": {"child", "son", "daughter", "has_child"},
        "wife": {"wife", "spouse", "married_to"},
        "husband": {"husband", "spouse", "married_to"},
        "spouse": {"wife", "husband", "spouse", "married_to"},
    }
    expanded_terms = set(tokens)
    for token in tokens:
        expanded_terms.update(relation_synonyms.get(token, ()))

    docs = _get_current_user_memories(user_id)
    scored = []
    normalized_query = " ".join(tokens)
    for doc in docs:
        content = str(doc.get("content") or "")
        entities = doc.get("entities") or []
        # Graph facts may contain the answer only in entity nodes/relationship,
        # not in the natural-language memory content.
        entity_parts = []
        for entity in entities:            
            if not isinstance(entity, dict):
                continue
            entity_parts.append(str(entity.get("relationship") or "").lower().replace("_", " "))
            for node_key in ("source", "target"):
                node = entity.get(node_key) or {}
                if isinstance(node, dict):
                    entity_parts.append(str(node.get("name") or "").lower())
                    entity_parts.append(str(node.get("entity_type") or "").lower())
        searchable = content.lower() + " " + " ".join(entity_parts)
        words = set(re.findall(r"[a-z0-9_+#.-]{2,}", searchable))
        words.update(part for part in re.findall(r"[a-z0-9_+#.-]{2,}", searchable.replace("_", " ")))
        # Lightweight morphology helps common question/fact inflections match
        # (work/works, sibling/siblings) without requiring an external model.
        for word in tuple(words):
            if len(word) > 4 and word.endswith("s"):
                words.add(word[:-1])
            if len(word) > 5 and word.endswith("ing"):
                words.add(word[:-3])
                words.add(word[:-3] + "e")
        overlap = expanded_terms & words
        if not overlap:
            continue

        score = sum(2.0 if len(term) >= 6 else 1.0 for term in overlap)
        score += 3.0 * sum(1 for term in tokens if term in words and len(term) >= 5)
        if normalized_query in content.lower():
            score += 10.0

        # Prefer structured relation matches when the question names a relation
        # or a synonym, and prefer records that actually contain a person/entity.
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            relation = str(entity.get("relationship") or "").lower().replace("_", " ")
            relation_words = set(re.findall(r"[a-z0-9_+#.-]{2,}", relation))
            if relation_words & expanded_terms:
                score += 6.0
            for node_key in ("source", "target"):
                node = entity.get(node_key) or {}
                if isinstance(node, dict):
                    node_name = str(node.get("name") or "").lower()
                    if node_name and any(term in node_name for term in tokens):
                        score += 2.0
        if doc.get("type") in {"profile", "project", "preference"}:
            score += 0.25
        scored.append((score, str((doc.get("temporal") or {}).get("valid_from") or ""), content))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    results, seen = [], set()
    for _, _, content in scored:
        if content and content not in seen:
            results.append(content)
            seen.add(content)
        if len(results) >= limit:
            break

    logger.info(
        "RETRIEVAL_TIMING user_id=%s path=mongo_entity_lexical total_ms=%.2f "
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
        explicit_supersedes = memory.pop("_supersedes_mongo_ids", []) or []
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
            "explicit_supersedes": [str(value) for value in explicit_supersedes],
        })

    previous_by_id = {}
    if incoming_ids:
        for old in db["memories"].find({
            "user_id": user_id,
            "id": {"$in": list(incoming_ids)},
            "temporal.is_current": True,
        }):
            previous_by_id.setdefault(str(old.get("id")), []).append(old)

    explicit_superseded_by_item = []
    explicit_old_ids = set()
    for item in prepared:
        old_docs = []
        for old_id in item["explicit_supersedes"]:
            if not ObjectId.is_valid(old_id):
                continue
            old = db["memories"].find_one({"_id": ObjectId(old_id), "user_id": user_id, "type": "project"})
            if old:
                old_docs.append(old)
                explicit_old_ids.add(str(old["_id"]))
        explicit_superseded_by_item.append(old_docs)

    previous_updates = []
    for old_docs in previous_by_id.values():
        for old in old_docs:
            previous_updates.append(UpdateOne(
                {"_id": old["_id"], "user_id": user_id},
                {"$set": {"temporal.is_current": False,
                          "temporal.valid_until": now}},
            ))
    for old_id in explicit_old_ids:
        previous_updates.append(UpdateOne({"_id": ObjectId(old_id), "user_id": user_id},
            {"$set": {"temporal.is_current": False, "temporal.valid_until": now}}))
    if previous_updates:
        db["memories"].bulk_write(previous_updates, ordered=False)

    memory_docs = []
    outbox_docs = []
    output_ids = []
    for item_index, item in enumerate(prepared):
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
            "supersedes_memory_ids": list(dict.fromkeys([str(old["_id"]) for old in previous] + [str(old["_id"]) for old in explicit_superseded_by_item[item_index]])),
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
        for marker in (            "tokens per minute", "request too large",
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
        "Return a valid JSON object matching the required memory schema; output JSON only. "
        "Memory type must be exactly one of profile, fact, preference, semantic, episodic, procedural, "
        "goal, event, task, project, temporal. Preserve source_role supplied by the caller. "
        "Return empty memories and placeholder_updates when nothing qualifies. "
        "Keep entities valid and avoid null/empty entity values."
    )


def _attach_explicit_project_supersession(user_id: str, user_input: str, memory_data: list[dict]) -> bool:
    """Link a corrected project to its explicitly named prior project.

    When extraction emits a new memory, attach the old Mongo IDs to that memory's
    outbox event. If extraction treats the correction as a duplicate, repair the
    existing current project and requeue its projections instead.
    """
    match = re.search(r"\b(?:now|changed to|updated to|replaced by|is now)\s+([A-Z][\w.-]*)\s*,?\s+not\s+([A-Z][\w.-]*)\b", user_input, re.I)
    if not match:
        return False
    new_name, old_name = match.group(1).casefold(), match.group(2).casefold()
    project_items = [item for item in memory_data if item.get("type") == "project" and any(
        isinstance(entity, dict) and str((entity.get("target") or {}).get("name") or "").casefold() == new_name
        for entity in (item.get("entities") or []))]
    from .mongodb_connection import db
    old_docs = list(db["memories"].find({"user_id": user_id, "type": "project", "temporal.is_current": {"$ne": False},
        "$or": [{"entities.target.name": {"$regex": "^" + re.escape(match.group(2)) + "$", "$options": "i"}},
                {"content": {"$regex": r"\b" + re.escape(match.group(2)) + r"\b", "$options": "i"}}]}, {"_id": 1, "content": 1, "entities": 1}))
    old_docs = [doc for doc in old_docs if any(
        isinstance(entity, dict) and str((entity.get("target") or {}).get("name") or "").casefold() == old_name
        for entity in (doc.get("entities") or [])) or re.search(r"\b" + re.escape(match.group(2)) + r"\b", str(doc.get("content") or ""), re.I)]
    if not old_docs:
        return False
    old_ids = [str(doc["_id"]) for doc in old_docs]

    if project_items:
        for item in project_items:
            item["_supersedes_mongo_ids"] = list(dict.fromkeys((item.get("_supersedes_mongo_ids") or []) + old_ids))
        return False

    # Extraction may return no memory because Orion already exists. In that case
    # close the old fact and requeue the existing Orion projection sync.
    current_new = list(db["memories"].find({
        "user_id": user_id, "type": "project", "temporal.is_current": {"$ne": False},
        "entities.target.name": {"$regex": "^" + re.escape(match.group(1)) + "$", "$options": "i"},
    }, {"_id": 1, "content": 1, "entities": 1}))
    if not current_new:
        return False
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    from pymongo import UpdateOne
    db["memories"].bulk_write([
        UpdateOne({"_id": doc["_id"], "user_id": user_id},
                  {"$set": {"temporal.is_current": False, "temporal.valid_until": now}})
        for doc in old_docs
    ], ordered=False)
    for new_doc in current_new:
        db["outbox"].update_one(
            {"user_id": user_id, "mongo_id": str(new_doc["_id"])},
            {"$addToSet": {"supersedes_memory_ids": {"$each": old_ids}},
             "$set": {"pinecone.status": "pending", "pinecone.attempts": 0,
                      "pinecone.last_error": None, "neo4j.status": "pending",
                      "neo4j.attempts": 0, "neo4j.last_error": None, "updated_at": now}},
        )
    with _USER_MEMORY_CACHE_LOCK:
        _USER_MEMORY_CACHE.pop(user_id, None)
    return True


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
                    + "\n\nReturn a valid JSON object matching the required memory schema. "
                    "The word JSON is required; output JSON only."
                    + f"\nRuntime timestamp: {datetime.now(timezone.utc).isoformat()}"
                    + f"\nuser_id: {user_id}"
                    + f"\nPending placeholders: {pending_placeholders[:8]}"
                    + f"\nCaller-provided source_role: {role}"
                )

                response = invoke_with_retry(memory_extractor, [
                    {"role": "system", "content": compact_prompt},
                    {"role": "user", "content": user_input},
                ])


        repaired_existing_project = False
        if response.memories:
            memory_data = []
            for memory in response.memories:
                item = memory.model_dump()
                item["source_role"] = role
                memory_data.append(item)
            _attach_explicit_project_supersession(user_id, user_input, memory_data)
            _save_memories_and_outbox(db, memory_data, user_id)
        else:
            repaired_existing_project = _attach_explicit_project_supersession(user_id, user_input, [])

        if response.memories or repaired_existing_project:
            save_pending_memory(user_id)

        if response.placeholder_updates:
            now = datetime.now(timezone.utc).isoformat()
            for update in response.placeholder_updates:
                fp = update.filled_placeholders
                db.outbox.update_many(
                    {"user_id": user_id, "placeholders": {"$elemMatch": {
                        "name": fp.name, "statuses.mongo_status": "pending",
                    }}},                    {"$set": {"placeholders.$[p].value": fp.value, "updated_at": now}},
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
        logger.info("ROUTER START | user_id=%s | source=%s | message_chars=%d",
                    user_id, role, len(user_input))
        try:
            decision = invoke_with_retry(
                MemoryDecisionGate,
                [{"role": "system", "content": memory_decision_prompt},
                 {"role": "user", "content": user_input}],
            )
        except Exception as exc:
            if not _is_request_too_large(exc):
                raise
            # The decision prompt may exceed provider limits even when the
            # extractor can succeed with its compact-prompt fallback. Do not
            # silently drop a user message solely because routing was oversized.
            logger.warning(
                "Memory decision request exceeded provider limit; bypassing gate "
                "and using extractor safeguards for user_id=%s",
                user_id,
            )
            _save_message_background(user_id, user_input, role, True)
            return
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


_QUESTION_START_RE = re.compile(
    r"^\s*(?:what|which|who|whom|whose|where|when|why|how|do|does|did|is|are|was|were|can|could|tell me|name)\b",
    re.IGNORECASE,
)


def _looks_like_question(text: str) -> bool:
    return text.strip().endswith("?") or bool(_QUESTION_START_RE.match(text))


_GREETING_RE = re.compile(
    r"^(?:hi|hello|hey|hiya|howdy|good\s+(?:morning|afternoon|evening)|"
    r"thanks|thank\s+you|thx|bye|goodbye|see\s+you|how\s+are\s+you)"
    r"[!.?,\s]*$",
    re.IGNORECASE,
)
_PERSONAL_MEMORY_INTENT_RE = re.compile(
    r"\b(?:my|mine|me|our|ours|remember|recall|previous|previously|before|"
    r"last time|earlier|history|historical|memory|memories|project|employer|"
    r"workplace|job|career|friend|colleague|knows me|what did i ask|"
    r"what have i told you|what do you know about me|task|website did i ask)\b",
    re.IGNORECASE,
)


def _is_low_value_chat_turn(message: str) -> bool:
    """Avoid memory retrieval and extraction routing for greetings/thanks and generic questions.

    Questions containing explicit personal-memory/history/task signals continue through
    retrieval, even when phrased as short or generic-looking questions.
    """
    text = message.strip()
    if not text:
        return False
    if _GREETING_RE.fullmatch(text):
        return True
    if _looks_like_question(text):
        return not bool(_PERSONAL_MEMORY_INTENT_RE.search(text))
    return False


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
        # Assistant messages skip retrieval, but still pass through extraction.
        # source_role lets downstream consumers distinguish generated content.
        _background_executor.submit(
            _save_message_background, user_id, user_input, role, True,
        )
        result["memory_save_scheduled"] = True
        return result

    if role != "user":
        _background_executor.submit(
            _save_message_background, user_id, user_input, role, True,
        )
        result["memory_save_scheduled"] = True
        return result

    # Greetings, acknowledgements, and general knowledge questions should not
    # pay for memory retrieval or an LLM save-decision call. Personal-memory,
    # task, project, and history questions are explicitly preserved below.
    if _is_low_value_chat_turn(user_input):
        return result

    # Collaborator questions can refer to a historical project. Search the
    # named person's records across temporal states rather than only current
    # project memories.
    is_question = _looks_like_question(user_input)
    collaborator_match = re.search(
        r"\b(?:project|working on|worked on)\b.{0,50}\bwith\s+((?-i:[A-Z])[\w.-]*)\b|"
        r"\bwith\s+((?-i:[A-Z])[\w.-]*)\b.{0,50}\b(?:project|working on|worked on)\b",
        user_input, re.I,
    ) if is_question else None
    if collaborator_match:
        name = next(group for group in collaborator_match.groups() if group)
        from .mongodb_connection import db
        docs = list(db["memories"].find({
            "user_id": user_id,
            "$or": [
                {"content": {"$regex": r"\b" + re.escape(name) + r"\b", "$options": "i"}},
                {"entities.source.name": {"$regex": "^" + re.escape(name) + "$", "$options": "i"}},
                {"entities.target.name": {"$regex": "^" + re.escape(name) + "$", "$options": "i"}},
            ],
        }, {"content": 1, "temporal": 1, "type": 1, "entities": 1}).sort("temporal.valid_from", -1).limit(20))
        result["retrieval"] = [doc["content"] for doc in docs if doc.get("content")]
        return result

    # Historical project questions must include superseded records. Attach the
    # resolved historical relation to the result: the stored memory's content
    # may describe Atlas but not repeat the anchor ("before Orion"), even though
    # this retrieval path selected it specifically for that predecessor query.
    if is_question and _requires_historical_project_lookup(user_input):
        historical_results = _retrieve_project_memories_fast(
            user_id, include_history=True, query=user_input
        )
        before_match = re.search(r"\bbefore\s+([A-Z][\w.-]*)\b", user_input, re.IGNORECASE)
        anchor = before_match.group(1) if before_match else None
        if anchor and historical_results:
            result["retrieval"] = [
                f"Project-history lookup for the question 'what came before {anchor}' "
                f"selected this historical project as the predecessor: {memory}"
                for memory in historical_results
            ]
        else:
            result["retrieval"] = historical_results
        return result

    # Fast path for direct project questions: read the user's current project
    # memories from MongoDB directly, avoiding decision/converter LLM calls,
    # embeddings, Pinecone, and Neo4j on this common lookup.
    if is_question and _requires_personal_project_lookup(user_input):
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
    if is_question and _requires_personal_workplace_lookup(user_input):
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
    if is_question and _requires_knows_user_lookup(user_input):
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

    if is_question and _requires_personal_project_lookup(user_input):
        logger.info("RETRIEVAL_PATH user_id=%s path=project_fast", user_id)
        result["retrieval"] = _retrieve_project_memories_fast(user_id)
        return result

    try:
        result["retrieval"] = _retrieve_memories_fast(user_id, user_input)
    except Exception:
        logger.exception("General memory retrieval failed")
        result["retrieval"] = []
    logger.info(
        "RETRIEVAL_TIMING user_id=%s path=general total_ms=%.2f result_count=%d",
        user_id, (time.perf_counter() - fast_started) * 1000,
        len(result["retrieval"]),
    )
    return result