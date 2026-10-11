import re
import time
from contextlib import contextmanager
import logging

from .mongodb_connection import db, ensure_indexes

logger = logging.getLogger(__name__)

PLACEHOLDER_RE = re.compile(r"\((user_[a-z][a-z0-9_]*)\)")

# Max memories sent to the extractor as context. Pending-placeholder
# memories are always kept first, so they are never cut off.
MAX_CONTEXT_DOCS = 12
MAX_CONTENT_CHARS = 800
MAX_ENTITIES_PER_MEMORY = 6


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------
@contextmanager
def timed(label: str):
    """Usage:  with timed("extractor LLM"): response = memory_extractor.invoke(...)"""
    start = time.perf_counter()
    try:
        yield
    finally:
        logger.debug("TIMING %s: %.3fs", label, time.perf_counter() - start)


# ---------------------------------------------------------------------------
# Context for SAVING (no LLM calls)
# ---------------------------------------------------------------------------
def _find_current_docs(user_id: str) -> list[dict]:
    ensure_indexes(memory_types=["memories"], include_outbox=False)
    current_filter = {
        "user_id": user_id,
        "temporal.is_current": True,
        "temporal.valid_until": None,
    }
    placeholder_pattern = r"\(user_[a-z][a-z0-9_]*\)"
    pending = list(
        db["memories"].find({
            **current_filter,
            "content": {"$regex": placeholder_pattern},
        }).sort("temporal.valid_from", -1).limit(MAX_CONTEXT_DOCS)
    )
    pending_ids = [doc["_id"] for doc in pending]
    remaining = MAX_CONTEXT_DOCS - len(pending)
    recent = []
    if remaining > 0:
        recent = list(
            db["memories"].find({
                **current_filter,
                "_id": {"$nin": pending_ids},
                "content": {"$not": {"$regex": placeholder_pattern}},
            }).sort("temporal.valid_from", -1).limit(remaining)
        )
    return pending + recent

def _compact_node(node):
    if not isinstance(node, dict):
        return node
    return {
        key: node.get(key)
        for key in ("name", "type")
        if node.get(key) is not None
    }


def _compact_entity(entity):
    if not isinstance(entity, dict):
        return entity
    compact = {}
    if entity.get("source") is not None:
        compact["source"] = _compact_node(entity["source"])
    if entity.get("relationship") is not None:
        compact["relationship"] = entity["relationship"]
    if entity.get("target") is not None:
        compact["target"] = _compact_node(entity["target"])
    return compact


def _compact(doc: dict) -> dict:
    """Keep the most useful context while bounding prompt size."""
    content = doc.get("content") or ""
    entities = doc.get("entities") or []
    return {
        "id": str(doc.get("id") or doc["_id"]),
        "type": doc.get("type"),
        "content": content[:MAX_CONTENT_CHARS],
        "entities": [
            _compact_entity(entity)
            for entity in entities[:MAX_ENTITIES_PER_MEMORY]
        ],
    }


def get_context_fast(user_id: str) -> dict:
    """Load the user's CURRENT memories straight from MongoDB.

    - No LLM calls (skips the query converter and the router).
    - The unified memories collection is queried directly.
    - Mongo is the source of truth and is always up to date, while Pinecone
      and Neo4j are filled later by your outbox worker.
    - Pending placeholders are read directly from the content, so P9 never
      depends on a search happening to return the right memory.

    If a user's store grows far beyond MAX_CONTEXT_DOCS, switch this to a
    Pinecone top_k search on the message embedding (still no LLM needed).
    """
    with timed("load context from MongoDB"):
        docs = _find_current_docs(user_id)

    pending, seen = [], set()
    pending_docs, other_docs = [], []
    for doc in docs:
        names = PLACEHOLDER_RE.findall(doc.get("content") or "")
        if names:
            pending_docs.append(doc)
            for n in names:
                if n not in seen:
                    seen.add(n)
                    pending.append(n)
        else:
            other_docs.append(doc)

    other_docs.sort(
        key=lambda d: (d.get("temporal") or {}).get("valid_from") or "",
        reverse=True,
    )

    chosen = (pending_docs + other_docs)[:MAX_CONTEXT_DOCS]

    return {
        "context": [_compact(d) for d in chosen],
        "pending_placeholders": pending,
    }


# ---------------------------------------------------------------------------
# Prompt builder (fixes the "if context else" precedence bug)
# ---------------------------------------------------------------------------
def build_extractor_prompt(
    system_message: str,
    current_time: str,
    user_id: str,
    context: list,
    pending_placeholders: list,
) -> str:
    """Always includes BOTH the context and the pending placeholders."""
    # Keep the large, stable system prompt byte-for-byte unchanged so
    # provider prompt caching can reuse its prefix across calls. Put dynamic
    # values after the stable instructions instead of replacing text in them.
    return (
        system_message
        + f"\n\nRuntime metadata (use for new valid_from timestamps): {current_time}"
        + f"\nuser_id: {user_id}"
        + f"\n\nContext: {context}"
        + f"\n\nPending placeholders: {pending_placeholders}"
    )

