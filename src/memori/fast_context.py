import re
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from src.memori.mongodb_connection import db

MEMORY_TYPES = [
    "profile", "fact", "preference", "semantic", "episodic",
    "procedural", "goal", "event", "task", "project", "temporal",
]

PLACEHOLDER_RE = re.compile(r"\((user_[a-z][a-z0-9_]*)\)")

# Max memories sent to the extractor as context. Pending-placeholder
# memories are always kept first, so they are never cut off.
MAX_CONTEXT_DOCS = 30


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
        print(f"TIMING: {label}: {time.perf_counter() - start:.2f}s")


# ---------------------------------------------------------------------------
# Context for SAVING (no LLM calls)
# ---------------------------------------------------------------------------
def _find_current_docs(memory_type: str, user_id: str) -> list[dict]:
    return list(
        db[memory_type]
        .find({
            "user_id": user_id,
            "temporal.is_current": True,
            "temporal.valid_until": None,
        })
        .sort("temporal.valid_from", -1)
        .limit(MAX_CONTEXT_DOCS)
    )


def _compact(doc: dict) -> dict:
    """Keep only what the extractor needs (smaller prompt = faster call)."""
    return {
        "id": str(doc.get("id") or doc["_id"]),
        "type": doc.get("type"),
        "content": doc.get("content"),
        "entities": doc.get("entities", []),
    }


def get_context_fast(user_id: str) -> dict:
    """Load the user's CURRENT memories straight from MongoDB.

    - No LLM calls (skips the query converter and the router).
    - 11 collections are queried in parallel.
    - Mongo is the source of truth and is always up to date, while Pinecone
      and Neo4j are filled later by your outbox worker.
    - Pending placeholders are read directly from the content, so P9 never
      depends on a search happening to return the right memory.

    If a user's store grows far beyond MAX_CONTEXT_DOCS, switch this to a
    Pinecone top_k search on the message embedding (still no LLM needed).
    """
    with ThreadPoolExecutor(max_workers=len(MEMORY_TYPES)) as pool:
        per_type = list(pool.map(lambda t: _find_current_docs(t, user_id), MEMORY_TYPES))

    docs = [d for group in per_type for d in group]

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
    return (
        system_message.replace("__CURRENT_DATETIME__", current_time)
        + f"\nuser_id: {user_id}"
        + f"\n\nContext: {context}"
        + f"\n\nPending placeholders: {pending_placeholders}"
    )


# ---------------------------------------------------------------------------
# How to use it in main.py
# ---------------------------------------------------------------------------
#
#   from src.memori.fast_context import timed, get_context_fast, build_extractor_prompt
#
#   def save_memory(user_id, user_input, saving_needs_context):
#       current_time = datetime.now(timezone.utc).isoformat()
#
#       if saving_needs_context:
#           with timed("context (mongo)"):
#               ctx = get_context_fast(user_id)
#           context = ctx["context"]
#           pending_placeholder = ctx["pending_placeholders"]
#       else:
#           context, pending_placeholder = [], []
#
#       system_prompt = build_extractor_prompt(
#           system_message, current_time, user_id, context, pending_placeholder
#       )
#
#       with timed("extractor LLM"):
#           response = memory_extractor.invoke([
#               {"role": "system", "content": system_prompt},
#               {"role": "user", "content": user_input},
#           ])
#       ...rest unchanged...
#
# And in main(), time the decision call too:
#
#       with timed("decision LLM"):
#           memory_decision = Memory_Decision(user_input)
#
# Keep retrieve_memory(...) ONLY for answering questions (needs_retrieval).