import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from bson import ObjectId

from .mongodb_connection import db
from .vector_str import index
from .embeddings import embeddings
from .neo4j_learn import set_memory_to_neo4j, set_memory_to_neo4j_supersedes

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="memori-worker")
_user_locks = {}
_user_locks_guard = threading.Lock()


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def _get_user_lock(user_id):
    with _user_locks_guard:
        return _user_locks.setdefault(user_id, threading.Lock())


def find_pending_memory(user_id):
    return list(db.outbox.find({
        "user_id": user_id,
        "$or": [
            {"pinecone.status": {"$in": ["pending", "error"]}},
            {"neo4j.status": {"$in": ["pending", "error"]}},
            {"placeholders.statuses.mongo_status": {"$in": ["pending", "error"]}},
            {"placeholders.statuses.pinecone_status": {"$in": ["pending", "error"]}},
            {"placeholders.statuses.neo4j_status": {"$in": ["pending", "error"]}},
        ],
    }))


def get_real_memories(user_id, mongo_id, memory_type):
    if not memory_type or not mongo_id:
        return None
    return db[memory_type].find_one({
        "user_id": user_id,
        "_id": ObjectId(str(mongo_id)),
    })


def set_memory_status(user_id, mongo_id, db_name, status):
    update = {"$set": {f"{db_name}.status": status}}
    if status == "completed":
        update["$set"][f"{db_name}.last_error"] = None
    db.outbox.update_one(
        {"user_id": user_id, "mongo_id": str(mongo_id)},
        update,
    )


def set_memory_error_or_attempts(user_id, mongo_id, db_name, last_error):
    db.outbox.update_one(
        {"user_id": user_id, "mongo_id": str(mongo_id)},
        {
            "$set": {
                f"{db_name}.status": "error",
                f"{db_name}.last_error": str(last_error),
            },
            "$inc": {f"{db_name}.attempts": 1},
        },
    )


def embed_and_upsert_memory(memory):
    temporal = memory.get("temporal") or {}
    metadata = {
        "user_id": memory["user_id"],
        "memory_type": memory["type"],
        "is_current": temporal.get("is_current", True),
        "content": memory["content"],
        "valid_from": temporal.get("valid_from"),
        "memory_id": str(memory["_id"]),
    }
    if temporal.get("valid_until"):
        metadata["valid_until"] = temporal["valid_until"]

    values = embeddings.embed_query(memory["content"])
    index.upsert(
        vectors=[{
            "id": str(memory["_id"]),
            "values": values,
            "metadata": metadata,
        }],
        namespace=memory["user_id"],
    )


def update_memory_pinecone(user_id, update_ids):
    for update_id in update_ids or []:
        if update_id:
            index.update(
                id=str(update_id),
                namespace=user_id,
                set_metadata={
                    "is_current": False,
                    "valid_until": utc_now_iso(),
                },
            )


def _process_pinecone(user_id, work_items):
    for memory, supersedes in work_items:
        mongo_id = memory["_id"]
        try:
            embed_and_upsert_memory(memory)
            update_memory_pinecone(user_id, supersedes)
            set_memory_status(user_id, mongo_id, "pinecone", "completed")
        except Exception as e:
            logger.exception("Pinecone failed for %s", mongo_id)
            set_memory_error_or_attempts(user_id, mongo_id, "pinecone", e)


def _process_neo4j(user_id, work_items):
    for memory, supersedes in work_items:
        mongo_id = memory["_id"]
        try:
            set_memory_to_neo4j(memory, user_id)
            set_memory_to_neo4j_supersedes(user_id, [supersedes])
            set_memory_status(user_id, mongo_id, "neo4j", "completed")
        except Exception as e:
            logger.exception("Neo4j failed for %s", mongo_id)
            set_memory_error_or_attempts(user_id, mongo_id, "neo4j", e)


def _save_pending_memory_worker(user_id):
    lock = _get_user_lock(user_id)

    with lock:
        pending = find_pending_memory(user_id)
        pinecone_work = []
        neo4j_work = []

        for outbox in pending:
            mongo_id = outbox.get("mongo_id")
            memory_type = outbox.get("memory_type")
            supersedes = [str(x) for x in (outbox.get("supersedes_memory_ids") or [])]

            real_memory = get_real_memories(user_id, mongo_id, memory_type)
            if not real_memory:
                for provider in ("pinecone", "neo4j"):
                    if outbox.get(provider, {}).get("status") in ("pending", "error"):
                        set_memory_error_or_attempts(
                            user_id,
                            mongo_id,
                            provider,
                            "Source memory no longer exists in MongoDB",
                        )
                continue

            if outbox.get("pinecone", {}).get("status") in ("pending", "error"):
                pinecone_work.append((real_memory, supersedes))

            if outbox.get("neo4j", {}).get("status") in ("pending", "error"):
                neo4j_work.append((real_memory, supersedes))

        _process_pinecone(user_id, pinecone_work)
        _process_neo4j(user_id, neo4j_work)


def _log_future(future):
    try:
        future.result()
    except Exception:
        logger.exception("Background memory worker failed")


def save_pending_memory(user_id):
    future = _executor.submit(_save_pending_memory_worker, user_id)
    future.add_done_callback(_log_future)
    return future




import time
from openai import APIStatusError


def invoke_with_retry(runnable, messages, retries=3):
    for attempt in range(retries):
        try:
            return runnable.invoke(messages)

        except APIStatusError as e:
            if e.status_code != 503 or attempt == retries - 1:
                raise

            wait = 2 ** attempt
            print(
                f"LLM temporarily unavailable (503). "
                f"Retrying in {wait}s..."
            )
            time.sleep(wait)