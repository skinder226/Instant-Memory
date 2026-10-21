import logging
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from bson import ObjectId
from openai import APIStatusError

from .mongodb_connection import db, ensure_indexes
from .vector_str import index
from .embeddings import embeddings
from .neo4j_learn import set_memory_to_neo4j, set_memory_to_neo4j_supersedes

logger = logging.getLogger(__name__)

# Separate executors prevent provider tasks from deadlocking the ingestion pool.
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="memori-worker")
_provider_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="memori-provider")
# A fixed set of locks prevents the lock registry growing forever with user IDs.
_USER_LOCKS = tuple(threading.Lock() for _ in range(64))
MAX_PROVIDER_ATTEMPTS = 5
PINECONE_BATCH_SIZE = 50


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def _get_user_lock(user_id):
    return _USER_LOCKS[hash(user_id) % len(_USER_LOCKS)]


def find_pending_memory(user_id):
    ensure_indexes(memory_types=[], include_outbox=True)
    return list(db.outbox.find({
        "user_id": user_id,
        "$or": [
            {"pinecone.status": "pending"},
            {"$and": [
                {"pinecone.status": "error"},
                {"$or": [
                    {"pinecone.attempts": {"$lt": MAX_PROVIDER_ATTEMPTS}},
                    {"pinecone.attempts": {"$exists": False}},
                ]},
            ]},
            {"neo4j.status": "pending"},
            {"$and": [
                {"neo4j.status": "error"},
                {"$or": [
                    {"neo4j.attempts": {"$lt": MAX_PROVIDER_ATTEMPTS}},
                    {"neo4j.attempts": {"$exists": False}},
                ]},
            ]},
            {"placeholders.statuses.mongo_status": {"$in": ["pending", "error"]}},
            {"placeholders.statuses.pinecone_status": {"$in": ["pending", "error"]}},
            {"placeholders.statuses.neo4j_status": {"$in": ["pending", "error"]}},
        ],
    }))

def get_real_memories(user_id, mongo_id, memory_type):
    if not memory_type or not mongo_id or not ObjectId.is_valid(str(mongo_id)):
        return None
    memory = db["memories"].find_one({
        "user_id": user_id,
        "_id": ObjectId(str(mongo_id)),
    })
    if memory and memory.get("type") != memory_type:
        return None
    return memory


def set_memory_status(user_id, mongo_id, db_name, status):
    update = {"$set": {db_name + ".status": status}}
    if status == "completed":
        update["$set"][db_name + ".last_error"] = None
    db.outbox.update_one(
        {"user_id": user_id, "mongo_id": str(mongo_id)},
        update,
    )


def set_memory_error_or_attempts(user_id, mongo_id, db_name, last_error):
    db.outbox.update_one(
        {"user_id": user_id, "mongo_id": str(mongo_id)},
        {
            "$set": {
                db_name + ".status": "error",
                db_name + ".last_error": str(last_error),
            },
            "$inc": {db_name + ".attempts": 1},
        },
    )


def _pinecone_vector(memory, values):
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
    return {"id": str(memory["_id"]), "values": values, "metadata": metadata}


def embed_and_upsert_memory(memory):
    values = embeddings.embed_query(memory["content"])
    index.upsert(
        vectors=[_pinecone_vector(memory, values)],
        namespace=memory["user_id"],
    )


def update_memory_pinecone(user_id, update_ids):
    ids = [str(item) for item in (update_ids or []) if item]
    for start in range(0, len(ids), 100):
        for update_id in ids[start:start + 100]:
            index.update(
                id=update_id,
                namespace=user_id,
                set_metadata={
                    "is_current": False,
                    "valid_until": utc_now_iso(),
                },
            )


def _process_pinecone(user_id, work_items):
    # Batch embeddings and Pinecone upserts; individual status updates remain
    # per outbox event so a failed item can be retried independently.
    for start in range(0, len(work_items), PINECONE_BATCH_SIZE):
        batch = work_items[start:start + PINECONE_BATCH_SIZE]
        batch_started = time.perf_counter()
        try:
            contents = [memory["content"] for memory, _ in batch]
            embedding_started = time.perf_counter()
            embedded = embeddings.embed_documents(contents)
            embedding_ms = (time.perf_counter() - embedding_started) * 1000
            vectors = [
                _pinecone_vector(memory, values)
                for (memory, _), values in zip(batch, embedded)
            ]
            upsert_started = time.perf_counter()
            index.upsert(vectors=vectors, namespace=user_id)
            upsert_ms = (time.perf_counter() - upsert_started) * 1000
            logger.info(
                "SAVE_TIMING database=pinecone operation=upsert user_id=%s items=%d "
                "embedding_ms=%.2f upsert_ms=%.2f total_ms=%.2f",
                user_id, len(batch), embedding_ms, upsert_ms,
                (time.perf_counter() - batch_started) * 1000,
            )
        except Exception as exc:
            logger.warning(
                "SAVE_TIMING database=pinecone operation=upsert user_id=%s items=%d "
                "status=error elapsed_ms=%.2f",
                user_id, len(batch), (time.perf_counter() - batch_started) * 1000,
            )
            logger.exception("Pinecone batch failed for user_id=%s", user_id)
            for memory, _ in batch:
                set_memory_error_or_attempts(user_id, memory["_id"], "pinecone", exc)
            continue

        for memory, supersedes in batch:
            mongo_id = memory["_id"]
            try:
                update_memory_pinecone(user_id, supersedes)
                set_memory_status(user_id, mongo_id, "pinecone", "completed")
            except Exception as exc:
                logger.exception("Pinecone supersede update failed for %s", mongo_id)
                set_memory_error_or_attempts(user_id, mongo_id, "pinecone", exc)


def _process_neo4j(user_id, work_items):
    for memory, supersedes in work_items:
        mongo_id = memory["_id"]
        neo4j_started = time.perf_counter()
        try:
            set_memory_to_neo4j(memory, user_id)
            set_memory_to_neo4j_supersedes(user_id, [supersedes])
            set_memory_status(user_id, mongo_id, "neo4j", "completed")
            logger.info(
                "SAVE_TIMING database=neo4j operation=save_memory user_id=%s "
                "memory_id=%s duration_ms=%.2f status=ok",
                user_id, mongo_id, (time.perf_counter() - neo4j_started) * 1000,
            )
        except Exception as exc:
            logger.warning(
                "SAVE_TIMING database=neo4j operation=save_memory user_id=%s "
                "memory_id=%s duration_ms=%.2f status=error",
                user_id, mongo_id, (time.perf_counter() - neo4j_started) * 1000,
            )
            logger.exception("Neo4j failed for %s", mongo_id)
            set_memory_error_or_attempts(user_id, mongo_id, "neo4j", exc)


def _save_pending_memory_worker(user_id):
    worker_started = time.perf_counter()
    with _get_user_lock(user_id):
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
                            user_id, mongo_id, provider,
                            "Source memory no longer exists in MongoDB",
                        )
                continue

            if outbox.get("pinecone", {}).get("status") == "pending" or (
                outbox.get("pinecone", {}).get("status") == "error"
                and outbox.get("pinecone", {}).get("attempts", 0) < MAX_PROVIDER_ATTEMPTS
            ):
                pinecone_work.append((real_memory, supersedes))

            if outbox.get("neo4j", {}).get("status") == "pending" or (
                outbox.get("neo4j", {}).get("status") == "error"
                and outbox.get("neo4j", {}).get("attempts", 0) < MAX_PROVIDER_ATTEMPTS
            ):
                neo4j_work.append((real_memory, supersedes))

        # These sinks are independent. Use a separate executor so waiting here
        # cannot starve the ingestion executor that scheduled this function.
        futures = []
        if pinecone_work:
            futures.append(_provider_executor.submit(_process_pinecone, user_id, pinecone_work))
        if neo4j_work:
            futures.append(_provider_executor.submit(_process_neo4j, user_id, neo4j_work))
        for future in futures:
            future.result()
        logger.info(
            "SAVE_TIMING operation=sync_pending user_id=%s pending=%d "
            "pinecone_items=%d neo4j_items=%d total_ms=%.2f",
            user_id, len(pending), len(pinecone_work), len(neo4j_work),
            (time.perf_counter() - worker_started) * 1000,
        )


def _log_future(future):
    try:
        future.result()
    except Exception:
        logger.exception("Background memory worker failed")


def save_pending_memory(user_id):
    future = _executor.submit(_save_pending_memory_worker, user_id)
    future.add_done_callback(_log_future)
    return future


def invoke_with_retry(runnable, messages, retries=3):
    """Retry transient 429/503 responses, honoring numeric Retry-After headers."""
    for attempt in range(retries):
        try:
            return runnable.invoke(messages)
        except APIStatusError as exc:
            if exc.status_code not in (429, 503) or attempt == retries - 1:
                raise

            headers = getattr(getattr(exc, "response", None), "headers", {}) or {}
            retry_after = headers.get("retry-after") or headers.get("Retry-After")
            wait = None
            if retry_after is not None:
                try:
                    wait = float(retry_after)
                except (TypeError, ValueError):
                    # HTTP-date Retry-After values are not numeric; fall back
                    # to bounded backoff rather than failing the save worker.
                    wait = None

            if wait is None:
                # Jitter avoids synchronized workers retrying in lockstep.
                wait = min(2 ** attempt, 8) + random.uniform(0, 0.25)
            if wait < 0:
                wait = 0
            if wait > 60:
                logger.warning(
                    "LLM temporarily unavailable (%s); Retry-After=%ss exceeds "
                    "the worker wait budget, deferring to the outbox retry cycle",
                    exc.status_code, wait,
                )
                raise

            logger.warning(
                "LLM temporarily unavailable (%s); retrying in %.2fs",
                exc.status_code, wait,
            )
            time.sleep(wait)
    raise RuntimeError("Retry loop exited without a result")
