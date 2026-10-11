from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

connection_string = os.getenv("mongodb_URI")

# Connect to MongoDB
client = MongoClient(connection_string)

# Select/Create database
db = client["memories"]

# Lazily create indexes on first real database operation. Indexes are ensured
# per collection so a simple retrieval existence-check does not create dozens
# of indexes across all collections.
import threading
from pymongo import ASCENDING, DESCENDING

_indexed_memory_types = set()
_outbox_basic_ready = False
_outbox_worker_ready = False
_indexes_lock = threading.Lock()
_MEMORY_TYPES = [
    "profile", "fact", "preference", "semantic", "episodic",
    "procedural", "goal", "event", "task", "project", "temporal",
]


def ensure_indexes(memory_types=None, include_outbox=True):
    global _outbox_basic_ready, _outbox_worker_ready
    requested_types = set(_MEMORY_TYPES if memory_types is None else memory_types)
    with _indexes_lock:
        for memory_type in requested_types - _indexed_memory_types:
            collection = db[memory_type]
            collection.create_index(
                [("user_id", ASCENDING), ("temporal.is_current", ASCENDING),
                 ("temporal.valid_until", ASCENDING), ("temporal.valid_from", DESCENDING)],
                name="user_current_temporal_idx",
            )
            collection.create_index(
                [("user_id", ASCENDING), ("id", ASCENDING),
                 ("temporal.is_current", ASCENDING)],
                name="user_memory_id_current_idx",
            )
            _indexed_memory_types.add(memory_type)

        if include_outbox and not _outbox_basic_ready:
            outbox = db["outbox"]
            # The compound index has user_id as its prefix and supports
            # lookup by the outbox event's Mongo ID.
            outbox.create_index(
                [("user_id", ASCENDING), ("mongo_id", ASCENDING)],
                name="user_mongo_id_idx",
            )
            _outbox_basic_ready = True

        if include_outbox is True and not _outbox_worker_ready:
            outbox = db["outbox"]
            outbox.create_index(
                [("user_id", ASCENDING), ("pinecone.status", ASCENDING),
                 ("pinecone.attempts", ASCENDING)],
                name="user_pinecone_status_attempts_idx",
            )
            outbox.create_index(
                [("user_id", ASCENDING), ("neo4j.status", ASCENDING),
                 ("neo4j.attempts", ASCENDING)],
                name="user_neo4j_status_attempts_idx",
            )
            outbox.create_index(
                [("user_id", ASCENDING), ("placeholders.statuses.mongo_status", ASCENDING)],
                name="user_placeholder_mongo_status_idx",
            )
            outbox.create_index(
                [("user_id", ASCENDING), ("placeholders.statuses.pinecone_status", ASCENDING)],
                name="user_placeholder_pinecone_status_idx",
            )
            outbox.create_index(
                [("user_id", ASCENDING), ("placeholders.statuses.neo4j_status", ASCENDING)],
                name="user_placeholder_neo4j_status_idx",
            )
            _outbox_worker_ready = True
