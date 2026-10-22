from pymongo import MongoClient, monitoring
import logging
import os
import threading
import time
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class _MongoTimingListener(monitoring.CommandListener):
    """Log MongoDB command latency without changing query results."""
    def __init__(self):
        self._started = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(event):
        # request_id is scoped to a connection; include connection_id so
        # concurrent commands on different sockets cannot overwrite timings.
        return (event.connection_id, event.request_id)

    def started(self, event):
        with self._lock:
            self._started[self._key(event)] = time.perf_counter()

    def succeeded(self, event):
        with self._lock:
            started = self._started.pop(self._key(event), None)
        if started is not None:
            logger.info(
                "DB_TIMING database=mongodb command=%s duration_ms=%.2f status=ok",
                event.command_name, (time.perf_counter() - started) * 1000,
            )

    def failed(self, event):
        with self._lock:
            started = self._started.pop(self._key(event), None)
        if started is not None:
            logger.warning(
                "DB_TIMING database=mongodb command=%s duration_ms=%.2f status=error failure=%s",
                event.command_name, (time.perf_counter() - started) * 1000,
                event.failure,
            )

connection_string = os.getenv("mongodb_URI") or os.getenv("MONGODB_URI")
# PyMongo's MongoClient(None) silently targets localhost. Use a lazy proxy that
# fails with a clear configuration error on first DB use instead of falling back.
_mongo_timing_listener = _MongoTimingListener()
_client = None
_client_lock = threading.Lock()


def _get_client():
    global _client, connection_string
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        uri = connection_string or os.getenv("mongodb_URI") or os.getenv("MONGODB_URI")
        if not uri:
            raise RuntimeError(
                "MongoDB URI is missing. Set mongodb_URI or MONGODB_URI before database operations."
            )
        connection_string = uri
        _client = MongoClient(
            uri,
            event_listeners=[_mongo_timing_listener],
            serverSelectionTimeoutMS=int(os.getenv("MONGODB_SERVER_SELECTION_TIMEOUT_MS", "5000")),
            connectTimeoutMS=int(os.getenv("MONGODB_CONNECT_TIMEOUT_MS", "5000")),
            socketTimeoutMS=int(os.getenv("MONGODB_SOCKET_TIMEOUT_MS", "15000")),
        )
        return _client


class _LazyMongoClient:
    def __getattr__(self, name):
        return getattr(_get_client(), name)

    def __getitem__(self, name):
        return _get_client()[name]


class _LazyMongoDatabase:
    def __getattr__(self, name):
        return getattr(_get_client()["memories"], name)

    def __getitem__(self, name):
        return _get_client()["memories"][name]


# Public proxies preserve the existing client.admin/db["collection"]/db.collection API.
client = _LazyMongoClient()
db = _LazyMongoDatabase()

# Lazily create indexes on first real database operation. Indexes are ensured
# per collection so a simple retrieval existence-check does not create dozens
# of indexes across all collections.
import threading
from pymongo import ASCENDING, DESCENDING

_memories_indexes_ready = False
_outbox_basic_ready = False
_outbox_worker_ready = False
_indexes_lock = threading.Lock()


def ensure_indexes(memory_types=None, include_outbox=True):
    """Create indexes for the unified memories collection and the outbox.

    memory_types remains accepted for backward compatibility. An empty list
    skips memory indexes, as used by outbox-only existence checks.
    """
    global _memories_indexes_ready, _outbox_basic_ready, _outbox_worker_ready
    with _indexes_lock:
        if not _memories_indexes_ready and memory_types != []:
            memories = db["memories"]
            memories.create_index(
                [("user_id", ASCENDING), ("temporal.is_current", ASCENDING),
                 ("temporal.valid_until", ASCENDING), ("temporal.valid_from", DESCENDING)],
                name="user_current_temporal_idx",
            )
            memories.create_index(
                [("user_id", ASCENDING), ("id", ASCENDING),
                 ("temporal.is_current", ASCENDING)],
                name="user_memory_id_current_idx",
            )
            memories.create_index(
                [("user_id", ASCENDING), ("type", ASCENDING),
                 ("temporal.is_current", ASCENDING)],
                name="user_type_current_idx",
            )
            memories.create_index(
                [("user_id", ASCENDING), ("temporal.is_current", ASCENDING),
                 ("entities.relationship", ASCENDING)],
                name="user_current_entity_relationship_idx",
            )
            _memories_indexes_ready = True

        if include_outbox and not _outbox_basic_ready:
            outbox = db["outbox"]
            outbox.create_index(
                [("user_id", ASCENDING), ("mongo_id", ASCENDING)],
                name="user_mongo_id_idx",
            )
            outbox.create_index(
                [("user_id", ASCENDING), ("memory_id", ASCENDING)],
                name="user_memory_id_idx",
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
