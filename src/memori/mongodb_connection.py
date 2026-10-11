from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

connection_string = os.getenv("mongodb_URI")

# Connect to MongoDB
client = MongoClient(connection_string)

# Select/Create database
db = client["memories"]

# Lazily create indexes on first real database operation. This keeps module
# import cheap while preventing user/context/outbox queries from degrading as
# the collections grow.
import threading
from pymongo import ASCENDING, DESCENDING

_indexes_ready = False
_indexes_lock = threading.Lock()


def ensure_indexes():
    global _indexes_ready
    if _indexes_ready:
        return
    with _indexes_lock:
        if _indexes_ready:
            return

        memory_types = [
            "profile", "fact", "preference", "semantic", "episodic",
            "procedural", "goal", "event", "task", "project", "temporal",
        ]
        for memory_type in memory_types:
            collection = db[memory_type]
            collection.create_index([("user_id", ASCENDING)], name="user_id_idx")
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

        outbox = db["outbox"]
        outbox.create_index([("user_id", ASCENDING)], name="user_id_idx")
        outbox.create_index(
            [("user_id", ASCENDING), ("mongo_id", ASCENDING)],
            name="user_mongo_id_idx",
        )
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
        _indexes_ready = True
