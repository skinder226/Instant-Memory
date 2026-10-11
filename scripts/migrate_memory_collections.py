"""Migrate legacy memory-type collections into the unified memories collection.

Run from the repository root after backing up MongoDB:
    uv run python scripts/migrate_memory_collections.py

After verifying the app, old collections can optionally be dropped:
    uv run python scripts/migrate_memory_collections.py --drop-legacy
"""
from __future__ import annotations

import argparse
import logging

from src.memori.mongodb_connection import db

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("memory-migration")

LEGACY_COLLECTIONS = (
    "profile", "fact", "preference", "semantic", "episodic",
    "procedural", "goal", "event", "task", "project", "temporal",
)


def migrate(drop_legacy: bool = False) -> None:
    target = db["memories"]
    migrated = 0
    skipped = 0

    for collection_name in LEGACY_COLLECTIONS:
        source = db[collection_name]
        count = source.count_documents({})
        logger.info("%s: scanning %d documents", collection_name, count)

        for source_doc in source.find({}):
            doc = dict(source_doc)
            # The old collection name is the authoritative type for legacy rows.
            doc["type"] = collection_name
            existing = target.find_one({"_id": doc["_id"]})

            if existing:
                if (
                    existing.get("user_id") != doc.get("user_id")
                    or existing.get("type") != collection_name
                ):
                    raise RuntimeError(
                        "Conflicting _id while migrating "
                        f"{collection_name}/{doc['_id']}; no legacy collection was dropped."
                    )
                skipped += 1
                continue

            target.insert_one(doc)
            migrated += 1

    # Verify every source document has a matching target document before any
    # optional destructive action.
    for collection_name in LEGACY_COLLECTIONS:
        source = db[collection_name]
        for source_doc in source.find({}, {"_id": 1, "user_id": 1}):
            target_doc = target.find_one(
                {"_id": source_doc["_id"], "user_id": source_doc.get("user_id"),
                 "type": collection_name},
                {"_id": 1},
            )
            if not target_doc:
                raise RuntimeError(
                    f"Verification failed for {collection_name}/{source_doc['_id']}; "
                    "legacy collections were retained."
                )

    logger.info("Migration verified. inserted=%d already_present=%d", migrated, skipped)

    if drop_legacy:
        confirmation = input(
            "Verification passed. Permanently drop the 11 legacy collections? Type DROP to continue: "
        )
        if confirmation != "DROP":
            logger.info("Cancelled dropping legacy collections; all data was retained.")
            return
        for collection_name in LEGACY_COLLECTIONS:
            db.drop_collection(collection_name)
        logger.info("Dropped all legacy memory-type collections.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drop-legacy",
        action="store_true",
        help="Drop the old collections after successful verification and explicit confirmation.",
    )
    args = parser.parse_args()
    migrate(drop_legacy=args.drop_legacy)
