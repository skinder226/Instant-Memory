"""Interactive launcher for the Memori application."""

import argparse
import re
from datetime import datetime, timezone

from rich import print

from src.memori.fast_context import timed, get_context_fast, build_extractor_prompt
from src.memori.retival_worker import retrieve_memory
from src.memori.update_placeholders import (
    update_mongo_placeholders,
    update_neo4j_placeholders,
    update_pinecone_placeholders,
)
from src.memori.worker import save_pending_memory
from src.memori.mongodb_connection import db
from src.memori.prompts import system_message, memory_decision_prompt
from src.memori.llms import memory_extractor, MemoryDecisionGate


def Memory_Decision(user_input):
    return MemoryDecisionGate.invoke([
        {"role": "system", "content": memory_decision_prompt},
        {"role": "user", "content": user_input},
    ])


def _all_memory_collections():
    return [
        "profile", "fact", "preference", "semantic", "episodic",
        "procedural", "goal", "event", "task", "project", "temporal",
    ]


def save_memory_and_outbox(db, memory_data, user_id):
    """Write Mongo source-of-truth first, then create an outbox event.

    Updates search every memory collection by user_id + stable memory id, so
    changing a memory's type does not orphan its previous version.
    """
    outbox = db["outbox"]
    incoming_id = memory_data.get("id")
    placeholders = memory_data.pop("placeholders", []) or []
    memory_data["user_id"] = user_id

    previous = []
    if incoming_id:
        for memory_type in _all_memory_collections():
            previous.extend(db[memory_type].find({
                "user_id": user_id,
                "id": incoming_id,
                "temporal.is_current": True,
            }))

    now = datetime.now(timezone.utc).isoformat()
    memory_data.setdefault("temporal", {})
    memory_data["temporal"]["valid_from"] = memory_data["temporal"].get("valid_from") or now
    memory_data["temporal"]["valid_until"] = None
    memory_data["temporal"]["is_current"] = True

    if not incoming_id:
        result = db[memory_data["type"]].insert_one(memory_data)
        stable_id = str(result.inserted_id)
    else:
        stable_id = str(incoming_id)

        for old in previous:
            old_collection = db[old["type"]]
            old_collection.update_one(
                {"_id": old["_id"], "user_id": user_id},
                {"$set": {
                    "temporal.is_current": False,
                    "temporal.valid_until": now,
                }},
            )

        memory_data["id"] = stable_id
        result = db[memory_data["type"]].insert_one(memory_data)

    mongo_id = str(result.inserted_id)
    db[memory_data["type"]].update_one(
        {"_id": result.inserted_id, "user_id": user_id},
        {"$set": {"id": stable_id}},
    )

    placeholder_docs = []
    for placeholder in placeholders:
        item = dict(placeholder)
        item["statuses"] = {
            "mongo_status": "pending",
            "pinecone_status": "pending",
            "neo4j_status": "pending",
        }
        placeholder_docs.append(item)

    outbox.insert_one({
        "event_id": mongo_id,
        "event_type": "MEMORY_CREATED" if not previous else "MEMORY_UPDATED",
        "memory_type": memory_data["type"],
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

    return stable_id, mongo_id


def _update_placeholder_values_in_outbox(user_id, placeholder_updates):
    now = datetime.now(timezone.utc).isoformat()
    for update in placeholder_updates:
        fp = update.filled_placeholders
        db.outbox.update_many(
            {
                "user_id": user_id,
                "placeholders": {"$elemMatch": {
                    "name": fp.name,
                    "statuses.mongo_status": "pending",
                }},
            },
            {"$set": {
                "placeholders.$[p].value": fp.value,
                "updated_at": now,
            }},
            array_filters=[{"p.name": fp.name, "p.statuses.mongo_status": "pending"}],
        )


def _pending_placeholder_outbox(user_id):
    return list(db.outbox.find({
        "user_id": user_id,
        "placeholders": {"$elemMatch": {
            "statuses.mongo_status": {"$in": ["pending", "error"]},
        }},
    }))


def save_memory(user_id, user_input, saving_needs_context):
    current_time = datetime.now(timezone.utc).isoformat()
    if saving_needs_context:
        with timed("context (mongo)"):
            ctx = get_context_fast(user_id)
        context = ctx["context"]
        pending_placeholders = ctx["pending_placeholders"]
    else:
        context, pending_placeholders = [], []

    prompt = build_extractor_prompt(
        system_message,
        current_time,
        user_id,
        context,
        pending_placeholders,
    )

    try:
        with timed("extractor LLM"):
            response = memory_extractor.invoke([
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_input},
            ])
    except Exception as e:
        print(f"Error during memory extraction: {e}")
        return False

    for memory in response.memories:
        memory_data = memory.model_dump()
        save_memory_and_outbox(db, memory_data, user_id)

    if response.memories:
        save_pending_memory(user_id)

    if response.placeholder_updates:
        _update_placeholder_values_in_outbox(user_id, response.placeholder_updates)

        pending = _pending_placeholder_outbox(user_id)
        mongo_pending = [
            item for item in pending
            if any(
                p.get("statuses", {}).get("mongo_status") in ("pending", "error")
                for p in item.get("placeholders", [])
            )
        ]

        docs = update_mongo_placeholders(
            mongo_pending,
            response.placeholder_updates,
        )

        if docs:
            update_neo4j_placeholders(
                user_id, response.placeholder_updates, docs
            )
            update_pinecone_placeholders(
                user_id, response.placeholder_updates, docs
            )

    return bool(response.memories or response.placeholder_updates)


def main(user_id):
    while True:
        user_input = input("\nEnter your input (or type 'exit' to quit): ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        with timed("decision LLM"):
            decision = Memory_Decision(user_input)

        if decision.needs_retrieval:
            print(retrieve_memory(user_id=user_id, user_query=user_input))

        if decision.needs_saving:
            save_memory(
                user_id=user_id,
                user_input=user_input,
                saving_needs_context=decision.saving_needs_context,
            )

    print("Exiting...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", required=True)
    args = parser.parse_args()
    main(args.user_id)
