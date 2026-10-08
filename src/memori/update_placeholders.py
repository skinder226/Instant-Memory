from bson import ObjectId
from .Neo4j_connect import Neo4jGraph
from .mongodb_connection import db
from .vector_str import index
from .embeddings import embeddings


def update_status(user_id, mongo_id, db_name, placeholder_name, status="completed", error=None):
    fields = {f"placeholders.$[p].statuses.{db_name}_status": status}
    if error:
        fields[f"placeholders.$[p].{db_name}_error"] = str(error)
    db["outbox"].update_one(
        {"user_id": user_id, "mongo_id": str(mongo_id)},
        {"$set": fields},
        array_filters=[
            {
                "p.name": placeholder_name,
                f"p.statuses.{db_name}_status": {"$in": ["pending", "error"]},
            }
        ],
    )


def _resolved_updates(placeholder_updates):
    result = []
    for update in placeholder_updates:
        fp = update.filled_placeholders if hasattr(update, "filled_placeholders") else update["filled_placeholders"]
        name = fp.name if hasattr(fp, "name") else fp["name"]
        value = fp.value if hasattr(fp, "value") else fp["value"]
        result.append((name, value))
    return result


def update_mongo_placeholders(mongo_pending_memories, placeholder_updates):
    resolutions = dict(_resolved_updates(placeholder_updates))
    updated = []

    for memory in mongo_pending_memories:
        user_id = memory["user_id"]
        mongo_id = memory["mongo_id"]
        collection = db[memory["memory_type"]]
        oid = ObjectId(mongo_id)

        for placeholder in memory.get("placeholders", []):
            name = placeholder.get("name")
            value = resolutions.get(name)

            if placeholder.get("statuses", {}).get("mongo_status") != "pending":
                continue
            if not value:
                continue

            doc = collection.find_one({"_id": oid, "user_id": user_id})
            if not doc:
                continue

            content = (doc.get("content") or "").replace(f"({name})", value)

            result = collection.update_one(
                {"_id": oid, "user_id": user_id},
                [{
                    "$set": {
                "content": content,
                        "entities": {
                            "$map": {
                                "input": {"$ifNull": ["$entities", []]},
                                "as": "entity",
                                "in": {
                                    "$mergeObjects": [
                                        "$$entity",
                                        {
                                            "source": {
                                                "$cond": [
                                                    {"$eq": ["$$entity.source.name", name]},
                                                    {"$mergeObjects": ["$$entity.source", {"name": value}]},
                                                    "$$entity.source",
                                                ]
                                            },
                                            "target": {
                                                "$cond": [
                                                    {"$eq": ["$$entity.target.name", name]},
                                                    {"$mergeObjects": ["$$entity.target", {"name": value}]},
                                                    "$$entity.target",
                                                ]
                                            },
                                        },
                                    ]
                                },
                            }
                        },
                    }
                }],
            )

            updated.append({
                "mongo_id": str(mongo_id),
                "user_id": user_id,
                    "content": content,
                "placeholder_name": name,
            })

            update_status(user_id, mongo_id, "mongo", name)

    return updated


def update_neo4j_placeholders(user_id, placeholder_updates, docs):
    resolutions = dict(_resolved_updates(placeholder_updates))
    rows = []

    for doc in docs:
        name = doc["placeholder_name"]
        value = resolutions.get(name)
        if value:
            rows.append({
                "memory_id": str(doc["mongo_id"]),
                "user_id": user_id,
                "name": name,
                "value": value,
            })

    if not rows:
        return

    query = """
    UNWIND $rows AS row
    MATCH (n {user_id: row.user_id})-[r {memory_id: row.memory_id}]->(t {user_id: row.user_id})
    WHERE n.name = row.name OR t.name = row.name
    SET n.name = CASE WHEN n.name = row.name THEN row.value ELSE n.name END,
        t.name = CASE WHEN t.name = row.name THEN row.value ELSE t.name END
    """

    try:
        Neo4jGraph.query(query, params={"rows": rows})
        for row in rows:
            update_status(user_id, row["memory_id"], "neo4j", row["name"])
    except Exception as e:
        for row in rows:
            update_status(user_id, row["memory_id"], "neo4j", row["name"], "error", str(e))
        raise


def update_pinecone_placeholders(user_id, placeholder_updates, docs):
    resolutions = dict(_resolved_updates(placeholder_updates))

    try:
        for doc in docs:
            name = doc["placeholder_name"]
            if name not in resolutions:
                continue

            content = doc.get("content", "")
            if not content:
                continue

            values = embeddings.embed_query(content)
            index.update(
                id=str(doc["mongo_id"]),
                values=values,
                set_metadata={"content": content},
                namespace=user_id,
            )
            update_status(user_id, doc["mongo_id"], "pinecone", name)

    except Exception as e:
        for doc in docs:
            name = doc.get("placeholder_name")
            if name:
                update_status(
                    user_id,
                    doc["mongo_id"],
                    "pinecone",
                    name,
                    "error",
                    str(e),
                )
        raise
