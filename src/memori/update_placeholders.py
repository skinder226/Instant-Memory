from bson import ObjectId
from pymongo import UpdateOne
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
    """Resolve all matching placeholders per memory with one read and one write."""
    resolutions = dict(_resolved_updates(placeholder_updates))
    updated = []
    status_operations = []

    for memory in mongo_pending_memories:
        user_id = memory["user_id"]
        mongo_id = str(memory["mongo_id"])
        collection_name = "memories"
        active_names = []
        for placeholder in memory.get("placeholders", []):
            name = placeholder.get("name")
            if (
                name in resolutions
                and placeholder.get("statuses", {}).get("mongo_status") in ("pending", "error")
            ):
                active_names.append(name)
        if not active_names or not ObjectId.is_valid(mongo_id):
            continue

        collection = db[collection_name]
        oid = ObjectId(mongo_id)
        doc = collection.find_one({"_id": oid, "user_id": user_id})
        if not doc:
            continue

        content = doc.get("content") or ""
        for name in active_names:
            content = content.replace("(" + name + ")", resolutions[name])

        entities = doc.get("entities") or []
        for entity in entities:
            for side in ("source", "target"):
                node = entity.get(side)
                if isinstance(node, dict) and node.get("name") in active_names:
                    node["name"] = resolutions[node["name"]]

        result = collection.update_one(
            {"_id": oid, "user_id": user_id},
            {"$set": {"content": content, "entities": entities}},
        )
        if not result.matched_count:
            continue

        for name in active_names:
            updated.append({
                "mongo_id": mongo_id,
                "user_id": user_id,
                "content": content,
                "placeholder_name": name,
            })
            status_operations.append(UpdateOne(
                {"user_id": user_id, "mongo_id": mongo_id},
                {"$set": {
                    "placeholders.$[p].statuses.mongo_status": "completed",
                    "placeholders.$[p].mongo_error": None,
                }},
                array_filters=[{
                    "p.name": name,
                    "p.statuses.mongo_status": {"$in": ["pending", "error"]},
                }],
            ))

    if status_operations:
        db["outbox"].bulk_write(status_operations, ordered=False)
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
    pending = []
    for doc in docs:
        name = doc.get("placeholder_name")
        if name in resolutions and doc.get("content"):
            pending.append(doc)

    # Pinecone update is one-vector-at-a-time. Fetch existing metadata in
    # batches and upsert vectors in batches so all metadata is preserved.
    batch_size = 50
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        ids = [str(doc["mongo_id"]) for doc in batch]
        try:
            contents = [doc["content"] for doc in batch]
            vectors = embeddings.embed_documents(contents)
            existing = index.fetch(ids=ids, namespace=user_id)
            existing_vectors = getattr(existing, "vectors", {}) or {}
            upserts = []
            for doc, values in zip(batch, vectors):
                memory_id = str(doc["mongo_id"])
                old_vector = existing_vectors.get(memory_id)
                if old_vector is None:
                    raise ValueError(
                        "Cannot resolve placeholder in missing Pinecone vector " + memory_id
                    )
                metadata = dict(getattr(old_vector, "metadata", None) or {})
                metadata["content"] = doc["content"]
                upserts.append({
                    "id": memory_id,
                    "values": values,
                    "metadata": metadata,
                })
            if upserts:
                index.upsert(vectors=upserts, namespace=user_id)
            for doc in batch:
                update_status(user_id, doc["mongo_id"], "pinecone", doc["placeholder_name"])
        except Exception as exc:
            for doc in batch:
                update_status(
                    user_id, doc["mongo_id"], "pinecone",
                    doc["placeholder_name"], "error", str(exc),
                )
            raise
