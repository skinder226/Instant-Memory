import re
from datetime import datetime, timezone
from dotenv import load_dotenv
from .Neo4j_connect import Neo4jGraph

load_dotenv()

_LABEL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_REL_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _safe_labels(labels):
    if isinstance(labels, str):
        labels = [labels]
    labels = [str(label) for label in labels or []]
    if not labels or any(not _LABEL_RE.fullmatch(label) for label in labels):
        raise ValueError(f"Invalid Neo4j label(s): {labels!r}")
    return ":".join(labels)


def set_memory_to_neo4j(memory, user_id):
    temporal = memory.get("temporal") or {}
    is_current = temporal.get("is_current", True)
    valid_from = temporal.get("valid_from")
    valid_until = temporal.get("valid_until")
    memory_id = str(memory["_id"])

    for entity in memory.get("entities", []):
        source = entity["source"]
        target = entity["target"]
        relationship_type = str(entity["relationship"])

        if not _REL_RE.fullmatch(relationship_type):
            raise ValueError(f"Invalid Neo4j relationship type: {relationship_type!r}")

        source_labels = _safe_labels(source.get("type"))
        target_labels = _safe_labels(target.get("type"))
        source_name = user_id if source["name"].lower() == "user" else source["name"]
        target_name = user_id if target["name"].lower() == "user" else target["name"]

        properties = dict(entity.get("properties") or {})
        properties.update({
            "memory_id": memory_id,
            "is_current": is_current,
            "valid_from": valid_from,
            "valid_until": valid_until,
            "user_id": user_id,
            "memory_type": memory["type"],
        })

        query = f"""
        MERGE (n:{source_labels} {{name: $source_name, user_id: $user_id}})
        MERGE (t:{target_labels} {{name: $target_name, user_id: $user_id}})
        MERGE (n)-[r:{relationship_type} {{memory_id: $memory_id}}]->(t)
        SET r += $properties
        RETURN n, t, r
        """

        Neo4jGraph.query(query, params={
            "source_name": source_name,
            "target_name": target_name,
            "user_id": user_id,
            "memory_id": memory_id,
            "properties": properties,
        })


def set_memory_to_neo4j_supersedes(user_id, neo4j_supersedes_memory_ids):
    ids = [
        str(memory_id)
        for group in (neo4j_supersedes_memory_ids or [])
        for memory_id in (group or [])
    ]
    if not ids:
        return

    query = """
    UNWIND $memory_ids AS memory_id
    MATCH (n {user_id: $user_id})-[r {memory_id: memory_id}]->(t {user_id: $user_id})
    SET r.is_current = false,
        r.valid_until = $valid_until
    """
    Neo4jGraph.query(query, params={
        "memory_ids": ids,
        "user_id": user_id,
        "valid_until": datetime.now(timezone.utc).isoformat(),
    })
