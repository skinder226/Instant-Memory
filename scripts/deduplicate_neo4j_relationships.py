"""One-time consolidation of duplicate Neo4j edges for a single user.

Run after deploying the canonical-edge writer:
    uv run python scripts/deduplicate_neo4j_relationships.py --user-id user_125

The script groups only relationships with the same user, source node, relationship
type, and target node. It retains the supporting memory IDs and current/temporal
state, and combines other relationship properties. Back up Neo4j before running.
"""
from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone

from src.memori.Neo4j_connect import Neo4jGraph

SAFE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _memory_ids(properties: dict) -> list[str]:
    values = properties.get("memory_ids")
    if not isinstance(values, list):
        values = [properties.get("memory_id")]
    return [str(value) for value in values if value is not None]


def deduplicate_user_relationships(user_id: str) -> int:
    groups = Neo4jGraph.query(
        """
        MATCH (a)-[r]->(b)
        WHERE r.user_id = $user_id
        WITH a, b, type(r) AS relationship_type, collect(properties(r)) AS props
        WHERE size(props) > 1
        RETURN
            a.name AS source_name,
            labels(a) AS source_labels,
            b.name AS target_name,
            labels(b) AS target_labels,
            relationship_type,
            props
        """,
        params={"user_id": user_id},
    ) or []

    merged_count = 0
    for group in groups:
        rel_type = str(group["relationship_type"])
        source_labels = sorted(set(group.get("source_labels") or []))
        target_labels = sorted(set(group.get("target_labels") or []))
        if not SAFE_NAME.fullmatch(rel_type):
            raise ValueError(f"Unsafe relationship type returned by Neo4j: {rel_type!r}")
        if any(not SAFE_NAME.fullmatch(label) for label in source_labels + target_labels):
            raise ValueError("Unsafe node label returned by Neo4j.")

        props_list = group.get("props") or []
        combined = {}
        memory_ids = []
        current = False
        valid_from_values = []
        valid_until_values = []
        for props in props_list:
            combined.update(props or {})
            for memory_id in _memory_ids(props or {}):
                if memory_id not in memory_ids:
                    memory_ids.append(memory_id)
            current = current or bool((props or {}).get("is_current", True))
            valid_from = (props or {}).get("valid_from")
            valid_until = (props or {}).get("valid_until")
            if valid_from:
                valid_from_values.append(str(valid_from))
            if valid_until:
                valid_until_values.append(str(valid_until))

        combined.pop("memory_id", None)
        combined["user_id"] = user_id
        combined["memory_ids"] = memory_ids
        combined["is_current"] = current
        combined["valid_from"] = min(valid_from_values) if valid_from_values else None
        combined["valid_until"] = None if current else (
            max(valid_until_values) if valid_until_values else datetime.now(timezone.utc).isoformat()
        )

        source_pattern = "".join(f":`{label}`" for label in source_labels)
        target_pattern = "".join(f":`{label}`" for label in target_labels)
        query = f"""
        MATCH (a{source_pattern} {{name: $source_name, user_id: $user_id}})
              -[r:`{rel_type}` {{user_id: $user_id}}]->
              (b{target_pattern} {{name: $target_name, user_id: $user_id}})
        WITH a, b, collect(r) AS rels
        WHERE size(rels) > 1
        FOREACH (rel IN rels | DELETE rel)
        CREATE (a)-[merged:`{rel_type}`]->(b)
        SET merged = $properties
        """
        result = Neo4jGraph.query(
            query,
            params={
                "user_id": user_id,
                "source_name": group["source_name"],
                "target_name": group["target_name"],
                "properties": combined,
            },
        )
        merged_count += 1

    return merged_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", required=True, help="Only consolidate this user's graph.")
    args = parser.parse_args()
    count = deduplicate_user_relationships(args.user_id)
    print(f"Consolidated {count} duplicate relationship group(s) for {args.user_id}.")


if __name__ == "__main__":
    main()
