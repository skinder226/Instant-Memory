from src.memori import neo4j_learn


def test_graph_writer_merges_logical_fact_without_memory_id_identity(monkeypatch):
    calls = []

    def fake_query(query, params=None):
        calls.append((query, params))
        return []

    monkeypatch.setattr(neo4j_learn.Neo4jGraph, "query", staticmethod(fake_query))
    neo4j_learn._INDEXED_LABELS.clear()

    base_memory = {
        "_id": "memory-one",
        "type": "profile",
        "temporal": {
            "is_current": True,
            "valid_from": "2026-10-10T00:00:00+00:00",
            "valid_until": None,
        },
        "entities": [{
            "source": {"name": "user", "type": ["Person", "User"]},
            "relationship": "works_with",
            "target": {"name": "Fasil", "type": ["Person"]},
        }],
    }
    updated_memory = {
        **base_memory,
        "_id": "memory-two",
        "type": "project",
        "entities": [dict(base_memory["entities"][0])],
    }

    neo4j_learn.set_memory_to_neo4j(base_memory, "user_125")
    neo4j_learn.set_memory_to_neo4j(updated_memory, "user_125")

    write_queries = [query for query, _ in calls if "MERGE (n)-[r:works_with" in query]
    assert len(write_queries) == 2
    assert all("MERGE (n)-[r:works_with {user_id: $user_id}]->(t)" in q for q in write_queries)
    assert all("memory_id: row.memory_id" not in q for q in write_queries)
    assert all("r.memory_ids" in q for q in write_queries)
