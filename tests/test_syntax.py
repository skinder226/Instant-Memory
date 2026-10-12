from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_python_sources_compile():
    files = [ROOT / "main.py", *((ROOT / "src").rglob("*.py")), *((ROOT / "scripts").rglob("*.py"))]
    for path in files:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_prompt_uses_runtime_timestamp_marker():
    prompt = (ROOT / "src/memori/prompts.py").read_text(encoding="utf-8")
    assert '__CURRENT_DATETIME__' in prompt
    assert 'datetime.now(timezone.utc).isoformat()' not in prompt



def test_extraction_prompt_no_longer_mentions_unused_scores():
    prompt = (ROOT / "src/memori/prompts.py").read_text(encoding="utf-8")
    assert "confidence" not in prompt.lower()
    assert "importance" not in prompt.lower()


def test_graph_writer_uses_canonical_relationship_identity():
    writer = (ROOT / "src/memori/neo4j_learn.py").read_text(encoding="utf-8")
    assert 'MERGE (n)-[r:{relationship_type} {{user_id: $user_id}}]->(t)' in writer
    assert "r.memory_ids" in writer
    assert "memory_id: row.memory_id" not in writer



def test_temporal_retrieval_intent_is_explicit_and_enforced():
    prompt = (ROOT / "src/memori/prompts.py").read_text(encoding="utf-8")
    worker = (ROOT / "src/memori/retival_worker.py").read_text(encoding="utf-8")

    assert "is_current=true" in prompt
    assert "is_current=false" in prompt
    assert "is_current=null" in prompt
    assert "is_current=retrieval_plan.is_current" in worker
    assert "memory_is_current != is_current" in worker
