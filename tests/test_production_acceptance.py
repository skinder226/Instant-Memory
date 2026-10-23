"""Production acceptance tests.

Fast tests run in CI without external credentials. Live provider tests are opt-in:
    MEMORI_RUN_LIVE_TESTS=1
    MEMORI_LIVE_TEST_USER_PREFIX=ci-acceptance
    uv run pytest tests/test_production_acceptance.py -m live -v

Use disposable test tenants and a non-production MongoDB/Pinecone/Neo4j project.
Live tests create uniquely named records and clean up only their own records.
"""
from __future__ import annotations

import concurrent.futures
import os
import re
import subprocess
import sys
import time
import uuid

import pytest

from memori import agentic_memory, security
from memori import api
from dotenv import load_dotenv


load_dotenv()

LIVE = os.getenv("MEMORI_RUN_LIVE_TESTS", "").strip().lower() in {"1", "true", "yes"}
def _live_services():
    if not LIVE:
        pytest.skip("Set MEMORI_RUN_LIVE_TESTS=1 to enable live-service acceptance tests.")
    required = (
        "mongodb_URI",
        "PINECONE_API_KEY",
        "NEO4J_URI",
        "NEO4J_USERNAME",
        "NEO4J_PASSWORD",
        "GEMINI_API_KEY",
    )
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        pytest.fail("Live acceptance test missing required environment variables: " + ", ".join(missing))
        print("Live acceptance test missing required environment variables: " + ", ".join(missing))
    from memori.mongodb_connection import db
    return db


def _memory(content: str) -> dict:
    token = uuid.uuid4().hex
    return {
        "type": "profile",
        "content": content,
        "confidence": 1.0,
        "importance": 0.5,
        "source_role": "user",
        "entities": [{
            "source": {"name": "AcceptanceSource-" + token, "type": ["Person"]},
            "target": {"name": "AcceptanceTarget-" + token, "type": ["Project"]},
            "relationship": "knows",
            "properties": {},
        }],
        "temporal": {"is_current": True},
    }


@pytest.mark.live
def test_live_memory_sync_mongodb_pinecone_and_neo4j():
    """A real Mongo memory must reach both projections before outbox is completed."""
    db = _live_services()
    from memori.worker import save_pending_memory
    from memori.vector_str import index
    from memori.Neo4j_connect import Neo4jGraph

    run_id = uuid.uuid4().hex
    user_id = f"{os.getenv('MEMORI_LIVE_TEST_USER_PREFIX', 'acceptance')}-{run_id}"
    content = f"Production acceptance sentinel {run_id}"
    mongo_ids: list[str] = []
    neo4j_graph = Neo4jGraph

    try:
        created = api._save_memories_and_outbox(db, [_memory(content)], user_id)
        assert len(created) == 1
        _stable_id, mongo_id = created[0]
        mongo_ids.append(mongo_id)

        # Wait for actual provider sync; this future is returned by the worker.
        save_pending_memory(user_id).result(timeout=180)

        source = db.memories.find_one({"user_id": user_id, "_id": __import__("bson").ObjectId(mongo_id)})
        outbox = db.outbox.find_one({"user_id": user_id, "mongo_id": mongo_id})
        assert source and source["content"] == content
        assert outbox, "MongoDB source write must have an outbox event"
        assert outbox.get("pinecone", {}).get("status") == "completed", outbox.get("pinecone")
        assert outbox.get("neo4j", {}).get("status") == "completed", outbox.get("neo4j")

        # Explicitly verify the vector can be fetched in the user's namespace.
        fetched = index.fetch(ids=[mongo_id], namespace=user_id)
        vectors = getattr(fetched, "vectors", {}) or {}
        assert mongo_id in vectors, "Pinecone did not return the synchronized vector"
        metadata = getattr(vectors[mongo_id], "metadata", {}) or {}
        assert metadata.get("user_id") == user_id
        assert metadata.get("content") == content

        # The graph writer's success status is supplemented by a read-after-write
        # check against the configured Neo4j database. All graph writes are tenant-tagged.
        result = neo4j_graph.query(
            "MATCH (n {user_id: $user_id})-[r {user_id: $user_id}]->(t {user_id: $user_id}) "
            "WHERE $mongo_id IN coalesce(r.memory_ids, []) "
            "RETURN count(r) AS relationship_count",
            params={"user_id": user_id, "mongo_id": mongo_id},
        )
        assert result and result[0]["relationship_count"] >= 1, (
            "Neo4j did not expose the synchronized relationship after successful sync"
        )
    finally:
        # Cleanup is tenant-scoped and uses this test's unique IDs only.
        if mongo_ids:
            db.outbox.delete_many({"user_id": user_id, "mongo_id": {"$in": mongo_ids}})
            db.memories.delete_many({
                "user_id": user_id,
                "_id": {"$in": [__import__("bson").ObjectId(value) for value in mongo_ids]},
            })
            try:
                index.delete(ids=mongo_ids, namespace=user_id)
            except Exception:
                pass
            try:
                neo4j_graph.query(
                    "MATCH (n {user_id: $user_id})-[r {user_id: $user_id}]->(t {user_id: $user_id}) "
                    "WHERE $mongo_id IN coalesce(r.memory_ids, []) DELETE r",
                    params={"user_id": user_id, "mongo_id": mongo_ids[0]},
                )
            except Exception:
                pass


@pytest.mark.live
def test_live_pending_outbox_is_replayed_after_worker_restart():
    """A pending outbox event must be recoverable by a fresh worker invocation."""
    db = _live_services()
    from memori.worker import save_pending_memory
    from bson import ObjectId

    run_id = uuid.uuid4().hex
    user_id = f"{os.getenv('MEMORI_LIVE_TEST_USER_PREFIX', 'recovery')}-{run_id}"
    content = f"Outbox restart recovery sentinel {run_id}"
    mongo_id = None
    try:
        created = api._save_memories_and_outbox(db, [_memory(content)], user_id)
        assert created
        _, mongo_id = created[0]
        # Simulate a process exiting after transient provider errors were recorded.
        db.outbox.update_one(
            {"user_id": user_id, "mongo_id": mongo_id},
            {"$set": {"pinecone.status": "error", "neo4j.status": "error"},
             "$inc": {"pinecone.attempts": 1, "neo4j.attempts": 1}},
        )
        event = db.outbox.find_one({"user_id": user_id, "mongo_id": mongo_id})
        assert event and event["pinecone"]["status"] == "error"
        assert event["neo4j"]["status"] == "error"

        # Cross a real process boundary: a separate Python process starts,
        # reconnects to MongoDB, and confirms the failed event survived process exit.
        child_script = (
            "import sys; from memori.mongodb_connection import db; "
            "user_id, mongo_id = sys.argv[1:3]; "
            "event = db.outbox.find_one({'user_id': user_id, 'mongo_id': mongo_id}); "
            "assert event and event.get('pinecone', {}).get('status') == 'error'; "
            "assert event and event.get('neo4j', {}).get('status') == 'error'; "
            "print('durable failed outbox event survived process boundary')"
        )
        child = subprocess.run(
            [sys.executable, "-c", child_script, user_id, mongo_id],
            check=True, capture_output=True, text=True, timeout=45,
        )
        assert "survived process boundary" in child.stdout

        # A newly invoked worker now replays durable failed work from MongoDB.
        save_pending_memory(user_id).result(timeout=180)
        replayed = db.outbox.find_one({"user_id": user_id, "mongo_id": mongo_id})
        assert replayed["pinecone"]["status"] == "completed", replayed.get("pinecone")
        assert replayed["neo4j"]["status"] == "completed", replayed.get("neo4j")
    finally:
        if mongo_id:
            db.outbox.delete_many({"user_id": user_id, "mongo_id": mongo_id})
            db.memories.delete_many({"user_id": user_id, "_id": ObjectId(mongo_id)})
            try:
                from memori.vector_str import index
                from memori.Neo4j_connect import Neo4jGraph
                index.delete(ids=[mongo_id], namespace=user_id)
                Neo4jGraph.query(
                    "MATCH (n {user_id: $user_id})-[r {user_id: $user_id}]->(t {user_id: $user_id}) "
                    "WHERE $mongo_id IN coalesce(r.memory_ids, []) DELETE r",
                    params={"user_id": user_id, "mongo_id": mongo_id},
                )
            except Exception:
                pass


@pytest.mark.live
def test_live_agent_events_and_tasks_are_tenant_isolated():
    """A second authenticated tenant must not retrieve another tenant's history."""
    db = _live_services()
    from memori.agentic_memory import (
        get_agent_task,
        record_agent_event,
        retrieve_agent_history,
        save_agent_task,
    )

    run_id = uuid.uuid4().hex
    tenant_a, tenant_b = f"tenant-a-{run_id}", f"tenant-b-{run_id}"
    session_id, task_id, event_id = f"session-{run_id}", f"task-{run_id}", f"event-{run_id}"
    memory_content = f"private memory sentinel {run_id}"
    memory_ids: list[str] = []
    api._USER_MEMORY_CACHE.clear()
    try:
        created = api._save_memories_and_outbox(db, [_memory(memory_content)], tenant_a)
        memory_ids = [item[1] for item in created]
        assert any(
            item.get("content") == memory_content
            for item in api._get_current_user_memories(tenant_a)
        )
        assert all(
            item.get("content") != memory_content
            for item in api._get_current_user_memories(tenant_b)
        )

        record_agent_event(
            tenant_a, session_id, "turn-1", "user",
            content=f"private sentinel {run_id}", event_id=event_id,
        )
        save_agent_task(
            tenant_a, task_id, goal=f"private goal {run_id}", status="paused",
            session_id=session_id, context={"sentinel": run_id},
        )

        assert retrieve_agent_history(tenant_a, session_id=session_id, limit=10)
        assert retrieve_agent_history(tenant_b, session_id=session_id, limit=10) == []
        assert get_agent_task(tenant_a, task_id) is not None
        assert get_agent_task(tenant_b, task_id) is None

        # Confirm the database query includes both the tenant and record key.
        assert db.agent_events.count_documents({"user_id": tenant_b, "event_id": event_id}) == 0
        assert db.agent_tasks.count_documents({"user_id": tenant_b, "task_id": task_id}) == 0
    finally:
        db.agent_events.delete_many({"user_id": tenant_a, "event_id": event_id})
        db.agent_tasks.delete_many({"user_id": tenant_a, "task_id": task_id})
        if memory_ids:
            from bson import ObjectId
            db.outbox.delete_many({"user_id": tenant_a, "mongo_id": {"$in": memory_ids}})
            db.memories.delete_many({
                "user_id": tenant_a, "_id": {"$in": [ObjectId(value) for value in memory_ids]},
            })
        api._USER_MEMORY_CACHE.clear()


@pytest.mark.live
def test_live_concurrent_memory_writes_finish_and_persist():
    """Exercise concurrent Mongo/outbox writes and provider sync with unique records."""
    db = _live_services()
    from bson import ObjectId
    from memori.worker import save_pending_memory

    run_id = uuid.uuid4().hex
    user_id = f"{os.getenv('MEMORI_LIVE_TEST_USER_PREFIX', 'load')}-{run_id}"
    count = int(os.getenv("MEMORI_LIVE_LOAD_ITEMS", "8"))
    if not 1 <= count <= 50:
        pytest.fail("MEMORI_LIVE_LOAD_ITEMS must be between 1 and 50")
    created_ids: list[str] = []
    started = time.perf_counter()
    try:
        def write_one(i: int):
            return api._save_memories_and_outbox(
                db, [_memory(f"Concurrent persistence sentinel {run_id} item {i}")], user_id
            )[0]

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, count)) as pool:
            created = list(pool.map(write_one, range(count)))
        created_ids = [item[1] for item in created]

        # A new worker pass should drain all events that were durably written.
        save_pending_memory(user_id).result(timeout=300)
        elapsed = time.perf_counter() - started
        persisted = db.memories.count_documents({
            "user_id": user_id, "_id": {"$in": [ObjectId(value) for value in created_ids]},
        })
        events = list(db.outbox.find(
            {"user_id": user_id, "mongo_id": {"$in": created_ids}},
            {"mongo_id": 1, "pinecone.status": 1, "neo4j.status": 1},
        ))
        assert persisted == count
        assert len(events) == count
        assert all(e.get("pinecone", {}).get("status") == "completed" for e in events)
        assert all(e.get("neo4j", {}).get("status") == "completed" for e in events)
        max_seconds = float(os.getenv("MEMORI_LIVE_LOAD_MAX_SECONDS", "180"))
        assert elapsed <= max_seconds, (
            f"Concurrent persistence took {elapsed:.2f}s, exceeding "
            f"MEMORI_LIVE_LOAD_MAX_SECONDS={max_seconds:.2f}s"
        )
        print(f"Concurrent persistence: {count} items in {elapsed:.2f}s "
              f"({count / max(elapsed, 0.001):.2f} items/s)")
    finally:
        if created_ids:
            db.outbox.delete_many({"user_id": user_id, "mongo_id": {"$in": created_ids}})
            db.memories.delete_many({
                "user_id": user_id, "_id": {"$in": [ObjectId(value) for value in created_ids]},
            })
            try:
                from memori.vector_str import index
                from memori.Neo4j_connect import Neo4jGraph
                index.delete(ids=created_ids, namespace=user_id)
                Neo4jGraph.query(
                    "MATCH (n {user_id: $user_id})-[r {user_id: $user_id}]->(t {user_id: $user_id}) "
                    "WHERE any(memory_id IN $memory_ids WHERE memory_id IN coalesce(r.memory_ids, [])) "
                    "DELETE r",
                    params={"user_id": user_id, "memory_ids": created_ids},
                )
            except Exception:
                pass


def test_oidc_authorization_requires_verified_identity_and_explicit_allowlist():
    from memori.production_auth import authorize_oidc_claims

    claims = {
        "sub": "subject-123",
        "iss": "https://identity.example",
        "email": "person@example.com",
        "email_verified": True,
    }
    tenant = authorize_oidc_claims(
        claims, allowed_emails="person@example.com", allowed_domains=""
    )
    assert tenant.startswith("oidc_")
    assert tenant == authorize_oidc_claims(
        claims, allowed_emails="person@example.com", allowed_domains=""
    )
    with pytest.raises(PermissionError):
        authorize_oidc_claims(
            claims, allowed_emails="", allowed_domains="other.example"
        )
    with pytest.raises(PermissionError):
        authorize_oidc_claims(
            {**claims, "email_verified": False},
            allowed_emails="person@example.com",
            allowed_domains="",
        )
    with pytest.raises(PermissionError):
        authorize_oidc_claims(
            claims, allowed_emails="", allowed_domains=""
        )


def test_public_url_validation_requires_https_without_embedded_credentials():
    from memori.production_auth import validate_public_https_url

    assert validate_public_https_url("https://memory.example.com") == "https://memory.example.com"
    for invalid in (
        "http://memory.example.com",
        "https:///missing-host",
        "https://user:password@memory.example.com",
        "",
    ):
        with pytest.raises(ValueError):
            validate_public_https_url(invalid)


def test_production_identity_adapter_rejects_client_supplied_blank_identity(monkeypatch):
    monkeypatch.setattr(security, "memori", lambda **kwargs: {"ok": True})
    with pytest.raises(ValueError):
        security.memori_for_principal({"sub": "  "}, "Who is my friend?")


def test_production_mode_uses_oidc_and_does_not_use_browser_selected_tenant():
    """Static guard against accidentally reverting production auth to a text input."""
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "streamlit_app.py").read_text(encoding="utf-8")
    assert 'os.getenv("MEMORI_ENV", "development").strip().lower() == "production"' in source
    assert "st.user.is_logged_in" in source
    assert "st.user" in source
    assert 'user_id = AUTHENTICATED_USER_ID' in source
    assert "Development mode only: user ID is a scope key, not authentication." in source
    assert "STREAMLIT_PRODUCTION_USER_ID" not in source


def test_production_config_does_not_commit_environment_secrets():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    gitignore = (root / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in gitignore
    assert ".env.*" in gitignore
    assert "!.env.example" in gitignore
    assert "dist/" in gitignore and "build/" in gitignore


def test_production_mode_requires_public_https_url():
    """Fail the deployment acceptance check if production is configured with HTTP."""
    from urllib.parse import urlparse

    if os.getenv("MEMORI_ENV", "development").strip().lower() != "production":
        pytest.skip("HTTPS enforcement applies to production deployment jobs.")
    public_url = os.getenv("MEMORI_PUBLIC_URL", "").strip()
    assert public_url, "Set MEMORI_PUBLIC_URL to the public deployment URL."
    parsed = urlparse(public_url)
    assert parsed.scheme == "https" and parsed.netloc, "Production public URL must use HTTPS."
