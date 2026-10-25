# Instant-Memory

Instant-Memory is a MongoDB-first memory pipeline with semantic retrieval through Pinecone and relationship/multi-hop retrieval through Neo4j.

## Public API

Use one function to retrieve memories for a message:

    from memori import memori

    result = memori(
        user_id="user_123",
        user_input="Where do I work?",
        source="user",
    )

    # Give the retrieved memories to your application's LLM.
    memories = result["retrieval"]
    answer = your_llm.invoke({
        "question": "Where do I work?",
        "memories": memories,
    })

## Execution behavior

- **Retrieval is synchronous.** If needed, memori waits for retrieval and returns the retrieved memory documents in result["retrieval"]. Your application LLM uses those documents to generate the final answer.
- **Memory extraction and saving are asynchronous.** If the message should be saved, memori schedules extraction and MongoDB/outbox writes in a background thread and returns without waiting for that work.
- **Pinecone and Neo4j synchronization** is queued separately after the MongoDB/outbox write.
- If retrieval is not needed, result["retrieval"] is an empty list.
- result["memory_save_scheduled"] indicates whether background processing was queued; it does not guarantee that processing later succeeds. Background failures are logged.

## Message source

The source parameter accepts user, assistant (or ai), tool, system, or another string. Unknown roles are normalized to other. The source role is stored on each extracted memory; it is not proof that a message's claims are true. Non-user messages are saved in the background but do not trigger user-question retrieval.


## Agentic memory (episodic events and resumable tasks)

Instant-Memory now exposes a separate API for storing structured agent events and
resumable task state. These APIs complement, rather than replace, `memori()` and
long-term semantic memory extraction.

```python
from memori import (
    record_agent_event,
    retrieve_agent_history,
    search_agent_events,
    save_agent_task,
    get_agent_task,
    list_resumable_agent_tasks,
)

# Persist the model's structured tool-call request.
record_agent_event(
    user_id="user_123",
    session_id="session_456",
    turn_id="turn_001",
    role="assistant",
    event_type="tool_call",
    tool_name="search_web",
    tool_call_id="call_abc",
    tool_arguments={"query": "FastAPI latest release"},
    content="Searching for the latest FastAPI release",
)

# Persist the matching tool result (redact secrets before storing).
record_agent_event(
    user_id="user_123",
    session_id="session_456",
    turn_id="turn_001",
    role="tool",
    event_type="tool_result",
    tool_name="search_web",
    tool_call_id="call_abc",
    tool_result={"summary": "Search result summary"},
)

# Read the recent interaction in chronological order.
events = retrieve_agent_history(
    user_id="user_123",
    session_id="session_456",
    limit=50,
)

# Search prior events across this user's sessions by a keyword or phrase.
past_attempts = search_agent_events(
    user_id="user_123",
    query="deployment failed",
    limit=20,
)

# Persist task progress so the host agent can resume after interruption.
save_agent_task(
    user_id="user_123",
    task_id="task_789",
    session_id="session_456",
    goal="Compare the latest FastAPI release with the project version",
    status="paused",
    current_step="Check the project's pinned dependency",
    completed_steps=["Search release notes"],
    pending_steps=["Inspect project dependency file", "Compare versions"],
    tool_history=[{"tool": "search_web", "status": "succeeded"}],
)
task = get_agent_task("user_123", "task_789")
resumable = list_resumable_agent_tasks("user_123")
```

The APIs use two MongoDB collections: `agent_events` (append-only structured
conversation/tool events) and `agent_tasks` (latest task state). Events are
scoped by `user_id`, `session_id`, `turn_id`, and optional `task_id`. Pass a
stable `event_id` when retrying a write to make that write idempotent. These
collections are separate from the `memories` and `outbox` collections.

Important integration notes:
- The host agent must call these APIs around its own model/tool loop. Instant-Memory
  does not itself execute tools or automatically intercept LangChain/LangGraph events.
- Store the provider's structured tool-call fields and preserve `tool_call_id` so
  tool results can be matched to their request.
- Redact API keys, credentials, personal secrets, and oversized tool output before
  recording events.
- Event history retrieval is scoped to a session or task and returns the latest
  events in chronological order. It is not a semantic search endpoint.
- Task state records progress; the host application remains responsible for safe
  resumption, authorization, idempotent tool execution, and updating task status.
- These new collections and APIs require integration/failure-recovery tests against
  your deployment before treating them as production-certified.

## Setup

1. Install Python 3.11 or newer (below 3.15).
2. Install dependencies from the repository:

       uv sync

3. Configure the existing provider and database environment variables in a local .env file. Do not commit secrets. The implementation uses MongoDB as the source of truth, Pinecone for semantic retrieval, and Neo4j for graph retrieval.
4. From the repository root, import the package as shown above.

## Development and quality checks

Tests and CI configuration are maintained in the GitHub repository to validate changes, but they are not included in the built PyPI distribution. Run tests from a development checkout with:

    uv run pytest

MongoDB is the source of truth. All memory types (profile, fact, preference, semantic, episodic, procedural, goal, event, task, project, and temporal) are stored as documents in one **memories collection**. The document's `type` field identifies its category. The separate `outbox` collection tracks Pinecone and Neo4j synchronization state and retry information. Completed outbox records are retained.

### Migrate an existing database

Back up MongoDB first, then run the migration from the repository root:

    uv run python scripts/migrate_memory_collections.py

The script copies the 11 legacy collections into `memories`, preserves each MongoDB `_id` so existing outbox/Pinecone/Neo4j references remain valid, and verifies the copied records. It is safe to rerun. The old collections are retained by default.

After checking the application and confirming the migration, you may optionally remove the old collections:

    uv run python scripts/migrate_memory_collections.py --drop-legacy

The script requires you to type `DROP` before it removes those collections. Do not use that option until you have a verified backup.


### Neo4j fact deduplication

The graph writer now uses the user, source node, relationship type, and target node as the logical edge identity. Supporting MongoDB memory IDs are retained in the relationship's `memory_ids` list, so repeating a fact in another memory does not create a new logical edge. Retrieval resolves the type of each supporting memory from MongoDB.

Before processing new messages with this code, back up Neo4j and consolidate duplicate edges already present in a user's graph:

    uv run python scripts/deduplicate_neo4j_relationships.py --user-id user_125

The script is scoped to the specified user. Review the resulting graph before running it for other users. It combines duplicate edges with the same endpoints and relationship type, keeps their memory IDs, and retains the fact as current if any duplicate edge was current. It does not merge distinct source/target pairs or different relationship types.


## LLM configuration

Configure `Mem_llm_provider`, `Mem_llm_Api`, `Mem_base_url`, and
`Mem_model` in your local `.env` file. You can optionally set
`Mem_router_model` to a smaller/faster model supported by the same provider.
The memory extractor continues using `Mem_model`; the decision gate, query
converter, and retrieval planner use `Mem_router_model` when it is set.

Example (replace the model name with one available to your provider):

    Mem_router_model=YOUR_SMALL_ROUTER_MODEL

The router model must support structured/function-calling output, because these
calls use Pydantic schemas.


## Production guide and hardening

Read [docs/PRODUCTION_GUIDE.md](docs/PRODUCTION_GUIDE.md) for the system architecture, retrieval and background-saving pipelines, data stores, performance trade-offs, setup, and production release checklist.

See [docs/PRODUCTION_HARDENING.md](docs/PRODUCTION_HARDENING.md) for the detailed production
release gates, failure-injection plan, tenant-isolation requirements, backup/restore
drills, and latency/monitoring targets.

### Regression tests and preflight

Run the synthetic retrieval benchmark (200 paraphrased questions; no external
services required), tenant-scoping tests, and the complete test suite:

```powershell
uv run pytest tests/test_retrieval_eval.py tests/test_tenant_scoping.py tests/test_security_adapter.py -q
uv run pytest
```

Run the deployment preflight to verify required environment variables are present
and MongoDB can be reached. It never prints secret values and does not write data:

```powershell
uv run python scripts/healthcheck.py
```

The preflight checks configuration and MongoDB only; it does not certify Pinecone,
Neo4j, or LLM provider availability.

### Integrating with an authenticated application

If your application authenticates users (for example, by validating a JWT), pass
the **verified principal** to the adapter:

```python
from memori.security import memori_for_principal

# principal must be produced by trusted auth middleware after token validation.
result = memori_for_principal(principal, user_input, source="user")
```

Do not construct `principal` from request JSON or an unverified user-supplied
`user_id`. The adapter derives the tenant from the verified `sub` claim (or
`user_id` fallback); it does not implement authentication itself.
