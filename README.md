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

## Setup

1. Install Python 3.11 or newer (below 3.15).
2. Install dependencies from the repository:

       uv sync

3. Configure the existing provider and database environment variables in a local .env file. Do not commit secrets. The implementation uses MongoDB as the source of truth, Pinecone for semantic retrieval, and Neo4j for graph retrieval.
4. From the repository root, import the package as shown above.

## Development

Run tests with:

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

After deploying this code, back up Neo4j and consolidate duplicate edges already present in a user's graph:

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
