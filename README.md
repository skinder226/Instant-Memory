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

MongoDB is the source of truth. Memory writes create outbox events for background synchronization to Pinecone and Neo4j. Every memory operation is scoped by user_id.
