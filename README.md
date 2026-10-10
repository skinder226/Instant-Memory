# Instant-Memory

Instant-Memory is a MongoDB-first memory pipeline with semantic retrieval through Pinecone and relationship/multi-hop retrieval through Neo4j.

## Public API

The intended application interface is one function:

```python
from memori import memori

# Process a user message: classify, retrieve when needed, and save new memories.
result = memori(
    user_id="user_123",
    user_input="My name is Skinder and I work at Acme.",
    source="user",
)

# Record an assistant-authored message with its original source role.
result = memori(
    user_id="user_123",
    user_input="The assistant suggested using PostgreSQL for this project.",
    source="assistant",  # "ai" is also accepted as an alias
)

# Other message sources are accepted and stored as source_role="other".
result = memori(
    user_id="user_123",
    user_input="Tool returned: build completed.",
    source="tool",
)
```

`source` accepts `user`, `assistant` (or `ai`), `tool`, `system`, or another string. Unknown roles are normalized to `other`. The source role is stored on each extracted memory; it is not proof that a message's claims are true.

For user messages, Memori decides whether to retrieve stored memories and/or save new durable information. Non-user messages are not used to answer user questions, but can be processed for memory extraction with their source role preserved.

The function returns a dictionary containing the user ID, normalized source, retrieval result (if any), number of memories saved, and number of placeholder updates.

## Setup

1. Install Python 3.11 or newer (below 3.15).
2. Install dependencies from the repository:

   ```bash
   uv sync
   ```

3. Configure the existing provider and database environment variables in a local `.env` file. Do not commit secrets. The current implementation uses MongoDB as the source of truth, Pinecone for semantic retrieval, and Neo4j for graph retrieval.
4. From the repository root, import the package as shown above.

## Development

Run the tests with:

```bash
uv run pytest
```

MongoDB is the source of truth. Memory writes create outbox events for background synchronization to Pinecone and Neo4j. Every memory operation is scoped by `user_id`.
