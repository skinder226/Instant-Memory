# Instant-Memory Production Guide

**Status: production-oriented implementation; not production-certified.** This guide describes the repository as implemented and lists the gates that must pass before serving real users. Passing unit tests or receiving a successful HTTP response does not prove durable memory processing, correct retrieval, tenant isolation, or recovery from provider outages.

## 1. What Instant-Memory is

Instant-Memory is a Python memory subsystem for an application/agent. It is not itself a complete hosted SaaS product or a public authenticated HTTP service.

Its main components are:

- **Python API** — `memori(user_id, user_input, source)`, exported by the `memori` package.
- **MongoDB** — authoritative store for memory documents, task/event records, and outbox state.
- **Pinecone** — vector index for semantic similarity retrieval.
- **Neo4j** — graph projection for entity relationships and multi-hop queries.
- **LLM provider** — decides whether a message needs retrieval, converts queries when needed, plans retrieval, and extracts/updates memory facts. The configured models/provider depend on environment variables.
- **Gemini embeddings** — current embedding implementation uses `gemini-embedding-001` by default.
- **Outbox synchronization** — records work to project MongoDB memories into Pinecone and Neo4j.
- **Agentic APIs** — optional APIs for episodic conversation/tool events and resumable task state.

The package can be integrated into an existing application. The host application remains responsible for user authentication, request authorization, its answer-generating LLM, deployment, quotas, and operational alerting.

## 2. Main data stores

| Store | Role | Source of truth? |
|---|---|---|
| MongoDB `memories` | Long-term memory documents, including facts, preferences, projects, temporal/current-state data and other memory types | **Yes** |
| MongoDB `outbox` | Synchronization status/retries for Pinecone and Neo4j projections, plus related processing metadata | Durable work tracking; not the canonical fact store |
| MongoDB `agent_events` | Structured episodic events, assistant/tool calls and results | Source for recorded agent history |
| MongoDB `agent_tasks` | Latest resumable task checkpoint/state | Source for recorded task state |
| Pinecone | Vector embeddings and metadata for semantic search | No; rebuildable projection |
| Neo4j | Entity/relationship graph for graph and multi-hop retrieval | No; rebuildable projection |

The memory types share the `memories` collection and are distinguished by fields such as `type`. The outbox is intentionally separate, and completed outbox records are retained unless an explicit retention policy is implemented. The agent-event/task collections are also separate.

## 3. Public API and message-source behavior

```python
from memori import memori

result = memori(
    user_id="stable-user-key-from-your-authenticated-session",
    user_input="Where do I work?",
    source="user",
)

memories = result["retrieval"]
# Pass memories to your own application LLM to generate its answer.
```

The exact returned result also includes status/metadata such as whether background saving was scheduled. Check the implementation and your installed version before depending on optional fields.

- `source="user"`: eligible for question retrieval and background memory processing.
- `source="assistant"` or `"ai"`, `"tool"`, `"system"`, or another role: saved/processed as supported, but does not trigger user-question retrieval.
- An empty retrieval list means no memories were returned; it does **not** by itself prove that the message was saved successfully.
- `memory_save_scheduled` means work was queued, not that extraction, MongoDB persistence, embedding, or graph synchronization has finished successfully.

Use a stable tenant key derived by your authenticated host application. Do not pass a client-supplied `user_id` directly from an untrusted request and treat it as authorization.

## 4. Retrieval pipeline (synchronous path)

For a user message, the public API determines whether it is a message that should trigger retrieval. When retrieval is needed, the call waits for retrieval and returns memory documents to the host application. The host application's LLM—not `memori()` itself—uses those memories to compose the user-facing answer.

Conceptual path:

1. **Input and role gate** — receive the user ID, message, and source role. Non-user messages skip question retrieval.
2. **Fast-path checks** — recognized personal/current-project questions may use optimized MongoDB lookups against current memories rather than running every LLM routing stage.
3. **Decision gate / query conversion / retrieval plan** — where needed, configured router models classify the message, normalize or convert the query, and plan the retrieval. `Mem_router_model` can select a smaller/faster model for routing stages; `Mem_model` is used for memory extraction.
4. **Candidate retrieval** — search relevant user-scoped memories through semantic/vector and/or graph retrieval paths, according to the retrieval plan and available projections.
5. **Filtering and ranking** — apply tenant scoping, temporal/current-state constraints, relevance and deduplication rules as implemented by the selected path.
6. **Return context** — return the retrieved memory documents synchronously. The calling application includes them in its own answer-generation prompt.

Important consistency detail: MongoDB is the source of truth, but Pinecone and Neo4j projections are updated asynchronously. A newly saved fact may not be available through vector/graph search until its outbox work completes. Some optimized direct-Mongo paths can see MongoDB facts sooner.

The repository contains specialized fast retrieval helpers, including current-project lookups and current-memory context. These are bounded/heuristic paths; they are not proof that every possible natural-language query will be recognized correctly. Retrieval quality must be measured with representative, labeled evaluation data.

## 5. Memory saving pipeline (background path)

Saving is designed not to hold up the user-facing retrieval/answer path. In the current implementation, a background executor runs ingestion after `memori()` schedules it.

Conceptual path:

1. **Schedule ingestion** — the API submits background processing and returns without waiting for all extraction and indexing work.
2. **Load bounded context** — the saver loads a small set of current MongoDB memories (currently up to 12 context documents, with compact content/entity fields) and prioritizes unresolved placeholders.
3. **Extract/interpret** — the configured memory LLM decides whether the message creates a fact, duplicates an existing fact, updates/supersedes a fact, or resolves a placeholder, using structured output/schema validation.
4. **Persist canonical memory** — store/update memory documents in MongoDB's unified `memories` collection, scoped to the tenant. Temporal/current-state metadata and entity/relationship fields support future retrieval.
5. **Record projection work** — create/update outbox records for downstream synchronization. The outbox tracks Pinecone/Neo4j statuses and attempts.
6. **Project to Pinecone** — embed the memory using the configured Gemini embedding model and upsert vector metadata.
7. **Project to Neo4j** — create/update user-scoped graph entities/relationships and supporting memory references.
8. **Retry and inspect failures** — failed projection work must remain visible and retryable; inspect worker logs/outbox status. MongoDB persistence does not imply that both projections are already current.

**Durability warning:** asynchronous work running in a process-local thread executor is not, by itself, a durable queue. A process crash can interrupt work that has been scheduled but not yet persisted to MongoDB/outbox. Before claiming crash-proof ingestion, verify that the raw inbound message or an ingestion job is durably written before acknowledging it, and that a separately managed worker can recover pending jobs after restart. This is a release gate, not something to assume from the phrase “background saving.”

The outbox should be idempotent, have bounded retries/backoff, dead-letter/failed status, and reconciliation. Do not delete completed outbox events without a separately tested retention policy.

## 6. What each supporting system does

### MongoDB

- Canonical memory content, user/tenant key, type, entities, temporal/current-state metadata.
- Outbox state and sync attempts.
- Optional episodic events and task checkpoints.
- Compound indexes support common user/current/type/relationship and outbox-status queries.
- Connection and index creation are lazy to reduce import-time setup and unnecessary index creation.

### Pinecone and embeddings

- Embedding provider currently defaults to Gemini `gemini-embedding-001`.
- Vector search finds semantically similar memories even when the query does not share exact words.
- Embedding model, vector dimension, index configuration and metadata schema must match. Do not change embedding dimensions or models without a migration/re-embedding plan.

### Neo4j

- Stores graph projections of entity relationships, enabling relationship traversal and multi-hop questions.
- Queries must always be tenant-scoped.
- MongoDB memory IDs/supporting IDs allow graph edges to be traced to source memories.
- Graph projection is derived data and should be repairable/rebuildable from canonical MongoDB records.

### LLMs

- Router/planner stages are for deciding whether and how to retrieve; extractor is for memory mutation/extraction.
- Configure `Mem_llm_provider`, `Mem_llm_Api`, `Mem_base_url`, `Mem_model`; optionally configure `Mem_router_model` for smaller/faster routing.
- The configured router model must support the structured/function-call output required by the schemas.
- Provider context/TPM limits, JSON/schema compliance, retries and request timeouts affect both latency and reliability. Use bounded prompts and monitor provider errors/cost.

### Episodic and procedural/agent state

The separate `agentic_memory.py` APIs record agent events (messages, tool-call requests/results) in `agent_events` and task checkpoints in `agent_tasks`. These APIs do not automatically intercept arbitrary LangChain/LangGraph agent loops; the host agent must call them explicitly and safely resume work. Event history is not the same thing as semantic vector retrieval.

### Current latency design

- Retrieval is synchronous; saving/index projection is scheduled in the background.
- Fast-path rules can skip the full router/query-conversion/planning chain for recognized question types.
- Router stages can use `Mem_router_model`, separate from the larger extractor model.
- MongoDB connection/client and embeddings clients are lazy and reused.
- MongoDB indexes support current-memory, type, relationship and outbox status lookups.
- Saving context is bounded (up to 12 current memory documents, up to 800 content characters and 6 entities per memory in compact context).
- Candidate/context sizes should remain bounded; do not send an entire user's memory store to the LLM.

### Known latency trade-offs

- A non-fast-path retrieval can involve multiple sequential LLM calls; model/network latency often dominates database query time.
- Pinecone/Neo4j consistency is eventually consistent with the outbox, so immediate semantic/graph retrieval of a new fact depends on projection completion.
- A fast return from `memori()` says nothing about background worker completion.
- Process-local caches and thread pools are not shared/durable across multiple application workers.
- Provider rate limits, large prompts, schema retries, embedding calls, graph writes, network/DNS and cold starts can dominate latency.
- The UI's token estimates use approximately one token per four characters unless exact provider usage is returned. They are not billing-grade usage data.

### How to measure performance

Measure each path separately: request-to-retrieval result, each router LLM call, MongoDB queries, vector search, graph traversal, extraction, MongoDB persistence, embedding, Pinecone upsert, Neo4j write, and outbox queue age/retry count. Report p50/p95/p99, error rate, queue depth, projection lag, provider rate-limit errors, tokens/cost, and cold-vs-warm behavior.

The repository previously had a local HTTP adapter smoke result of 181 requests with 0 HTTP failures, 109 ms average, 82 ms median, p95 430 ms, p99 760 ms. This is historical local smoke-test evidence only: it is not a benchmark of live end-to-end memory extraction/retrieval and does not establish production capacity. Re-run a representative end-to-end workload in staging before using any performance claim.

## 8. Install, configure, and run

Use Python 3.11–3.14 as declared in `pyproject.toml`.

```powershell
uv sync --all-groups
uv run python scripts/healthcheck.py
uv run pytest -q
```

Configure credentials using a local `.env` for development or a deployment secret manager in production. Never commit secrets.

Common variables include:

- `mongodb_URI` (or `MONGODB_URI`)
- `Mem_llm_provider`, `Mem_llm_Api`, `Mem_base_url`, `Mem_model`
- Optional `Mem_router_model`
- `GEMINI_API_KEY` or `GOOGLE_API_KEY`
- `PINECONE_API_KEY` and the Pinecone index/configuration used by the implementation
- `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`

Use the actual names and requirements in the current configuration code and healthcheck; never copy placeholder values into production. `uv run python scripts/healthcheck.py` checks preflight configuration/connectivity supported by that script; it does not certify the full end-to-end pipeline.

## 9. Test and release gates

CI in `.github/workflows/tests.yml` runs on pushes and pull requests, tests Python 3.11–3.14, compiles source, runs pytest, and builds the distributable package. CI passing is necessary, not sufficient.

Run:

```powershell
uv run pytest tests/test_retrieval_eval.py tests/test_tenant_scoping.py tests/test_security_adapter.py -q
uv run pytest -q
uv run python scripts/healthcheck.py
```

For live integration acceptance, use disposable staging services only:

```powershell
$env:MEMORI_RUN_LIVE_TESTS = "1"
$env:MEMORI_LIVE_TEST_USER_PREFIX = "staging-acceptance"
$env:MEMORI_LIVE_LOAD_ITEMS = "8"
$env:MEMORI_LIVE_LOAD_MAX_SECONDS = "180"
uv run pytest tests/test_production_acceptance.py -m live -v
Remove-Item Env:MEMORI_RUN_LIVE_TESTS
```

Before serving production users, verify all of the following:

- [ ] The full test suite and package build pass on the exact release commit.
- [ ] Live acceptance tests pass against disposable MongoDB, Pinecone, Neo4j and configured LLM/embedding services.
- [ ] A raw inbound message/job is durable before acknowledgement; restart/crash recovery has been tested.
- [ ] Outbox retries, idempotency, dead-letter visibility and reconciliation are tested by failure injection.
- [ ] Authentication is handled by the host; tenant ID comes from a verified principal.
- [ ] Cross-tenant retrieval, mutation, graph/vector queries, outbox retries, events and task reads/writes are tested.
- [ ] Deletion/export/retention policies cover MongoDB, Pinecone, Neo4j, events, tasks and outbox as required.
- [ ] Backups and restore drills have been completed.
- [ ] Rate limits, per-tenant token/cost budgets, timeouts, concurrency caps and queue bounds are configured.
- [ ] Monitoring/alerts cover error rates, background failures, stuck/dead-letter outbox items, queue lag, provider limits and database health.
- [ ] A realistic workload establishes agreed p95/p99 SLOs and costs.
- [ ] Staging pilot users validate retrieval quality and edit/update/supersession behavior before broad release.

## 10. Production verdict

The repository has production-oriented foundations: a package API, MongoDB as canonical storage, an outbox design, semantic and graph projections, automated CI, tests, a preflight healthcheck, and written hardening gates.

It should **not be described as production-ready solely because these components exist**. The current process-local background executor is not a durable queue, live multi-service recovery and tenant-isolation must be demonstrated in the deployment, and no fresh full-suite/live acceptance run was performed while this guide was written. Treat production readiness as conditional on the release gates above.

## Further documentation

- [README](../README.md)
- [Production hardening checklist](PRODUCTION_HARDENING.md)
