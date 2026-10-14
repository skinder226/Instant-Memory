# Production hardening and release gates

This document is an implementation checklist, not a claim that the repository has
already passed production certification. The library does not itself authenticate
end users or provide a public HTTP server. The hosting application owns request
authentication, authorization, deployment, alerts, and backup policies.

## 1. Retrieval evaluation

A synthetic 200-query regression suite lives in `tests/test_retrieval_eval.py`.
It covers people, family relationships, work, projects, preferences, dates,
corrections, multihop-like facts, tools, goals, hobbies, education, pets,
languages, devices, and deadlines. It does not require external services.

Run:

```powershell
uv run pytest tests/test_retrieval_eval.py -q
uv run pytest
```

This suite is a baseline, not a substitute for a labeled dataset built from
consented, anonymized production-like examples. Track Recall@k / MRR and answer
grounding separately. Add explicit negative questions for facts that do not
exist, expired facts, and conflicting old/current facts. Never use real personal
memory dumps in CI.

## 2. Source of truth and consistency

- MongoDB is authoritative. Treat Pinecone and Neo4j as rebuildable projections.
- Keep each MongoDB memory mutation and its outbox event in one MongoDB transaction
  where deployment topology supports transactions; otherwise use an idempotent
  write protocol and reconciliation job.
- Make outbox work idempotent; use bounded exponential backoff with jitter, a
  maximum attempt count, and visible dead-letter status.
- Never mark provider sync completed until the provider confirms the write.
- Add periodic reconciliation that compares current Mongo records against
  provider metadata, and supports rebuilding projections from MongoDB.
- Test corrections, supersession, expired facts, placeholder completion, duplicate
  deliveries, process restart mid-job, and concurrent writes to the same memory.
- Do not delete completed outbox records without a separately tested retention policy.

Failure-injection tests still need live disposable MongoDB/Pinecone/Neo4j instances;
the synthetic retrieval tests do not prove distributed consistency.

## 3. Tenant isolation and authentication

- Authenticate every API request in the hosting service and derive tenant identity
  from a verified JWT/session principal. Never trust `user_id` from request JSON
  as the authorization decision.
- `memori_for_principal` in `memori.security` provides an adapter that takes an
  already-verified principal and derives the Memori user ID from its `sub` (or
  `user_id`) claim. It is not an authentication provider and must not be called
  with a client-controlled object.
- Keep a tenant filter on every MongoDB read/update/delete, every outbox worker
  lookup, every Neo4j query, and every Pinecone query. Pinecone namespace and
  metadata filter should both be scoped where supported.
- Add automated cross-tenant tests for retrieval, updates, placeholder resolution,
  graph traversal, outbox retries, and vector queries before release.
- Use separate least-privilege credentials for development, staging, and production.

## 4. Recovery, concurrency, and latency

- Run integration tests against disposable services with MongoDB, Neo4j, Pinecone,
  and LLM providers independently unavailable.
- Verify a provider outage leaves MongoDB writes intact and sync events retryable.
- Verify process termination during each outbox stage can be recovered after restart.
- Load test with representative tenant counts, memory counts, and concurrent turns.
  Report p50, p95, p99 latency, error rate, queue depth, retries, and provider cost.
- Set request/provider timeouts and bounded queues; ensure background task exceptions
  are observable and have a retry/dead-letter path.
- Test cold-cache and warm-cache retrieval separately. Current process-local caches
  are not shared across workers and must not be treated as durable state.

## 5. Operations and safeguards

### Secrets
- Store credentials in deployment secret storage; do not commit `.env`, logs, or
  real memory exports. Rotate credentials and use least-privilege scopes.
- Validate required environment variables at startup and fail closed for production
  misconfiguration. Do not print secret values.

### Backups and restore
- Enable MongoDB managed backups or scheduled encrypted `mongodump` backups.
- Document retention, encryption, access control, and recovery-point/recovery-time
  objectives. Perform a restore drill into an isolated database before launch.
- Treat Pinecone and Neo4j as projections only if they can be rebuilt from MongoDB
  and the outbox; test that rebuild procedure.

### Health, logs, and alerts
- Add liveness/readiness endpoints in the hosting application. Readiness should
  check MongoDB and report projection/provider status separately; do not make
  user requests fail solely because a rebuildable projection is down.
- Emit structured logs with request/correlation ID, tenant-safe identifiers,
  route, duration, provider status, retry count, and result counts. Do not log
  raw prompts, secrets, or sensitive memory contents by default.
- Alert on elevated p95 latency, retrieval misses in evaluation, provider errors,
  outbox age/queue growth, dead-letter events, database connection failures, and
  backup failures.

## Release gate

Do not label the system production-ready until:
- full test suite passes in CI;
- retrieval evaluation meets an agreed threshold on a held-out labeled set;
- tenant-isolation tests pass across all storage providers;
- failure-injection and restart/replay tests pass;
- load tests meet agreed p95/p99 and error-rate targets;
- backup restoration and projection rebuild are demonstrated;
- the hosting API's authentication, authorization, rate limits, and monitoring are verified.
