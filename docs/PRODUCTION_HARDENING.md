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

## Live acceptance tests

The fast test suite uses mocks and must not be treated as proof of provider connectivity.
Opt-in live acceptance tests exercise MongoDB, Pinecone, Neo4j, durable outbox replay,
tenant-scoped agent history/task reads, and concurrent persistence:

```powershell
# Use a disposable staging environment, never production user data.
$env:MEMORI_RUN_LIVE_TESTS = "1"
$env:MEMORI_LIVE_TEST_USER_PREFIX = "staging-acceptance"
$env:MEMORI_LIVE_LOAD_ITEMS = "8"
$env:MEMORI_LIVE_LOAD_MAX_SECONDS = "180"
uv run pytest tests/test_production_acceptance.py -m live -v
Remove-Item Env:MEMORI_RUN_LIVE_TESTS
```

Configure the same provider credentials used by the staging application:
`mongodb_URI`, `PINECONE_API_KEY`, `NEO4J_URI`, `NEO4J_USERNAME`,
`NEO4J_PASSWORD`, and `GEMINI_API_KEY`. Tests generate unique tenant and record IDs,
and remove only records tagged with those IDs. A failed cleanup may leave test records
in the disposable environment, so inspect the test tenant before reusing it.

The load test defaults to eight concurrent source-memory writes and checks that the
write-plus-sync run completes within 180 seconds. This is a smoke threshold, not a
capacity SLO; set an agreed p95/p99 target and run a larger, separately monitored load
test before production. The HTTPS test is a deployment gate when
`MEMORI_ENV=production`: set `MEMORI_PUBLIC_URL` to the externally reachable HTTPS URL.

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

### Production Streamlit authorization configuration

Set these values in the deployment's secret/configuration manager (not in committed files):

- `MEMORI_ENV=production`
- `MEMORI_PUBLIC_URL=https://your-real-hostname.example`
- `MEMORI_OIDC_ALLOWED_EMAILS=admin@your-company.example,operator@your-company.example`
  **or** `MEMORI_OIDC_ALLOWED_EMAIL_DOMAINS=your-company.example`
- Configure Streamlit `[auth]` OIDC client ID, client secret, redirect URI, and issuer using the host's secret store or `.streamlit/secrets.toml` managed outside version control.

Prefer exact email allowlists for small deployments. Domain allowlists grant access to every verified account in that domain; use them only for a controlled organization domain. The application fails closed if the HTTPS URL, allowlist, issuer, subject, verified email, or authorization check is missing/invalid. Keep the OIDC client secret in the host's secret store and rotate it if exposed. Validate the external TLS certificate and redirect/callback URL from a separate machine after deployment; `MEMORI_PUBLIC_URL` is a configuration gate, not proof of a live TLS handshake.


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

## Deployment sign-off runbook

These checks require access to the actual hosting provider and production-like infrastructure; a repository commit cannot perform them on your behalf.

1. **Identity and access:** configure OIDC and the explicit email/domain allowlist above. Test an allowed user, a signed-in but unlisted user, an unverified email, logout, session expiry, and attempts to alter the tenant in browser state. The unlisted user must be denied before workspace data is queried.
2. **HTTPS:** verify the real deployed URL from outside the network, certificate chain, HTTP-to-HTTPS redirect, secure cookies, and OIDC callback URL. Do not mark this complete based only on the environment variable.
3. **Secrets:** use host-managed secrets, least-privilege database credentials, and separate staging/production projects. Scan repository history and deployment logs for accidental credentials; rotate any exposed credentials.
4. **Crash/replay:** run `uv run pytest tests/test_production_acceptance.py -m live -v` against disposable staging services. The restart test checks outbox durability from a separate Python process and then replays it with a fresh worker invocation. For an operational drill, terminate the deployed worker after an event is persisted but before sync finishes, restart it, and verify both provider statuses reach `completed`.
5. **Load:** increase `MEMORI_LIVE_LOAD_ITEMS` only after provider quotas and disposable-service cleanup are confirmed. Run a separately monitored load test representative of expected peak traffic and record p50/p95/p99 latency, throughput, error rate, provider rate limits/cost, and outbox oldest-event age. Eight writes in the smoke test are not a capacity target.
6. **Backup/restore:** enable encrypted managed MongoDB backups, document retention and RPO/RTO, restore to an isolated database, and rebuild/test Pinecone and Neo4j projections from authoritative MongoDB memories/outbox before sign-off.
7. **Operations:** add hosting-level liveness/readiness, error tracking, structured redacted logs, and alerts for MongoDB connectivity, provider failures, outbox age/dead-letter growth, latency, and backup failures.

## Release gate

Do not label the system production-ready until:
- full test suite passes in CI;
- retrieval evaluation meets an agreed threshold on a held-out labeled set;
- tenant-isolation tests pass across all storage providers;
- failure-injection and restart/replay tests pass;
- load tests meet agreed p95/p99 and error-rate targets;
- backup restoration and projection rebuild are demonstrated;
- the hosting API's authentication, authorization, rate limits, and monitoring are verified.
