# Production latency and reliability validation

## 1. Run deterministic regression tests

From the repository root on Windows PowerShell:

```powershell
uv run pytest tests/test_production.py tests/test_production_reliability.py tests/test_retrieval_fast_paths.py -v
uv run pytest -v
```

These unit tests do not prove live MongoDB, Pinecone, Neo4j, or provider availability. Keep this distinction in release notes.

## 2. Measure the real public API

Use a dedicated test tenant with representative current memories, historical project records, friend/relationship facts, and completed task history. Do not benchmark against a real user's account unless authorized.

```powershell
uv run python scripts/benchmark_production.py --user-id user_125 --runs 50 --no-background-saves --quiet-logs
```

Measure the cold snapshot separately:

```powershell
uv run python scripts/benchmark_production.py --user-id user_125 --runs 50 --cold-cache --no-background-saves --quiet-logs
```

Choose one route at a time to diagnose regressions:

```powershell
uv run python scripts/benchmark_production.py --user-id user_125 --scenario workplace --runs 30 --no-background-saves
uv run python scripts/benchmark_production.py --user-id user_125 --scenario project --runs 30 --no-background-saves
uv run python scripts/benchmark_production.py --user-id user_125 --scenario history --runs 30 --no-background-saves
uv run python scripts/benchmark_production.py --user-id user_125 --scenario past_task --runs 30 --no-background-saves
```

The script calls the real `memori()` API and reports min, median, p95, p99, max, retrieval count, and whether background saving was scheduled. Use `--no-background-saves` for repeated route-latency measurements: it intercepts new submissions to the API background executor so the benchmark does not create a burst of provider calls. It does not cancel jobs already queued/running before the benchmark or suppress work submitted to other executors. It never prints retrieved memory contents. Timings depend on the tenant's data, service health, network, and whether each route uses a cache. Use the same tenant and environment when comparing commits. The greeting and generic-question routes are expected to bypass retrieval; their result count should be zero by design.

## 3. Required release checks

- [ ] All unit tests pass on the supported Python version.
- [ ] Cold-cache and warm-cache latency measured and recorded for each important route.
- [ ] Retrieval answers checked against a hand-labeled test set; track precision@k / recall@k, not just response time.
- [ ] Past-task questions return the original matching task, not the current retrospective chat.
- [ ] Tenant isolation tested with at least two users whose memory contents intentionally differ.
- [ ] Provider outage / rate limit tested; persisted Mongo memory and pending outbox work remain recoverable.
- [ ] Outbox worker restarted after partial Pinecone/Neo4j failure; retries converge without duplicate or cross-tenant records.
- [ ] Concurrent requests tested at expected production load; record throughput, p95/p99 latency, error rate, queue depth, and provider cost.
- [ ] Logs checked for secrets or unnecessary personal-memory contents.

Do not mark the system production-ready from unit tests alone. Live dependency failure/recovery, load testing, and retrieval-quality evaluation must pass in the deployment environment.
