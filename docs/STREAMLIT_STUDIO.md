# Instant-Memory Studio (Streamlit)

## Start the interactive UI

From the repository root:

```powershell
uv sync
uv run streamlit run streamlit_app.py
```

Open the local URL Streamlit prints (normally `http://localhost:8501`).

Required `.env` values:

- `mongodb_URI`
- `Mem_llm_Api` (Groq API key in the current UI implementation)
- `Mem_model` (a model supported by the configured Groq account)
- `GEMINI_API_KEY` or `GOOGLE_API_KEY` for the current Gemini embedding implementation
- Pinecone and Neo4j settings required by the configured memory pipeline

Do not paste credentials into chat, commit them, or display them in screenshots. If a key has been exposed, revoke and rotate it.

## What the UI does

- Chat directly with a LangChain agent from Streamlit.
- On each user turn, call `AgenticMemoryRuntime.run()`, which invokes `memori(user_id, message, source="user")`, loads recent session/task events and a task checkpoint, assembles a bounded prompt, and invokes the agent.
- Record event history and tool calls/results in `agent_events`.
- Save resumable task checkpoints in `agent_tasks`.
- Inspect bounded context snapshots, tool arguments/results, task state, and the event timeline.
- Resume a selected task in a new Streamlit session.
- Run read-only diagnostics and an explicit event/task write-read check.

## Included tools and real-world action boundary

The included LLM tools are intentionally limited to:

1. `update_task_progress` — writes the user's task checkpoint.
2. `search_past_events` — lexical search over that user's stored event history.
3. `list_resumable_tasks` — lists that user's pending/running/paused tasks.

They do **not** browse the internet, send email, book services, deploy software, or modify external systems. Add such integrations as explicit, least-privilege tools with authorization, confirmation for consequential actions, timeouts, retries, and auditable results. Never describe a planned action as completed unless a tool confirmed it.

## Suggested acceptance test

1. Start the UI and send: `Remember that my preferred project codename is Orion.`
2. Wait for the background save to finish, then send: `What is my preferred project codename?`
3. Open **Retrieved context** and verify the retrieved-memory count/context payload.
4. Ask: `Plan a three-step release checklist and save the progress.` Verify a task checkpoint is updated.
5. Ask: `What unfinished tasks do I have?` and inspect the tool trace.
6. Stop and restart Streamlit, resume the task in the sidebar, and ask it to continue.
7. Run **Diagnostics**. A passing MongoDB event/task test does not prove Pinecone, Neo4j, or embedding credentials are healthy; test the complete memory pipeline separately.

The background memory saver may not finish before the next message. If a just-stated fact is not retrieved immediately, inspect the outbox/worker logs and verify that the memory was persisted and indexed before treating it as a retrieval defect.

## Production mode and remaining limitations

By default this is a local development/test console. Setting `MEMORI_ENV=production`
switches the app to Streamlit's OIDC login flow. The app derives the tenant key from
the verified identity provider issuer and subject; it does not accept a browser
selected user ID in production mode.

Before enabling production mode:

1. Configure Streamlit's `[auth]` settings and your identity provider using the
   current Streamlit OIDC deployment documentation. Set the redirect URI to the
   public HTTPS URL for this deployment.
2. Set `MEMORI_ENV=production` in the deployment environment.
3. Configure all required service credentials using the deployment secret manager.
4. Deploy behind HTTPS and test sign-in, sign-out, session expiry, and tenant
   isolation using two separate test identities.

Do not use the old `STREAMLIT_ACCESS_PASSWORD` or
`STREAMLIT_PRODUCTION_USER_ID` variables as a production authentication mechanism:
the current application code uses OIDC, not the former shared-password gate.
The OIDC subject/issuer mapping is an application tenant key; verify that your
identity provider's claims and account lifecycle meet your access-control policy.

Before deployment:
- Use TLS, secret management, network restrictions, least-privilege database credentials, and secure deployment configuration.
- Add rate limits, request timeouts, concurrency limits, and model-cost controls.
- Redact secrets before persistence, not just when displaying logs. Tool outputs and retrieved memories may contain sensitive data.
- Add retention/deletion policies and export/access controls for personal data.
- Add structured logs, metrics, alerts, retry/dead-letter handling, and health checks for MongoDB, embeddings, Pinecone, Neo4j, and outbox workers.
- Test tenant isolation, duplicate delivery, temporal updates, stale graph/vector records, concurrency, and recovery after partial failures.
- Keep prompts/context bounded and avoid logging more personal data than needed.
- Add automated unit/integration tests to CI; manually inspect every external tool permission before enabling it.

No system can guarantee “perfect” memory. The goal is measurable reliability through tests, observability, bounded context, safe failure handling, and explicit quality targets.
