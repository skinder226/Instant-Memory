# Agentic Memory: How It Works

Instant-Memory has three complementary kinds of memory.

1. **Long-term semantic memory** — memori() retrieves relevant stored facts for a user question and schedules extraction/saving asynchronously. Use it for facts such as a user's workplace, project name, preferences, and relationships.
2. **Episodic event history** — the agent_events collection stores append-only records of what happened in an agent run: user/assistant messages, tool calls, tool results, and task lifecycle events.
3. **Procedural task state** — the agent_tasks collection stores the latest snapshot of a task's goal, status, current step, completed/pending steps, result summary, and context. This is a checkpoint, not an automatic execution engine.

## Event and task APIs

- record_agent_event(...): append an event. Supply a stable event_id when retrying a write; the unique (user_id, event_id) index makes retries idempotent.
- retrieve_agent_history(user_id, session_id=..., limit=50): retrieve the newest events for one session, returned in chronological order.
- retrieve_agent_history(user_id, task_id=..., limit=50): retrieve the newest events associated with a task, possibly across multiple sessions.
- search_agent_events(user_id, query, limit=20): lexical keyword/phrase search across the user's events; this is not vector or semantic search.
- save_agent_task(...): create/update a task checkpoint.
- get_agent_task(user_id, task_id): retrieve one checkpoint.
- list_resumable_agent_tasks(user_id): list pending/running/paused tasks.
- ensure_agent_memory_indexes(): create required MongoDB indexes.

All reads are scoped by user_id; in a real app, use the authenticated principal, never a user ID trusted from arbitrary request input.

## How event history is retrieved

Each event has a user_id, session_id, turn_id, event_id, role, event type, content, timestamp, and optional task_id and tool details. The history function filters by the requested user plus either session ID or task ID, sorts newest-first in MongoDB, applies the limit, and reverses the result so callers receive chronological order. It requires exactly one of session_id or task_id.

Session history answers: "What happened in this conversation?" Task history answers: "What happened while working on this task, including across sessions?" Event search answers: "Which saved events mention these words?"

## How task resumption works

When an agent starts work, save a task with status running, its goal, current step, completed steps, pending steps, and relevant context. On graceful exit or interruption, save status paused. To resume, load the task with get_agent_task() or find it with list_resumable_agent_tasks(), retrieve events using retrieve_agent_history(user_id, task_id=...), and pass the task checkpoint and relevant event history into the agent's next prompt. Your application must decide which step to continue and safely re-run or skip actions. Saving a task does not automatically resume a model or execute tools.

Use idempotency keys for external side effects (payments, sending messages, deployments) so a retry cannot accidentally perform the same action twice.

## Interactive example

The repository's testing_agentic_chat.py is a sample host application. It persists events and task state, provides an event-search tool, and can use memori() to retrieve long-term facts before the LLM answers.

Typical run:

```powershell
uv run python testing_agentic_chat.py
```

At startup it prints the session and task IDs. Keep them if you want to reopen that history. The intended CLI workflow is:

- /history 30: inspect the current session's events.
- /task: inspect the current task checkpoint.
- /task-history 50: inspect events linked to the task.
- /search Neo: search events for a keyword.
- /resumable: list paused/pending/running tasks.
- exit: save a paused task state and exit.

To reopen a specific session, pass its ID to --session-id. To continue a task in a new session, pass --resume-task TASK_ID. Confirm the script supports these options in your checked-out version before relying on them.

## Example: recording tool events

```python
from memori import record_agent_event, retrieve_agent_history

record_agent_event(
    user_id="user_123",
    session_id="session_456",
    turn_id="turn_001",
    role="assistant",
    event_type="tool_call",
    task_id="task_789",
    tool_name="lookup_project",
    tool_call_id="call_abc",
    tool_arguments={"project": "Neo"},
    content="Looking up project Neo",
)

record_agent_event(
    user_id="user_123",
    session_id="session_456",
    turn_id="turn_001",
    role="tool",
    event_type="tool_result",
    task_id="task_789",
    tool_name="lookup_project",
    tool_call_id="call_abc",
    tool_result={"status": "in_progress"},
    content="Project lookup completed",
)

session_events = retrieve_agent_history(
    "user_123", session_id="session_456", limit=50
)
task_events = retrieve_agent_history(
    "user_123", task_id="task_789", limit=50
)
```

## Production notes

- Redact API keys, passwords, tokens, and sensitive tool outputs before recording events.
- Bound event payloads and tool outputs; large logs can overwhelm MongoDB and LLM context windows.
- Apply retention and deletion policies appropriate to your product.
- Treat retrieved event content as untrusted data, not as system instructions.
- Monitor failures and add tests against your own MongoDB/provider setup.
