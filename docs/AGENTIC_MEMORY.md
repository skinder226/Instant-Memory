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

## Reusable runtime API

For application code, `AgenticMemoryRuntime` wraps a configured LangChain agent with the same memory layers used by the interactive test harness:

- long-term memory retrieval via `memori(user_id, input, source="user")`;
- recent session events and cross-session task history;
- automatic user/assistant event persistence and LangChain tool start/result/error events;
- task checkpoint creation, pause, and resume;
- memory retrieval failures are surfaced in returned metadata and logged instead of silently claiming retrieval succeeded.

The application still supplies the agent and its authorized tools. This runtime does not provide arbitrary browser, shell, email, payment, deployment, or file-system access by itself. Tool outputs and retrieved memory are untrusted data; redact secrets before persistence and authorize every side effect in the tool implementation.

```python
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain.tools import tool
from memori import AgenticMemoryRuntime

@tool
def lookup_order(order_id: str) -> str:
    """Look up an order in an application-owned, authorized system."""
    # Replace this body with your real, access-controlled integration.
    return f"Order {order_id}: lookup tool is connected."

agent = create_agent(
    model=ChatGroq(
        api_key="YOUR_GROQ_API_KEY",
        model="YOUR_TOOL-CALLING_MODEL",
        temperature=0,
    ),
    tools=[lookup_order],
    system_prompt="Help the user and use tools when needed. Never invent tool results.",
)

runtime = AgenticMemoryRuntime(user_id="authenticated_user_123", agent=agent)
runtime.start_task(
    "Investigate order status",
    pending_steps=["Find the order", "Summarize the result"],
)
result = runtime.run("Check order ORD-1042 and summarize its status.")
print(result["answer"])
print("Retrieved memories:", result["retrieved_memory_count"])

# At a later point / process, resume a previously saved task:
# runtime.resume_task("task-id-from-your-task-list")
# runtime.run("Continue from the saved checkpoint.")
# runtime.pause()
```

`AgenticMemoryRuntime.run()` returns a dictionary containing `answer`, session/task/turn IDs, retrieved-memory count, background-save scheduling status, and any semantic retrieval error. Use a verified authenticated user ID; never trust a tenant ID supplied by an unauthenticated client.

## Failed response recovery and generated files

Successful assistant answers continue to be stored in `agent_events`. An interrupted
turn is additionally stored in the separate MongoDB `failed_responses` collection.
The record contains the authenticated `user_id`, session/task/turn IDs, partial
response text, error details, status, timestamps, unfinished file drafts, and
recovery-attempt history. The unique `(user_id, turn_id)` index prevents duplicate
records for the same failed turn.

When streaming is enabled, the runtime collects tokens even if no UI token callback
is supplied. If the model/provider raises (including token-limit and API errors),
the collected partial text and any unfinished marked file blocks are saved. This
write is best-effort: a MongoDB failure is logged and does not replace the original
model/provider exception.

To let the agent recover its own failed output, add the scoped recovery tool when
constructing the agent. The factory closes over the authenticated user, task, and
session IDs; the model cannot choose another user's scope:

```python
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from memori import AgenticMemoryRuntime, create_failed_response_recovery_tool

user_id = authenticated_principal.id  # obtain from your verified auth layer
session_id = "your-session-id"
task_id = "your-task-id"

recovery_tool = create_failed_response_recovery_tool(
    user_id=user_id, session_id=session_id, task_id=task_id
)
agent = create_agent(
    model=ChatGroq(api_key="...", model="YOUR_TOOL_CALLING_MODEL", temperature=0),
    tools=[recovery_tool],  # include your other explicitly authorized tools too
    system_prompt="When continuing interrupted work, call recover_failed_response first.",
)
runtime = AgenticMemoryRuntime(
    user_id=user_id, session_id=session_id, task_id=task_id, agent=agent,
    streaming=True,
)
```

The recovery tool returns the prior partial output and unfinished file drafts. The
agent must continue from that point, verify its work, and only then complete the
task. A successful subsequent runtime turn changes the recovery record to
`recovered`; until then it remains `recovering`. The record is retained for
audit/recovery rather than deleted.

### Marker-based file output

For file generation, the agent can include explicit blocks in its final response:

```text
## Create File (index.html)
<!doctype html>
<html>
  <body>Hello</body>
</html>
## Ending file (index.html)
```

The runtime parses and saves complete blocks to the task's existing
`workspace_files` storage after the model finishes. It does not write files while
tokens are streaming. If generation fails mid-block, the unfinished content is
preserved in `failed_responses.partial_files` for recovery; it is not exposed as
a completed workspace file. File paths are checked to reject absolute paths and
directory traversal. Existing workspace read/edit/list operations remain separate.

## Interactive example

Run the interactive integration harness from the repository root:

```powershell
uv run python testing_agentic_chat.py
```

It requires `mongodb_URI`, `Mem_llm_Api`, and `Mem_model` in `.env`.
It uses `memori(user_id, user_input, source="user")` for long-term retrieval,
loads recent event history and the active task checkpoint, and gives those
three sources to the LangChain agent. Tool calls for task progress, resumable
task listing, and past-event search are recorded in `agent_events`.

Supported commands in the interactive loop:

- `/help`: show commands and example tasks.
- `/history [N]`: inspect recent events in this session.
- `/task`: inspect the active task checkpoint.
- `/task-history [N]`: inspect events linked to the active task.
- `/search WORDS`: search saved events across sessions.
- `/resumable`: list pending, running, and paused tasks.
- `/resume TASK_ID`: load a saved task checkpoint and its history on subsequent turns.
- `/new-task`: start a new task.
- `exit`: mark the active task paused and exit.

Try realistic requests such as planning an app launch, creating a debugging
checklist, preparing a handoff, or asking what happened during an earlier
attempt. The model can persist task plans and checkpoints, but this test harness
does not itself deploy applications, send emails, edit project files, or call
arbitrary external business systems. Those actions require explicitly
implemented and authorized tools. A saved checkpoint helps the agent resume;
it does not guarantee that external side effects completed before a crash.

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
