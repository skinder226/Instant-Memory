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

Recovery is built into `AgenticMemoryRuntime`; the host application does not need
to import or register a recovery tool. On the next turn, the runtime checks for a
failed response belonging to the same authenticated user, session, and task,
marks it as recovering, and adds its partial output and unfinished file drafts to
the agent context with instructions to continue rather than restart. A successful
subsequent turn marks that recovery as `recovered`. Keep the same `user_id`
and `session_id` for every message in one chat. When `task_id` is omitted, the
runtime now looks up and reuses the latest task in that user/session (including
completed tasks), so an app can recreate the runtime between turns without silently
starting a new task. An explicitly supplied `task_id` still takes precedence. A
new chat must receive a new session ID; never share session IDs across users or chats.

The runtime always attaches a token-capture callback. To preserve partial text,
the underlying chat model/provider must itself have streaming enabled and emit
tokens before raising (for example, on a token-limit or API error). If the provider
does not emit tokens, the failure record can still capture the error, but its
`partial_response` will be empty. MongoDB write failures are logged and do not
replace the original provider exception. Common credential patterns in exception
text are redacted before storage.

```python
from memori import AgenticMemoryRuntime

runtime = AgenticMemoryRuntime(
    user_id=authenticated_principal.id,  # verified server-side identity
    session_id="stable-session-id",
    task_id="stable-task-id",
    agent=agent,  # your existing agent and its authorized tools
)
# On a retry, reuse the same runtime or recreate it with the same IDs.
result = runtime.run("Continue the task")
```

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

The runtime instructs the agent to wrap each generated file in the exact
`## Create File (relative/path.ext)` and `## Ending file (relative/path.ext)`
markers. It parses and saves complete blocks to the `workspace_files` collection
after the model finishes; it does not write files while tokens are streaming. On
every subsequent turn it lists and reads the current task's saved workspace files
and injects their paths and contents into the agent context (bounded to 30 files and
24,000 characters). This lets follow-up requests such as “change the header” operate
on the saved project instead of asking the user to paste code.
For compatibility, a single Markdown code block is also saved when the response
explicitly says which filename to save it as (for example, "Save this as
`calculator.html`"). Unlabelled code examples are not treated as files.

If generation fails mid-block, unfinished content is preserved in
`failed_responses.partial_files` for recovery; it is not exposed as a completed
workspace file. File paths are checked to reject absolute paths and directory
traversal. Existing workspace read/edit/list operations remain available, and
`delete_workspace_file(user_id, task_id, path)` provides scoped deletion.

The `workspace_files` collection is created in the **same MongoDB database
configured for Memori**, through `mongodb_connection.db`; there is no separate
workspace database or alternate workspace MongoDB URI. The runtime result
includes `workspace_files` for successful writes and `workspace_file_errors`
when a write fails. Pass `base_dir` to workspace functions only when you
explicitly want local filesystem storage.

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


## Durable long messages and file tools

The runtime stores user messages, assistant answers, and partial assistant output in
MongoDB collection `agent_message_files` whenever the UTF-8 message is larger than
8 KB. The returned metadata includes an opaque path such as
`messages/<turn-id>-user.txt` or `messages/<turn-id>-assistant.txt`.
Messages are scoped by both `user_id` and `session_id`; paths cannot be used to
read another user's or another chat's content. Messages are limited to 1 MB.
Short messages continue to be stored in the ordinary event history.

To let the model list/read/write/edit/delete project files and read/delete saved
long messages, register the scoped tools **when creating the LangChain agent**.
The package cannot add tools to an already-compiled agent graph after it has
been constructed.

```python
from langchain.agents import create_agent
from memori import AgenticMemoryRuntime, create_workspace_tools

user_id = authenticated_principal.id  # derive on the server; do not trust client input
session_id = stable_session_id        # keep unchanged for all turns in this chat

tools = create_workspace_tools(user_id=user_id, session_id=session_id)
agent = create_agent(
    model=llm,
    tools=tools,  # add your other authorized application tools here
    system_prompt="Use project and saved-message tools when needed. Never invent file contents.",
)
runtime = AgenticMemoryRuntime(
    user_id=user_id,
    session_id=session_id,
    agent=agent,
    streaming=True,
)
```

The factory registers these tools:
- `list_project_files()`
- `read_project_file(path, start_line?, end_line?)`
- `write_project_file(path, content)`
- `edit_project_file_lines(path, start_line, end_line, replacement)`
- `delete_project_file(path)`
- `list_saved_messages()`
- `read_saved_message(path, start_line?, end_line?)`
- `delete_saved_message(path)`

Line ranges are 1-based and inclusive: `start_line=1, end_line=5` reads lines
1 through 5. The model must use the exact path returned by the runtime or a list
tool; it must not invent paths. Delete operations are scoped to the current
authenticated user/session or task. The model should only delete content when
the user explicitly requests deletion.

### Dynamic activity events

`AgenticMemoryRuntime.stream()` now yields:
- `{"type":"activity","stage":"retrieving_memory",...}` while memory/history is retrieved;
- `memory_retrieved` and `agent_working` activity updates;
- `tool_started`, `tool_completed`, and `tool_failed` events with tool names;
- `token` events for streamed answer text;
- one final `result` event.

These are task-progress and tool-activity updates, not hidden chain-of-thought.
A FastAPI SSE adapter must forward activity events to the browser, for example:

```python
elif event_type == "activity":
    yield "event: activity\ndata: " + json.dumps({
        "stage": item.get("stage"),
        "tool": item.get("tool"),
        "message": item.get("message"),
    }) + "\n\n"
```

The browser should render each `activity` event as a live status item, and
continue rendering `token` and `result` events as before. If the API adapter
drops unknown event types, the activity UI will remain empty even though the
runtime emits updates.
