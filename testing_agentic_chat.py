"""Interactive agentic-memory test harness for realistic, multi-turn tasks.

Run from the repository root:
    uv run python testing_agentic_chat.py

Requires .env:
    mongodb_URI=...
    Mem_llm_Api=...
    Mem_model=...

This is an interactive integration harness, not an unrestricted computer agent.
It can use long-term Instant-Memory retrieval, persistent event history, and
MongoDB-backed task checkpoints. Its task tools modify only agent_tasks records.
Type /help for commands or exit to pause the active task.
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv

load_dotenv()

_REQUIRED = ("mongodb_URI", "Mem_llm_Api", "Mem_model")
_missing = [key for key in _REQUIRED if not os.getenv(key)]
if _missing:
    print("ERROR: Missing .env setting(s): " + ", ".join(_missing))
    raise SystemExit(1)

from langchain.agents import create_agent  # noqa: E402
from langchain.tools import tool  # noqa: E402

try:
    from langchain_groq import ChatGroq
except ImportError as exc:  # pragma: no cover
    print("ERROR: Install the configured LangChain provider package.")
    raise SystemExit(1) from exc

from memori import (  # noqa: E402
    get_agent_task,
    list_resumable_agent_tasks,
    memori,
    record_agent_event,
    retrieve_agent_history,
    save_agent_task,
    search_agent_events,
)

USER_ID = os.getenv("MEMORI_TEST_USER_ID", "agentic_memory_test_user")
RUN_ID = uuid.uuid4().hex[:10]
SESSION_ID = f"interactive-session-{RUN_ID}"
ACTIVE_TASK_ID = f"task-{uuid.uuid4().hex[:12]}"
TURN_ID = "startup"
TURN_NUMBER = 0
TASK_GOAL = "Interactive agent session"
MAX_CONTEXT_EVENTS = 18


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short(value: Any, limit: int = 1600) -> str:
    try:
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = str(value)
    return text[:limit]


def _event(kind: str, role: str, content: str = "", **kwargs: Any) -> None:
    """Best-effort event persistence; do not silently hide a failed event write."""
    record_agent_event(
        user_id=USER_ID,
        session_id=SESSION_ID,
        turn_id=TURN_ID,
        role=role,
        content=_short(content, 5000),
        task_id=ACTIVE_TASK_ID,
        event_type=kind,
        event_id=f"{RUN_ID}-{TURN_ID}-{kind}-{uuid.uuid4().hex[:8]}",
        **kwargs,
    )


def _save_checkpoint(
    *,
    goal: str | None = None,
    status: str = "running",
    current_step: str | None = None,
    completed_steps: list[str] | None = None,
    pending_steps: list[str] | None = None,
    result_summary: str | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    global TASK_GOAL
    if goal and goal.strip():
        TASK_GOAL = goal.strip()
    old = get_agent_task(USER_ID, ACTIVE_TASK_ID) or {}
    return save_agent_task(
        USER_ID,
        ACTIVE_TASK_ID,
        goal=TASK_GOAL,
        status=status,
        session_id=SESSION_ID,
        current_step=current_step if current_step is not None else old.get("current_step"),
        completed_steps=completed_steps if completed_steps is not None else old.get("completed_steps", []),
        pending_steps=pending_steps if pending_steps is not None else old.get("pending_steps", []),
        tool_history=old.get("tool_history", []),
        result_summary=result_summary if result_summary is not None else old.get("result_summary"),
        context={**(old.get("context") or {}), **(context or {}), "last_updated_by_harness": _now()},
    )


@tool
def update_task_progress(
    goal: str,
    status: str,
    current_step: str,
    completed_steps: list[str],
    pending_steps: list[str],
    result_summary: str = "",
) -> str:
    """Persist progress for the active real-world work task.

    Use this when starting a multi-step task and after meaningful progress.
    status must be one of pending, running, paused, completed, failed, cancelled.
    Include completed and pending steps so the task can be resumed later.
    """
    allowed = {"pending", "running", "paused", "completed", "failed", "cancelled"}
    status = status.strip().lower()
    if status not in allowed:
        return f"Invalid status. Choose one of: {', '.join(sorted(allowed))}."
    call_id = f"tool-{uuid.uuid4().hex[:12]}"
    _event(
        "tool_call", "assistant", f"Updating task progress: {goal}",
        tool_name="update_task_progress", tool_call_id=call_id,
        tool_arguments={
            "goal": goal, "status": status, "current_step": current_step,
            "completed_steps": completed_steps, "pending_steps": pending_steps,
        },
    )
    try:
        saved = _save_checkpoint(
            goal=goal, status=status, current_step=current_step,
            completed_steps=completed_steps, pending_steps=pending_steps,
            result_summary=result_summary or None,
        )
        output = {
            "task_id": ACTIVE_TASK_ID, "goal": saved.get("goal"),
            "status": saved.get("status"), "current_step": saved.get("current_step"),
            "completed_steps": saved.get("completed_steps", []),
            "pending_steps": saved.get("pending_steps", []),
        }
        _event(
            "tool_result", "tool", f"Task checkpoint saved with status {status}.",
            tool_name="update_task_progress", tool_call_id=call_id,
            tool_result=output,
        )
        return json.dumps(output, ensure_ascii=False)
    except Exception as exc:
        _event(
            "tool_result", "tool", f"Task checkpoint failed: {type(exc).__name__}",
            tool_name="update_task_progress", tool_call_id=call_id,
            tool_result={"error": type(exc).__name__},
        )
        return f"Could not save task checkpoint: {type(exc).__name__}: {exc}"


@tool
def list_my_resumable_tasks() -> str:
    """List this user's pending, running, and paused tasks to find work to resume."""
    call_id = f"tool-{uuid.uuid4().hex[:12]}"
    _event("tool_call", "assistant", "Listing resumable tasks",
           tool_name="list_my_resumable_tasks", tool_call_id=call_id)
    try:
        tasks = list_resumable_agent_tasks(USER_ID, limit=30)
        output = [
            {
                "task_id": task.get("task_id"),
                "goal": task.get("goal"),
                "status": task.get("status"),
                "current_step": task.get("current_step"),
                "pending_steps": task.get("pending_steps", []),
                "updated_at": str(task.get("updated_at", "")),
            }
            for task in tasks
        ]
        _event("tool_result", "tool", f"Found {len(output)} resumable task(s).",
               tool_name="list_my_resumable_tasks", tool_call_id=call_id,
               tool_result={"count": len(output), "tasks": output})
        return json.dumps(output, ensure_ascii=False)
    except Exception as exc:
        _event("tool_result", "tool", f"Task listing failed: {type(exc).__name__}",
               tool_name="list_my_resumable_tasks", tool_call_id=call_id,
               tool_result={"error": type(exc).__name__})
        return f"Could not list tasks: {type(exc).__name__}: {exc}"


@tool
def search_my_past_events(query: str) -> str:
    """Search the user's saved agent event history for earlier actions, tool results, or decisions."""
    call_id = f"tool-{uuid.uuid4().hex[:12]}"
    _event("tool_call", "assistant", f"Searching past events for: {query}",
           tool_name="search_my_past_events", tool_call_id=call_id,
           tool_arguments={"query": query})
    try:
        matches = search_agent_events(USER_ID, query, limit=10)
        output = [
            {
                "created_at": str(item.get("created_at", "")),
                "session_id": item.get("session_id"),
                "task_id": item.get("task_id"),
                "role": item.get("role"),
                "event_type": item.get("event_type"),
                "content": _short(item.get("content", ""), 900),
                "tool_name": item.get("tool_name"),
            }
            for item in matches
        ]
        _event("tool_result", "tool", f"Past-event search returned {len(output)} result(s).",
               tool_name="search_my_past_events", tool_call_id=call_id,
               tool_result={"match_count": len(output), "results": output})
        return json.dumps(output, ensure_ascii=False)
    except Exception as exc:
        _event("tool_result", "tool", f"Event search failed: {type(exc).__name__}",
               tool_name="search_my_past_events", tool_call_id=call_id,
               tool_result={"error": type(exc).__name__})
        return f"Event search failed: {type(exc).__name__}: {exc}"


def _history_context() -> str:
    events = retrieve_agent_history(USER_ID, session_id=SESSION_ID, limit=MAX_CONTEXT_EVENTS)
    if not events:
        return "No prior events in this session."
    lines = []
    for item in events:
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        lines.append(
            f"[{item.get('role', 'unknown')}/{item.get('event_type', 'message')}] "
            f"{content[:800]}"
        )
    return "\n".join(lines[-MAX_CONTEXT_EVENTS:]) or "No readable prior events."


def _task_context() -> str:
    task = get_agent_task(USER_ID, ACTIVE_TASK_ID)
    if not task:
        return "No task checkpoint exists yet."
    return json.dumps(
        {
            "task_id": ACTIVE_TASK_ID,
            "goal": task.get("goal"),
            "status": task.get("status"),
            "current_step": task.get("current_step"),
            "completed_steps": task.get("completed_steps", []),
            "pending_steps": task.get("pending_steps", []),
            "result_summary": task.get("result_summary"),
        },
        ensure_ascii=False,
        default=str,
    )


def _agent_response(agent: Any, user_input: str, long_term_memories: list[Any]) -> str:
    prompt = (
        "LONG-TERM MEMORIES RETRIEVED BY INSTANT-MEMORY (use only when relevant):\n"
        f"{_short(long_term_memories, 7000)}\n\n"
        "RECENT SESSION EVENTS (episodic history):\n"
        f"{_history_context()}\n\n"
        "ACTIVE TASK CHECKPOINT (procedural state):\n"
        f"{_task_context()}\n\n"
        "CURRENT USER REQUEST:\n"
        f"{user_input}"
    )
    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    messages = result.get("messages", [])
    final_message = next(
        (
            message for message in reversed(messages)
            if getattr(message, "type", "") == "ai"
            and getattr(message, "content", "")
            and not getattr(message, "tool_calls", None)
        ),
        None,
    )
    if final_message is None:
        final_message = next(
            (
                message for message in reversed(messages)
                if getattr(message, "type", "") == "ai" and getattr(message, "content", "")
            ),
            None,
        )
    return str(final_message.content) if final_message else "I could not produce a text response."


def _show_history(limit: int = 30, task_id: str | None = None) -> None:
    events = retrieve_agent_history(
        USER_ID,
        task_id=task_id if task_id else None,
        session_id=None if task_id else SESSION_ID,
        limit=max(1, min(limit, 100)),
    )
    if not events:
        print("No saved events found.")
        return
    for item in events:
        print(
            f"{item.get('created_at')} | {item.get('role')}/{item.get('event_type')} | "
            f"task={item.get('task_id')} | {_short(item.get('content', ''), 220)}"
        )


def _help() -> None:
    print(
        """
Commands:
  /help                 Show this help
  /history [N]          Show the last N events in this session
  /task                 Show the active task checkpoint
  /task-history [N]     Show events associated with the active task
  /search WORDS         Search events across this user's sessions
  /resumable             List pending/running/paused tasks
  /resume TASK_ID       Switch to an existing task and load its checkpoint/history
  /new-task             Start a fresh task in this session
  exit                  Pause the active task and exit

Try realistic requests such as:
  "Plan a launch for my new app, break it into steps, and track progress."
  "Help me debug a failing test; save what we have tried and what remains."
  "Prepare a project handoff checklist and mark completed steps."
  "What did we try last time when the deployment failed?"
  "Resume the task I was working on."
"""
    )


def main() -> int:
    global ACTIVE_TASK_ID, TURN_ID, TURN_NUMBER, TASK_GOAL

    llm = ChatGroq(
        api_key=os.getenv("Mem_llm_Api"),
        model=os.getenv("Mem_model"),
        temperature=0,
    )
    agent = create_agent(
        model=llm,
        tools=[update_task_progress, list_my_resumable_tasks, search_my_past_events],
        system_prompt=(
            "You are Instant-Memory's interactive agentic-memory test agent. You help with "
            "realistic multi-step application tasks such as project planning, debugging plans, "
            "release checklists, incident follow-ups, and handoffs. You have three context "
            "sources: long-term memories, recent events, and a task checkpoint; distinguish them. "
            "Never invent a saved fact or claim a tool action happened unless its tool succeeded. "
            "For any multi-step task, call update_task_progress to save a clear goal, current step, "
            "completed steps, pending steps, and status. Update the checkpoint after meaningful "
            "progress and mark status completed only when the work is actually complete; if the "
            "user asks to pause, mark it paused. Use search_my_past_events for questions about what "
            "happened in earlier sessions. Use list_my_resumable_tasks when the user asks what work "
            "can be resumed. If a task needs external changes (email, payments, deployment, files, "
            "production systems), do not claim to perform them: this harness has no such tools. "
            "Provide a useful plan or draft and clearly state which actions are simulated versus "
            "actually persisted. Save concise checkpoints rather than giant logs."
        ),
    )

    # Ensure the task exists before the first user turn.
    _save_checkpoint(
        goal=TASK_GOAL,
        status="running",
        current_step="Waiting for the first user request",
        completed_steps=[],
        pending_steps=["Understand the user's task"],
        context={"run_id": RUN_ID, "session_id": SESSION_ID},
    )

    print("\nInstant-Memory — Interactive Agentic Memory Test")
    print(f"User ID:    {USER_ID}")
    print(f"Session ID: {SESSION_ID}")
    print(f"Task ID:    {ACTIVE_TASK_ID}")
    print("Long-term memory + event history + task checkpoints are enabled.")
    print("Type /help for commands. Type exit to pause and leave history in MongoDB.\n")

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                print("\nInput closed.")
                break
            except KeyboardInterrupt:
                print("\nInterrupted.")
                break

            if not user_input:
                continue
            command = user_input.strip()
            lower = command.lower()
            if lower in {"exit", "quit"}:
                break
            if lower == "/help":
                _help()
                continue
            if lower.startswith("/history"):
                parts = command.split(maxsplit=1)
                try:
                    limit = int(parts[1]) if len(parts) > 1 else 30
                except ValueError:
                    limit = 30
                _show_history(limit)
                continue
            if lower == "/task":
                print(_task_context())
                continue
            if lower.startswith("/task-history"):
                parts = command.split(maxsplit=1)
                try:
                    limit = int(parts[1]) if len(parts) > 1 else 30
                except ValueError:
                    limit = 30
                _show_history(limit, task_id=ACTIVE_TASK_ID)
                continue
            if lower.startswith("/search "):
                query = command[len("/search "):].strip()
                if query:
                    matches = search_agent_events(USER_ID, query, limit=10)
                    for item in matches:
                        print(f"{item.get('created_at')} | {_short(item.get('content', ''), 300)}")
                    if not matches:
                        print("No matching events.")
                continue
            if lower == "/resumable":
                tasks = list_resumable_agent_tasks(USER_ID, limit=30)
                if not tasks:
                    print("No resumable tasks found.")
                for task in tasks:
                    print(
                        f"{task.get('task_id')} | {task.get('status')} | "
                        f"{task.get('goal')} | next={task.get('current_step')}"
                    )
                continue
            if lower.startswith("/resume "):
                target_id = command[len("/resume "):].strip()
                task = get_agent_task(USER_ID, target_id)
                if not task:
                    print("Task not found for this user.")
                    continue
                ACTIVE_TASK_ID = target_id
                TASK_GOAL = task.get("goal") or "Resumed task"
                save_agent_task(
                    USER_ID, ACTIVE_TASK_ID,
                    goal=TASK_GOAL, status="running", session_id=SESSION_ID,
                    current_step=task.get("current_step"),
                    completed_steps=task.get("completed_steps", []),
                    pending_steps=task.get("pending_steps", []),
                    tool_history=task.get("tool_history", []),
                    result_summary=task.get("result_summary"),
                    context={**(task.get("context") or {}), "resumed_in_session": SESSION_ID},
                )
                _event("task_step", "system", f"Resumed task {ACTIVE_TASK_ID}: {TASK_GOAL}")
                print(f"Resumed task {ACTIVE_TASK_ID}: {TASK_GOAL}")
                print(_task_context())
                continue
            if lower == "/new-task":
                ACTIVE_TASK_ID = f"task-{uuid.uuid4().hex[:12]}"
                TASK_GOAL = "New interactive task"
                _save_checkpoint(
                    goal=TASK_GOAL, status="running",
                    current_step="Waiting for the task request",
                    completed_steps=[], pending_steps=["Understand the user's task"],
                    context={"run_id": RUN_ID, "session_id": SESSION_ID},
                )
                print(f"Started new task: {ACTIVE_TASK_ID}")
                continue

            TURN_NUMBER += 1
            TURN_ID = f"turn-{TURN_NUMBER}-{uuid.uuid4().hex[:6]}"
            _event("message", "user", user_input)
            try:
                # memori() is the existing long-term memory layer. Retrieval is
                # synchronous; its own memory extraction/saving is background work.
                memory_result = memori(USER_ID, user_input, source="user")
                long_term_memories = memory_result.get("retrieval", []) or []
                _event(
                    "task_step", "system",
                    f"Long-term memory lookup returned {len(long_term_memories)} item(s).",
                    metadata={"memory_count": len(long_term_memories)},
                )
                _save_checkpoint(
                    status="running",
                    current_step=f"Processing user turn {TURN_NUMBER}",
                    pending_steps=["Respond to the request", "Persist task progress"],
                    context={"last_user_request": user_input[:1200], "turn_number": TURN_NUMBER},
                )
                print("AI: ", end="", flush=True)
                answer = _agent_response(agent, user_input, long_term_memories)
                print(answer)
                _event("message", "assistant", answer)
                current = get_agent_task(USER_ID, ACTIVE_TASK_ID) or {}
                # Keep the model's checkpoint if it updated it; otherwise store
                # a minimal checkpoint for this turn.
                if current.get("current_step") == f"Processing user turn {TURN_NUMBER}":
                    _save_checkpoint(
                        status="running",
                        current_step=f"Ready for the next request",
                        completed_steps=[f"Answered turn {TURN_NUMBER}"],
                        pending_steps=["Wait for the next user request"],
                        result_summary=answer[:1000],
                    )
                print()
            except Exception as exc:
                print(f"\nERROR: {type(exc).__name__}: {exc}")
                print("The loop will continue; inspect the provider/database configuration.")
                try:
                    _event("task_failed", "system", f"Turn failed: {type(exc).__name__}: {exc}")
                    _save_checkpoint(
                        status="paused",
                        current_step=f"Turn {TURN_NUMBER} failed",
                        pending_steps=["Inspect the error", "Retry the task"],
                        result_summary=f"{type(exc).__name__}: {exc}"[:1000],
                    )
                except Exception:
                    pass
                print()

    finally:
        try:
            task = get_agent_task(USER_ID, ACTIVE_TASK_ID)
            if task and task.get("status") not in {"completed", "failed", "cancelled"}:
                _save_checkpoint(
                    status="paused",
                    current_step=task.get("current_step") or "Chat closed",
                    pending_steps=task.get("pending_steps") or ["Resume this task later"],
                    result_summary=task.get("result_summary"),
                )
                _event("task_paused", "system", f"Interactive session paused task {ACTIVE_TASK_ID}.")
        except Exception as exc:
            print(f"WARNING: Could not save paused task: {type(exc).__name__}: {exc}")

        print("\nSession ended. Saved events and task state remain in MongoDB.")
        print(f"Session ID: {SESSION_ID}")
        print(f"Active task ID: {ACTIVE_TASK_ID}")
        print("Use /resume TASK_ID in a later run to continue a saved task.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
