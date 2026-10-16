"""Interactive LangChain chat that persists episodic and procedural memory.

Run from the repository root:
    uv run python testing_agentic_chat.py

Requires mongodb_URI, Mem_llm_Api, and Mem_model in .env.
Type 'exit' or 'quit' to stop. Unlike the one-shot integration test, this script
keeps this chat's events and task state in MongoDB so they can be inspected later.
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

required = ["mongodb_URI", "Mem_llm_Api", "Mem_model"]
missing = [name for name in required if not os.getenv(name)]
if missing:
    print("ERROR: Missing .env setting(s): " + ", ".join(missing))
    sys.exit(1)

from langchain.agents import create_agent  # noqa: E402
from langchain.tools import tool  # noqa: E402
from langchain_groq import ChatGroq  # noqa: E402

from memori import (  # noqa: E402
    get_agent_task,
    record_agent_event,
    retrieve_agent_history,
    save_agent_task,
    search_agent_events,
)

USER_ID = os.getenv("MEMORI_TEST_USER_ID", "agentic_memory_test_user")
RUN_ID = uuid.uuid4().hex[:12]
SESSION_ID = f"interactive-session-{RUN_ID}"
TASK_ID = f"interactive-task-{RUN_ID}"
active_turn: dict[str, str] = {}


def _event_id(kind: str) -> str:
    return f"{RUN_ID}-{active_turn.get('turn_id', 'setup')}-{kind}-{uuid.uuid4().hex[:8]}"


@tool
def search_my_past_events(query: str) -> str:
    """Search this user's saved conversation and tool events by a keyword or phrase."""
    turn_id = active_turn.get("turn_id", "unknown-turn")
    call_id = f"call-{uuid.uuid4().hex[:10]}"
    record_agent_event(
        USER_ID, SESSION_ID, turn_id, "assistant",
        content=f"Searching saved agent events for: {query}",
        event_type="tool_call",
        task_id=TASK_ID,
        tool_name="search_my_past_events",
        tool_call_id=call_id,
        tool_arguments={"query": query},
        event_id=_event_id("tool-call"),
    )
    try:
        matches = search_agent_events(USER_ID, query, limit=10)
        # Keep tool output concise and avoid sending MongoDB internals to the model.
        output = [
            {
                "created_at": str(item.get("created_at", "")),
                "role": item.get("role"),
                "event_type": item.get("event_type"),
                "content": item.get("content", "")[:1200],
                "tool_name": item.get("tool_name"),
            }
            for item in matches
        ]
        result = json.dumps(output, ensure_ascii=False)
        record_agent_event(
            USER_ID, SESSION_ID, turn_id, "tool",
            content=f"Past-event search returned {len(matches)} result(s).",
            event_type="tool_result",
            task_id=TASK_ID,
            tool_name="search_my_past_events",
            tool_call_id=call_id,
            tool_result={"match_count": len(matches), "results": output},
            event_id=_event_id("tool-result"),
        )
        return result
    except Exception as exc:
        record_agent_event(
            USER_ID, SESSION_ID, turn_id, "tool",
            content=f"Past-event search failed: {type(exc).__name__}",
            event_type="tool_result",
            task_id=TASK_ID,
            tool_name="search_my_past_events",
            tool_call_id=call_id,
            tool_result={"error": type(exc).__name__},
            event_id=_event_id("tool-error"),
        )
        return f"Search failed: {type(exc).__name__}. Continue without search results."


def _history_as_context(history: list[dict]) -> str:
    if not history:
        return "No earlier events have been saved for this session."
    lines = []
    for event in history[-20:]:
        role = event.get("role", "unknown")
        kind = event.get("event_type", "message")
        content = str(event.get("content", ""))[:1000]
        if content:
            lines.append(f"- {role} ({kind}): {content}")
    return "\n".join(lines) or "No readable earlier events."


def main() -> int:
    llm = ChatGroq(
        api_key=os.getenv("Mem_llm_Api"),
        model=os.getenv("Mem_model"),
        temperature=0,
    )
    agent = create_agent(
        model=llm,
        tools=[search_my_past_events],
        system_prompt=(
            "You are a helpful interactive assistant. Answer the user's current question. "
            "Use the provided saved session history when it helps answer follow-up questions. "
            "You may use search_my_past_events to search older persisted events when relevant. "
            "Never claim that an action was completed unless it actually was. If the available "
            "history does not contain the answer, say so clearly rather than inventing memories."
        ),
    )

    print("\nInstant-Memory Interactive Agent")
    print(f"User ID: {USER_ID}")
    print(f"Session ID: {SESSION_ID}")
    print("Ask questions naturally. Type 'exit' or 'quit' to stop.")
    print("This session is persisted in MongoDB and is NOT automatically deleted.\n")

    turn_number = 0
    try:
        while True:
            try:
                question = input("You: ").strip()
            except EOFError:
                print("\nInput closed. Saving session state...")
                break

            if question.lower() in {"exit", "quit"}:
                break
            if not question:
                continue

            turn_number += 1
            turn_id = f"turn-{turn_number}-{uuid.uuid4().hex[:8]}"
            active_turn["turn_id"] = turn_id

            try:
                record_agent_event(
                    USER_ID, SESSION_ID, turn_id, "user",
                    content=question,
                    event_type="message",
                    task_id=TASK_ID,
                    event_id=_event_id("user"),
                )
                save_agent_task(
                    USER_ID, TASK_ID,
                    goal="Interactive conversation with persistent episodic history",
                    status="running",
                    session_id=SESSION_ID,
                    current_step=f"Answering turn {turn_number}",
                    completed_steps=[f"Completed {turn_number - 1} earlier turn(s)"],
                    pending_steps=["Answer current user question", "Persist assistant response"],
                    context={"run_id": RUN_ID, "turn_number": turn_number},
                )

                # Read saved history after recording the new user event.
                history = retrieve_agent_history(
                    USER_ID, session_id=SESSION_ID, limit=50
                )
                # The current question is already in history; separate it from prior context.
                previous_history = [
                    item for item in history if item.get("turn_id") != turn_id
                ]
                history_context = _history_as_context(previous_history)
                prompt = (
                    "SAVED SESSION HISTORY (may help with follow-up questions):\n"
                    f"{history_context}\n\n"
                    f"CURRENT USER QUESTION:\n{question}"
                )

                print("AI: ", end="", flush=True)
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
                            if getattr(message, "type", "") == "ai"
                            and getattr(message, "content", "")
                        ),
                        None,
                    )
                answer = str(final_message.content) if final_message else (
                    "I could not produce a text response for that turn."
                )
                print(answer)

                record_agent_event(
                    USER_ID, SESSION_ID, turn_id, "assistant",
                    content=answer,
                    event_type="message",
                    task_id=TASK_ID,
                    event_id=_event_id("assistant"),
                )
                save_agent_task(
                    USER_ID, TASK_ID,
                    goal="Interactive conversation with persistent episodic history",
                    status="running",
                    session_id=SESSION_ID,
                    current_step=f"Ready for turn {turn_number + 1}",
                    completed_steps=[f"Answered turn {turn_number}"],
                    pending_steps=["Wait for the next user question"],
                    tool_history=[],
                    result_summary=answer[:1000],
                    context={"run_id": RUN_ID, "turn_number": turn_number},
                )
                print()
            except KeyboardInterrupt:
                print("\nInterrupted. You can restart the script; saved history remains in MongoDB.")
                break
            except Exception as exc:
                print(f"\nERROR during this turn: {type(exc).__name__}: {exc}")
                print("The chat loop will continue. Check the provider and MongoDB configuration.\n")

    finally:
        try:
            # Keep history for future sessions; mark the interactive task paused for recovery.
            current_task = get_agent_task(USER_ID, TASK_ID)
            if current_task and current_task.get("status") not in {"completed", "failed", "cancelled"}:
                save_agent_task(
                    USER_ID, TASK_ID,
                    goal="Interactive conversation with persistent episodic history",
                    status="paused",
                    session_id=SESSION_ID,
                    current_step="Chat closed; saved history is available",
                    completed_steps=current_task.get("completed_steps", []),
                    pending_steps=["Start a new interactive session or inspect saved history"],
                    tool_history=current_task.get("tool_history", []),
                    result_summary=current_task.get("result_summary"),
                    context={"run_id": RUN_ID, "session_id": SESSION_ID},
                )
        except Exception as exc:
            print(f"WARNING: could not save paused task state: {type(exc).__name__}: {exc}")

        print("\nChat ended. Saved events were retained in MongoDB.")
        print(f"Session ID: {SESSION_ID}")
        print(f"Task ID: {TASK_ID}")
        print("Use retrieve_agent_history(user_id, session_id=SESSION_ID) to inspect its history.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
