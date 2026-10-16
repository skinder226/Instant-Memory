"""Real LangChain-agent integration test for Instant-Memory.

Run from the repository root:
    uv run python testing_agentic_integration.py

Requires mongodb_URI, Mem_llm_Api, and Mem_model in .env. This invokes the
configured LLM and writes temporary agent_events/agent_tasks records to MongoDB.
Only records tagged with this run's unique IDs are removed afterwards.
"""
from __future__ import annotations

import os
import sys
import uuid
from dotenv import load_dotenv

load_dotenv()

required = ["mongodb_URI", "Mem_llm_Api", "Mem_model"]
missing = [name for name in required if not os.getenv(name)]
if missing:
    print("ERROR: Missing .env setting(s): " + ", ".join(missing))
    print("This integration test needs MongoDB and your configured Groq model.")
    sys.exit(1)

from langchain.agents import create_agent  # noqa: E402
from langchain.tools import tool  # noqa: E402
from langchain_groq import ChatGroq  # noqa: E402

from memori import (  # noqa: E402
    get_agent_task,
    record_agent_event,
    retrieve_agent_history,
    save_agent_task,
)
from memori.mongodb_connection import db  # noqa: E402


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS  {message}")


def main() -> int:
    user_id = os.getenv("MEMORI_TEST_USER_ID", "agentic_memory_test_user")
    run_id = uuid.uuid4().hex[:12]
    session_id = f"langchain-session-{run_id}"
    turn_id = f"langchain-turn-{run_id}"
    task_id = f"langchain-task-{run_id}"
    user_event_id = f"langchain-event-user-{run_id}"
    assistant_event_id = f"langchain-event-assistant-{run_id}"
    tool_call_event_id = f"langchain-event-tool-call-{run_id}"
    tool_result_event_id = f"langchain-event-tool-result-{run_id}"
    expected_code = f"LM-{run_id[:6].upper()}"
    event_ids = [user_event_id, assistant_event_id, tool_call_event_id, tool_result_event_id]
    tool_calls_seen: list[str] = []

    print("\nInstant-Memory + LangChain agent integration test")
    print(f"Test user: {user_id}")
    print("A real LLM call will be made; temporary MongoDB records are cleaned up.\n")

    @tool
    def lookup_test_record(code: str) -> str:
        """Look up a test record by code. Call this tool to retrieve the requested record."""
        tool_call_id = f"call-{run_id}"
        tool_calls_seen.append(code)

        # Instrument the real tool execution: record invocation, run it, record result.
        record_agent_event(
            user_id, session_id, turn_id, "assistant",
            content=f"Calling lookup_test_record for code {code}",
            event_type="tool_call",
            task_id=task_id,
            tool_name="lookup_test_record",
            tool_call_id=tool_call_id,
            tool_arguments={"code": code},
            event_id=tool_call_event_id,
        )

        # Local deterministic test data; no external system is modified.
        if code == expected_code:
            result = {"code": expected_code, "status": "ready", "owner": "integration-test"}
        else:
            result = {"code": code, "status": "not_found"}

        record_agent_event(
            user_id, session_id, turn_id, "tool",
            content=f"lookup_test_record returned status={result['status']}",
            event_type="tool_result",
            task_id=task_id,
            tool_name="lookup_test_record",
            tool_call_id=tool_call_id,
            tool_result=result,
            event_id=tool_result_event_id,
        )
        return str(result)

    try:
        record_agent_event(
            user_id, session_id, turn_id, "user",
            content=f"Find the status for test code {expected_code}. You must use the lookup_test_record tool.",
            event_type="message",
            task_id=task_id,
            event_id=user_event_id,
        )
        save_agent_task(
            user_id, task_id,
            goal=f"Look up integration test record {expected_code}",
            status="running",
            session_id=session_id,
            current_step="Ask LangChain agent to use lookup_test_record",
            completed_steps=[],
            pending_steps=["Call lookup tool", "Record final response"],
            context={"test_run": run_id, "expected_code": expected_code},
        )
        print("Running LangChain agent with configured LLM...")

        llm = ChatGroq(
            api_key=os.getenv("Mem_llm_Api"),
            model=os.getenv("Mem_model"),
            temperature=0,
        )
        # Force a tool call so the integration test checks actual tool execution.
        tool_enabled_llm = llm.bind_tools([lookup_test_record], tool_choice="required")
        agent = create_agent(
            model=tool_enabled_llm,
            tools=[lookup_test_record],
            system_prompt=(
                "You are running an integration test. You must call lookup_test_record "
                "exactly once using the test code in the user request. Then report the "
                "status returned by the tool. Do not guess the result."
            ),
        )

        result = agent.invoke({
            "messages": [{
                "role": "user",
                "content": f"Find the status for test code {expected_code}. Use the lookup_test_record tool.",
            }]
        })
        messages = result.get("messages", [])
        final_message = next(
            (message for message in reversed(messages) if getattr(message, "type", "") == "ai" and getattr(message, "content", "")),
            None,
        )
        final_text = str(final_message.content) if final_message else ""
        check(bool(tool_calls_seen), "LangChain executed the lookup_test_record tool")
        check(expected_code in tool_calls_seen, "tool received the expected test code")
        check(bool(final_text), "LangChain produced a final assistant response")

        record_agent_event(
            user_id, session_id, turn_id, "assistant",
            content=final_text,
            event_type="message",
            task_id=task_id,
            metadata={"test_run": run_id},
            event_id=assistant_event_id,
        )
        save_agent_task(
            user_id, task_id,
            goal=f"Look up integration test record {expected_code}",
            status="completed",
            session_id=session_id,
            current_step="Completed lookup and recorded final response",
            completed_steps=["Call lookup tool", "Record final response"],
            pending_steps=[],
            tool_history=[{"tool": "lookup_test_record", "status": "succeeded"}],
            result_summary=final_text[:1000],
            context={"test_run": run_id, "expected_code": expected_code},
        )

        history = retrieve_agent_history(user_id, session_id=session_id, limit=20)
        saved_task = get_agent_task(user_id, task_id)
        retrieved_ids = {event.get("event_id") for event in history}
        check(
            {user_event_id, tool_call_event_id, tool_result_event_id, assistant_event_id}.issubset(retrieved_ids),
            "history contains user, tool-call, tool-result, and assistant events",
        )
        check(
            saved_task is not None and saved_task.get("status") == "completed",
            "task state persists as completed",
        )

        print("\nAgent final response:")
        print(final_text)
        print("\nALL INTEGRATION CHECKS PASSED")
        print("Verified: real LangChain model call -> tool execution -> event persistence -> history/task retrieval.")
        return 0

    except Exception as exc:
        print(f"\nTEST FAILED: {type(exc).__name__}: {exc}")
        print("Check MongoDB connectivity, provider credentials/model availability, and tool-calling support.")
        return 1

    finally:
        try:
            db["agent_events"].delete_many(
                {"user_id": user_id, "event_id": {"$in": event_ids}}
            )
            db["agent_tasks"].delete_one({"user_id": user_id, "task_id": task_id})
            print("Cleaned up temporary integration-test records.")
        except Exception as cleanup_exc:
            print(f"WARNING: automatic cleanup failed: {cleanup_exc}")
            print(f"Manually inspect user_id={user_id!r}, task_id={task_id!r}, run_id={run_id!r}.")


if __name__ == "__main__":
    raise SystemExit(main())
