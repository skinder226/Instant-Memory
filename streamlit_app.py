"""Streamlit observability dashboard for Instant-Memory agent runs.

Run with: uv run streamlit run streamlit_app.py
Requires mongodb_URI in .env. This dashboard is read-only.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient, DESCENDING
from pymongo.errors import PyMongoError

load_dotenv()

st.set_page_config(page_title="Instant-Memory · Agent Monitor", page_icon="🧠", layout="wide")
st.title("🧠 Instant-Memory Agent Monitor")
st.caption("Live view of retrieved context, agent events, tool calls/results, and resumable task checkpoints. Read-only dashboard.")

uri = os.getenv("mongodb_URI")
if not uri:
    st.error("Missing mongodb_URI. Add it to your .env file, then restart Streamlit.")
    st.stop()

@st.cache_resource
def get_db(connection_uri: str):
    client = MongoClient(connection_uri, serverSelectionTimeoutMS=5000, connectTimeoutMS=10000)
    client.admin.command("ping")
    return client["memories"]

try:
    db = get_db(uri)
    user_ids = sorted(set(db.agent_events.distinct("user_id")) | set(db.agent_tasks.distinct("user_id")))
except Exception as exc:
    st.error(f"Could not connect to MongoDB: {type(exc).__name__}: {exc}")
    st.stop()

with st.sidebar:
    st.header("Filters")
    default_user = os.getenv("MEMORI_TEST_USER_ID", user_ids[0] if user_ids else "")
    user_id = st.selectbox("User ID", options=([""] + user_ids), index=(user_ids.index(default_user) + 1 if default_user in user_ids else 0))
    refresh = st.checkbox("Auto-refresh (5 seconds)", value=False)
    event_limit = st.slider("Events to load", min_value=20, max_value=300, value=100, step=20)
    st.caption("Only events for the selected user are shown. Avoid sharing screenshots containing private memory or tool output.")

if refresh:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=5000, key="agent_monitor_refresh")
    except ImportError:
        st.info("For automatic refresh, install streamlit-autorefresh. Manual refresh is still available.")
        if st.button("Refresh now"):
            st.rerun()

if not user_id:
    st.info("No user selected. Run the agent once to create events, then select a user.")
    st.stop()

events = list(db.agent_events.find({"user_id": user_id}, {"_id": 0}).sort("created_at", DESCENDING).limit(event_limit))
events.reverse()
tasks = list(db.agent_tasks.find({"user_id": user_id}, {"_id": 0}).sort("updated_at", DESCENDING).limit(100))
event_types = sorted({str(e.get("event_type", "unknown")) for e in events})
sessions = sorted({str(e.get("session_id", "")) for e in events if e.get("session_id")}, reverse=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Events loaded", len(events))
col2.metric("Tool calls", sum(e.get("event_type") == "tool_call" for e in events))
col3.metric("Tool results", sum(e.get("event_type") == "tool_result" for e in events))
col4.metric("Open tasks", sum(t.get("status") in {"pending", "running", "paused"} for t in tasks))

tab_overview, tab_context, tab_tools, tab_tasks, tab_events = st.tabs(
    ["Overview", "Retrieved context", "Tool calls", "Tasks & checkpoints", "Event timeline"]
)

def as_text(value, max_chars=20000):
    try:
        value = json.dumps(value, indent=2, ensure_ascii=False, default=str)
    except Exception:
        value = str(value)
    return value[:max_chars] + ("\n… (truncated)" if len(value) > max_chars else "")

def event_label(event):
    timestamp = event.get("created_at")
    if isinstance(timestamp, datetime):
        timestamp = timestamp.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return f"{timestamp or 'time unknown'} · {event.get('event_type', 'event')} · {event.get('role', '?')}"

with tab_overview:
    st.subheader("Recent activity")
    selected_session = st.selectbox("Session", ["All sessions"] + sessions, key="overview_session")
    filtered = [e for e in events if selected_session == "All sessions" or e.get("session_id") == selected_session]
    if not filtered:
        st.info("No events found for this filter.")
    else:
        for event in reversed(filtered[-30:]):
            with st.container(border=True):
                st.markdown(f"**{event_label(event)}**")
                st.write(event.get("content") or "(no text)")
                st.caption(f"session={event.get('session_id', '—')} · task={event.get('task_id', '—')} · turn={event.get('turn_id', '—')}")

with tab_context:
    st.subheader("What context was retrieved and sent to the agent?")
    context_events = [
        e for e in events
        if e.get("event_type") == "task_step"
        and (e.get("metadata") or {}).get("kind") == "context_assembly"
    ]
    if not context_events:
        st.warning("No context-assembly events are stored yet. Pull the latest runtime update and run the agent again. Older runs may not have logged the exact context.")
    else:
        choices = {event_label(e) + " · " + str(e.get("turn_id", "")): e for e in reversed(context_events)}
        choice = st.selectbox("Agent turn", list(choices), key="context_choice")
        selected = choices[choice]
        metadata = selected.get("metadata") or {}
        stats = metadata.get("counts") or {}
        a, b, c, d = st.columns(4)
        a.metric("Long-term memories", stats.get("long_term_memories", "—"))
        b.metric("Session events", stats.get("session_events", "—"))
        c.metric("Task events", stats.get("task_events", "—"))
        d.metric("Context size", f"{metadata.get('context_chars', '—')} chars")
        if metadata.get("retrieval_error"):
            st.error("Long-term memory retrieval error: " + str(metadata["retrieval_error"]))
        st.markdown("**Context payload**")
        st.json(metadata.get("context", {}), expanded=False)
        with st.expander("Exact prompt sent to the agent"):
            st.code(metadata.get("prompt", "Exact prompt was not stored for this event."), language="text")
        st.caption("This view can contain private user memories and tool history. Keep the dashboard local and restrict access.")

with tab_tools:
    st.subheader("Tool execution trace")
    tool_events = [e for e in events if e.get("event_type") in {"tool_call", "tool_result"}]
    if not tool_events:
        st.info("No tool calls have been recorded for this user yet.")
    else:
        tool_filter = st.multiselect("Tool name", sorted({e.get("tool_name") or "unknown" for e in tool_events}), default=sorted({e.get("tool_name") or "unknown" for e in tool_events}))
        for event in reversed(tool_events):
            if (event.get("tool_name") or "unknown") not in tool_filter:
                continue
            title = f"{event_label(event)} · {event.get('tool_name') or 'unknown tool'}"
            with st.expander(title, expanded=event.get("event_type") == "tool_call"):
                st.markdown("**Event content**")
                st.write(event.get("content") or "—")
                if event.get("tool_arguments") is not None:
                    st.markdown("**Arguments / input**")
                    st.code(as_text(event["tool_arguments"]), language="json")
                if event.get("tool_result") is not None:
                    st.markdown("**Result / error**")
                    st.code(as_text(event["tool_result"]), language="json")
                st.caption(f"tool_call_id={event.get('tool_call_id', '—')} · task={event.get('task_id', '—')}")

with tab_tasks:
    st.subheader("Task checkpoints")
    if not tasks:
        st.info("No agent tasks stored for this user yet.")
    else:
        status_filter = st.multiselect("Task status", sorted({str(t.get("status", "unknown")) for t in tasks}), default=sorted({str(t.get("status", "unknown")) for t in tasks}))
        for task in tasks:
            if str(task.get("status", "unknown")) not in status_filter:
                continue
            title = f"{task.get('goal', 'Untitled task')} · {task.get('status', 'unknown')} · {task.get('task_id', '')}"
            with st.expander(title, expanded=task.get("status") == "running"):
                x, y = st.columns(2)
                x.markdown("**Current step**")
                x.write(task.get("current_step") or "—")
                y.markdown("**Last result summary**")
                y.write(task.get("result_summary") or "—")
                st.markdown("**Completed steps**")
                st.json(task.get("completed_steps") or [], expanded=False)
                st.markdown("**Pending steps**")
                st.json(task.get("pending_steps") or [], expanded=False)
                st.markdown("**Saved context**")
                st.json(task.get("context") or {}, expanded=False)
                st.caption(f"task={task.get('task_id')} · session={task.get('session_id', '—')} · updated={task.get('updated_at', '—')}")

with tab_events:
    st.subheader("Event timeline")
    type_filter = st.multiselect("Event types", event_types, default=event_types)
    session_filter = st.selectbox("Session filter", ["All sessions"] + sessions, key="timeline_session")
    visible = [e for e in reversed(events) if e.get("event_type", "unknown") in type_filter and (session_filter == "All sessions" or e.get("session_id") == session_filter)]
    for event in visible:
        with st.container(border=True):
            st.markdown(f"**{event_label(event)}**")
            st.write(event.get("content") or "(no text)")
            if event.get("tool_name"):
                st.caption(f"Tool: {event.get('tool_name')} · Call ID: {event.get('tool_call_id', '—')}")
            with st.expander("Full event document"):
                st.json(event, expanded=False)

st.divider()
st.caption("Instant-Memory observability · MongoDB collections: agent_events, agent_tasks · Dashboard is read-only.")
