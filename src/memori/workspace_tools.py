"""LangChain tools for reading and maintaining a user's active project workspace."""
from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool

from .agentic_memory import (
    find_latest_active_agent_task,
    get_agent_task,
    save_agent_task,
)
from .code_workspace import (
    delete_workspace_file,
    edit_workspace_lines,
    list_workspace_files,
    read_workspace_file,
    write_workspace_file,
)
from .message_files import delete_agent_message_file, list_agent_message_files, read_agent_message_file


def create_workspace_tools(user_id: str, session_id: str) -> list:
    """Create user/session-scoped project and saved-message tools.

    Resolve the active task at tool invocation time. The runtime creates its task
    checkpoint before calling the model, so these tools operate on that task.
    Never accept user_id, session_id, or task_id from model-generated arguments.
    """
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("user_id must be a non-empty string.")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be a non-empty string.")
    scoped_user = user_id.strip()
    scoped_session = session_id.strip()

    def active_task_id() -> str:
        task = find_latest_active_agent_task(scoped_user, scoped_session)
        if not task or not task.get("task_id"):
            raise RuntimeError("No active project task exists in this conversation yet.")
        return str(task["task_id"])

    @tool
    def plan_task_steps(steps: list[str]) -> dict:
        """Create or replace the ordered implementation plan for the active task. Call before starting a multi-step build."""
        task_id = active_task_id()
        if not isinstance(steps, list) or not 2 <= len(steps) <= 12:
            raise ValueError("Provide between 2 and 12 concrete ordered steps.")
        normalized = [str(step).strip() for step in steps]
        if any(not step or len(step) > 300 for step in normalized):
            raise ValueError("Every task step must contain 1-300 characters.")
        if len({step.casefold() for step in normalized}) != len(normalized):
            raise ValueError("Task plan steps must be unique.")
        current = get_agent_task(scoped_user, task_id)
        if current is None:
            raise RuntimeError("Active task checkpoint could not be loaded.")
        return save_agent_task(
            scoped_user, task_id,
            goal=current.get("goal") or "Project task",
            status="running", session_id=scoped_session,
            current_step=normalized[0],
            completed_steps=current.get("completed_steps", []),
            pending_steps=normalized,
            tool_history=current.get("tool_history", []),
            result_summary=current.get("result_summary"),
            context={**(current.get("context") or {}), "plan_created": True},
        )

    @tool
    def update_task_step(step: str, status: str) -> dict:
        """Update one planned step. status must be in_progress or completed."""
        task_id = active_task_id()
        step = str(step).strip()
        status = str(status).strip().lower()
        if not step:
            raise ValueError("step must not be empty.")
        if status not in {"in_progress", "completed"}:
            raise ValueError("status must be in_progress or completed.")
        current = get_agent_task(scoped_user, task_id)
        if current is None:
            raise RuntimeError("Active task checkpoint could not be loaded.")
        pending = list(current.get("pending_steps", []))
        completed = list(current.get("completed_steps", []))
        match = next((item for item in pending if str(item).casefold() == step.casefold()), None)
        if match is None and not any(str(item).casefold() == step.casefold() for item in completed):
            raise ValueError("Step was not found in the current plan. Call plan_task_steps first.")
        if status == "completed" and match is not None:
            pending.remove(match)
            completed.append(match)
        next_step = step if status == "in_progress" else (pending[0] if pending else "All planned steps completed")
        next_status = "completed" if status == "completed" and not pending else "running"
        return save_agent_task(
            scoped_user, task_id,
            goal=current.get("goal") or "Project task",
            status=next_status, session_id=scoped_session,
            current_step=next_step,
            completed_steps=completed,
            pending_steps=pending,
            tool_history=current.get("tool_history", []),
            result_summary=current.get("result_summary"),
            context=current.get("context", {}),
        )

    @tool
    def list_project_files() -> list[dict]:
        """List files saved in the current project workspace."""
        return list_workspace_files(scoped_user, active_task_id())

    @tool
    def read_project_file(path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
        """Read a project file. Optional start_line/end_line are 1-based inclusive line numbers."""
        content = read_workspace_file(scoped_user, active_task_id(), path)
        lines = content.splitlines(keepends=True)
        start = 1 if start_line is None else start_line
        end = len(lines) if end_line is None else end_line
        if not isinstance(start, int) or not isinstance(end, int) or start < 1 or end < start:
            raise ValueError("line range must be valid and 1-based.")
        if end > len(lines):
            raise ValueError(f"line range exceeds file length ({len(lines)} lines).")
        return "".join(lines[start - 1:end])

    @tool
    def write_project_file(path: str, content: str) -> dict:
        """Create or replace a project file. Supply the complete file content."""
        return write_workspace_file(scoped_user, active_task_id(), path, content)

    @tool
    def edit_project_file_lines(path: str, start_line: int, end_line: int, replacement: str) -> dict:
        """Replace a 1-based inclusive line range in a project file."""
        return edit_workspace_lines(
            scoped_user, active_task_id(), path, start_line, end_line, replacement
        )

    @tool
    def delete_project_file(path: str) -> bool:
        """Delete one project file from the current task workspace."""
        return delete_workspace_file(scoped_user, active_task_id(), path)

    @tool
    def list_saved_messages() -> list[dict]:
        """List long user/assistant messages saved in this conversation, including their paths and line counts."""
        return list_agent_message_files(scoped_user, scoped_session)

    @tool
    def read_saved_message(path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
        """Read a previously saved long user/assistant message; line numbers are 1-based and inclusive."""
        result = read_agent_message_file(
            scoped_user, scoped_session, path, start_line=start_line, end_line=end_line
        )
        return result["content"]

    @tool
    def delete_saved_message(path: str) -> bool:
        """Delete a saved long message using its exact messages/... path."""
        return delete_agent_message_file(scoped_user, scoped_session, path)

    return [
        plan_task_steps,
        update_task_step,
        list_project_files,
        read_project_file,
        write_project_file,
        edit_project_file_lines,
        delete_project_file,
        list_saved_messages,
        read_saved_message,
        delete_saved_message,
    ]
