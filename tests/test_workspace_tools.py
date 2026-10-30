def test_workspace_tool_factory_scopes_tools_to_user_and_session(monkeypatch, tmp_path):
    from memori import workspace_tools

    monkeypatch.setattr(
        workspace_tools,
        "find_latest_active_agent_task",
        lambda user_id, session_id: {"task_id": "task-owned"},
    )
    monkeypatch.setattr(
        workspace_tools,
        "write_workspace_file",
        lambda user_id, task_id, path, content: {
            "user_id": user_id, "task_id": task_id, "path": path, "content": content
        },
    )
    tools = {item.name: item for item in workspace_tools.create_workspace_tools("user-a", "session-a")}
    result = tools["write_project_file"].invoke({"path": "index.html", "content": "<html/>"})
    assert result == {
        "user_id": "user-a", "task_id": "task-owned",
        "path": "index.html", "content": "<html/>",
    }



def test_read_project_file_uses_inclusive_one_based_lines(monkeypatch):
    from memori import workspace_tools

    monkeypatch.setattr(
        workspace_tools,
        "find_latest_active_agent_task",
        lambda user_id, session_id: {"task_id": "task-owned"},
    )
    monkeypatch.setattr(
        workspace_tools,
        "read_workspace_file",
        lambda user_id, task_id, path: "one\\ntwo\\nthree\\n",
    )
    tools = {item.name: item for item in workspace_tools.create_workspace_tools("user-a", "session-a")}
    result = tools["read_project_file"].invoke(
        {"path": "notes.txt", "start_line": 2, "end_line": 3}
    )
    assert result == "two\\nthree\\n"


def test_workspace_tools_do_not_accept_tenant_ids_from_model_arguments(monkeypatch):
    from memori import workspace_tools

    monkeypatch.setattr(
        workspace_tools,
        "find_latest_active_agent_task",
        lambda user_id, session_id: {"task_id": "task-owned"},
    )
    seen = {}
    def fake_write(user_id, task_id, path, content):
        seen.update(user_id=user_id, task_id=task_id)
        return {"path": path}
    monkeypatch.setattr(workspace_tools, "write_workspace_file", fake_write)
    tools = {item.name: item for item in workspace_tools.create_workspace_tools("trusted-user", "trusted-session")}
    tools["write_project_file"].invoke(
        {"path": "index.html", "content": "x"}
    )
    assert seen == {"user_id": "trusted-user", "task_id": "task-owned"}
