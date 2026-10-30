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
