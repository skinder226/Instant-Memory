"""Regression tests for resumable project-file editing and static checks."""
import json
from pathlib import Path

import pytest

from memori.code_workspace import (
    apply_workspace_patch,
    check_workspace,
    list_workspace_files,
    read_workspace_file,
    workspace_root,
    write_workspace_file,
)


def test_workspaces_are_persistent_and_isolated_by_user_and_task(tmp_path):
    write_workspace_file("user-a", "task-1", "index.html", "<html></html>", base_dir=tmp_path)
    assert read_workspace_file("user-a", "task-1", "index.html", base_dir=tmp_path) == "<html></html>"
    assert list_workspace_files("user-a", "task-1", base_dir=tmp_path)[0]["path"] == "index.html"
    assert list_workspace_files("user-b", "task-1", base_dir=tmp_path) == []
    assert list_workspace_files("user-a", "task-2", base_dir=tmp_path) == []


@pytest.mark.parametrize("path", ["../escape.html", "/tmp/escape.html", "sub/../../escape.js"])
def test_workspace_rejects_path_traversal(tmp_path, path):
    with pytest.raises(ValueError):
        write_workspace_file("u", "t", path, "bad", base_dir=tmp_path)


def test_workspace_rejects_unsupported_file_types(tmp_path):
    with pytest.raises(ValueError, match="unsupported file type"):
        write_workspace_file("u", "t", ".env", "SECRET=x", base_dir=tmp_path)


def test_patch_changes_only_unique_matching_snippet(tmp_path):
    write_workspace_file("u", "t", "app.js", "const label = 'Old';\n", base_dir=tmp_path)
    result = apply_workspace_patch(
        "u", "t", "app.js", "'Old'", "'New'", base_dir=tmp_path
    )
    assert result["path"] == "app.js"
    assert read_workspace_file("u", "t", "app.js", base_dir=tmp_path) == "const label = 'New';\n"


def test_patch_refuses_missing_or_ambiguous_match(tmp_path):
    write_workspace_file("u", "t", "app.js", "x = 1\nx = 1\n", base_dir=tmp_path)
    with pytest.raises(ValueError, match="exactly one match"):
        apply_workspace_patch("u", "t", "app.js", "x = 1", "x = 2", base_dir=tmp_path)
    with pytest.raises(ValueError, match="exactly one match"):
        apply_workspace_patch("u", "t", "app.js", "not present", "x", base_dir=tmp_path)


def test_static_checks_report_valid_and_invalid_files(tmp_path):
    write_workspace_file(
        "u", "t", "index.html",
        "<!doctype html><html><head></head><body><h1>Hello</h1></body></html>",
        base_dir=tmp_path,
    )
    write_workspace_file("u", "t", "config.json", json.dumps({"ok": True}), base_dir=tmp_path)
    write_workspace_file("u", "t", "broken.json", "{", base_dir=tmp_path)
    write_workspace_file("u", "t", "broken.py", "def broken(:\n", base_dir=tmp_path)

    report = check_workspace("u", "t", base_dir=tmp_path)
    assert report["workspace_files"] == 4
    assert report["passed"] == 2
    assert report["failed"] == 2
    assert {item["path"] for item in report["results"] if item["status"] == "failed"} == {
        "broken.json", "broken.py"
    }


def test_workspace_root_does_not_expose_raw_user_or_task_ids(tmp_path):
    root = workspace_root("email@example.com", "task/../../secret", tmp_path)
    assert root.is_relative_to(tmp_path.resolve())
    assert "email@example.com" not in str(root)
    assert "secret" not in str(root)


def test_workspace_enforces_file_size_limit(tmp_path):
    with pytest.raises(ValueError, match="exceeds"):
        write_workspace_file("u", "t", "index.html", "x" * 1_000_001, base_dir=tmp_path)
