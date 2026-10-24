"""Persistent per-user/task artifact workspace for resumable agent sessions.

Stores essays, notes, plans, reports, code and other text artifacts outside LLM context.
Agents can read, patch or edit specific lines instead of regenerating whole artifacts.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


MAX_FILE_BYTES = 1_000_000
MAX_FILES = 500
_ALLOWED_SUFFIXES = {
    ".html", ".htm", ".css", ".js", ".jsx", ".ts", ".tsx", ".json",
    ".md", ".txt", ".rst", ".csv", ".tex", ".py", ".yaml", ".yml", ".toml", ".svg", ".xml",
}
_SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|token|password|secret)\s*[:=]\s*[^\s,;]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\bgsk_[A-Za-z0-9_-]{12,}\b"),
)


def _slug(value: str) -> str:
    """Make a stable opaque path component; do not use user-provided IDs as paths."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def workspace_root(user_id: str, task_id: str, base_dir: str | Path | None = None) -> Path:
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("user_id must be a non-empty string")
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("task_id must be a non-empty string")
    base = Path(base_dir or os.getenv("MEMORI_WORKSPACE_ROOT", ".memori-workspaces")).expanduser().resolve()
    root = (base / _slug(user_id) / _slug(task_id)).resolve()
    if root != base and base not in root.parents:
        raise ValueError("workspace path escapes configured root")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_file(root: Path, relative_path: str, *, allow_new: bool = False) -> Path:
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError("path must be a non-empty relative path")
    candidate_path = Path(relative_path)
    if candidate_path.is_absolute() or any(part in {"..", ""} for part in candidate_path.parts):
        raise ValueError("path must stay inside the task workspace")
    if candidate_path.suffix.lower() not in _ALLOWED_SUFFIXES:
        raise ValueError(f"unsupported file type: {candidate_path.suffix or '(no extension)'}")
    root = root.resolve()
    candidate = (root / candidate_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("path escapes task workspace")
    if candidate.exists() and candidate.is_symlink():
        raise ValueError("symbolic links are not allowed")
    if not allow_new and not candidate.is_file():
        raise FileNotFoundError(relative_path)
    return candidate


def list_workspace_files(user_id: str, task_id: str, *, base_dir: str | Path | None = None) -> list[dict[str, Any]]:
    root = workspace_root(user_id, task_id, base_dir)
    results: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() not in _ALLOWED_SUFFIXES:
            continue
        rel = path.relative_to(root).as_posix()
        results.append({"path": rel, "bytes": path.stat().st_size})
        if len(results) >= MAX_FILES:
            break
    return results


def read_workspace_file(
    user_id: str, task_id: str, relative_path: str, *, base_dir: str | Path | None = None
) -> str:
    root = workspace_root(user_id, task_id, base_dir)
    path = _safe_file(root, relative_path)
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise ValueError(f"file exceeds {MAX_FILE_BYTES} bytes")
    return path.read_text(encoding="utf-8")


def write_workspace_file(
    user_id: str,
    task_id: str,
    relative_path: str,
    content: str,
    *,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    if not isinstance(content, str):
        raise TypeError("content must be a string")
    if len(content.encode("utf-8")) > MAX_FILE_BYTES:
        raise ValueError(f"file exceeds {MAX_FILE_BYTES} bytes")
    root = workspace_root(user_id, task_id, base_dir)
    path = _safe_file(root, relative_path, allow_new=True)
    if not path.exists() and sum(1 for p in root.rglob("*") if p.is_file()) >= MAX_FILES:
        raise ValueError(f"workspace cannot contain more than {MAX_FILES} files")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size}


def apply_workspace_patch(
    user_id: str,
    task_id: str,
    relative_path: str,
    old_text: str,
    new_text: str,
    *,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Replace one exact, unique snippet; refuse ambiguous or missing matches."""
    if not old_text:
        raise ValueError("old_text must not be empty; use write_workspace_file for new files")
    current = read_workspace_file(user_id, task_id, relative_path, base_dir=base_dir)
    matches = current.count(old_text)
    if matches != 1:
        raise ValueError(f"patch requires exactly one match; found {matches}")
    updated = current.replace(old_text, new_text, 1)
    return write_workspace_file(user_id, task_id, relative_path, updated, base_dir=base_dir)


def edit_workspace_lines(
    user_id: str,
    task_id: str,
    relative_path: str,
    start_line: int,
    end_line: int,
    replacement: str = "",
    *,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Replace a 1-based inclusive line range; empty replacement deletes those lines."""
    if not isinstance(start_line, int) or not isinstance(end_line, int):
        raise TypeError("start_line and end_line must be integers")
    if start_line < 1 or end_line < start_line:
        raise ValueError("line range must be valid and 1-based")
    current = read_workspace_file(user_id, task_id, relative_path, base_dir=base_dir)
    lines = current.splitlines(keepends=True)
    if end_line > len(lines):
        raise ValueError(f"line range exceeds artifact length ({len(lines)} lines)")
    replacement_lines = replacement.splitlines(keepends=True)
    if replacement and not replacement.endswith(("\\n", "\\r")):
        replacement_lines[-1] += "\\n" if (end_line < len(lines) or current.endswith("\\n")) else ""
    updated_lines = lines[: start_line - 1] + replacement_lines + lines[end_line:]
    result = write_workspace_file(
        user_id, task_id, relative_path, "".join(updated_lines), base_dir=base_dir
    )
    result.update({"edited_lines": [start_line, end_line], "replacement_line_count": len(replacement_lines)})
    return result


class _HTMLCheck(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.errors: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(message)


def check_workspace(user_id: str, task_id: str, *, base_dir: str | Path | None = None) -> dict[str, Any]:
    """Run bounded, non-executing syntax/structure checks and report results."""
    root = workspace_root(user_id, task_id, base_dir)
    files = list_workspace_files(user_id, task_id, base_dir=base_dir)
    results: list[dict[str, str]] = []
    for item in files:
        rel = item["path"]
        try:
            text = read_workspace_file(user_id, task_id, rel, base_dir=base_dir)
            suffix = Path(rel).suffix.lower()
            if suffix == ".json":
                json.loads(text)
            elif suffix in {".html", ".htm"}:
                parser = _HTMLCheck()
                parser.feed(text)
                parser.close()
                if not re.search(r"<(?:!doctype\s+html|html\b)", text, re.I):
                    raise ValueError("missing HTML document declaration or <html> root")
                if not re.search(r"</html\s*>", text, re.I):
                    raise ValueError("missing closing </html> tag")
            elif suffix == ".py":
                compile(text, rel, "exec")
            results.append({"path": rel, "status": "passed", "message": "Static check passed"})
        except Exception as exc:
            results.append({"path": rel, "status": "failed", "message": f"{type(exc).__name__}: {exc}"})
    failures = sum(item["status"] == "failed" for item in results)
    return {
        "workspace_files": len(files),
        "passed": len(results) - failures,
        "failed": failures,
        "results": results,
        "note": "Static checks do not execute JavaScript, install dependencies, or replace browser/build tests.",
    }


def redact_workspace_text(value: str) -> str:
    """Best-effort redaction for displaying workspace snippets in agent tool results."""
    result = value
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub("[REDACTED]", result)
    return result
