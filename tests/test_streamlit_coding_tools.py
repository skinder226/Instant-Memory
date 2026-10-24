"""Regression checks for the coding workflow exposed in Streamlit Studio."""
from pathlib import Path


def test_streamlit_agent_registers_persistent_code_workspace_tools():
    source = (Path(__file__).resolve().parents[1] / "streamlit_app.py").read_text(encoding="utf-8")
    for tool_name in (
        "workspace_list_files",
        "workspace_read_file",
        "workspace_write_file",
        "workspace_apply_patch",
        "workspace_check",
    ):
        assert f"def {tool_name}(" in source
        assert tool_name in source[source.index("tools=["):source.index("system_prompt=(")]


def test_streamlit_html_preview_uses_iframe_not_deprecated_component():
    source = (Path(__file__).resolve().parents[1] / "streamlit_app.py").read_text(encoding="utf-8")
    assert "st.iframe(html_document" in source
    assert "components.html(" not in source
    assert "import streamlit.components.v1 as components" not in source
