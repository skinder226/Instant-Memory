"""Production-readiness regression checks for the Instant-Memory API.

Run from the repository root:
    uv run pytest tests/test_production_readiness.py -q

These tests do not contact MongoDB, Pinecone, Neo4j, or an LLM provider.
"""
from memori.api import (
    _compact_extractor_system_message,
    _is_request_too_large,
    _requires_historical_project_lookup,
    _requires_personal_project_lookup,
    _source_role,
)


def test_compact_extractor_prompt_mentions_json_for_json_mode_providers():
    """Some OpenAI-compatible providers reject JSON mode unless prompt says JSON."""
    prompt = _compact_extractor_system_message()
    assert "json" in prompt.casefold(), (
        "Compact fallback prompt must explicitly request JSON. "
        "The current background-save fallback can fail with: "
        "'messages must contain the word json ... response_format=json_object'."
    )


def test_compact_extractor_prompt_is_bounded_and_keeps_required_rules():
    prompt = _compact_extractor_system_message()
    assert len(prompt) < 3000, "Compact fallback prompt should stay small."
    assert "Analyze ONLY the latest supplied message" in prompt
    assert "placeholder_updates" in prompt
    assert "source_role" in prompt


def test_source_role_normalization():
    assert _source_role("user") == "user"
    assert _source_role("AI") == "assistant"
    assert _source_role("assistant") == "assistant"
    assert _source_role("tool") == "tool"


def test_oversized_provider_request_is_recognized():
    error = Exception(
        "Error code: 413 - Request too large for model; tokens per minute (TPM)"
    )
    assert _is_request_too_large(error)


def test_current_project_questions_trigger_fast_lookup():
    assert _requires_personal_project_lookup("What is my current project?")
    assert _requires_personal_project_lookup("Which project am I working on?")
    assert not _requires_personal_project_lookup("What is my favorite language?")


def test_historical_project_questions_are_distinguished():
    assert _requires_historical_project_lookup("What was my project before Orion?")
    assert _requires_historical_project_lookup("What was my previous project?")
    assert not _requires_historical_project_lookup("What is my current project?")
