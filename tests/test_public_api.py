from memori.api import _source_role


def test_user_source_is_preserved():
    assert _source_role("user") == "user"


def test_ai_alias_maps_to_assistant():
    assert _source_role("ai") == "assistant"


def test_assistant_role_is_preserved():
    assert _source_role("assistant") == "assistant"


def test_tool_role_is_preserved():
    assert _source_role("tool") == "tool"


def test_unknown_role_maps_to_other():
    assert _source_role("custom-provider") == "other"


def test_empty_source_is_rejected():
    try:
        _source_role(" ")
    except ValueError:
        pass
    else:
        raise AssertionError("Empty source should raise ValueError")



def test_oversized_groq_tpm_error_is_detected():
    from memori.api import _is_request_too_large

    error = Exception(
        "Error code: 413 - Request too large for model; tokens per minute (TPM)"
    )
    assert _is_request_too_large(error)


def test_compact_extractor_prompt_is_small_and_has_core_rules():
    from memori.api import _compact_extractor_system_message

    prompt = _compact_extractor_system_message()
    assert len(prompt) < 3000
    assert "Analyze ONLY the latest supplied message" in prompt
    assert "placeholder_updates" in prompt
    assert "source_role" in prompt
