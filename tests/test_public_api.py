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


def test_ai_source_is_not_saved_or_retrieved(monkeypatch):
    from memori import api

    scheduled = []
    monkeypatch.setattr(
        api._background_executor,
        "submit",
        lambda fn, *args, **kwargs: scheduled.append((fn, args, kwargs)),
    )

    result = api.memori("tenant-a", "The assistant summarized a saved preference.", source="ai")

    assert result["retrieval"] == []
    assert result["memory_save_scheduled"] is False
    assert scheduled == []


def test_tool_source_is_not_saved_or_retrieved(monkeypatch):
    from memori import api

    scheduled = []
    monkeypatch.setattr(
        api._background_executor,
        "submit",
        lambda fn, *args, **kwargs: scheduled.append((fn, args, kwargs)),
    )

    result = api.memori("tenant-a", "Tool output mentions Microsoft.", source="tool")

    assert result["retrieval"] == []
    assert result["memory_save_scheduled"] is False
    assert scheduled == []


def test_oversized_decision_prompt_falls_back_to_extractor(monkeypatch):
    import sys
    import types

    from memori import api

    saved = []
    fake_llms = types.ModuleType("memori.llms")
    fake_llms.MemoryDecisionGate = object()
    fake_worker = types.ModuleType("memori.worker")
    fake_worker.invoke_with_retry = lambda *args, **kwargs: (_ for _ in ()).throw(
        Exception("413 Request too large; tokens per minute exceeded")
    )
    monkeypatch.setitem(sys.modules, "memori.llms", fake_llms)
    monkeypatch.setitem(sys.modules, "memori.worker", fake_worker)
    monkeypatch.setattr(api, "_is_request_too_large", lambda exc: "413" in str(exc))
    monkeypatch.setattr(
        api,
        "_save_message_background",
        lambda *args, **kwargs: saved.append((args, kwargs)),
    )

    api._decide_and_save_background("tenant-a", "Remember my current preference.", "user")

    assert len(saved) == 1
    assert saved[0][0] == ("tenant-a", "Remember my current preference.", "user", True)

