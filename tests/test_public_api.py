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
