from memori.agentic_runtime import _redact


def test_redacts_sensitive_mapping_keys_recursively():
    payload = {
        "api_key": "secret-value",
        "nested": {"Authorization": "Bearer abc.def.ghi", "safe": "hello"},
        "items": [{"password": "p@ss", "name": "tool"}],
    }

    result = _redact(payload)

    assert result["api_key"] == "[REDACTED]"
    assert result["nested"]["Authorization"] == "[REDACTED]"
    assert result["nested"]["safe"] == "hello"
    assert result["items"][0]["password"] == "[REDACTED]"
    assert result["items"][0]["name"] == "tool"


def test_redacts_common_secret_patterns_in_strings():
    result = _redact(
        "Authorization: Bearer abc.def.ghi; api_key=supersecret "
        "mongodb://worker:topsecret@db.example.com/app"
    )

    assert "abc.def.ghi" not in result
    assert "supersecret" not in result
    assert "worker:topsecret" not in result
    assert "[REDACTED]" in result
