from memori import security


def test_principal_adapter_uses_verified_subject(monkeypatch):
    calls = []
    monkeypatch.setattr(security, "memori", lambda **kwargs: calls.append(kwargs) or {"ok": True})
    result = security.memori_for_principal({"sub": "tenant-a"}, "Who is my friend?")
    assert result == {"ok": True}
    assert calls == [{"user_id": "tenant-a", "user_input": "Who is my friend?", "source": "user"}]


def test_principal_adapter_rejects_missing_identity(monkeypatch):
    monkeypatch.setattr(security, "memori", lambda **kwargs: {"ok": True})
    try:
        security.memori_for_principal({}, "hello")
    except ValueError as exc:
        assert "principal" in str(exc).lower()
    else:
        raise AssertionError("Missing identity must be rejected")


def test_principal_adapter_does_not_accept_blank_identity():
    try:
        security.memori_for_principal({"sub": "  "}, "hello")
    except ValueError:
        pass
    else:
        raise AssertionError("Blank identity must be rejected")
