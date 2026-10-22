"""MongoDB configuration safety tests; no live database required."""
import pytest

from memori import mongodb_connection as mongo


def test_missing_uri_fails_clearly_instead_of_defaulting_to_localhost(monkeypatch):
    monkeypatch.setattr(mongo, "connection_string", None)
    monkeypatch.setattr(mongo, "_client", None)
    monkeypatch.delenv("mongodb_URI", raising=False)
    monkeypatch.delenv("MONGODB_URI", raising=False)

    with pytest.raises(RuntimeError, match="MongoDB URI is missing"):
        mongo.db["memories"]
