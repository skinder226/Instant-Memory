"""Regression tests for explicit project corrections and historical lookup routing."""
from memori import api


def test_historical_project_questions_are_recognized():
    assert api._requires_historical_project_lookup("What was my test project before Orion")
    assert api._requires_historical_project_lookup("Which project did I previously work on?")
    assert not api._requires_historical_project_lookup("What is my current test project?")


class _OldProjectCollection:
    def find(self, query, projection):
        assert query["user_id"] == "user-a"
        assert query["type"] == "project"
        return [{
            "_id": "old-mongo-id",
            "content": "The user is working on a project called Atlas with Omar.",
            "entities": [{"target": {"name": "Atlas"}}],
        }]


class _FakeDB:
    def __getitem__(self, name):
        assert name == "memories"
        return _OldProjectCollection()


def test_explicit_project_correction_links_only_named_old_project(monkeypatch):
    from memori import mongodb_connection

    monkeypatch.setattr(mongodb_connection, "db", _FakeDB())
    memories = [{
        "type": "project",
        "content": "The user is working on a test project called Orion.",
        "entities": [{"target": {"name": "Orion"}}],
    }]

    api._attach_explicit_project_supersession(
        "user-a", "My test project is now Orion, not Atlas", memories
    )

    assert memories[0]["_supersedes_mongo_ids"] == ["old-mongo-id"]


def test_project_correction_does_not_guess_when_old_name_is_missing(monkeypatch):
    from memori import mongodb_connection

    monkeypatch.setattr(mongodb_connection, "db", _FakeDB())
    memories = [{
        "type": "project",
        "content": "The user is working on a test project called Orion.",
        "entities": [{"target": {"name": "Orion"}}],
    }]

    api._attach_explicit_project_supersession(
        "user-a", "My test project is now Orion", memories
    )

    assert "_supersedes_mongo_ids" not in memories[0]
