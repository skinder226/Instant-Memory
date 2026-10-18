"""Regression tests for deterministic retrieval helpers; no external services required."""
from memori import api


def test_people_lookup_matches_relationship_words_in_memory_text(monkeypatch):
    monkeypatch.setattr(
        api,
        "_get_current_user_memories",
        lambda user_id: [
            {
                "content": "Fasil is the user's friend and works with the user.",
                "entities": [],
                "temporal": {"valid_from": "2026-10-01T00:00:00+00:00"},
            }
        ],
    )

    assert api._retrieve_people_who_know_user("tenant-a") == [
        "Fasil is the user's friend and works with the user."
    ]


def test_people_lookup_does_not_return_unrelated_text(monkeypatch):
    monkeypatch.setattr(
        api,
        "_get_current_user_memories",
        lambda user_id: [
            {
                "content": "The user prefers Python for backend development.",
                "entities": [],
                "temporal": {"valid_from": "2026-10-01T00:00:00+00:00"},
            }
        ],
    )

    assert api._retrieve_people_who_know_user("tenant-a") == []
