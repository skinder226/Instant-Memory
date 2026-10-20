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


def test_project_fast_lookup_uses_shared_user_snapshot(monkeypatch):
    snapshot = [
        {"content": "The user is building Neo.", "type": "project", "temporal": {"valid_from": "2026-10-01"}},
        {"content": "The user works at Microsoft.", "type": "profile", "temporal": {"valid_from": "2026-10-02"}},
        {"content": "Old unrelated note.", "type": "note", "temporal": {"valid_from": "2026-09-01"}},
    ]
    monkeypatch.setattr(api, "_get_current_user_memories", lambda user_id: snapshot)
    assert api._retrieve_project_memories_fast("tenant-a") == [
        "The user is building Neo.",
    ]


def test_workplace_fast_lookup_uses_shared_snapshot_and_prefers_structured_fact(monkeypatch):
    snapshot = [
        {
            "content": "The user works at Microsoft.",
            "type": "profile",
            "entities": [{"relationship": "works_at"}],
            "temporal": {"valid_from": "2026-10-01"},
        },
        {
            "content": "The user is working at Contoso.",
            "type": "profile",
            "entities": [],
            "temporal": {"valid_from": "2026-10-02"},
        },
        {
            "content": "The user likes Python.",
            "type": "preference",
            "entities": [],
            "temporal": {"valid_from": "2026-10-03"},
        },
    ]
    monkeypatch.setattr(api, "_get_current_user_memories", lambda user_id: snapshot)
    assert api._retrieve_workplace_memories_fast("tenant-a") == [
        "The user works at Microsoft.",
        "The user is working at Contoso.",
    ]