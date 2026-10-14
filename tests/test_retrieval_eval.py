"""Deterministic retrieval evaluation against synthetic, tenant-scoped memory data.

Run with: uv run pytest tests/test_retrieval_eval.py -q
This does not require MongoDB, Pinecone, Neo4j, or an LLM.
"""
from __future__ import annotations

import pytest

from memori import api


# Synthetic fixtures only. Never load real user memories into CI.
FACTS = [
    ("person_friend", "The user's friend is Fasil.", "friend"),
    ("person_sibling", "Sara is the user's sibling.", "sibling"),
    ("person_parent", "The user's mother is Amina.", "mother"),
    ("person_child", "The user's son is Hamza.", "son"),
    ("person_spouse", "The user's wife is Ayesha.", "wife"),
    ("workplace", "The user works at Microsoft.", "workplace"),
    ("project", "The user is building the Neo project with Fasil at Microsoft.", "project"),
    ("preference", "The user prefers Python for backend development.", "preference"),
    ("date_event", "The user's meeting with Fasil is on 2026-11-05.", "date"),
    ("correction", "The user's current city is Lahore.", "correction"),
    ("multihop", "Fasil works with the user on Neo at Microsoft.", "multihop"),
    ("skill", "The user uses MongoDB and Neo4j for memory storage.", "database"),
    ("goal", "The user's goal is to ship a reliable memory system.", "goal"),
    ("hobby", "The user enjoys hiking on weekends.", "hobby"),
    ("education", "The user studied computer science.", "education"),
    ("pet", "The user's cat is named Miso.", "pet"),
    ("language", "The user speaks Urdu and English.", "language"),
    ("device", "The user develops on Windows.", "device"),
    ("deadline", "The Neo project deadline is 2026-12-15.", "deadline"),
    ("team", "Sara reviews the user's project designs.", "team"),
]

# Ten paraphrases per fact = 200 retrieval queries across diverse memory types.
QUERY_TEMPLATES = {
    "person_friend": [
        "Who is my friend?", "Tell me about my friends", "Which person is my friend?",
        "Who do I know as a friend?", "Name my friend", "Who is a friend of mine?",
        "Do I have a friend mentioned?", "Who is my close friend?", "Which friend do I have?",
        "What person is my friend?",
    ],
    "person_sibling": [
        "Do I have siblings?", "Who is my sibling?", "Is Sara related to me as family?",
        "Do I have a brother or sister?", "Tell me about my siblings", "Who in my family is my sibling?",
        "Is Sara my sister or sibling?", "Which sibling is mentioned?", "Do I have any sisters?",
        "Who is sibling of the user?",
    ],
    "person_parent": [
        "Who is my mother?", "Name my parent", "What is my mother's name?",
        "Do you know my mom?", "Who is my mum?", "Tell me about my parents",
        "Which person is my mother?", "What family facts are saved about my mother?",
        "Who is Amina to me?", "Is my mother mentioned?",
    ],
    "person_child": [
        "Who is my son?", "Do I have a child?", "What is my child's name?",
        "Tell me about my children", "Who is Hamza?", "Which son do I have?",
        "Is a son mentioned in my memories?", "What do you know about my family child?",
        "Name my boy", "Who is my son called?",
    ],
    "person_spouse": [
        "Who is my wife?", "Who am I married to?", "What is my spouse's name?",
        "Tell me about my wife", "Who is Ayesha to me?", "Do I have a spouse?",
        "Which partner is mentioned?", "What do you know about my marriage?",
        "Who is the user's wife?", "Name my spouse",
    ],
    "workplace": [
        "Where do I work?", "Which company employs me?", "Who is my employer?",
        "Where am I working?", "What company do I work for?", "Tell me about my workplace",
        "What is my current job company?", "Which organization do I work at?",
        "Am I employed at Microsoft?", "Where is my job?",
    ],
    "project": [
        "What project am I building?", "Which project am I working on?",
        "Tell me about my project", "What am I developing?", "What is Neo?",
        "Which product am I making?", "What project do I have with Fasil?",
        "What am I building at Microsoft?", "Name my current project",
        "What software project is in my memories?",
    ],
    "preference": [
        "Which programming language do I prefer?", "What is my backend preference?",
        "Do I prefer Python?", "Tell me my coding preferences", "What language do I like for backend?",
        "What is my preferred backend language?", "Which language suits my preference?",
        "What tech preference have I shared?", "Do I like Python for backend development?",
        "What programming tool do I prefer?",
    ],
    "date_event": [
        "When is my meeting with Fasil?", "What date is the meeting?", "When do I meet Fasil?",
        "Tell me the meeting date", "What is scheduled for 2026-11-05?",
        "Which date did I mention for the meeting?", "When is the event with Fasil?",
        "What date is my appointment with my friend?", "On what day is my meeting?",
        "Do you remember the date of my meeting?",
    ],
    "correction": [
        "What is my current city?", "Where do I live now?", "Which city am I in?",
        "What city is saved for me?", "Where is my current location?", "Tell me my city",
        "What city should be treated as current?", "Which place do I currently live in?",
        "What is the latest city fact?", "In which city am I based?",
    ],
    "multihop": [
        "Who works with me on Neo?", "Which project do Fasil and I share?",
        "Where does Fasil work with me?", "Who is on my project at Microsoft?",
        "What connects Fasil, Neo, and Microsoft?", "Who collaborates with me on my project?",
        "Which colleague helps with Neo?", "Who works with the user on Neo?",
        "What project do I work on with Fasil at my employer?", "Tell me the people and company related to Neo",
    ],
    "skill": [
        "Which databases do I use?", "What do I use for memory storage?",
        "Do I use MongoDB?", "Which graph database do I use?", "What databases are in my stack?",
        "Tell me my storage technologies", "Which tools store my memories?",
        "Do I use Neo4j and MongoDB?", "What database technologies have I mentioned?",
        "What data stores do I work with?",
    ],
    "goal": [
        "What is my goal?", "What am I trying to ship?", "Tell me my project goal",
        "What outcome do I want?", "What is my main objective?", "What do I want to achieve?",
        "What goal did I share?", "What am I aiming for with the memory system?",
        "What should I accomplish?", "What is the goal of my work?",
    ],
    "hobby": [
        "What do I do for fun?", "What is my hobby?", "What do I enjoy on weekends?",
        "Do I like hiking?", "Tell me about my interests", "What activity do I enjoy?",
        "What do I do on Saturdays and Sundays?", "Which hobby have I mentioned?",
        "What are my leisure activities?", "What do I enjoy doing?",
    ],
    "education": [
        "What did I study?", "What is my educational background?", "Did I study computer science?",
        "What subject did I learn?", "Tell me about my education", "What degree subject is mentioned?",
        "Which field did I study?", "What was my area of study?", "What did I learn at university?",
        "What education fact do you remember?",
    ],
    "pet": [
        "What is my cat called?", "Do I have a pet?", "What is my pet's name?",
        "Tell me about my cat", "Who is Miso?", "Which animal do I own?",
        "What name did I give my cat?", "Do I have a cat?", "Name my pet",
        "What pet information is stored?",
    ],
    "language": [
        "Which languages do I speak?", "Do I speak Urdu?", "What languages do I know?",
        "Tell me about my language skills", "Which spoken languages are saved?",
        "What language can I communicate in?", "Do I speak English?", "What languages have I mentioned?",
        "Which languages do I use?", "What are my known languages?",
    ],
    "device": [
        "Which operating system do I use?", "Do I develop on Windows?", "What is my development OS?",
        "Tell me about my computer environment", "Which OS do I work on?",
        "What system do I develop with?", "What operating system is saved?",
        "Am I using Windows?", "Which platform do I code on?", "What is my dev environment?",
    ],
    "deadline": [
        "When is the Neo deadline?", "What is my project deadline?", "When must Neo be finished?",
        "What date is the project due?", "Tell me the deadline", "When is Neo due?",
        "What date did I set for the project?", "When does the project need shipping?",
        "What is the due date for Neo?", "Which deadline is saved?",
    ],
    "team": [
        "Who reviews my designs?", "What does Sara do on my team?", "Who reviews project design?",
        "Does Sara review my work?", "Tell me about my project reviewers", "Who gives feedback on my designs?",
        "Which person reviews my designs?", "Who is involved in design reviews?",
        "What role does Sara have?", "Who checks my project designs?",
    ],
}


@pytest.fixture
def memory_snapshot():
    docs = []
    for memory_id, content, category in FACTS:
        entities = []
        if memory_id == "person_sibling":
            entities = [{"source": {"name": "user_125", "type": ["User"]},
                         "relationship": "sibling_of",
                         "target": {"name": "Sara", "type": ["Person"]}}]
        elif memory_id == "person_friend":
            entities = [{"source": {"name": "user_125", "type": ["User"]},
                         "relationship": "has_friend",
                         "target": {"name": "Fasil", "type": ["Person"]}}]
        docs.append({
            "user_id": "user_125", "content": content, "type": category,
            "temporal": {"is_current": True, "valid_from": "2026-10-01T00:00:00+00:00"},
            "entities": entities,
        })
    return docs


@pytest.mark.parametrize(
    "fact_id,query",
    [
        (fact_id, query)
        for fact_id, queries in QUERY_TEMPLATES.items()
        for query in queries
    ],
)
def test_retrieval_returns_expected_memory(monkeypatch, memory_snapshot, fact_id, query):
    """Each test asserts its ground-truth memory appears in top-8 retrieval."""
    expected = next(content for memory_id, content, _ in FACTS if memory_id == fact_id)
    monkeypatch.setattr(api, "_get_current_user_memories", lambda user_id: (
        memory_snapshot if user_id == "user_125" else []
    ))
    result = api._retrieve_memories_fast("user_125", query, limit=8)
    assert expected in result, f"query={query!r} expected={expected!r} got={result!r}"


def test_retrieval_does_not_read_another_users_snapshot(monkeypatch, memory_snapshot):
    monkeypatch.setattr(api, "_get_current_user_memories", lambda user_id: (
        memory_snapshot if user_id == "user_125" else []
    ))
    assert api._retrieve_memories_fast("user_999", "Who is my friend?") == []
