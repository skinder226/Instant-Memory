from src.memori.schema import ContextDecision, MemoryItem, RetrievalPlan


def test_memory_item_scores_and_entities():
    item = MemoryItem(
        type="fact",
        content="The user uses Python.",
        source_role="user",
        confidence=0.9,
        importance=0.8,
        entities=[],
    )
    assert item.confidence == 0.9
    assert item.importance == 0.8


def test_retrieval_plan_allows_history_filter_and_six_hops():
    plan = RetrievalPlan(
        query_type="graph",
        query="all companies I worked at",
        memory_type="profile",
        is_current=None,
        max_hops=6,
        top_k=1,
    )
    assert plan.is_current is None
    assert plan.max_hops == 6
    assert plan.top_k == 1


def test_context_decision_exists_and_validates():
    assert ContextDecision(needs_context=True).needs_context is True
    assert ContextDecision(needs_context=False).needs_context is False
