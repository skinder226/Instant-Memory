from memori.schema import ContextDecision, MemoryItem, RetrievalPlan


def test_memory_item_has_no_unused_scoring_fields():
    item = MemoryItem(
        type="fact",
        content="The user uses Python.",
        source_role="user",
        entities=[],
    )
    assert "confidence" not in item.model_dump()
    assert "importance" not in item.model_dump()


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
