"""Unit tests for the runtime's opt-in streaming contract."""
from __future__ import annotations

import pytest

from memori.agentic_runtime import AgenticMemoryRuntime


class _FakeRuntime(AgenticMemoryRuntime):
    def __init__(self, *, streaming: bool, fail: bool = False):
        # Keep this unit test independent from MongoDB and model providers.
        self.streaming = streaming
        self.fail = fail

    def run(self, user_input, *, task_goal=None, on_token=None, on_progress=None):
        if self.fail:
            raise TimeoutError("simulated provider timeout")
        if on_progress is not None:
            on_progress({"type": "activity", "stage": "agent_working", "message": "Working"})
        if on_token is not None:
            on_token("Hello ")
            on_token("world")
        return {"answer": "Hello world", "user_id": "test-user"}


def test_streaming_is_disabled_by_default_contract():
    runtime = _FakeRuntime(streaming=False)
    with pytest.raises(RuntimeError, match="Streaming is disabled"):
        list(runtime.stream("hello"))


def test_stream_yields_tokens_then_final_result():
    runtime = _FakeRuntime(streaming=True)
    events = list(runtime.stream("hello"))
    assert [event["type"] for event in events] == ["activity", "token", "token", "result"]
    assert events[0]["stage"] == "agent_working"
    assert "".join(event["content"] for event in events if event["type"] == "token") == "Hello world"
    assert events[-1]["result"]["answer"] == "Hello world"


def test_stream_propagates_worker_exception():
    runtime = _FakeRuntime(streaming=True, fail=True)
    with pytest.raises(TimeoutError, match="simulated provider timeout"):
        list(runtime.stream("hello"))
