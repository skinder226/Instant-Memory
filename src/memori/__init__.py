"""Public interface for the Instant-Memory package.

The library deliberately does not configure application logging. Applications
can configure the `memori` logger through their own logging configuration.
"""

import logging

# Avoid Python's "No handler found" behavior without taking control of the
# host application's logging destinations or log levels.
logging.getLogger("memori").addHandler(logging.NullHandler())

from .api import memori
from .agentic_runtime import AgenticMemoryRuntime
from .failed_responses import create_failed_response_recovery_tool
from .agentic_memory import (
    get_agent_task,
    list_resumable_agent_tasks,
    list_agent_tasks,
    record_agent_event,
    retrieve_agent_history,
    save_agent_task,
    search_agent_events,
)

__all__ = [
    "memori",
    "AgenticMemoryRuntime",
    "create_failed_response_recovery_tool",
    "record_agent_event",
    "retrieve_agent_history",
    "save_agent_task",
    "search_agent_events",
    "get_agent_task",
    "list_resumable_agent_tasks",
    "list_agent_tasks",
]
