"""Public interface for the Instant-Memory package."""

import logging
import sys

# Show memori's INFO logs (ROUTER START/DONE, RETRIEVAL_TIMING, save results) in the terminal.
_log = logging.getLogger("memori")
if not _log.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    _log.addHandler(_handler)
_log.setLevel(logging.INFO)
_log.propagate = False

from .api import memori
from .agentic_runtime import AgenticMemoryRuntime
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
    "record_agent_event",
    "retrieve_agent_history",
    "save_agent_task",
    "search_agent_events",
    "get_agent_task",
    "list_resumable_agent_tasks",
    "list_agent_tasks",
]