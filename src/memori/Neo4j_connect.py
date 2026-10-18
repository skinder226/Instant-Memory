"""Lazy Neo4j graph connection.

Importing the package and collecting unit tests must not require live service
credentials. The connection is created on first graph operation.
"""
from __future__ import annotations

import os
import threading

from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph as _Neo4jGraph

load_dotenv()


class _LazyNeo4jGraph:
    """Thread-safe lazy proxy preserving the Neo4jGraph query interface."""

    def __init__(self) -> None:
        self._graph = None
        self._lock = threading.Lock()

    def _get_graph(self):
        if self._graph is None:
            with self._lock:
                if self._graph is None:
                    uri = os.getenv("NEO4J_URI")
                    username = os.getenv("NEO4J_USERNAME")
                    password = os.getenv("NEO4J_PASSWORD")
                    missing = [
                        name for name, value in (
                            ("NEO4J_URI", uri),
                            ("NEO4J_USERNAME", username),
                            ("NEO4J_PASSWORD", password),
                        ) if not value
                    ]
                    if missing:
                        raise RuntimeError(
                            "Neo4j is not configured; set " + ", ".join(missing)
                            + " before using graph retrieval or synchronization."
                        )
                    self._graph = _Neo4jGraph(
                        url=uri,
                        username=username,
                        password=password,
                        refresh_schema=False,
                    )
        return self._graph

    def query(self, *args, **kwargs):
        return self._get_graph().query(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._get_graph(), name)


Neo4jGraph = _LazyNeo4jGraph()
