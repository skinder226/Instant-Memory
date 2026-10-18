"""Lazy Neo4j connection so importing the package does not require live credentials."""
from __future__ import annotations

import os
import threading

from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph as _Neo4jGraph

load_dotenv()


class _LazyNeo4jGraph:
    """Initialize the real graph client only on the first database operation."""

    def __init__(self) -> None:
        self._graph = None
        self._lock = threading.Lock()

    def _get_graph(self):
        if self._graph is None:
            with self._lock:
                if self._graph is None:
                    url = os.getenv("NEO4J_URI")
                    username = os.getenv("NEO4J_USERNAME")
                    password = os.getenv("NEO4J_PASSWORD")
                    if not url:
                        raise RuntimeError(
                            "Neo4j is required for this operation but NEO4J_URI is not configured."
                        )
                    self._graph = _Neo4jGraph(
                        url=url,
                        username=username,
                        password=password,
                        refresh_schema=False,
                    )
        return self._graph

    def query(self, query: str, params: dict | None = None, **kwargs):
        """Forward a query to the lazily initialized LangChain Neo4j client."""
        return self._get_graph().query(query, params=params, **kwargs)


# Preserve the existing Neo4jGraph.query(...) call sites without connecting at import.
Neo4jGraph = _LazyNeo4jGraph()
