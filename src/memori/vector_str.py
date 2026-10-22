"""Lazy Pinecone client and index initialization.

Importing memory retrieval/worker modules must not require external service
credentials. Validate configuration at the first operation that actually needs
Pinecone, so offline unit tests and non-vector retrieval can run independently.
"""
import os
import threading

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "memori-agent")
INDEX_DIMENSION = int(os.getenv("PINECONE_DIMENSION", "3072"))

_client = None
_index = None
_client_lock = threading.Lock()
_index_lock = threading.Lock()


def _get_client():
    """Create the Pinecone client only when a vector operation is requested."""
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "PINECONE_API_KEY is required for Pinecone vector operations"
            )
        _client = Pinecone(api_key=api_key)
        return _client


def _get_index():
    """Connect to or create the configured index on first actual use."""
    global _index
    if _index is not None:
        return _index
    with _index_lock:
        if _index is not None:
            return _index
        client = _get_client()
        if not client.has_index(INDEX_NAME):
            client.create_index(
                name=INDEX_NAME,
                dimension=INDEX_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=os.getenv("PINECONE_CLOUD", "aws"),
                    region=os.getenv("PINECONE_REGION", "us-east-1"),
                ),
            )
        _index = client.Index(INDEX_NAME)
        return _index


class _LazyIndex:
    """Proxy preserving the existing index.query/upsert/update interface."""

    def __getattr__(self, name):
        return getattr(_get_index(), name)


index = _LazyIndex()
