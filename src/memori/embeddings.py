"""Lazy Gemini embedding client.

The memory worker is also used by offline unit tests and non-embedding code.
Do not validate provider credentials merely by importing the package; validate
them when an embedding operation is actually requested.
"""
import os
import threading

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

_embeddings = None
_embeddings_lock = threading.Lock()


def _get_embeddings():
    global _embeddings
    if _embeddings is not None:
        return _embeddings
    with _embeddings_lock:
        if _embeddings is not None:
            return _embeddings
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY or GOOGLE_API_KEY is required for embedding operations"
            )
        _embeddings = GoogleGenerativeAIEmbeddings(
            model=os.getenv("MEMORI_EMBEDDING_MODEL", "gemini-embedding-001"),
            api_key=api_key,
        )
        return _embeddings


class _LazyEmbeddings:
    """Proxy that preserves the existing embed_query/embed_documents interface."""

    def __getattr__(self, name):
        return getattr(_get_embeddings(), name)


embeddings = _LazyEmbeddings()
