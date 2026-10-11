import os
import threading
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is not configured")

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "memori-agent")
INDEX_DIMENSION = int(os.getenv("PINECONE_DIMENSION", "3072"))

pc = Pinecone(api_key=PINECONE_API_KEY)
_index = None
_index_lock = threading.Lock()


def _get_index():
    """Connect to the index on first actual use, not during module import."""
    global _index
    if _index is not None:
        return _index
    with _index_lock:
        if _index is not None:
            return _index
        if not pc.has_index(INDEX_NAME):
            pc.create_index(
                name=INDEX_NAME,
                dimension=INDEX_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=os.getenv("PINECONE_CLOUD", "aws"),
                    region=os.getenv("PINECONE_REGION", "us-east-1"),
                ),
            )
        _index = pc.Index(INDEX_NAME)
        return _index


class _LazyIndex:
    """Proxy preserving the existing index.query/upsert/update interface."""
    def __getattr__(self, name):
        return getattr(_get_index(), name)


index = _LazyIndex()
