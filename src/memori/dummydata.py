from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
import os
from src.memori.embeddings import embeddings
from src.memori.vector_str import index


memories = [
    {
        "memory_id": "mem_009",
        "content": "Ali used python and python",
    },
]




vectors = []

for memory in memories:

    vector = embeddings.embed_query(
        memory["content"]
    )

    vectors.append({
        "id": memory["memory_id"],
        "values": vector,
        "metadata": {
            "user_id": "user_125",
            "memory_id": memory["memory_id"],
            "memory_type": "fact",
            "is_current": True,
            "content": memory["content"],
        },
    })


index.upsert(
    vectors=vectors,
    namespace="user_125",
)

print(f"Inserted {len(vectors)} memories.")