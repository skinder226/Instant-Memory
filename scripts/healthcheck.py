"""Deployment preflight: validate required settings and ping MongoDB.

Run from the repository root with: uv run python scripts/healthcheck.py
This never prints secret values and does not write application data.
"""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()


def main() -> int:
    required = {
        "MongoDB URI": "mongodb_URI",
        "LLM provider": "Mem_llm_provider",
        "LLM API key": "Mem_llm_Api",
        "LLM base URL": "Mem_base_url",
        "LLM model": "Mem_model",
        "Pinecone API key": "PINECONE_API_KEY",
        "Neo4j URI": "NEO4J_URI",
        "Neo4j username": "NEO4J_USERNAME",
        "Neo4j password": "NEO4J_PASSWORD",
        "Gemini embeddings API key": "GEMINI_API_KEY",
    }
    missing = [label for label, env_name in required.items() if not os.getenv(env_name)]
    for label, env_name in required.items():
        print(f"{'OK' if os.getenv(env_name) else 'MISSING'} configuration: {label}")

    mongo_ok = False
    uri = os.getenv("mongodb_URI")
    if uri:
        client = MongoClient(
            uri,
            serverSelectionTimeoutMS=int(os.getenv("MEMORI_HEALTHCHECK_TIMEOUT_MS", "5000")),
            connectTimeoutMS=int(os.getenv("MEMORI_HEALTHCHECK_TIMEOUT_MS", "5000")),
        )
        try:
            client.admin.command("ping")
            print("OK connectivity: MongoDB ping")
            mongo_ok = True
        except Exception as exc:
            print(f"FAIL connectivity: MongoDB ping ({type(exc).__name__})")
        finally:
            client.close()
    else:
        print("SKIP connectivity: MongoDB ping (URI missing)")

    if missing or not mongo_ok:
        print("Preflight failed. Fix missing configuration/connectivity before deployment.")
        return 1
    print("Preflight passed for configuration and MongoDB only.")
    print("Note: Pinecone, Neo4j, and LLM provider connectivity were not called.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
