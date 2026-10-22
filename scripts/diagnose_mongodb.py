"""Diagnose MongoDB Atlas connectivity without printing connection credentials.

Run from the repository root:
    uv run python scripts/diagnose_mongodb.py

This checks the configured MongoDB URI by issuing a real ping, then reports
replica-set topology. It intentionally does not print the URI or raw exceptions.
"""
from __future__ import annotations

import os
import sys
import time

from dotenv import load_dotenv
from pymongo.errors import PyMongoError


def main() -> int:
    load_dotenv()
    if not os.getenv("mongodb_URI", "").strip():
        print("FAIL: mongodb_URI is missing. Add it to your local .env or deployment secrets.")
        return 2

    # Import only after validating configuration so an absent URI cannot silently
    # fall back to a local MongoDB instance.
    from memori.mongodb_connection import client

    started = time.perf_counter()
    try:
        client.admin.command("ping")
    except PyMongoError as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        print(f"MongoDB ping: FAIL ({type(exc).__name__}, {elapsed_ms:.1f} ms)")
        print("Check Atlas cluster health, Network Access IP allowlist, DNS, outbound TCP 27017, TLS, and URI/replica-set settings.")
        try:
            topology = client.topology_description
            print("Topology:", topology.topology_type_name)
            for server in topology.server_descriptions().values():
                error = server.error
                print(
                    f"Server {server.address}: type={server.server_type_name}, "
                    f"error={type(error).__name__ if error else 'none'}"
                )
        except Exception:
            print("Topology details unavailable.")
        return 1

    elapsed_ms = (time.perf_counter() - started) * 1000
    topology = client.topology_description
    print(f"MongoDB ping: OK ({elapsed_ms:.1f} ms)")
    print("Topology:", topology.topology_type_name)
    primary_found = False
    for server in topology.server_descriptions().values():
        if server.server_type_name == "RSPrimary":
            primary_found = True
        error = server.error
        print(
            f"Server {server.address}: type={server.server_type_name}, "
            f"error={type(error).__name__ if error else 'none'}"
        )

    if topology.topology_type_name == "ReplicaSetNoPrimary" or not primary_found:
        print("WARNING: ping succeeded but a replica-set primary was not visible in the current topology snapshot.")
        print("Wait briefly and rerun before treating this snapshot as a confirmed outage.")
        return 1

    print("MongoDB connectivity check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
