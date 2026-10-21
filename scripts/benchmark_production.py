"""Measure real Memori API latency against configured services.

Usage:
    uv run python scripts/benchmark_production.py --user-id user_125 --runs 10

This script intentionally calls the real public API. Configure MongoDB and any
providers needed by the selected route first. It prints timings and result counts,
never retrieved memory text. Results are diagnostic, not a CI performance gate.
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from collections.abc import Callable
from typing import Any

from memori import api


SCENARIOS: dict[str, str] = {
    "greeting": "hey",
    "generic_question": "What is photosynthesis?",
    "workplace": "Where do I work?",
    "project": "What project am I working on?",
    "relationship": "Who is my friend?",
    "history": "What project did I work on before Orion?",
    "past_task": "What website did I ask you to create last time?",
}


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = round((len(ordered) - 1) * percent / 100)
    return ordered[index]


def summarize(samples_ms: list[float]) -> dict[str, float]:
    return {
        "min_ms": round(min(samples_ms), 2),
        "median_ms": round(statistics.median(samples_ms), 2),
        "p95_ms": round(percentile(samples_ms, 95), 2),
        "p99_ms": round(percentile(samples_ms, 99), 2),
        "max_ms": round(max(samples_ms), 2),
    }


class _BenchmarkNoSaveExecutor:
    """Record save submissions without executing provider calls during a benchmark."""

    def __init__(self) -> None:
        self.submissions = 0

    def submit(self, *args: Any, **kwargs: Any) -> None:
        self.submissions += 1
        return None


def run_scenario(
    user_id: str, query: str, runs: int, cold_cache: bool, no_background_saves: bool
) -> dict[str, Any]:
    samples: list[float] = []
    result_counts: list[int] = []
    save_scheduled_runs = 0
    skipped_save_submissions = 0
    original_executor = api._background_executor
    benchmark_executor = _BenchmarkNoSaveExecutor()
    if no_background_saves:
        api._background_executor = benchmark_executor

    try:
        for _ in range(runs):
            if cold_cache:
                with api._USER_MEMORY_CACHE_LOCK:
                    api._USER_MEMORY_CACHE.pop(user_id, None)

            started = time.perf_counter()
            result = api.memori(user_id, query, source="user")
            samples.append((time.perf_counter() - started) * 1000)
            result_counts.append(len(result.get("retrieval") or []))
            save_scheduled_runs += bool(result.get("memory_save_scheduled"))
    finally:
        if no_background_saves:
            skipped_save_submissions = benchmark_executor.submissions
            api._background_executor = original_executor

    return {
        "query": query,
        "cache_mode": "snapshot-cold-per-run" if cold_cache else "natural",
        "runs": runs,
        **summarize(samples),
        "retrieval_count_last_run": result_counts[-1],
        "retrieval_count_min": min(result_counts),
        "retrieval_count_max": max(result_counts),
        "memory_save_scheduled_runs": save_scheduled_runs,
        "background_save_submissions_skipped": skipped_save_submissions,
        "note": (
            "Cold mode clears only the in-process current-memory snapshot before each call; "
            "it does not reset database contents, network pools, or provider/service caches."
            + (
                " Background save submissions were intercepted and not executed."
                if no_background_saves else
                " Background saves were enabled and may call external LLM providers."
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", required=True, help="Existing test tenant ID")
    parser.add_argument("--runs", type=int, default=10, help="Runs per scenario (1..100)")
    parser.add_argument(
        "--scenario", choices=["all", *SCENARIOS.keys()], default="all",
        help="Scenario to measure",
    )
    parser.add_argument(
        "--cold-cache", action="store_true",
        help="Clear the in-process current-memory snapshot before each run",
    )
    parser.add_argument(
        "--no-background-saves", action="store_true",
        help="Measure request routing/retrieval without executing background save jobs; recommended for repeat benchmarks",
    )
    parser.add_argument(
        "--quiet-logs", action="store_true",
        help="Reduce logs while measuring; warnings and errors remain visible",
    )
    args = parser.parse_args()

    if not 1 <= args.runs <= 100:
        parser.error("--runs must be between 1 and 100")
    if args.quiet_logs:
        logging.getLogger("memori").setLevel(logging.WARNING)
        logging.getLogger("memori.api").setLevel(logging.WARNING)

    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    report: dict[str, Any] = {
        "user_id": args.user_id,
        "runs_per_scenario": args.runs,
        "note": "Use a dedicated test tenant; route results depend on its stored memories and service health.",
        "scenarios": {},
    }
    for name in names:
        query = SCENARIOS[name]
        report["scenarios"][name] = run_scenario(
            args.user_id, query, args.runs, args.cold_cache, args.no_background_saves
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
