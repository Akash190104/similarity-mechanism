"""Extract data from Inspect AI eval logs for plotting.

Reads structured results stored in Inspect's eval logs and converts
them into the dict formats that the existing plotting scripts expect.

Usage:
    from inspect_similarity.plotting.from_logs import (
        load_tournament_results,
        load_benchmark_results,
        load_elicitation_results,
    )

    # From Inspect log file
    tournament = load_tournament_results("logs/2026-04-08T.../eval.json")
    benchmark = load_benchmark_results("logs/2026-04-08T.../eval.json")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_eval_log(log_path: str | Path) -> dict:
    """Read an Inspect eval log and extract sample-level store data.

    Returns a dict with keys from the first sample's store.
    """
    try:
        from inspect_ai.log import read_eval_log

        log = read_eval_log(str(log_path))
        samples = log.samples or []
        if not samples:
            return {}

        # Inspect stores our custom data in sample.store
        first_sample = samples[0]
        return dict(first_sample.store) if first_sample.store else {}

    except ImportError:
        # Fallback: read raw JSON
        with open(log_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        samples = data.get("samples", [])
        if not samples:
            return {}
        return samples[0].get("store", {})


# ---------------------------------------------------------------------------
# Tournament results
# ---------------------------------------------------------------------------

def load_tournament_results(log_path: str | Path) -> dict[str, Any]:
    """Extract tournament results from an Inspect eval log.

    Returns:
        Dict with keys:
        - "agent_average_payoff": {agent_type: float}
        - "matchup_payoffs": serialized MatchupPayoffs dict
    """
    store = _read_eval_log(log_path)
    return {
        "agent_average_payoff": store.get("agent_average_payoff", {}),
        "matchup_payoffs": store.get("matchup_payoffs", {}),
    }


# ---------------------------------------------------------------------------
# Benchmark results
# ---------------------------------------------------------------------------

def load_benchmark_results(log_path: str | Path) -> dict[str, Any]:
    """Extract benchmark results from an Inspect eval log.

    Returns:
        For single benchmark: the benchmark_output dict
        For battery: the battery_output dict (keyed by benchmark name)
    """
    store = _read_eval_log(log_path)

    if "battery_output" in store:
        return store["battery_output"]
    if "benchmark_output" in store:
        return store["benchmark_output"]
    return store


def load_similarity_matrix(log_path: str | Path) -> dict[str, float]:
    """Extract the pairwise similarity matrix from a benchmark eval log.

    Returns:
        Dict like {"agent_a vs agent_b": 75.5, ...}
    """
    results = load_benchmark_results(log_path)

    # Single benchmark
    if "similarity_matrix" in results:
        return results["similarity_matrix"]

    # Battery: merge all matrices
    merged = {}
    for bench_name, bench_data in results.items():
        if isinstance(bench_data, dict) and "similarity_matrix" in bench_data:
            for k, v in bench_data["similarity_matrix"].items():
                merged[f"{bench_name}:{k}"] = v
    return merged


# ---------------------------------------------------------------------------
# Elicitation results
# ---------------------------------------------------------------------------

def load_elicitation_results(log_path: str | Path) -> dict[str, Any]:
    """Extract elicitation results from an Inspect eval log.

    Returns:
        Dict with keys: single_results, pairwise_results, seed, etc.
    """
    store = _read_eval_log(log_path)
    return store.get("elicitation_results", store.get("result", {}))


# ---------------------------------------------------------------------------
# Config-based results
# ---------------------------------------------------------------------------

def load_config_results(log_path: str | Path) -> dict[str, Any]:
    """Extract results from a from_config eval log.

    Returns the full result dict which includes a "type" field
    ("tournament" or "elicitation") plus the corresponding data.
    """
    store = _read_eval_log(log_path)
    return store.get("result", store)


# ---------------------------------------------------------------------------
# Convenience: list all logs in a directory
# ---------------------------------------------------------------------------

def list_eval_logs(log_dir: str | Path = "logs") -> list[Path]:
    """Find all Inspect eval log files in a directory."""
    log_dir = Path(log_dir)
    if not log_dir.exists():
        return []

    logs = []
    for p in log_dir.rglob("*.json"):
        # Inspect log files typically have specific naming
        if p.name.startswith("eval") or "log" in p.name.lower():
            logs.append(p)
    return sorted(logs, key=lambda p: p.stat().st_mtime, reverse=True)
