#!/usr/bin/env python3
"""Pre-cache benchmark results for a given agent config.

Runs every benchmark in the catalogue on the specified agent(s) and saves
results to data/benchmark_cache/ in the same format used by the similarity
mechanism, so subsequent experiment runs get instant cache hits.

Usage:
    python script/cache_benchmarks.py --agents configs/agents/gpt54_solo.yaml --max-workers 100
    python script/cache_benchmarks.py --agents configs/agents/gpt54_solo.yaml --benchmark newcomb,cabin --max-workers 50
"""

import argparse
import copy
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.base import BenchmarkResult
from benchmarks.registry import BENCHMARK_REGISTRY, BENCHMARKS_EXCLUDE_FROM_ALL
from src.registry.agent_registry import create_agent

import yaml

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "benchmark_cache"

# Benchmarks that need special constructor args beyond max_items/max_workers
SPECIAL_KWARGS = {
    "hle": dict(
        judge_model="google/gemini-3.1-flash-lite-preview",
        judge_provider="OpenRouter",
    ),
    "similarity_game": None,  # skip — requires a game config
}


def cache_key(benchmark_name: str, agent_id: str) -> str:
    safe = re.sub(r"[^\w\-.]", "_", agent_id)
    return f"benchmark_cache_{benchmark_name}_{safe}.json"


def load_cached(benchmark_name: str, agent_id: str) -> BenchmarkResult | None:
    p = CACHE_DIR / cache_key(benchmark_name, agent_id)
    if p.exists():
        return BenchmarkResult.from_dict(json.loads(p.read_text()))
    return None


def save_cached(benchmark_name: str, agent_id: str, result: BenchmarkResult) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = CACHE_DIR / cache_key(benchmark_name, agent_id)
    out.write_text(json.dumps(result.serialize(), indent=2))
    print(f"  [Cache SAVE] {out.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-cache benchmark results for agents")
    parser.add_argument("--agents", required=True, help="Path to agents config YAML")
    parser.add_argument(
        "--benchmark", default="all",
        help="Comma-separated benchmark names, or 'all' (default: all)",
    )
    parser.add_argument("--max-workers", type=int, default=None, help="Concurrent LLM calls per benchmark")
    parser.add_argument("--max-items", type=int, default=None, help="Limit questions per benchmark")
    args = parser.parse_args()

    # Resolve agent config path
    config_path = Path("configs") / args.agents if not Path(args.agents).exists() else Path(args.agents)
    with open(config_path) as f:
        agent_cfgs = yaml.safe_load(f)

    # Create agents
    agents = []
    for cfg in agent_cfgs:
        agent_config = copy.deepcopy(cfg)
        agent_config["player_id"] = 1
        agents.append(create_agent(agent_config))

    # Determine benchmarks to run
    if args.benchmark == "all":
        bench_names = [
            k for k in BENCHMARK_REGISTRY
            if k not in BENCHMARKS_EXCLUDE_FROM_ALL and SPECIAL_KWARGS.get(k) is not None or k not in SPECIAL_KWARGS
        ]
    else:
        bench_names = [b.strip() for b in args.benchmark.split(",")]

    # Filter out similarity_game (needs game config)
    bench_names = [b for b in bench_names if b != "similarity_game"]

    print(f"Agents: {[a.agent_type for a in agents]}")
    print(f"Benchmarks: {bench_names}")
    print(f"Max workers: {args.max_workers}")
    print(f"Cache dir: {CACHE_DIR}\n")

    for bench_name in bench_names:
        print(f"\n{'='*60}")
        print(f"BENCHMARK: {bench_name}")
        print(f"{'='*60}")

        for agent in agents:
            agent_id = agent.agent_type
            cached = load_cached(bench_name, agent_id)
            if cached is not None:
                print(f"  [Cache HIT] {agent_id} — skipping")
                continue

            print(f"  Running {bench_name} on {agent_id}...")

            # Build kwargs
            kwargs = {"max_workers": args.max_workers, "store_traces": True}
            if args.max_items is not None:
                kwargs["max_items"] = args.max_items
            extra = SPECIAL_KWARGS.get(bench_name, {})
            if extra:
                kwargs.update(extra)

            from benchmarks.registry import create_benchmark
            bench = create_benchmark(bench_name, **kwargs)
            result = bench.run(agent)
            save_cached(bench_name, agent_id, result)
            print(f"  Scores: {result.scores}")

    print(f"\nDone! All results cached in {CACHE_DIR}")


if __name__ == "__main__":
    main()
