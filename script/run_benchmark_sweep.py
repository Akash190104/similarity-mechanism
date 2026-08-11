#!/usr/bin/env python3
"""Sweep similarity percentage across all benchmarks to produce a cooperation heatmap.

For each of N benchmarks and configurable similarity levels (default 0-100% in 10% steps),
tells each agent that it has been measured at X% similarity on that benchmark,
then records their action distribution (solo elicitation, no paired gameplay).

This is a "spoofed" experiment — no actual benchmark is run. The similarity score is fake.

Usage:
    python script/run_benchmark_sweep.py --config main/benchmark_similarity_sweep.yaml --trials 3
    python script/run_benchmark_sweep.py --config main/benchmark_similarity_sweep.yaml --trials 1 --benchmarks gpqa,newcomb --percentages 0,50,100

Outputs:
    Writes to `outputs/<yyyy>/<mm>/<dd>/<HH:MM:SS>/` (or `--output-dir`):
    `benchmark_sweep_results.json` (results grouped by benchmark and
    similarity percentage) plus heatmap PNGs from `plot_similarity_heatmap`.
"""

import argparse
import copy
import random
import sys
from pathlib import Path

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.registry import BENCHMARK_INFO, BENCHMARKS_EXCLUDE_FROM_ALL, build_benchmark_prompt
from src.config_loader import ConfigLoader
from src.logger_manager import LOGGER
from src.mechanisms.similarity_elicitation import SimilarityElicitation
from src.registry.agent_registry import create_agent
from src.registry.game_registry import GAME_REGISTRY
from src.utils.concurrency import run_tasks, set_default_max_workers


def set_seed(seed: int = 42) -> None:
    if torch is not None:
        torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep similarity % across benchmarks for a cooperation heatmap"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Config YAML file name"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Trials per (agent, benchmark, percentage) combo",
    )
    parser.add_argument(
        "--percentages",
        type=str,
        default=None,
        help="Comma-separated similarity percentages (e.g. '0,25,50,75,100')",
    )
    parser.add_argument(
        "--increment",
        type=int,
        default=10,
        help="Percentage increment (default: 10)",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default=None,
        help="Comma-separated benchmark keys to test (default: all)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Custom output directory"
    )

    args = parser.parse_args()
    set_seed(args.seed)

    # Determine percentages
    if args.percentages is not None:
        percentages = [int(p.strip()) for p in args.percentages.split(",")]
    else:
        percentages = list(range(0, 101, args.increment))
        if 100 not in percentages:
            percentages.append(100)

    # Determine benchmarks
    if args.benchmarks is not None:
        benchmark_keys = [b.strip() for b in args.benchmarks.split(",")]
        for b in benchmark_keys:
            if b not in BENCHMARK_INFO:
                parser.error(f"Unknown benchmark: {b}")
    else:
        benchmark_keys = [k for k in BENCHMARK_INFO if k not in BENCHMARKS_EXCLUDE_FROM_ALL]

    # Load config
    loader = ConfigLoader()
    config = loader.load_main_config(args.config)

    concurrency_cfg = config.get("concurrency", {}) or {}
    set_default_max_workers(concurrency_cfg.get("max_workers"))

    if args.output_dir:
        LOGGER.set_log_dir(Path(args.output_dir))

    # Setup game
    game_class = GAME_REGISTRY[config["game"]["type"]]
    game = game_class(**config["game"].get("kwargs", {}))

    # Setup agents (each independently, player_id=1)
    agents = []
    for cfg in config["agents"]:
        agent_config = copy.deepcopy(cfg)
        agent_config["player_id"] = 1
        agents.append(create_agent(agent_config))

    # Build flat task list: (agent, benchmark, pct, trial)
    tasks = [
        (agent, bench, pct, trial + 1)
        for agent in agents
        for bench in benchmark_keys
        for pct in percentages
        for trial in range(args.trials)
    ]

    total = len(tasks)
    print(f"\n{'=' * 80}")
    print(
        f"BENCHMARK SIMILARITY SWEEP: {len(benchmark_keys)} benchmarks × "
        f"{len(percentages)} levels × {len(agents)} agents × {args.trials} trials"
    )
    print(f"Total tasks: {total}")
    print(f"{'=' * 80}")

    pbar = tqdm(total=total, desc="Benchmark sweep", unit="task")

    def run_single(task):
        agent, bench, pct, trial_num = task
        custom_prompt = build_benchmark_prompt(bench)
        elicitation = SimilarityElicitation(
            base_game=game,
            similarity_pct=pct,
            prompt_mode="custom",
            custom_prompt=custom_prompt,
        )
        result = elicitation.elicit_single(agent)
        coop_rate = result.action_distribution.get("A0", 0)
        pbar.update(1)
        pbar.set_postfix(
            {"bench": bench[:8], "pct": f"{pct}%", "agent": agent.name[:15]}
        )
        return {
            "agent_name": agent.name,
            "benchmark": bench,
            "similarity_pct": pct,
            "trial": trial_num,
            "action_distribution": result.action_distribution,
            "cooperation_rate": coop_rate,
            "trace_id": result.trace_id,
        }

    all_results = run_tasks(tasks, run_single)
    pbar.close()

    # Organize results by benchmark and percentage
    from collections import defaultdict

    results_by_benchmark: dict[str, dict[str, list]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in all_results:
        bench = r["benchmark"]
        pct = r["similarity_pct"]
        results_by_benchmark[bench][str(pct)].append(r)

    output = {
        "config": config,
        "seed": args.seed,
        "trials": args.trials,
        "percentages": percentages,
        "benchmarks": benchmark_keys,
        "results_by_benchmark": {
            bench: {"results_by_percentage": pct_data}
            for bench, pct_data in results_by_benchmark.items()
        },
    }

    LOGGER.log_record(output, "benchmark_sweep_results.json")
    results_path = LOGGER.log_dir / "benchmark_sweep_results.json"
    print(f"\nResults saved to {results_path}")

    # Generate heatmap plots
    from script.plot import plot_benchmark_sweep

    print("\nGenerating heatmaps...")
    plot_benchmark_sweep(argparse.Namespace(
        results=str(results_path),
        output_dir=str(LOGGER.log_dir),
        label=None,
        flip="auto",
    ))
    print("\nDone!")


if __name__ == "__main__":
    main()
