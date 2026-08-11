#!/usr/bin/env python3
"""Find grounded similarity for LLM agent pairs.

Sweeps across similarity percentages, measures similarity via elicitation,
and computes the weighted average as the grounded similarity.

Usage:
    python script/run_fixed_point.py --config main/fixed_point_search.yaml --trials 3
    python script/run_fixed_point.py --config main/fixed_point_search.yaml --metric cooperation
    python script/run_fixed_point.py --config main/fixed_point_search.yaml --metric both

Outputs:
    Writes to `outputs/<yyyy>/<mm>/<dd>/<HH:MM:SS>/` (or `--output-dir`):
    `fixed_point_results.json` (per-pair sweep, primary fixed point, and
    validation stats) plus plots produced by `script.plot_fixed_point.plot_all`.
"""

import argparse
import copy
import itertools
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PROJECT_DIR
from script.plot import plot_fixed_point
from src.config_loader import ConfigLoader
from src.logger_manager import LOGGER
from script.fixed_point import FixedPointFinder
from src.registry.agent_registry import create_agent
from src.registry.game_registry import GAME_REGISTRY
from src.utils.concurrency import run_tasks, set_default_max_workers


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find fixed-point similarity for agent pairs"
    )
    parser.add_argument("--config", type=str, required=True, help="Config YAML file name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--trials", type=int, default=3,
        help="Trials per sweep evaluation point (default: 3)",
    )
    parser.add_argument(
        "--validation-trials", type=int, default=5,
        help="Trials for validating the fixed point (default: 5)",
    )
    parser.add_argument(
        "--increment", type=int, default=10,
        help="Percentage increment for sweep (default: 10)",
    )
    parser.add_argument(
        "--percentages", type=str, default=None,
        help="Comma-separated similarity percentages (overrides --increment)",
    )
    parser.add_argument(
        "--metric", type=str, default="agreement",
        choices=["agreement", "cooperation", "both"],
        help="Similarity metric: agreement (action agreement rate), "
             "cooperation (avg cooperation rate), or both (default: agreement)",
    )
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()
    set_seed(args.seed)

    # Determine percentages
    if args.percentages is not None:
        percentages = [int(p.strip()) for p in args.percentages.split(",")]
    else:
        percentages = list(range(0, 101, args.increment))
        if 100 not in percentages:
            percentages.append(100)

    # Load config
    loader = ConfigLoader()
    config = loader.load_main_config(args.config)

    # Setup concurrency
    concurrency_cfg = config.get("concurrency", {}) or {}
    set_default_max_workers(concurrency_cfg.get("max_workers"))

    if args.output_dir:
        LOGGER.set_log_dir(Path(args.output_dir))

    # Setup game
    game_class = GAME_REGISTRY[config["game"]["type"]]
    game = game_class(**config["game"].get("kwargs", {}))

    # Setup agents
    agents = []
    for cfg in config["agents"]:
        agent_config = copy.deepcopy(cfg)
        agent_config["player_id"] = 1
        agents.append(create_agent(agent_config))

    # Similarity config
    sim_config = config.get("similarity", {})

    # Create finder
    finder = FixedPointFinder(
        base_game=game,
        prompt_mode=sim_config.get("prompt_mode", "percentage"),
        trials_per_point=args.trials,
        metric=args.metric,
        domain=sim_config.get("domain", ""),
        custom_prompt=sim_config.get("custom_prompt", ""),
    )

    # Generate all agent pairs (self-play + cross-play)
    pairs = list(itertools.combinations_with_replacement(range(len(agents)), 2))

    print("\n" + "=" * 80)
    print("GROUNDED SIMILARITY SEARCH")
    print(f"  Agents: {[a.name for a in agents]}")
    print(f"  Pairs: {len(pairs)} (self + cross)")
    print(f"  Percentages: {percentages}")
    print(f"  Trials per point: {args.trials}")
    print(f"  Validation trials: {args.validation_trials}")
    print(f"  Metric: {args.metric}")
    total_calls = len(pairs) * len(percentages) * args.trials * 2
    print(f"  Estimated API calls (sweep): {total_calls}")
    print("=" * 80)

    all_results = {
        "config": config,
        "seed": args.seed,
        "trials_per_point": args.trials,
        "validation_trials": args.validation_trials,
        "percentages": percentages,
        "pairs": [],
    }

    # Process each pair
    for pair_idx, (i, j) in enumerate(pairs):
        agent_a = agents[i]
        agent_b = agents[j]
        pair_label = f"{agent_a.name} vs {agent_b.name}" if i != j else f"{agent_a.name} (self-play)"

        print(f"\n{'=' * 60}")
        print(f"Pair {pair_idx + 1}/{len(pairs)}: {pair_label}")
        print(f"{'=' * 60}")

        # Run full pipeline: sweep -> weighted average -> validate
        result = finder.find_fixed_point(
            agent_a, agent_b,
            percentages=percentages,
            validation_trials=args.validation_trials,
        )

        print(f"\n  Grounded similarity: {result.primary_fixed_point:.1f}%")
        if result.metric == "both" and result.primary_fixed_point_alt is not None:
            print(f"  Grounded similarity (cooperation): {result.primary_fixed_point_alt:.1f}%")
        if result.validation_rate is not None:
            print(f"  Validation: {result.validation_rate:.1f}% "
                  f"(+/- {result.validation_std:.1f}%) "
                  f"@ told similarity {result.validation_similarity_pct}%")

        all_results["pairs"].append(result.serialize())

    # Save results
    LOGGER.log_record(all_results, "fixed_point_results.json")
    print(f"\n{'=' * 80}")
    print(f"Results saved to: {LOGGER.log_dir / 'fixed_point_results.json'}")
    print(f"{'=' * 80}")

    # Generate plots
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)
    plot_fixed_point(argparse.Namespace(
        results_json=str(LOGGER.log_dir / "fixed_point_results.json"),
        output_dir=str(LOGGER.log_dir),
    ))

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for pair_data in all_results["pairs"]:
        label = f"{_simplify(pair_data['agent_a'])} vs {_simplify(pair_data['agent_b'])}"
        fp = pair_data.get("primary_fixed_point")
        val = pair_data.get("validation", {})
        val_rate = f"{val['rate_mean']:.1f}%" if val and val.get("rate_mean") is not None else "N/A"
        print(f"  {label}: grounded similarity = {fp:.1f}%, validated at {val_rate}")
    print("=" * 80)


def _simplify(name: str) -> str:
    return name.split("/")[-1].split("(")[0]


if __name__ == "__main__":
    main()
