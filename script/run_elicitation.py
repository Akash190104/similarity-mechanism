#!/usr/bin/env python3
"""Entry point for running similarity elicitation (non-tournament mode).

Elicits strategy choices from agents under similarity framing without
tournament play. Supports single-agent and pairwise elicitation.

Usage:
    python script/run_elicitation.py --config configs/main/similarity_elicitation.yaml

Outputs:
    Writes to `outputs/<yyyy>/<mm>/<dd>/<HH:MM:SS>/` (or `--output-dir`):
    `elicitation_results.json` containing config, seed, single-agent and
    pairwise action distributions (with JS divergence per pair).
"""

import argparse
import copy
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config_loader import ConfigLoader
from src.logger_manager import LOGGER
from src.mechanisms.similarity_elicitation import SimilarityElicitation
from src.registry.agent_registry import create_agent
from src.registry.game_registry import GAME_REGISTRY


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run similarity elicitation (non-tournament mode)"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Config YAML file name"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "pairwise", "both"],
        default="both",
        help="Elicitation mode: single (each agent solo), pairwise (compare pairs), or both",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of trials to run for each agent (default: 1)",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    # Load config
    loader = ConfigLoader()
    config = loader.load_main_config(args.config)

    if args.output_dir:
        LOGGER.set_log_dir(Path(args.output_dir))

    # Setup game
    game_class = GAME_REGISTRY[config["game"]["type"]]
    game = game_class(**config["game"].get("kwargs", {}))

    # Setup agents (all as player_id=1 since no tournament seating needed)
    agents = []
    for cfg in config["agents"]:
        agent_config = copy.deepcopy(cfg)
        agent_config["player_id"] = 1
        agents.append(create_agent(agent_config))

    # Setup similarity elicitation
    sim_config = config.get("similarity", {})
    elicitation = SimilarityElicitation(
        base_game=game,
        similarity_pct=sim_config.get("similarity_pct", 70),
        prompt_mode=sim_config.get("prompt_mode", "percentage"),
        domain=sim_config.get("domain", ""),
        custom_prompt=sim_config.get("custom_prompt", ""),
    )

    results = {
        "config": config,
        "seed": args.seed,
        "trials": args.trials,
        "single_results": [],
        "pairwise_results": [],
    }

    # Single-agent elicitation
    if args.mode in ("single", "both"):
        print("\n" + "=" * 60)
        print("SINGLE-AGENT ELICITATION")
        print("=" * 60)
        for agent in agents:
            for trial in range(args.trials):
                print(f"\nEliciting from {agent.name} (Trial {trial + 1}/{args.trials})...")
                result = elicitation.elicit_single(agent)
                result_dict = result.serialize()
                result_dict["trial"] = trial + 1
                results["single_results"].append(result_dict)
                print(f"  Actions: {result.action_distribution}")

    # Pairwise elicitation
    if args.mode in ("pairwise", "both"):
        print("\n" + "=" * 60)
        print("PAIRWISE ELICITATION")
        print("=" * 60)
        for agent_a, agent_b in itertools.combinations(agents, 2):
            for trial in range(args.trials):
                print(f"\nComparing {agent_a.name} vs {agent_b.name} (Trial {trial + 1}/{args.trials})...")
                result = elicitation.elicit_pairwise(agent_a, agent_b)
                result_dict = result.serialize()
                result_dict["trial"] = trial + 1
                results["pairwise_results"].append(result_dict)
                print(f"  {agent_a.name}: {result.agent_a.action_distribution}")
                print(f"  {agent_b.name}: {result.agent_b.action_distribution}")
                print(f"  JS Divergence: {result.strategy_divergence:.4f}")

    # Save results
    LOGGER.log_record(results, "elicitation_results.json")
    print(f"\nResults saved to {LOGGER.log_dir / 'elicitation_results.json'}")


if __name__ == "__main__":
    main()
