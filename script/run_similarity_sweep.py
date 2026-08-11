#!/usr/bin/env python3
"""Run similarity sweep across multiple similarity percentages.

Delegates to the Similarity mechanism with similarity_source="sweep".
Plays the actual game at each similarity level and computes payoffs.

Usage:
    python script/run_similarity_sweep.py --config main/similarity_testing.yaml --trials 5 --seed 42

Outputs:
    Writes to `outputs/<yyyy>/<mm>/<dd>/<HH:MM:SS>/` (or `--output-dir`):
    `config.json`, `matchup_payoffs.json`, `agent_average_payoff.json`,
    plus the mechanism's `records.jsonl` and `game_log.txt`.
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config_loader import ConfigLoader
from src.logger_manager import LOGGER
from src.mechanisms.similarity import Similarity
from src.registry.agent_registry import create_players_with_player_id
from src.registry.game_registry import GAME_REGISTRY
from src.utils.concurrency import set_default_max_workers


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run similarity sweep across multiple similarity percentages"
    )
    parser.add_argument("--config", type=str, required=True, help="Config YAML file name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--trials", type=int, default=1,
        help="Number of trials per agent per similarity level",
    )
    parser.add_argument(
        "--percentages", type=str, default=None,
        help="Comma-separated similarity percentages (e.g., '0,25,50,75,100')",
    )
    parser.add_argument(
        "--increment", type=int, default=10,
        help="Percentage increment (default: 10)",
    )
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()
    set_seed(args.seed)

    # Determine percentages
    if args.percentages is not None:
        percentages = [int(p.strip()) for p in args.percentages.split(",")]
    else:
        percentages = None  # let mechanism use increment

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

    # Get prompt settings from mechanism or similarity config
    mech_kwargs = {}
    if "mechanism" in config:
        mech_kwargs = (config["mechanism"].get("kwargs", {}) or {}).copy()
    sim_config = config.get("similarity", {})

    prompt_mode = (
        mech_kwargs.get("prompt_mode")
        or sim_config.get("prompt_mode", "percentage_updated")
    )

    # Create mechanism with sweep source
    tournament_workers = concurrency_cfg.get("tournament_workers", 1)
    mechanism = Similarity(
        base_game=game,
        similarity_source="sweep",
        prompt_mode=prompt_mode,
        domain=sim_config.get("domain", mech_kwargs.get("domain", "")),
        custom_prompt=sim_config.get("custom_prompt", mech_kwargs.get("custom_prompt", "")),
        increment=args.increment,
        percentages=percentages,
        trials_per_level=args.trials,
        tournament_workers=tournament_workers,
    )

    # Create players
    players = create_players_with_player_id(config["agents"], game.num_players)

    # Log config
    config["seed"] = args.seed
    LOGGER.log_record(config, "config.json")

    # Run tournament
    payoffs = mechanism.run_tournament(players)
    LOGGER.log_record(record=payoffs.to_json(), file_name="matchup_payoffs.json")

    # Report averages
    print("\n" + "=" * 60)
    print("MODEL AVERAGE PAYOFFS")
    print("=" * 60)
    agent_avg = payoffs.agent_average_payoff()
    for agent, avg_payoff in sorted(agent_avg.items()):
        if avg_payoff is None:
            print(f"  {agent}: Never played")
        else:
            print(f"  {agent}: {avg_payoff:.4f}")
    LOGGER.log_record(agent_avg, "agent_average_payoff.json")
    print("=" * 60)

    print(f"\nResults saved to {LOGGER.log_dir}")


if __name__ == "__main__":
    main()
