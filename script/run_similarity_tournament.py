#!/usr/bin/env python3
"""Run a three-phase similarity tournament.

Phase 1 (Elicit): Ask each agent for their strategy at multiple similarity levels.
Phase 2 (Compute): Use JS divergence to compute one similarity score per agent pair.
Phase 3 (Validate): Play the actual game at the computed similarity.

Delegates to:
- SimilarityGame benchmark for phases 1-2 (elicit + compute similarity)
- Similarity mechanism with similarity_source="fixed" for phase 3 (validate)

Usage:
    python script/run_similarity_tournament.py --config main/similarity_testing.yaml --seed 42

Outputs:
    Writes to `outputs/<yyyy>/<mm>/<dd>/<HH:MM:SS>/` (or `--output-dir`):
    `similarity_computation.json` (per-pair similarity + elicitation results)
    and `similarity_tournament_results.json` (config, seed, computed
    similarity per pair), alongside the game records and log.
"""

from __future__ import annotations

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.similarity_game import SimilarityGame
from src.config_loader import ConfigLoader
from src.logger_manager import LOGGER
from src.mechanisms.similarity import Similarity
from src.registry.agent_registry import create_agent, create_players_with_player_id
from src.registry.game_registry import GAME_REGISTRY
from src.utils.concurrency import set_default_max_workers


def set_seed(seed: int = 42) -> None:
    if torch is not None:
        torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    parser = argparse.ArgumentParser(description="Run three-phase similarity tournament")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to main config file (relative to configs/)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--elicitation-trials", type=int, default=5,
        help="Trials per similarity level in elicitation phase",
    )
    parser.add_argument(
        "--validation-trials", type=int, default=10,
        help="Trials per matchup in validation phase",
    )
    parser.add_argument(
        "--increment", type=int, default=10,
        help="Similarity percentage increment (default: 10)",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-workers", type=int, default=None)

    args = parser.parse_args()
    set_seed(args.seed)

    # Load config
    loader = ConfigLoader()
    config = loader.load_main_config(args.config)

    concurrency_cfg = config.get("concurrency", {}) or {}
    max_workers = args.max_workers or concurrency_cfg.get("max_workers")
    set_default_max_workers(max_workers)

    if args.output_dir:
        LOGGER.set_log_dir(Path(args.output_dir))

    # Setup game
    game_class = GAME_REGISTRY[config["game"]["type"]]
    game = game_class(**config["game"].get("kwargs", {}))

    # Get prompt settings
    mech_kwargs = {}
    if "mechanism" in config:
        mech_kwargs = (config["mechanism"].get("kwargs", {}) or {}).copy()
    prompt_mode = mech_kwargs.get("prompt_mode", "percentage_updated")

    # Setup agents for benchmark elicitation
    agent_configs = []
    for cfg in config["agents"]:
        agent_config = copy.deepcopy(cfg)
        agent_config["player_id"] = 1
        agent_configs.append(agent_config)

    agents = [create_agent(cfg) for cfg in agent_configs]

    print(f"\n{'=' * 80}")
    print("SIMILARITY TOURNAMENT (three-phase)")
    print(f"{'=' * 80}")
    print(f"Game: {config['game']['type']}")
    print(f"Agents: {len(agents)}")
    print(f"Elicitation trials per level: {args.elicitation_trials}")
    print(f"Validation trials per pair: {args.validation_trials}")
    print(f"Increment: {args.increment}%")
    print(f"{'=' * 80}\n")

    # ── Phase 1 & 2: Elicit + Compute similarity ────────────────────────
    bench = SimilarityGame(
        base_game=game,
        prompt_mode=prompt_mode,
        increment=args.increment,
        trials=args.elicitation_trials,
        max_workers=max_workers,
    )

    print("PHASE 1: Eliciting strategies at all similarity levels...")
    bench_results = {}
    for agent in agents:
        print(f"  Running on {agent.name}...")
        bench_results[agent.name] = bench.run(agent)

    print(f"\n{'=' * 80}")
    print("PHASE 2: Computing pairwise similarity from elicited distributions...")
    print(f"{'=' * 80}")

    agent_names = [a.name for a in agents]
    sim_matrix: dict[tuple[str, str], float] = {}
    for i, name_a in enumerate(agent_names):
        for j, name_b in enumerate(agent_names):
            if i <= j:
                if i == j:
                    sim = 100.0
                else:
                    sim = bench.compute_similarity(
                        bench_results[name_a], bench_results[name_b]
                    )
                sim_matrix[(name_a, name_b)] = sim
                sim_matrix[(name_b, name_a)] = sim
                if i != j:
                    print(f"  {name_a} vs {name_b}: {sim:.2f}%")

    LOGGER.log_record(
        {
            "computed_similarity": {
                f"{a} vs {b}": s
                for (a, b), s in sorted(sim_matrix.items())
                if a <= b
            },
            "elicitation_results": {
                name: result.serialize() for name, result in bench_results.items()
            },
        },
        "similarity_computation.json",
    )

    # ── Phase 3: Validation ──────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("PHASE 3: Validation (playing games at computed similarity)...")
    print(f"{'=' * 80}")

    tournament_workers = concurrency_cfg.get("tournament_workers", 1)

    # For each unique pair, run validation at their computed similarity
    for i, name_a in enumerate(agent_names):
        for j, name_b in enumerate(agent_names):
            if i > j:
                continue

            pair_sim = sim_matrix.get((name_a, name_b), 50.0)
            pair_sim_int = int(round(pair_sim))

            print(f"\n  {name_a} vs {name_b} @ {pair_sim_int}% "
                  f"({args.validation_trials} trials)...")

            # Create players with proper player_ids for this pair
            pair_players = []
            for cfg in config["agents"]:
                agent_cfg = copy.deepcopy(cfg)
                test_agent = create_agent({**agent_cfg, "player_id": 1})
                if test_agent.name == name_a and not any(
                    p.name == name_a for p in pair_players
                ):
                    agent_cfg["player_id"] = 1
                    pair_players.append(create_agent(agent_cfg))
                elif test_agent.name == name_b and not any(
                    p.name == name_b for p in pair_players
                ):
                    agent_cfg["player_id"] = 2
                    pair_players.append(create_agent(agent_cfg))

            if len(pair_players) < 2:
                print(f"    Skipping (couldn't form pair)")
                continue

            mechanism = Similarity(
                base_game=game,
                similarity_source="sweep",
                similarity_pct=pair_sim_int,
                prompt_mode=prompt_mode,
                num_rounds=1,
                percentages=[pair_sim_int],
                trials_per_level=args.validation_trials,
                tournament_workers=tournament_workers,
            )

            payoffs = mechanism.run_tournament(pair_players)
            avg = payoffs.agent_average_payoff()
            for agent_name, avg_payoff in sorted(avg.items()):
                print(f"    {agent_name}: {avg_payoff:.4f}" if avg_payoff else f"    {agent_name}: N/A")

    # Save all results
    all_results = {
        "config": config,
        "seed": args.seed,
        "elicitation_trials": args.elicitation_trials,
        "validation_trials": args.validation_trials,
        "computed_similarity": {
            f"{a} vs {b}": s
            for (a, b), s in sorted(sim_matrix.items())
            if a <= b
        },
    }

    LOGGER.log_record(all_results, "similarity_tournament_results.json")

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print("\nComputed similarity per pair:")
    for (a, b), sim in sorted(sim_matrix.items()):
        if a <= b and a != b:
            print(f"  {a} vs {b}: {sim:.2f}%")

    print(f"\nResults saved to {LOGGER.log_dir}")


if __name__ == "__main__":
    main()
