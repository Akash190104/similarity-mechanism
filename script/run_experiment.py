#!/usr/bin/env python3
"""Main entry point for running game-theoretic experiments with a configured mechanism.

Loads a main YAML config from `configs/main/`, instantiates the specified game,
mechanism, and agents, runs the tournament, and executes any configured
evaluations (model averages, evolutionary dynamics, deviation rating).

Usage:
    python script/run_experiment.py --config main/similarity_testing.yaml --seed 42

Outputs:
    Writes to `outputs/<yyyy>/<mm>/<dd>/<HH:MM:SS>/` (or `--output-dir`):
    config.json, matchup_payoffs.json, agent_average_payoff.json,
    population_history.json (if evolutionary_dynamics is enabled),
    deviation_ratings.json (if deviation_rating is enabled),
    plus records.jsonl/game_log.txt and any auto-generated plots.
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from config import DATA_DIR
from src.agents.agent_manager import Agent
from src.config_loader import ConfigLoader
from src.logger_manager import LOGGER
from src.ranking_evaluations.deviation_rating import DeviationRating
from src.ranking_evaluations.matchup_payoffs import MatchupPayoffs
from src.ranking_evaluations.replicator_dynamics import \
    DiscreteReplicatorDynamics
from src.ranking_evaluations.reputation_payoffs import ReputationPayoffs
from src.registry.agent_registry import create_players_with_player_id
from src.registry.game_registry import GAME_REGISTRY
from src.registry.mechanism_registry import MECHANISM_REGISTRY
from src.utils.concurrency import set_default_max_workers

# =============================================================================
# DEFAULT EVALUATION PARAMETERS
# =============================================================================

# Evolutionary Dynamics defaults
DEFAULT_EVOL_INITIAL_POPULATION = "uniform"
DEFAULT_EVOL_STEPS = 25
DEFAULT_EVOL_LR_METHOD = "constant"
DEFAULT_EVOL_LR_NU = 0.1

# Deviation Rating defaults
DEFAULT_DEVIATION_TOLERANCE = 1e-14


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def set_seed(seed=42):
    """Set the random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(filename: str) -> dict:
    """
    Load and parse a YAML configuration file.

    Supports both legacy monolithic configs and modular configs.
    """
    loader = ConfigLoader()
    return loader.load_main_config(filename)


def _resolve_workers(concurrency_cfg: dict, agents_cfg: list, key: str) -> int | None:
    """Resolve a worker count from `key` (explicit) → per-model overrides → flat default.

    Order:
    1. Explicit ``key`` (``max_workers`` or ``tournament_workers``) wins.
    2. Otherwise, sum across agents using ``model_workers[model]`` when present,
       falling back to ``workers_per_model`` per agent. This makes the result
       model-count-invariant *and* respects per-model overrides.
    """
    explicit = concurrency_cfg.get(key)
    if explicit is not None:
        return explicit
    model_workers = concurrency_cfg.get("model_workers") or {}
    per_model_default = concurrency_cfg.get("workers_per_model")
    if not model_workers and per_model_default is None:
        return None
    total = 0
    for cfg in agents_cfg:
        model_id = (cfg.get("llm") or {}).get("model", "")
        cap = model_workers.get(model_id) or per_model_default
        if cap:
            total += cap
    return total or None


def setup_game_and_mechanism(config: dict):
    """
    Initialize game and mechanism from config.

    Args:
        config: Configuration dictionary

    Returns:
        tuple: (game, mechanism)
    """
    game_class = GAME_REGISTRY[config["game"]["type"]]
    mechanism_class = MECHANISM_REGISTRY[config["mechanism"]["type"]]

    game = game_class(**config["game"].get("kwargs", {}))
    mech_kwargs = (config["mechanism"].get("kwargs", {}) or {}).copy()

    # Add tournament_workers from concurrency config
    concurrency_cfg = config.get("concurrency", {}) or {}
    tw = _resolve_workers(concurrency_cfg, config.get("agents", []) or [], "tournament_workers")
    if tw is not None:
        mech_kwargs["tournament_workers"] = tw

    # Forward optional per-model concurrency overrides to the mechanism.
    if concurrency_cfg.get("model_workers"):
        mech_kwargs["model_workers"] = concurrency_cfg["model_workers"]

    mechanism = mechanism_class(base_game=game, **mech_kwargs)

    print(
        f"Running {config['game']['type']} with mechanism {config['mechanism']['type']}.\n"
    )

    return game, mechanism


def _load_matchup_payoffs_from_file(path: Path) -> MatchupPayoffs:
    """Load pre-computed matchup payoffs from a JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Matchup payoff file {path} was not found.")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    print(f"Loaded precomputed matchup payoffs from {path}.")
    return MatchupPayoffs.from_json(payload)


def run_mechanism(
    mechanism, players: list[Agent], args
) -> MatchupPayoffs | ReputationPayoffs:
    """
    Run the mechanism tournament or load pre-computed payoffs.

    Args:
        mechanism: Mechanism instance
        players: List of Agent instances
        args: Command-line arguments

    Returns:
        PayoffsBase instance (either MatchupPayoffs or ReputationPayoffs)
    """
    if args.matchup_payoffs:
        payoffs = _load_matchup_payoffs_from_file(
            DATA_DIR / args.matchup_payoffs
        )
    else:
        print("No precomputed matchup payoff provided; running tournament...")
        payoffs = mechanism.run_tournament(players)
        LOGGER.log_record(
            record=payoffs.to_json(),
            file_name="matchup_payoffs.json",
        )

    return payoffs


def report_model_averages(payoffs: MatchupPayoffs | ReputationPayoffs) -> None:
    """
    Report model average payoffs for all agents.

    This works for both MatchupPayoffs and ReputationPayoffs.

    Args:
        payoffs: PayoffsBase instance
    """
    print("\n" + "="*60)
    print("MODEL AVERAGE PAYOFFS")
    print("="*60)

    agent_avg = payoffs.agent_average_payoff()

    # Pretty print the results
    for agent, avg_payoff in sorted(agent_avg.items()):
        if avg_payoff is None:
            print(f"  {agent}: Never played")
        else:
            print(f"  {agent}: {avg_payoff:.4f}")
    LOGGER.log_record(agent_avg, "agent_average_payoff.json")
    print("="*60 + "\n")


def run_evolutionary_dynamics(
    payoffs: MatchupPayoffs, players: list[Agent], eval_kwargs: dict
) -> None:
    """
    Run evolutionary dynamics evaluation.

    Args:
        payoffs: MatchupPayoffs instance
        players: List of Agent instances
        eval_kwargs: Evaluation-specific kwargs from config
    """
    print("\n" + "="*60)
    print("RUNNING EVOLUTIONARY DYNAMICS")
    print("="*60 + "\n")

    replicator_dynamics = DiscreteReplicatorDynamics(
        players=players,
        matchup_payoffs=payoffs,
    )

    population_history = replicator_dynamics.run_dynamics(
        initial_population=eval_kwargs.get("initial_population", DEFAULT_EVOL_INITIAL_POPULATION),
        steps=int(eval_kwargs.get("steps", DEFAULT_EVOL_STEPS)),
        lr_method=eval_kwargs.get("lr_method", DEFAULT_EVOL_LR_METHOD),
        lr_nu=float(eval_kwargs.get("lr_nu", DEFAULT_EVOL_LR_NU)),
    )

    # Log the population history
    LOGGER.log_record(population_history, "population_history.json")

    print("\n" + "="*60 + "\n")


def run_deviation_rating(payoffs: MatchupPayoffs, eval_kwargs: dict) -> None:
    """
    Run deviation rating evaluation.

    Args:
        payoffs: MatchupPayoffs instance
        eval_kwargs: Evaluation-specific kwargs from config
    """
    print("\n" + "="*60)
    print("RUNNING DEVIATION RATING")
    print("="*60 + "\n")

    # Ensure payoff tensor is built
    if payoffs._payoff_tensor is None:
        payoffs.build_payoff_tensor()

    deviation_rating = DeviationRating(
        matchup_payoffs=payoffs,
        tolerance=float(
            eval_kwargs.get("tolerance", DEFAULT_DEVIATION_TOLERANCE)
        ),
    )

    ratings = deviation_rating.compute_ratings()

    # Pretty print the results
    print("\nDeviation Ratings:")
    for model, rating in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
        print(f"  {model}: {rating:.6f}")

    LOGGER.log_record(ratings, "deviation_ratings.json")

    print("\n" + "="*60 + "\n")


def run_evaluations(
    payoffs: MatchupPayoffs | ReputationPayoffs,
    players: list[Agent],
    config: dict,
) -> None:
    """
    Run all configured evaluations.

    Args:
        payoffs: MatchupPayoffs instance
        players: List of Agent instances
        config: Configuration dictionary
    """
    # Always report model averages first
    report_model_averages(payoffs)

    # For reputation: ONLY model averages possible
    if isinstance(payoffs, ReputationPayoffs):
        print("\n" + "!"*60)
        print("! REPUTATION MECHANISM DETECTED")
        print("! Only model averages available (no tensor-based evaluations)")
        print("!"*60 + "\n")
        return

    # For matchup-based: run configured evaluations
    evaluation_config = config.get("evaluation", {})
    methods = evaluation_config.get("methods", [])

    if not methods:
        print("\nNo evaluation methods configured in config['evaluation']['methods']")
        print("Only model averages will be reported.\n")
        return

    # Run each evaluation method
    for eval_method in methods:
        eval_type = eval_method.get("type")
        eval_kwargs = eval_method.get("kwargs", {})

        if eval_type == "evolutionary_dynamics":
            run_evolutionary_dynamics(payoffs, players, eval_kwargs)
        elif eval_type == "deviation_rating":
            run_deviation_rating(payoffs, eval_kwargs)
        else:
            print(f"\nWARNING: Unknown evaluation type '{eval_type}' - skipping")


def main():
    """
    Main experiment pipeline:
    1. Load config
    2. Setup game and mechanism
    3. Run mechanism (tournament)
    4. Run evaluations
    """
    parser = argparse.ArgumentParser(
        description="Run game-theoretic experiments with configurable evaluations"
    )
    parser.add_argument("--config", type=str, required=True, help="Config YAML file name")
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases figure saving"
    )
    parser.add_argument(
        "--matchup-payoffs",
        type=str,
        default=None,
        help="Path to a JSON file containing precomputed matchup payoffs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory for this experiment (overrides default timestamped directory)"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name for this experiment (used as subdirectory under output-dir)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Auto-plot sweep results after experiment completes"
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Max benchmark questions (injected into mechanism's benchmark_kwargs)"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Override the benchmark name in the mechanism config (e.g. newcomb, cabin, ggb)"
    )
    parser.add_argument(
        "--no-benchmark-cache",
        action="store_true",
        default=False,
        help="Disable automatic benchmark result caching (default: cache at data/benchmark_cache/)"
    )
    parser.add_argument(
        "--no-coop-plot",
        action="store_true",
        default=False,
        help="Skip the per-player cooperation-rate plot with SEM error bars"
    )

    args = parser.parse_args()

    # 0. Set random seed for reproducibility
    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")

    # 1. Setup custom logging directory if provided
    if args.output_dir:
        if args.experiment_name:
            experiment_dir = Path(args.output_dir) / args.experiment_name
        else:
            experiment_dir = Path(args.output_dir)
        LOGGER.set_log_dir(experiment_dir)
        print(f"Logging to: {experiment_dir}")

    # 2. Load config
    config = load_config(filename=args.config)

    # Setup concurrency
    concurrency_cfg = config.get("concurrency", {}) or {}
    set_default_max_workers(
        _resolve_workers(concurrency_cfg, config.get("agents", []) or [], "max_workers")
    )

    # Inject --benchmark override into mechanism kwargs
    if args.benchmark is not None:
        mech_kwargs = config.get("mechanism", {}).get("kwargs") or {}
        mech_kwargs["benchmark"] = args.benchmark
        config.setdefault("mechanism", {})["kwargs"] = mech_kwargs

    # Disable benchmark caching if requested
    if args.no_benchmark_cache:
        mech_kwargs = config.get("mechanism", {}).get("kwargs") or {}
        mech_kwargs["benchmark_cache_dir"] = None
        config.setdefault("mechanism", {})["kwargs"] = mech_kwargs

    # Inject --max-items into mechanism's benchmark_kwargs and subjective_max_items
    if args.max_items is not None:
        mech_kwargs = config.get("mechanism", {}).get("kwargs") or {}
        bench_kwargs = mech_kwargs.get("benchmark_kwargs") or {}
        bench_kwargs["max_items"] = args.max_items
        mech_kwargs["benchmark_kwargs"] = bench_kwargs
        # Also cap how many questions the subjective judge sees
        if "subjective_max_items" not in mech_kwargs or mech_kwargs["subjective_max_items"] is None:
            mech_kwargs["subjective_max_items"] = args.max_items
        config.setdefault("mechanism", {}).setdefault("kwargs", {}).update(mech_kwargs)

    # 3. Setup game, agents and mechanism
    game, mechanism = setup_game_and_mechanism(config)
    players = create_players_with_player_id(config["agents"], game.num_players)

    # Log effective configuration including concurrency settings and seed
    config["seed"] = args.seed
    LOGGER.log_record(config, "config.json")

    # 4. Run mechanism (tournament)
    payoffs = run_mechanism(mechanism, players, args)

    # 5. Run evaluations
    run_evaluations(payoffs, players, config)

    # 6. Auto-plot if requested
    if args.plot:
        sweep_file = LOGGER.log_dir / "sweep_results_by_percentage.json"
        if sweep_file.exists():
            from script.plot_trust_game_sweep import extract_payoffs, load_sweep, plot_matchup_payoffs, plot_model_payoffs
            data = load_sweep(sweep_file)
            percentages, matchup_payoffs, model_payoffs = extract_payoffs(data)
            plot_matchup_payoffs(percentages, matchup_payoffs, LOGGER.log_dir)
            plot_model_payoffs(percentages, model_payoffs, LOGGER.log_dir)
        else:
            print(f"No sweep results found at {sweep_file}, skipping plots.")

    # 7. Cooperation rate with SEM error bars (per-similarity for sweeps,
    # per-player otherwise)
    has_sweep = (LOGGER.log_dir / "sweep_results_by_percentage.json").exists()
    has_bench_sweep = (LOGGER.log_dir / "benchmark_sweep_results.json").exists()
    has_records = (LOGGER.log_dir / "records.jsonl").exists()
    if not args.no_coop_plot and (has_bench_sweep or has_sweep or has_records):
        from script.plot_cooperation_with_sem import run as run_coop_plot
        try:
            outputs = run_coop_plot(LOGGER.log_dir)
            print(f"Cooperation summary: {outputs['csv']}")
            print(f"Cooperation plot:    {outputs['plot']}")
        except Exception as exc:
            print(f"WARNING: cooperation plot failed: {exc}")

    return LOGGER.log_dir


if __name__ == "__main__":
    main()
