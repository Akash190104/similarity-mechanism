"""Inspect AI entry point that loads existing YAML configs.

This lets you run experiments using the same YAML configs as before:

    inspect eval inspect_similarity/tasks/from_config.py \
        -T config_path=main/similarity_testing.yaml \
        -T seed=42

    inspect eval inspect_similarity/tasks/from_config.py \
        -T config_path=main/similarity_elicitation.yaml \
        -T seed=42

The config is loaded via the existing ConfigLoader, which detects whether
it's a tournament config (has mechanism_config) or an elicitation config
(has similarity key), and dispatches accordingly.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Ensure project root is on sys.path (Inspect loads task files directly)
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import Generate, TaskState, solver

from src.config_loader import ConfigLoader
from src.logger_manager import LOGGER


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _disable_tqdm_mp_lock() -> None:
    """Prevent tqdm from spawning a multiprocessing resource tracker."""
    import tqdm.std as _tqdm_std

    if not hasattr(_tqdm_std.TqdmDefaultWriteLock, "_mp_patched"):

        @classmethod
        def _noop_create_mp_lock(cls):
            cls.mp_lock = None

        _tqdm_std.TqdmDefaultWriteLock.create_mp_lock = _noop_create_mp_lock
        _tqdm_std.TqdmDefaultWriteLock._mp_patched = True


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Solver: loads config and dispatches
# ---------------------------------------------------------------------------

@solver
def config_solver(config_path: str, seed: int, max_items: int | None, benchmark: str | None):
    """Load YAML config and run the appropriate pipeline."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        _disable_tqdm_mp_lock()
        _set_seed(seed)

        loader = ConfigLoader()
        config = loader.load_main_config(config_path)

        # Detect experiment type
        has_mechanism = "mechanism" in config
        has_similarity = "similarity" in config

        if has_mechanism:
            # Tournament experiment
            result = _run_tournament(config, seed, max_items, benchmark)
        elif has_similarity:
            # Elicitation experiment
            result = _run_elicitation(config, seed)
        else:
            raise ValueError(
                f"Config at {config_path} doesn't appear to be a tournament "
                f"(needs mechanism_config) or elicitation (needs similarity) config."
            )

        state.store.set("result", result)
        state.output = ModelOutput.from_content(
            model="from_config",
            content=json.dumps(result, indent=2, default=str),
        )
        return state

    return solve


def _run_tournament(config: dict, seed: int, max_items: int | None, benchmark: str | None) -> dict:
    """Run tournament pipeline (mirrors run_experiment.py)."""
    import copy

    from inspect_similarity.agents import create_inspect_agents
    from src.registry.game_registry import GAME_REGISTRY
    from src.registry.mechanism_registry import MECHANISM_REGISTRY
    from src.utils.concurrency import set_default_max_workers

    # Setup concurrency
    concurrency_cfg = config.get("concurrency", {}) or {}
    set_default_max_workers(concurrency_cfg.get("max_workers"))

    # Inject overrides
    mech_kwargs = (config.get("mechanism", {}).get("kwargs") or {}).copy()
    if benchmark is not None:
        mech_kwargs["benchmark"] = benchmark
    if max_items is not None:
        bench_kw = mech_kwargs.get("benchmark_kwargs") or {}
        bench_kw["max_items"] = max_items
        mech_kwargs["benchmark_kwargs"] = bench_kw
        if "subjective_max_items" not in mech_kwargs or mech_kwargs.get("subjective_max_items") is None:
            mech_kwargs["subjective_max_items"] = max_items

    # Create game
    game_class = GAME_REGISTRY[config["game"]["type"]]
    game = game_class(**config["game"].get("kwargs", {}))

    # Create mechanism
    if "tournament_workers" in concurrency_cfg:
        mech_kwargs["tournament_workers"] = concurrency_cfg["tournament_workers"]
    mechanism_class = MECHANISM_REGISTRY[config["mechanism"]["type"]]
    mechanism = mechanism_class(base_game=game, **mech_kwargs)

    # Create players via Inspect
    players = create_inspect_agents(config["agents"], game.num_players)

    # Log config
    config["seed"] = seed
    LOGGER.log_record(config, "config.json")

    # Run tournament
    payoffs = mechanism.run_tournament(players)
    LOGGER.log_record(payoffs.to_json(), "matchup_payoffs.json")

    avg = payoffs.agent_average_payoff()
    LOGGER.log_record(avg, "agent_average_payoff.json")

    return {
        "type": "tournament",
        "agent_average_payoff": avg,
    }


def _run_elicitation(config: dict, seed: int) -> dict:
    """Run elicitation pipeline (mirrors run_elicitation.py)."""
    import copy
    import itertools

    from inspect_similarity.agents import create_inspect_agents
    from src.mechanisms.similarity_elicitation import SimilarityElicitation
    from src.registry.game_registry import GAME_REGISTRY

    # Create game
    game_class = GAME_REGISTRY[config["game"]["type"]]
    game = game_class(**config["game"].get("kwargs", {}))

    # Create agents
    agents = []
    for cfg in config["agents"]:
        agent_config = copy.deepcopy(cfg)
        agent_config["player_id"] = 1
        agents.extend(create_inspect_agents([agent_config], num_players=1))

    # Setup elicitation
    sim_config = config.get("similarity", {})
    elicitation = SimilarityElicitation(
        base_game=game,
        similarity_pct=sim_config.get("similarity_pct", 70),
        prompt_mode=sim_config.get("prompt_mode", "percentage"),
        domain=sim_config.get("domain", ""),
        custom_prompt=sim_config.get("custom_prompt", ""),
    )

    results = {"seed": seed, "single_results": [], "pairwise_results": []}

    # Single elicitation
    for agent in agents:
        result = elicitation.elicit_single(agent)
        results["single_results"].append(result.serialize())

    # Pairwise elicitation
    for agent_a, agent_b in itertools.combinations(agents, 2):
        result = elicitation.elicit_pairwise(agent_a, agent_b)
        results["pairwise_results"].append(result.serialize())

    LOGGER.log_record(results, "elicitation_results.json")
    return {"type": "elicitation", **results}


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

@scorer(metrics=[mean()])
def config_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        result = state.store.get("result", {})
        exp_type = result.get("type", "unknown")

        if exp_type == "tournament":
            avg_payoffs = result.get("agent_average_payoff", {})
            vals = [v for v in avg_payoffs.values() if v is not None]
            value = sum(vals) / max(1, len(vals)) if vals else 0.0
        elif exp_type == "elicitation":
            pairwise = result.get("pairwise_results", [])
            if pairwise:
                value = sum(r.get("strategy_divergence", 0) for r in pairwise) / len(pairwise)
            else:
                value = 0.0
        else:
            value = 0.0

        return Score(
            value=value,
            answer=json.dumps(result, indent=2, default=str),
            metadata=result,
        )

    return score


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

@task
def from_config(
    config_path: str = "main/similarity_testing.yaml",
    seed: int = 42,
    max_items: int | None = None,
    benchmark: str | None = None,
) -> Task:
    """Run an experiment from an existing YAML config file.

    Detects whether the config is a tournament or elicitation config
    and dispatches accordingly. All existing game/mechanism/benchmark
    logic runs unchanged.
    """
    dataset = MemoryDataset(
        samples=[
            Sample(
                input=f"Config: {config_path}",
                target="config",
                metadata={"config_path": config_path, "seed": seed},
            )
        ],
        name=f"from_config_{config_path.replace('/', '_')}",
    )

    return Task(
        dataset=dataset,
        solver=config_solver(
            config_path=config_path,
            seed=seed,
            max_items=max_items,
            benchmark=benchmark,
        ),
        scorer=config_scorer(),
        metadata={
            "config_path": config_path,
            "seed": seed,
            "max_items": max_items,
            "benchmark": benchmark,
        },
    )
