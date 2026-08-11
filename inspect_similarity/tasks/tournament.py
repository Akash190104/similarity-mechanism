"""Inspect AI task that wraps the existing tournament pipeline.

Usage:
    # Direct CLI usage with explicit params
    inspect eval inspect_similarity/tasks/tournament.py \
        -T game=PrisonersDilemma \
        -T mechanism=Similarity \
        -T agents_config=agents/cheap_llms_3.yaml \
        -T seed=42

    # With mechanism kwargs (JSON string)
    inspect eval inspect_similarity/tasks/tournament.py \
        -T game=PrisonersDilemma \
        -T mechanism=Similarity \
        -T mechanism_kwargs='{"similarity_source":"sweep","prompt_mode":"percentage_updated"}' \
        -T seed=42
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

from inspect_similarity.agents import create_inspect_agents
from src.config_loader import ConfigLoader
from src.logger_manager import LOGGER
from src.registry.game_registry import GAME_REGISTRY
from src.registry.mechanism_registry import MECHANISM_REGISTRY
from src.utils.concurrency import set_default_max_workers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _disable_tqdm_mp_lock() -> None:
    """Prevent tqdm from spawning a multiprocessing resource tracker.

    Inside Inspect's async sandbox the file-descriptor table is restricted,
    so ``multiprocessing.resource_tracker.ensure_running()`` fails with
    ``ValueError: bad value(s) in fds_to_keep``.  Setting ``mp_lock = None``
    makes tqdm fall back to its threading lock only.
    """
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
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def _load_agents_yaml(agents_config: str) -> list[dict]:
    """Load agent configs from a YAML path (relative to configs/)."""
    import yaml
    from config import CONFIG_DIR
    from pathlib import Path

    path = Path(CONFIG_DIR) / agents_config
    if not path.exists():
        path = Path(agents_config)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Solver: runs the full tournament
# ---------------------------------------------------------------------------

@solver
def tournament_solver(
    game_type: str,
    game_kwargs: dict,
    mechanism_type: str,
    mechanism_kwargs: dict,
    agent_cfgs: list[dict],
    concurrency_cfg: dict,
    seed: int,
):
    """Solver that runs the mechanism tournament and stores results in state."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        _disable_tqdm_mp_lock()
        _set_seed(seed)
        set_default_max_workers(concurrency_cfg.get("max_workers"))

        # Create game
        game_class = GAME_REGISTRY[game_type]
        game = game_class(**game_kwargs)

        # Create mechanism
        mech_kw = (mechanism_kwargs or {}).copy()
        if "tournament_workers" in concurrency_cfg:
            mech_kw["tournament_workers"] = concurrency_cfg["tournament_workers"]
        mechanism_class = MECHANISM_REGISTRY[mechanism_type]
        mechanism = mechanism_class(base_game=game, **mech_kw)

        # Create players via Inspect adapter
        players = create_inspect_agents(agent_cfgs, game.num_players)

        # Run tournament
        payoffs = mechanism.run_tournament(players)

        # Store results in TaskState
        avg = payoffs.agent_average_payoff()
        state.store.set("matchup_payoffs", payoffs.to_json())
        state.store.set("agent_average_payoff", avg)

        # Log to disk as well (existing logger)
        LOGGER.log_record(payoffs.to_json(), "matchup_payoffs.json")
        LOGGER.log_record(avg, "agent_average_payoff.json")

        # Set output for scorer
        state.output = ModelOutput.from_content(
            model="tournament", content=json.dumps(avg, indent=2)
        )

        return state

    return solve


# ---------------------------------------------------------------------------
# Scorer: extracts metrics from tournament results
# ---------------------------------------------------------------------------

@scorer(metrics=[mean()])
def tournament_scorer():
    """Score based on average payoff across agents."""

    async def score(state: TaskState, target: Target) -> Score:
        avg_payoffs = state.store.get("agent_average_payoff", {})

        if not avg_payoffs:
            return Score(value=0.0, explanation="No payoffs recorded")

        overall_avg = sum(
            v for v in avg_payoffs.values() if v is not None
        ) / max(1, len([v for v in avg_payoffs.values() if v is not None]))

        return Score(
            value=overall_avg,
            answer=json.dumps(avg_payoffs, indent=2),
            metadata={
                "agent_average_payoff": avg_payoffs,
                "matchup_payoffs": state.store.get("matchup_payoffs"),
            },
        )

    return score


# ---------------------------------------------------------------------------
# Task definition
# ---------------------------------------------------------------------------

@task
def similarity_tournament(
    # Game
    game: str = "PrisonersDilemma",
    game_kwargs: str = "{}",
    # Mechanism
    mechanism: str = "Similarity",
    mechanism_kwargs: str = "{}",
    # Agents
    agents_config: str = "agents/cheap_llms_3.yaml",
    # Experiment
    seed: int = 42,
    # Concurrency
    max_workers: int | None = None,
    tournament_workers: int = 1,
) -> Task:
    """Run a game-theoretic tournament via Inspect AI.

    This wraps the existing ``mechanism.run_tournament()`` pipeline.
    All game/mechanism/benchmark logic runs unchanged — only the LLM
    layer is routed through Inspect's model abstraction.
    """
    game_kw = json.loads(game_kwargs) if isinstance(game_kwargs, str) else game_kwargs
    mech_kw = json.loads(mechanism_kwargs) if isinstance(mechanism_kwargs, str) else mechanism_kwargs

    agent_cfgs = _load_agents_yaml(agents_config)
    concurrency_cfg = {
        "max_workers": max_workers,
        "tournament_workers": tournament_workers,
    }

    # Single-sample dataset — the tournament IS the evaluation unit
    dataset = MemoryDataset(
        samples=[
            Sample(
                input=f"Tournament: {game} + {mechanism}",
                target="tournament",
                metadata={
                    "game": game,
                    "mechanism": mechanism,
                    "seed": seed,
                },
            )
        ],
        name=f"{game}_{mechanism}_tournament",
    )

    return Task(
        dataset=dataset,
        solver=tournament_solver(
            game_type=game,
            game_kwargs=game_kw,
            mechanism_type=mechanism,
            mechanism_kwargs=mech_kw,
            agent_cfgs=agent_cfgs,
            concurrency_cfg=concurrency_cfg,
            seed=seed,
        ),
        scorer=tournament_scorer(),
        metadata={
            "game": game,
            "mechanism": mechanism,
            "seed": seed,
            "agents_config": agents_config,
        },
    )
