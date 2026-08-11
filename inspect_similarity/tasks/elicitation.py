"""Inspect AI task for similarity elicitation (non-tournament mode).

Elicits strategy choices from agents under similarity framing without
tournament play. Supports single-agent and pairwise modes.

Usage:
    inspect eval inspect_similarity/tasks/elicitation.py \
        -T game=PrisonersDilemma \
        -T agents_config=agents/cheap_llms_3.yaml \
        -T mode=pairwise \
        -T seed=42
"""

from __future__ import annotations

import copy
import itertools
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Ensure project root is on sys.path (Inspect loads task files directly)
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import Generate, TaskState, solver

from config import CONFIG_DIR
from inspect_similarity.agents import create_inspect_agents
from src.logger_manager import LOGGER
from src.mechanisms.similarity_elicitation import SimilarityElicitation
from src.registry.game_registry import GAME_REGISTRY


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


def _load_agents_yaml(agents_config: str) -> list[dict]:
    path = Path(CONFIG_DIR) / agents_config
    if not path.exists():
        path = Path(agents_config)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

@solver
def elicitation_solver(
    game_type: str,
    game_kwargs: dict,
    agent_cfgs: list[dict],
    similarity_pct: int,
    prompt_mode: str,
    domain: str,
    custom_prompt: str,
    mode: str,
    trials: int,
    seed: int,
):
    """Run similarity elicitation on agents."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        _disable_tqdm_mp_lock()
        _set_seed(seed)

        # Create game
        game_class = GAME_REGISTRY[game_type]
        game = game_class(**game_kwargs)

        # Create agents (all as player_id=1)
        agents = []
        for cfg in agent_cfgs:
            agent_config = copy.deepcopy(cfg)
            agent_config["player_id"] = 1
            agents.extend(create_inspect_agents([agent_config], num_players=1))

        # Setup elicitation
        elicitation = SimilarityElicitation(
            base_game=game,
            similarity_pct=similarity_pct,
            prompt_mode=prompt_mode,
            domain=domain,
            custom_prompt=custom_prompt,
        )

        results = {
            "seed": seed,
            "trials": trials,
            "single_results": [],
            "pairwise_results": [],
        }

        # Single-agent elicitation
        if mode in ("single", "both"):
            for agent in agents:
                for trial in range(trials):
                    result = elicitation.elicit_single(agent)
                    result_dict = result.serialize()
                    result_dict["trial"] = trial + 1
                    results["single_results"].append(result_dict)

        # Pairwise elicitation
        if mode in ("pairwise", "both"):
            for agent_a, agent_b in itertools.combinations(agents, 2):
                for trial in range(trials):
                    result = elicitation.elicit_pairwise(agent_a, agent_b)
                    result_dict = result.serialize()
                    result_dict["trial"] = trial + 1
                    results["pairwise_results"].append(result_dict)

        state.store.set("elicitation_results", results)
        LOGGER.log_record(results, "elicitation_results.json")

        state.output = ModelOutput.from_content(
            model="elicitation",
            content=json.dumps(results, indent=2, default=str),
        )
        return state

    return solve


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

@scorer(metrics=[mean()])
def elicitation_scorer():
    """Score based on JS divergence (lower = more similar strategies)."""

    async def score(state: TaskState, target: Target) -> Score:
        results = state.store.get("elicitation_results", {})
        pairwise = results.get("pairwise_results", [])

        if pairwise:
            avg_div = sum(r["strategy_divergence"] for r in pairwise) / len(pairwise)
        else:
            avg_div = 0.0

        return Score(
            value=avg_div,
            answer=json.dumps(results, indent=2, default=str),
            metadata=results,
        )

    return score


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

@task
def similarity_elicitation(
    game: str = "PrisonersDilemma",
    game_kwargs: str = "{}",
    agents_config: str = "agents/cheap_llms_3.yaml",
    similarity_pct: int = 70,
    prompt_mode: str = "percentage_updated",
    domain: str = "",
    custom_prompt: str = "",
    mode: str = "both",
    trials: int = 1,
    seed: int = 42,
) -> Task:
    """Elicit agent strategies under similarity framing (non-tournament)."""
    game_kw = json.loads(game_kwargs) if isinstance(game_kwargs, str) else game_kwargs
    agent_cfgs = _load_agents_yaml(agents_config)

    dataset = MemoryDataset(
        samples=[
            Sample(
                input=f"Elicitation: {game} @ {similarity_pct}% similarity",
                target="elicitation",
                metadata={
                    "game": game,
                    "similarity_pct": similarity_pct,
                    "mode": mode,
                    "seed": seed,
                },
            )
        ],
        name=f"elicitation_{game}_{similarity_pct}pct",
    )

    return Task(
        dataset=dataset,
        solver=elicitation_solver(
            game_type=game,
            game_kwargs=game_kw,
            agent_cfgs=agent_cfgs,
            similarity_pct=similarity_pct,
            prompt_mode=prompt_mode,
            domain=domain,
            custom_prompt=custom_prompt,
            mode=mode,
            trials=trials,
            seed=seed,
        ),
        scorer=elicitation_scorer(),
        metadata={
            "game": game,
            "similarity_pct": similarity_pct,
            "prompt_mode": prompt_mode,
            "mode": mode,
            "seed": seed,
        },
    )
