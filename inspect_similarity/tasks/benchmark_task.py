"""Inspect AI tasks for running benchmarks and computing pairwise similarity.

Usage:
    # Run a single benchmark
    inspect eval inspect_similarity/tasks/benchmark_task.py@benchmark_eval \
        -T benchmark_name=newcomb \
        -T agents_config=agents/cheap_llms_3.yaml \
        -T max_items=20

    # Run a battery of benchmarks
    inspect eval inspect_similarity/tasks/benchmark_task.py@benchmark_battery \
        -T benchmarks=newcomb,cabin,dilemmas \
        -T agents_config=agents/cheap_llms_3.yaml \
        -T max_items=20
"""

from __future__ import annotations

import copy
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

from benchmarks.base import Benchmark, BenchmarkResult
from benchmarks.registry import create_benchmark as registry_create_benchmark
from config import CONFIG_DIR
from inspect_similarity.agents import create_inspect_agents
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


def _load_agents_yaml(agents_config: str) -> list[dict]:
    path = Path(CONFIG_DIR) / agents_config
    if not path.exists():
        path = Path(agents_config)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _create_benchmark(name: str, **kwargs) -> Benchmark:
    """Create a benchmark via the existing registry."""
    return registry_create_benchmark(name, **kwargs)


# ---------------------------------------------------------------------------
# Solver: runs benchmark on all agents
# ---------------------------------------------------------------------------

@solver
def benchmark_solver(
    benchmark_name: str,
    benchmark_kwargs: dict,
    agent_cfgs: list[dict],
    seed: int,
):
    """Run a benchmark on each agent and compute pairwise similarity."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        _disable_tqdm_mp_lock()
        _set_seed(seed)

        # Create benchmark
        bench = _create_benchmark(benchmark_name, **benchmark_kwargs)

        # Create agents (all as player_id=1, no seating needed)
        agents = []
        for cfg in agent_cfgs:
            agent_config = copy.deepcopy(cfg)
            agent_config["player_id"] = 1
            agents.extend(create_inspect_agents([agent_config], num_players=1))

        # Run benchmark on each agent
        results: dict[str, BenchmarkResult] = {}
        for agent in agents:
            result = bench.run(agent)
            results[agent.name] = result

        # Compute pairwise similarity
        agent_names = [a.name for a in agents]
        matrix: dict[str, float] = {}
        for i, a in enumerate(agent_names):
            for j, b in enumerate(agent_names):
                if i <= j:
                    if i == j:
                        matrix[f"{a} vs {b}"] = 100.0
                    else:
                        sim = bench.compute_similarity(results[a], results[b])
                        matrix[f"{a} vs {b}"] = sim
                        matrix[f"{b} vs {a}"] = sim

        # Store results
        output_data = {
            "benchmark": benchmark_name,
            "agents": agent_names,
            "results": {
                name: result.serialize() for name, result in results.items()
            },
            "similarity_matrix": {
                k: v for k, v in sorted(matrix.items())
                if " vs " in k and k.split(" vs ")[0] <= k.split(" vs ")[1]
            },
        }

        state.store.set("benchmark_output", output_data)
        LOGGER.log_record(output_data, f"benchmark_results_{benchmark_name}.json")

        state.output = ModelOutput.from_content(
            model="benchmark",
            content=json.dumps(output_data["similarity_matrix"], indent=2),
        )

        return state

    return solve


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

@scorer(metrics=[mean()])
def benchmark_scorer():
    """Score based on average pairwise similarity."""

    async def score(state: TaskState, target: Target) -> Score:
        output = state.store.get("benchmark_output", {})
        matrix = output.get("similarity_matrix", {})

        if not matrix:
            return Score(value=0.0, explanation="No similarity computed")

        # Average pairwise similarity (exclude self-comparisons)
        pair_sims = [
            v for k, v in matrix.items()
            if k.split(" vs ")[0] != k.split(" vs ")[1]
        ]
        avg_sim = sum(pair_sims) / max(1, len(pair_sims)) if pair_sims else 0.0

        return Score(
            value=avg_sim,
            answer=json.dumps(matrix, indent=2),
            metadata=output,
        )

    return score


# ---------------------------------------------------------------------------
# Task: single benchmark
# ---------------------------------------------------------------------------

@task
def benchmark_eval(
    benchmark_name: str = "newcomb",
    agents_config: str = "agents/cheap_llms_3.yaml",
    max_items: int | None = None,
    max_workers: int | None = None,
    seed: int = 42,
) -> Task:
    """Run a single benchmark on all agents and compute pairwise similarity."""
    agent_cfgs = _load_agents_yaml(agents_config)

    bench_kwargs: dict[str, Any] = {}
    if max_items is not None:
        bench_kwargs["max_items"] = max_items
    if max_workers is not None:
        bench_kwargs["max_workers"] = max_workers

    dataset = MemoryDataset(
        samples=[
            Sample(
                input=f"Benchmark: {benchmark_name}",
                target="benchmark",
                metadata={"benchmark": benchmark_name, "seed": seed},
            )
        ],
        name=f"benchmark_{benchmark_name}",
    )

    return Task(
        dataset=dataset,
        solver=benchmark_solver(
            benchmark_name=benchmark_name,
            benchmark_kwargs=bench_kwargs,
            agent_cfgs=agent_cfgs,
            seed=seed,
        ),
        scorer=benchmark_scorer(),
        metadata={
            "benchmark": benchmark_name,
            "agents_config": agents_config,
            "max_items": max_items,
            "seed": seed,
        },
    )


# ---------------------------------------------------------------------------
# Task: battery of benchmarks
# ---------------------------------------------------------------------------

@solver
def battery_solver(
    benchmark_names: list[str],
    benchmark_kwargs: dict,
    agent_cfgs: list[dict],
    seed: int,
):
    """Run multiple benchmarks and aggregate similarity matrices."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        _disable_tqdm_mp_lock()
        _set_seed(seed)

        # Create agents once
        agents = []
        for cfg in agent_cfgs:
            agent_config = copy.deepcopy(cfg)
            agent_config["player_id"] = 1
            agents.extend(create_inspect_agents([agent_config], num_players=1))

        all_outputs = {}
        for bench_name in benchmark_names:
            bench = _create_benchmark(bench_name, **benchmark_kwargs)

            results: dict[str, BenchmarkResult] = {}
            for agent in agents:
                result = bench.run(agent)
                results[agent.name] = result

            agent_names = [a.name for a in agents]
            matrix: dict[str, float] = {}
            for i, a in enumerate(agent_names):
                for j, b in enumerate(agent_names):
                    if i < j:
                        sim = bench.compute_similarity(results[a], results[b])
                        matrix[f"{a} vs {b}"] = sim
                    elif i == j:
                        matrix[f"{a} vs {b}"] = 100.0

            output = {
                "benchmark": bench_name,
                "agents": agent_names,
                "results": {n: r.serialize() for n, r in results.items()},
                "similarity_matrix": matrix,
            }
            all_outputs[bench_name] = output
            LOGGER.log_record(output, f"benchmark_results_{bench_name}.json")

        state.store.set("battery_output", all_outputs)
        LOGGER.log_record(all_outputs, "benchmark_results_all.json")

        state.output = ModelOutput.from_content(
            model="benchmark_battery",
            content=json.dumps(list(all_outputs.keys())),
        )
        return state

    return solve


@task
def benchmark_battery(
    benchmarks: str = "newcomb,cabin,dilemmas",
    agents_config: str = "agents/cheap_llms_3.yaml",
    max_items: int | None = None,
    max_workers: int | None = None,
    seed: int = 42,
) -> Task:
    """Run a battery of benchmarks on all agents."""
    benchmark_names = [b.strip() for b in benchmarks.split(",")]
    agent_cfgs = _load_agents_yaml(agents_config)

    bench_kwargs: dict[str, Any] = {}
    if max_items is not None:
        bench_kwargs["max_items"] = max_items
    if max_workers is not None:
        bench_kwargs["max_workers"] = max_workers

    dataset = MemoryDataset(
        samples=[
            Sample(
                input=f"Battery: {benchmarks}",
                target="battery",
                metadata={"benchmarks": benchmark_names, "seed": seed},
            )
        ],
        name=f"benchmark_battery",
    )

    return Task(
        dataset=dataset,
        solver=battery_solver(
            benchmark_names=benchmark_names,
            benchmark_kwargs=bench_kwargs,
            agent_cfgs=agent_cfgs,
            seed=seed,
        ),
        scorer=benchmark_scorer(),
        metadata={
            "benchmarks": benchmark_names,
            "agents_config": agents_config,
            "max_items": max_items,
            "seed": seed,
        },
    )
