"""Fixed-point similarity finder for LLM agents.

Sweeps across similarity percentages, measures similarity via elicitation,
and computes the weighted average as the "grounded similarity" —
the single number representing how similar the agents actually behave.

Supports two metrics:
  - "agreement": action agreement rate = sum(p_i * q_i). Both cooperate = 100%,
    both defect = 100%, opposite = 0%.
  - "cooperation": average cooperation rate (A0%) across both agents.

This is a module used by `script/run_fixed_point.py`; not a standalone CLI.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.agents.agent_manager import Agent
from src.games.base import Game
from src.mechanisms.similarity_elicitation import SimilarityElicitation
from src.utils.concurrency import run_tasks

VALID_METRICS = ("agreement", "cooperation", "both")


@dataclass
class FixedPointResult:
    """Result from a fixed-point search for one agent pair."""

    agent_a_name: str
    agent_b_name: str
    game_type: str
    metric: str  # "agreement", "cooperation", or "both"

    # Sweep data
    sweep_percentages: list[int] = field(default_factory=list)
    sweep_rates: list[float] = field(default_factory=list)
    sweep_std: list[float] = field(default_factory=list)
    sweep_raw: dict[int, list[float]] = field(default_factory=dict)

    # When metric="both", store the second metric's sweep
    sweep_rates_alt: list[float] = field(default_factory=list)

    # Fixed point
    primary_fixed_point: float | None = None
    primary_fixed_point_alt: float | None = None  # only when metric="both"

    # Validation
    validation_similarity_pct: int | None = None
    validation_rate: float | None = None
    validation_std: float | None = None

    def serialize(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "agent_a": self.agent_a_name,
            "agent_b": self.agent_b_name,
            "game_type": self.game_type,
            "metric": self.metric,
            "sweep": {
                "percentages": self.sweep_percentages,
                "rates_mean": self.sweep_rates,
                "rates_std": self.sweep_std,
                "raw_trials": {
                    str(k): v for k, v in self.sweep_raw.items()
                },
            },
            "primary_fixed_point": self.primary_fixed_point,
        }
        if self.metric == "both":
            result["sweep"]["rates_mean_alt"] = self.sweep_rates_alt
            result["primary_fixed_point_alt"] = self.primary_fixed_point_alt
        if self.validation_similarity_pct is not None:
            result["validation"] = {
                "similarity_pct": self.validation_similarity_pct,
                "rate_mean": self.validation_rate,
                "rate_std": self.validation_std,
            }
        else:
            result["validation"] = None
        return result


def _agreement_rate(dist_a: dict[str, float], dist_b: dict[str, float]) -> float:
    """Compute action agreement rate between two distributions.

    agreement = sum(p_i * q_i) * 100, where p and q are normalized to [0,1].
    Returns a percentage in [0, 100].
    """
    all_keys = set(dist_a.keys()) | set(dist_b.keys())
    agreement = 0.0
    for k in all_keys:
        pa = dist_a.get(k, 0) / 100.0
        pb = dist_b.get(k, 0) / 100.0
        agreement += pa * pb
    return agreement * 100.0


def _cooperation_rate(dist_a: dict[str, float], dist_b: dict[str, float]) -> float:
    """Average cooperation rate (A0%) across both agents."""
    return (dist_a.get("A0", 0) + dist_b.get("A0", 0)) / 2.0


def _weighted_average(rates: np.ndarray) -> float:
    """Weighted average where the rate itself is the weight."""
    total = rates.sum()
    if total > 0:
        return float(np.dot(rates, rates) / total)
    return 0.0


class FixedPointFinder:
    """Find grounded similarity for LLM agent pairs."""

    def __init__(
        self,
        base_game: Game,
        *,
        prompt_mode: str = "percentage",
        trials_per_point: int = 3,
        metric: str = "agreement",
        domain: str = "",
        custom_prompt: str = "",
    ) -> None:
        if metric not in VALID_METRICS:
            raise ValueError(f"metric must be one of {VALID_METRICS}, got {metric!r}")
        self.base_game = base_game
        self.prompt_mode = prompt_mode
        self.trials_per_point = trials_per_point
        self.metric = metric
        self.domain = domain
        self.custom_prompt = custom_prompt

    def _get_action_dist(self, agent: Agent, similarity_pct: int) -> dict[str, int]:
        """Run one elicitation and return the full action distribution.

        Returns e.g. {"A0": 60, "A1": 40} (percentages summing to 100).
        """
        elicitation = SimilarityElicitation(
            base_game=self.base_game,
            similarity_pct=similarity_pct,
            prompt_mode=self.prompt_mode,
            domain=self.domain,
            custom_prompt=self.custom_prompt,
        )
        result = elicitation.elicit_single(agent)
        return result.action_distribution

    def _compute_rate(
        self, dist_a: dict[str, float], dist_b: dict[str, float],
    ) -> float:
        """Compute the primary metric from two distributions."""
        if self.metric == "cooperation":
            return _cooperation_rate(dist_a, dist_b)
        return _agreement_rate(dist_a, dist_b)

    def sweep(
        self,
        agent_a: Agent,
        agent_b: Agent,
        percentages: list[int],
    ) -> FixedPointResult:
        """Evaluate f(s) across all percentages, fully parallelized."""
        result = FixedPointResult(
            agent_a_name=agent_a.name,
            agent_b_name=agent_b.name,
            game_type=type(self.base_game).__name__,
            metric=self.metric,
            sweep_percentages=percentages,
        )

        # Build all tasks: (agent, pct, trial, agent_label)
        tasks = []
        for pct in percentages:
            for trial in range(self.trials_per_point):
                tasks.append((agent_a, pct, trial, "a"))
                tasks.append((agent_b, pct, trial, "b"))

        def run_single(task):
            agent, pct, _trial, _label = task
            return (pct, _trial, _label, self._get_action_dist(agent, pct))

        all_results = run_tasks(tasks, run_single, desc="Sweep elicitations")

        # Organize: group by (pct, trial) -> distributions for agent_a and agent_b
        by_pct_trial: dict[tuple[int, int], dict[str, dict]] = defaultdict(dict)
        for pct, trial, label, dist in all_results:
            by_pct_trial[(pct, trial)][label] = dist

        # Compute per-percentage rates
        for pct in percentages:
            trial_rates = []
            trial_rates_alt = []
            for trial in range(self.trials_per_point):
                dists = by_pct_trial[(pct, trial)]
                dist_a = dists.get("a", {})
                dist_b = dists.get("b", {})
                trial_rates.append(self._compute_rate(dist_a, dist_b))
                if self.metric == "both":
                    trial_rates_alt.append(_cooperation_rate(dist_a, dist_b))
            result.sweep_raw[pct] = trial_rates
            result.sweep_rates.append(float(np.mean(trial_rates)))
            result.sweep_std.append(float(np.std(trial_rates)))
            if self.metric == "both":
                result.sweep_rates_alt.append(float(np.mean(trial_rates_alt)))

        return result

    def validate(
        self,
        agent_a: Agent,
        agent_b: Agent,
        fixed_point: float,
        result: FixedPointResult,
        *,
        validation_trials: int = 5,
    ) -> FixedPointResult:
        """Run extra trials at the fixed point to confirm."""
        pct = int(round(fixed_point))
        pct = max(0, min(100, pct))

        # Build tasks and run in parallel with progress
        tasks = []
        for trial in range(validation_trials):
            tasks.append((agent_a, pct, trial, "a"))
            tasks.append((agent_b, pct, trial, "b"))

        def run_single(task):
            agent, p, _trial, _label = task
            return (p, _trial, _label, self._get_action_dist(agent, p))

        all_val = run_tasks(tasks, run_single, desc="Validation")

        by_trial: dict[int, dict[str, dict]] = defaultdict(dict)
        for _, trial, label, dist in all_val:
            by_trial[trial][label] = dist

        values = []
        for trial in range(validation_trials):
            dists = by_trial[trial]
            values.append(self._compute_rate(dists.get("a", {}), dists.get("b", {})))

        result.validation_similarity_pct = pct
        result.validation_rate = float(np.mean(values))
        result.validation_std = float(np.std(values))
        return result

    def find_fixed_point(
        self,
        agent_a: Agent,
        agent_b: Agent,
        percentages: list[int] | None = None,
        *,
        validation_trials: int = 5,
    ) -> FixedPointResult:
        """Sweep across similarity percentages, compute grounded similarity.

        The grounded similarity is the weighted average rate across all
        similarity levels. Higher rate at a given level counts more.
        """
        if percentages is None:
            percentages = list(range(0, 101, 10))

        # Phase 1: Sweep
        result = self.sweep(agent_a, agent_b, percentages)

        # Phase 2: Weighted average = grounded similarity
        rates = np.array(result.sweep_rates)
        result.primary_fixed_point = round(_weighted_average(rates), 1)

        if self.metric == "both":
            alt = np.array(result.sweep_rates_alt)
            result.primary_fixed_point_alt = round(_weighted_average(alt), 1)

        # Phase 3: Validate at that similarity percentage
        result = self.validate(
            agent_a,
            agent_b,
            result.primary_fixed_point,
            result,
            validation_trials=validation_trials,
        )

        return result
