"""Similarity Game benchmark.

Three-phase process that computes how similar two agents' decision-making
actually is under the similarity framing:

1. Elicit: Ask each agent for their strategy at multiple similarity levels
   (0%, 10%, ..., 100%) to build a "similarity response profile".
2. Compute: Compare profiles using Jensen-Shannon divergence averaged
   across levels to produce a single similarity score per pair.
3. (Optional) Validate: Play the actual game at the computed similarity.

The benchmark fits the standard Benchmark interface:
- ``run(agent)`` elicits strategies at all levels → BenchmarkResult
- ``compute_similarity(result_a, result_b)`` → similarity score in [0, 100]

Validation (phase 3) can be done by running the Similarity mechanism with
``similarity_source="fixed"`` and the computed similarity percentage.
"""

from collections import defaultdict
from typing import Any

import numpy as np

from benchmarks.base import Benchmark, BenchmarkResult
from src.agents.agent_manager import Agent
from src.games.base import Game
from src.mechanisms.similarity_elicitation import SimilarityElicitation
from src.mechanisms.similarity_utils import js_divergence
from src.utils.concurrency import run_tasks


class SimilarityGame(Benchmark):
    """Benchmark that elicits agent strategies at multiple similarity levels.

    For each agent, the benchmark prompts them with the game + similarity
    framing at each configured similarity level and records their mixed
    strategy distribution. Pairwise similarity is then computed as:

        similarity = (1 - avg_JS_divergence) * 100

    where JS divergence is averaged across all similarity levels.
    """

    name = "similarity_game"

    def __init__(
        self,
        base_game: Game,
        *,
        prompt_mode: str = "percentage_updated",
        increment: int = 10,
        percentages: list[int] | None = None,
        trials: int = 1,
        domain: str = "",
        custom_prompt: str = "",
        max_workers: int | None = None,
        max_items: int | None = None,  # unused, for interface compat
    ) -> None:
        self.base_game = base_game
        self.prompt_mode = prompt_mode
        self.domain = domain
        self.custom_prompt = custom_prompt
        self.trials = trials
        self.max_workers = max_workers

        if percentages is not None:
            self.percentages = percentages
        else:
            self.percentages = list(range(0, 101, increment))
            if 100 not in self.percentages:
                self.percentages.append(100)

        # Build questions list for interface compat
        self.questions = [
            {"similarity_pct": pct, "trial": t + 1}
            for pct in self.percentages
            for t in range(self.trials)
        ]

    def run(self, agent: Agent) -> BenchmarkResult:
        """Elicit strategy distributions at each similarity level.

        Args:
            agent: The agent to elicit from.

        Returns:
            BenchmarkResult where scores contains per-level distributions
            and an aggregate profile.
        """
        # Build tasks: (pct, trial)
        tasks = [
            (pct, trial + 1)
            for pct in self.percentages
            for trial in range(self.trials)
        ]

        def elicit_single(task):
            pct, trial_num = task
            elicitation = SimilarityElicitation(
                base_game=self.base_game,
                similarity_pct=pct,
                prompt_mode=self.prompt_mode,
                domain=self.domain,
                custom_prompt=self.custom_prompt,
            )
            result = elicitation.elicit_single(agent)
            return {
                "similarity_pct": pct,
                "trial": trial_num,
                "action_distribution": result.action_distribution,
                "trace_id": result.trace_id,
            }

        if self.max_workers and self.max_workers > 1:
            results = run_tasks(tasks, elicit_single, max_workers=self.max_workers)
        else:
            results = [elicit_single(t) for t in tasks]

        # Organize by percentage
        by_pct: dict[int, list[dict[str, int]]] = defaultdict(list)
        for r in results:
            by_pct[r["similarity_pct"]].append(r["action_distribution"])

        # Compute average distribution per level
        avg_distributions: dict[int, dict[str, float]] = {}
        for pct in self.percentages:
            dists = by_pct[pct]
            if not dists:
                continue
            all_keys = set()
            for d in dists:
                all_keys.update(d.keys())
            avg = {}
            for k in sorted(all_keys):
                avg[k] = float(np.mean([d.get(k, 0) for d in dists]))
            avg_distributions[pct] = avg

        return BenchmarkResult(
            agent_name=agent.name,
            benchmark_name=self.name,
            scores={
                "avg_distributions": {str(k): v for k, v in avg_distributions.items()},
                "percentages": self.percentages,
                "trials": self.trials,
            },
            raw_responses=results,
        )

    def compute_similarity(
        self, result_a: BenchmarkResult, result_b: BenchmarkResult
    ) -> float:
        """Chance-corrected JSD-based similarity, rescaled to [0, 100].

        Uses per-trial distributions from raw_responses (not pre-averaged
        avg_distributions) so the chance baseline is well-defined.

        JSD_obs = mean over per-trial pairs at the SAME similarity_pct of
                  JS divergence between agent A's and agent B's distributions.
        JSD_exp = mean over ALL cross-pct cross-pairs (independence baseline).
        kappa   = 1 − JSD_obs / JSD_exp        ∈ [−1, +1]
        score   = (kappa + 1) / 2 * 100        ∈ [0, 100]

        100 = identical paired distributions, 50 ≈ chance under independence
        across similarity levels (random agents land here), 0 = anti-correlated.

        Falls back to the previous avg-then-compare formula if raw_responses
        aren't available on either result.
        """
        raw_a = result_a.raw_responses
        raw_b = result_b.raw_responses

        if not raw_a or not raw_b:
            avg_a = result_a.scores["avg_distributions"]
            avg_b = result_b.scores["avg_distributions"]
            divergences = [
                js_divergence(avg_a[p], avg_b[p])
                for p in avg_a if p in avg_b
            ]
            if not divergences:
                return 0.0
            return round((1 - float(np.mean(divergences))) * 100, 2)

        a_by_pct: dict[int, list[dict]] = defaultdict(list)
        b_by_pct: dict[int, list[dict]] = defaultdict(list)
        for r in raw_a:
            a_by_pct[r["similarity_pct"]].append(r["action_distribution"])
        for r in raw_b:
            b_by_pct[r["similarity_pct"]].append(r["action_distribution"])

        common_pcts = sorted(a_by_pct.keys() & b_by_pct.keys())
        if not common_pcts:
            return 0.0

        # JSD_obs: per-trial paired comparison at same pct
        obs = []
        for p in common_pcts:
            for a_d, b_d in zip(a_by_pct[p], b_by_pct[p]):
                obs.append(js_divergence(a_d, b_d))

        # JSD_exp: all cross-pct cross-pairs (independence over pct)
        all_a = [d for p in common_pcts for d in a_by_pct[p]]
        all_b = [d for p in common_pcts for d in b_by_pct[p]]
        exp = [js_divergence(a, b) for a in all_a for b in all_b]

        obs_m = float(np.mean(obs))
        exp_m = float(np.mean(exp))
        if exp_m == 0:
            return 100.0  # both agents constant and identical → perfect agreement

        kappa = 1 - obs_m / exp_m
        return round((kappa + 1) / 2 * 100, 2)
