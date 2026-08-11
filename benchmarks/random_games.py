"""Random die roll and coin toss benchmarks.

These benchmarks measure how similarly two models generate "random" outputs.
Each model is asked to produce a sequence of random die rolls or coin tosses,
and similarity is the fraction of positions where both models gave the same result.
"""

import re
from typing import Any, override

from benchmarks.base import Benchmark, BenchmarkResult
from src.utils.concurrency import run_tasks


DIE_ROLL_PROMPT = """\
Please roll a fair six-sided die {num_trials} times and report each result.

Format your answer as a comma-separated list of numbers (1-6), with no other text.
For example, if rolling 5 times: 3, 1, 5, 2, 6

Your response should contain exactly {num_trials} numbers, each between 1 and 6."""

COIN_TOSS_PROMPT = """\
Please flip a fair coin {num_trials} times and report each result.

Format your answer as a comma-separated list using H (heads) or T (tails), with no other text.
For example, if flipping 5 times: H, T, H, H, T

Your response should contain exactly {num_trials} results, each either H or T."""


class RandomDieRoll(Benchmark):
    """Random die roll benchmark.

    Asks each model to produce a sequence of die rolls and compares
    how often two models produce the same value at the same position.
    """

    name = "random_die_roll"

    def __init__(
        self,
        num_trials: int = 100,
        max_workers: int | None = None,
    ) -> None:
        self.num_trials = num_trials
        self.max_workers = max_workers
        # Compatibility with runner's len(bench.questions) display
        self.questions = list(range(num_trials))

    @override
    def run(self, agent: Any) -> BenchmarkResult:
        prompt = DIE_ROLL_PROMPT.format(num_trials=self.num_trials)
        response, trace_id = agent.invoke(prompt)
        rolls = self._parse_rolls(response, self.num_trials)

        return BenchmarkResult(
            agent_name=agent.name,
            benchmark_name=self.name,
            scores={
                "answer_vector": {str(i): v for i, v in enumerate(rolls)},
                "rolls": rolls,
            },
            raw_responses=[{
                "prompt": prompt,
                "response": response,
                "parsed_rolls": rolls,
                "trace_id": trace_id,
            }],
        )

    @override
    def compute_similarity(
        self, result_a: BenchmarkResult, result_b: BenchmarkResult
    ) -> float:
        """Compute similarity as raw positional agreement percentage.

        Returns:
            Similarity score in [0, 100] where 100 = perfect agreement.
        """
        rolls_a = result_a.scores["rolls"]
        rolls_b = result_b.scores["rolls"]
        length = min(len(rolls_a), len(rolls_b))
        if length == 0:
            return 0.0

        matches = sum(1 for i in range(length) if rolls_a[i] == rolls_b[i])
        return matches / length * 100

    @staticmethod
    def _parse_rolls(response: str, expected: int) -> list[int]:
        """Extract die roll values (1-6) from response."""
        numbers = [int(m) for m in re.findall(r"\b([1-6])\b", response)]
        # Take the first `expected` valid numbers
        numbers = numbers[:expected]
        return numbers


class RandomCoinToss(Benchmark):
    """Random coin toss benchmark.

    Asks each model to produce a sequence of coin tosses and compares
    how often two models produce the same result at the same position.
    """

    name = "random_coin_toss"

    def __init__(
        self,
        num_trials: int = 100,
        max_workers: int | None = None,
    ) -> None:
        self.num_trials = num_trials
        self.max_workers = max_workers
        self.questions = list(range(num_trials))

    @override
    def run(self, agent: Any) -> BenchmarkResult:
        prompt = COIN_TOSS_PROMPT.format(num_trials=self.num_trials)
        response, trace_id = agent.invoke(prompt)
        tosses = self._parse_tosses(response, self.num_trials)

        return BenchmarkResult(
            agent_name=agent.name,
            benchmark_name=self.name,
            scores={
                "answer_vector": {str(i): v for i, v in enumerate(tosses)},
                "tosses": tosses,
            },
            raw_responses=[{
                "prompt": prompt,
                "response": response,
                "parsed_tosses": tosses,
                "trace_id": trace_id,
            }],
        )

    @override
    def compute_similarity(
        self, result_a: BenchmarkResult, result_b: BenchmarkResult
    ) -> float:
        """Compute similarity as raw positional agreement percentage.

        Returns:
            Similarity score in [0, 100] where 100 = perfect agreement.
        """
        tosses_a = result_a.scores["tosses"]
        tosses_b = result_b.scores["tosses"]
        length = min(len(tosses_a), len(tosses_b))
        if length == 0:
            return 0.0

        matches = sum(1 for i in range(length) if tosses_a[i] == tosses_b[i])
        return matches / length * 100

    @staticmethod
    def _parse_tosses(response: str, expected: int) -> list[str]:
        """Extract coin toss values (H/T) from response."""
        tokens = re.findall(r"\b([HT])\b", response, re.IGNORECASE)
        tokens = [t.upper() for t in tokens]
        return tokens[:expected]
