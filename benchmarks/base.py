"""Abstract base class for benchmarks and result storage."""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class BenchmarkResult:
    """Structured result from running a benchmark on an agent."""

    agent_name: str
    benchmark_name: str
    scores: dict[str, Any]
    raw_responses: list[dict] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def serialize(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkResult":
        return cls(
            agent_name=data["agent_name"],
            benchmark_name=data["benchmark_name"],
            scores=data["scores"],
            raw_responses=data.get("raw_responses", []),
            timestamp=data.get("timestamp", ""),
        )


class Benchmark(ABC):
    """Unified interface for running benchmarks and computing similarity.

    Subclasses implement specific benchmark protocols (Big Five, PsyBench, etc.)
    and define how to compute similarity between two agents' results.
    """

    name: str

    @abstractmethod
    def run(self, agent: Any) -> BenchmarkResult:
        """Run this benchmark on an agent.

        Args:
            agent: An Agent instance (or any object with an `invoke(prompt)` method).

        Returns:
            BenchmarkResult with scores and raw responses.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_similarity(
        self, result_a: BenchmarkResult, result_b: BenchmarkResult
    ) -> float:
        """Compute similarity score between two agents' benchmark results.

        Args:
            result_a: First agent's results.
            result_b: Second agent's results.

        Returns:
            Similarity score in [0, 100]. 100 = identical, 0 = maximally different.
        """
        raise NotImplementedError
