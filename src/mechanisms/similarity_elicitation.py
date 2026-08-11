"""Solo and pairwise strategy elicitation under similarity framing.

This module runs agents outside of tournament play to observe their
strategy choices when told about opponent similarity, without actually
playing against anyone.
"""

from dataclasses import asdict, dataclass
from typing import Any

from src.agents.agent_manager import Agent
from src.games.base import Game
from src.mechanisms.prompts import SIMILARITY_MECHANISM_PROMPT_SINGLE
from src.mechanisms.similarity_utils import build_similarity_framing, js_divergence


@dataclass
class ElicitationResult:
    """Result from eliciting a single agent's strategy."""

    agent_name: str
    similarity_pct: int
    prompt_mode: str
    action_distribution: dict[str, int]  # e.g., {"A0": 60, "A1": 40}
    raw_response: str
    trace_id: str

    def serialize(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PairwiseResult:
    """Result from eliciting two agents independently and comparing."""

    agent_a: ElicitationResult
    agent_b: ElicitationResult
    strategy_divergence: float  # Jensen-Shannon divergence

    def serialize(self) -> dict[str, Any]:
        return {
            "agent_a": self.agent_a.serialize(),
            "agent_b": self.agent_b.serialize(),
            "strategy_divergence": self.strategy_divergence,
        }


class SimilarityElicitation:
    """Run agents solo or side-by-side to observe strategy choices under similarity framing."""

    def __init__(
        self,
        base_game: Game,
        *,
        similarity_pct: int = 70,
        prompt_mode: str = "percentage_updated",
        domain: str = "",
        custom_prompt: str = "",
        difference_framing: bool | str = False,
    ) -> None:
        self.base_game = base_game
        self.similarity_pct = similarity_pct
        self.prompt_mode = prompt_mode
        self.domain = domain
        self.custom_prompt = custom_prompt
        self.difference_framing = difference_framing

    def _build_elicitation_prompt(self) -> str:
        """Build the full prompt for elicitation (game description + similarity twist)."""
        framing = build_similarity_framing(
            similarity_pct=self.similarity_pct,
            prompt_mode=self.prompt_mode,
            domain=self.domain,
            custom_prompt=self.custom_prompt,
            difference_framing=self.difference_framing,
        )
        return SIMILARITY_MECHANISM_PROMPT_SINGLE.format(
            similarity_framing=framing,
        )

    def elicit_single(self, agent: Agent) -> ElicitationResult:
        """Give one agent the game + similarity prompt, record their action distribution.

        The agent sees the game description and the similarity framing, then
        outputs a mixed strategy. No opponent plays.

        Args:
            agent: The agent to elicit a strategy from.

        Returns:
            ElicitationResult with the agent's action distribution.
        """
        extra_info = self._build_elicitation_prompt()

        trace_id, mix_probs = self.base_game.prompt_player_mix_probs(
            agent, extra_info=extra_info
        )

        # Convert int-keyed probs to action token keys
        action_dist = {}
        for idx, prob in mix_probs.items():
            token = self.base_game.action_class.from_index(idx).to_token()
            action_dist[token] = prob

        return ElicitationResult(
            agent_name=agent.name,
            similarity_pct=self.similarity_pct,
            prompt_mode=self.prompt_mode,
            action_distribution=action_dist,
            raw_response="",  # response is logged via game_log.txt
            trace_id=trace_id,
        )

    def elicit_pairwise(
        self, agent_a: Agent, agent_b: Agent
    ) -> PairwiseResult:
        """Give both agents the same scenario independently and compare.

        Each agent independently receives the game description + similarity
        framing, outputs a mixed strategy. Their distributions are compared
        using Jensen-Shannon divergence.

        Args:
            agent_a: First agent.
            agent_b: Second agent.

        Returns:
            PairwiseResult with both results and JS divergence.
        """
        result_a = self.elicit_single(agent_a)
        result_b = self.elicit_single(agent_b)

        divergence = js_divergence(
            result_a.action_distribution,
            result_b.action_distribution,
        )

        return PairwiseResult(
            agent_a=result_a,
            agent_b=result_b,
            strategy_divergence=divergence,
        )
