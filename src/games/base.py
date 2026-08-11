import json
import random
import re
import textwrap
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Self, Sequence, cast, override

from src.agents.agent_manager import Agent
from src.utils.concurrency import run_tasks


class Action(Enum):
    """Base class for actions in the game"""

    def to_token(self) -> str:
        """Convert the action to a token (eg, A1) starting from A0 for LLM parsing."""
        idx = list(type(self)).index(self)
        return f"A{idx}"

    @classmethod
    def from_token(cls, token: str) -> Self:
        """Parse an action from a token like "A0" or "A1"."""
        try:
            idx = int(token.lstrip("A"))
            action = list(cls)[idx]
        except Exception as exp:
            raise ValueError(f"Unknown action token {token!r}") from exp
        return action

    @classmethod
    def from_index(cls, index: int) -> Self:
        """Get action from its index."""
        try:
            action = list(cls)[index]
        except Exception as exp:
            raise ValueError(f"Unknown action index {index!r}") from exp
        return action

    @property
    def is_mediator(self) -> bool:
        """Check if this specific action is the Mediator action."""
        return self.name == "MEDIATOR"

    @classmethod
    def game_actions(cls) -> list[Self]:
        """Return all playable moves (excluding the Mediator action)."""
        return [act for act in cls if not act.is_mediator]


@dataclass
class Move:
    """
    A record of one player's action in a single round.
    """

    player: Agent
    action: Action
    points: float
    trace_id: str
    mediated: bool = False
    mix_probs: dict[int, float] | None = None

    def serialize(self) -> dict[str, Any]:
        """Convert the Move to a dictionary, mostly for logging and record purpose."""
        # Build dict manually to avoid deepcopy issues with Agent's HTTP clients
        d = {
            "player": self.player.name,
            "action": str(self.action),
            "points": self.points,
            "trace_id": self.trace_id,
        }
        if self.mediated:
            d["mediated"] = True
        if self.mix_probs is not None:
            d["mix_probs"] = {f"A{k}": v for k, v in self.mix_probs.items()}
        return d

class Game(ABC):
    """
    Base class for all games in the tournament.
    """

    def __init__(
        self,
        prompt: str,
        action_class: type[Action],
        *,
        num_players: int,
        is_symmetric: bool = True,
    ) -> None:
        self.prompt = prompt
        self.num_players = num_players
        self.number_to_position = {1: "first", 2: "second", 3: "third", 4: "fourth"}
        self.is_symmetric = is_symmetric
        self.action_class: type[Action] = action_class
        self.num_actions = len(self.action_class)
        self.default_output_instruction = textwrap.dedent(
            """
        Instruction:
        - Choose a probability distribution over the provided actions each round.
        - Output must contain a valid JSON object at the end.
        - Keys must be the action names exactly as given.
        - Values must be percentage points given in integers.
        - The values must sum to exactly 100.

        Format requirement:
        Return exactly one JSON object, for example:
        {"A0": <INT>, "A1": <INT>, ...}
        """
        )

    def add_mediator_action(self) -> None:
        """Dynamically replace action_cls with a version containing MEDIATOR."""
        members = {action.name: action.value for action in self.action_class}
        members["MEDIATOR"] = "MEDIATOR"

        new_enum = Enum(
            self.action_class.__name__,
            members,
            type=Action,
        )

        self.action_class = cast(type[Action], new_enum)

    def get_player_prompt(self, player_id: int) -> str:
        """Get game prompt from specific player's perspective. Per default, this is the same for all players in symmetric games."""
        return self.prompt + f"\nThere are {self.num_players} players in this game, numbered Player 0 through Player {self.num_players - 1}. In case player identification becomes relevant, you are playing in the position of Player {player_id} in this game.\n"

    @abstractmethod
    def play(
        self,
        additional_info: list[str] | str,
        players: Sequence[Agent],
        action_map: Callable = lambda x: x,
    ) -> list[Move]:
        """Play the game."""
        raise NotImplementedError

    def prompt_player_mix_probs(
        self,
        player: Agent,
        extra_info: str | None = None,
        output_instruction: str | None = None,
    ) -> tuple[str, dict[int, float]]:
        """
        Given the mechanism's additional info and the base game prompt,
        format the full prompt and query the player.

        Returns the player's raw response.
        """
        prompt = self.get_player_prompt(player.player_id)

        if extra_info:
            prompt += extra_info

        if output_instruction is None:
            output_instruction = self.default_output_instruction
        prompt += "\n" + output_instruction

        _response, trace_id, mix_probs = player.chat_with_retries(
            prompt, self._parse_mixed_probs
        )
        return trace_id, mix_probs

    def _parse_mixed_probs(
        self,
        response: str,
    ) -> dict[int, float]:
        """
        Parse mixed strategy pairs like '<A0=60>|<A1=25>|<A2=15>'.
        Rules:
        - integers only
        - each in [0,100]
        - sum exactly 100
        """
        matches = re.findall(r"\{.*?\}", response, re.DOTALL)
        if not matches:
            raise ValueError(
                f"No JSON object found in the response {response!r}"
            )
        json_str = matches[-1]
        json_obj = json.loads(json_str)

        result = {}
        total = 0
        for k, v in json_obj.items():
            if not isinstance(v, int):
                raise ValueError(f"Value for {k} must be an integer, got {v!r}")
            if not 0 <= v <= 100:
                raise ValueError(
                    f"Value for {k} must be between 0 and 100, got {v}"
                )
            idx = int(k[1:])  # strip the leading 'A'
            result[idx] = v
            total += v

        got_keys = set(result.keys())
        missing = set(range(self.num_actions)) - got_keys
        if missing:
            raise ValueError(f"Action key mismatch. Missing: {sorted(missing)}")

        if total != 100:
            raise ValueError(f"Probabilities must sum to 100 (got {total}).")

        return result

    @staticmethod
    def _choose_from_mix_strategy(probs: dict[int, float]) -> int:
        keys = list(probs.keys())
        weights = list(probs.values())
        return random.choices(keys, weights=weights, k=1)[0]

    def _collect_actions(
        self,
        players: Sequence[Agent],
        info: Sequence[str],
    ) -> dict[Agent, tuple[Action, str, bool, dict[int, float]]]:
        if len(players) != len(info):
            raise ValueError(f"Count mismatch: {len(players)} vs {len(info)}.")

        def query(player: Agent, extra_info: str) -> tuple[int, str, dict[int, float]]:
            trace_id, mix_probs = self.prompt_player_mix_probs(
                player, extra_info=extra_info
            )
            action_idx = self._choose_from_mix_strategy(mix_probs)
            return action_idx, trace_id, mix_probs

        return {
            player: (
                self.action_class.from_index(action_idx),
                trace_id,
                False,  # mediated=False, could be modified with action_map
                mix_probs,
            )
            for player, (action_idx, trace_id, mix_probs) in zip(
                players, run_tasks(zip(players, info), lambda p: query(*p))
            )
        }


class GridGame(Game):
    """
    A base class for grid-based games.
    """

    def __init__(
        self,
        payoff_matrix: Mapping[str, Sequence[float]],
        action_class: type[Action],
        *,
        num_players: int,
        is_symmetric: bool,
        ordinal_payoffs: bool = False,
    ) -> None:
        self.action_class = action_class
        self.raw_payoff_matrix = payoff_matrix
        self.payoff_matrix = self._parse_payoff_matrix(payoff_matrix)
        self.ordinal_payoffs = ordinal_payoffs
        # Build the prompt now that payoff_matrix is ready
        actions_block = "\n".join(
            [f"- {act.to_token()}" for act in action_class]
        )
        prompt_template = textwrap.dedent(
            """
        Setup:
        You are playing a decision-making game with another player.
        Your objective is to maximize your total points received in the game described in length below.

        Actions available to each player:
        {actions_block}

        Basic game rules:
        1. You and the other player each choose a probability for each action, simultaneously.
        2. After both decisions are locked in, the final action will be drawn from the probability distributions.
        3. Both players receive the points specified in the payoff description below.

        Payoff description:
        {payoff_description}
        """
        )

        def _desc(pid: int = 0) -> str:
            return (
                self._payoff_description_ordinal(pid)
                if ordinal_payoffs
                else self._payoff_description(pid)
            )

        prompt = prompt_template.format(
            actions_block=actions_block,
            payoff_description=_desc(),
        )

        super().__init__(
            prompt,
            action_class,
            num_players=num_players,
            is_symmetric=is_symmetric,
        )

        # For asymmetric games, store per-player prompts so each player
        # sees the payoff matrix from their own perspective.
        if not is_symmetric:
            self._player_prompts: dict[int, str] = {}
            for pid in range(num_players):
                self._player_prompts[pid] = prompt_template.format(
                    actions_block=actions_block,
                    payoff_description=_desc(pid),
                )

    def _parse_payoff_matrix(
        self,
        raw_payoff: Mapping[str, Sequence[float]],
    ) -> dict[
        tuple[Action, Action],
        tuple[float, float],
    ]:
        """
        Convert a raw payoff matrix with string keys into typed action pairs.
        """
        payoffs = {}
        for key, (p1, p2) in raw_payoff.items():
            a1 = self.action_class(key[0])
            a2 = self.action_class(key[1])
            payoffs[(a1, a2)] = (p1, p2)
        return payoffs

    @override
    def get_player_prompt(self, player_id: int) -> str:
        if not self.is_symmetric and hasattr(self, "_player_prompts"):
            base = self._player_prompts.get(player_id, self.prompt)
        else:
            base = self.prompt
        return base + f"\nThere are {self.num_players} players in this game, numbered Player 0 through Player {self.num_players - 1}. In case player identification becomes relevant, you are playing in the position of Player {player_id} in this game.\n"

    def _payoff_description(self, player_perspective: int = 0) -> str:
        lines = []
        if player_perspective == 0:
            for (a, b), (pts_a, pts_b) in self.payoff_matrix.items():
                lines.append(
                    f"\t- If you choose {a.to_token()} and the other player chooses {b.to_token()}: "
                    f"you get {pts_a} points, the other player gets {pts_b} points."
                )
        else:
            # From player 2's perspective: swap roles
            for (a, b), (pts_a, pts_b) in self.payoff_matrix.items():
                lines.append(
                    f"\t- If you choose {b.to_token()} and the other player chooses {a.to_token()}: "
                    f"you get {pts_b} points, the other player gets {pts_a} points."
                )
        return "\n".join(lines)

    def _payoff_description_ordinal(self, player_perspective: int = 0) -> str:
        """Show preference orderings instead of cardinal payoff values."""

        # Collect outcomes from the player's perspective
        if player_perspective == 0:
            outcomes = [
                (a.to_token(), b.to_token(), float(pts_a), float(pts_b))
                for (a, b), (pts_a, pts_b) in self.payoff_matrix.items()
            ]
        else:
            outcomes = [
                (b.to_token(), a.to_token(), float(pts_b), float(pts_a))
                for (a, b), (pts_a, pts_b) in self.payoff_matrix.items()
            ]

        def _label(rank: int, total: int, subject: str) -> str:
            """Descriptive label for a given rank out of total distinct preference levels."""
            is_you = subject == "you"
            if rank == 1:
                return (
                    "The outcome you prefer the most"
                    if is_you else
                    f"The outcome {subject} prefers the most"
                )
            if rank == total:
                return (
                    "The outcome you prefer the least"
                    if is_you else
                    f"The outcome {subject} prefers the least"
                )
            mid = (total + 1) / 2
            if rank <= mid:
                return (
                    "An outcome that you do prefer, yet is not the best"
                    if is_you else
                    f"An outcome that {subject} does prefer, yet is not the best"
                )
            return (
                "An outcome that you do not prefer, yet is not the worst"
                if is_you else
                f"An outcome that {subject} does not prefer, yet is not the worst"
            )

        def _rank_lines(sorted_items: list, header: str, subject: str) -> list[str]:
            # Count distinct payoff levels to determine total ranks
            seen: list[float] = []
            for item in sorted_items:
                if not seen or item[2] != seen[-1]:
                    seen.append(item[2])
            total_ranks = len(seen)

            lines = [f"\t{header}:"]
            rank = 1
            i = 0
            while i < len(sorted_items):
                my_tok, other_tok, my_pts, _ = sorted_items[i]
                j = i + 1
                while j < len(sorted_items) and sorted_items[j][2] == my_pts:
                    j += 1
                tie_suffix = " (tied)" if j - i > 1 else ""
                lbl = _label(rank, total_ranks, subject) + tie_suffix
                for k in range(i, j):
                    t_my, t_oth, _, _ = sorted_items[k]
                    lines.append(f"\t\t{lbl}: You choose {t_my}, other player chooses {t_oth}")
                rank += 1
                i = j
            return lines

        sorted_by_player = sorted(outcomes, key=lambda x: -x[2])
        sorted_by_other = sorted(outcomes, key=lambda x: -x[3])

        lines = _rank_lines(sorted_by_player, "Your preference ordering", "you")
        lines.append("")
        lines += _rank_lines(sorted_by_other, "The other player's preference ordering", "the other player")
        return "\n".join(lines)

    @override
    def add_mediator_action(self) -> None:
        super().add_mediator_action()
        self.payoff_matrix = self._parse_payoff_matrix(self.raw_payoff_matrix)
