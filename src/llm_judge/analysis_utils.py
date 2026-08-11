"""Helpers for parsing agent names and applying filters in judge pipelines.

Ported from ``coopeval.visualization.analysis_utils`` — only the functions the
judge CLI scripts depend on are included here. Matplotlib/coopeval game
dependencies are deliberately omitted.
"""

from __future__ import annotations

import re
from typing import Iterable

PLAYER_ID_SUFFIX_RE = re.compile(r"#P(?P<player_id>\d+)$")

DEFAULT_SKIP_GAMES: tuple[str, ...] = ()


def should_skip_game_name(
    game_name: str, skip_games: Iterable[str] | None = None
) -> bool:
    """Return True if the provided game name appears in the skip list."""
    source = (
        DEFAULT_SKIP_GAMES
        if skip_games is None
        else tuple(
            value.strip() for value in skip_games if value and value.strip()
        )
    )
    if not source:
        return False
    return game_name.strip() in set(source)


def detect_agent_type(player_name: str) -> str:
    """Infer whether the serialized player used CoT or IO prompting."""
    for pattern in ("CoT", "IO"):
        if f"({pattern})" in player_name:
            return pattern
    raise ValueError(f"Could not detect agent type from: {player_name}")


def extract_player_id(player_name: str) -> int:
    """Return the trailing #P suffix as an integer (e.g., '#P2' -> 2)."""
    match = PLAYER_ID_SUFFIX_RE.search(player_name)
    if not match:
        raise ValueError(f"Invalid player name format: {player_name}")
    return int(match.group("player_id"))


def extract_agent_name(player_name: str) -> str:
    """Drop the trailing ``#P<id>`` seat suffix from a serialized player name."""
    base = PLAYER_ID_SUFFIX_RE.sub("", player_name)
    return base.strip()


def extract_model_name(player_name: str) -> str:
    """Strip agent annotations and seat suffix from serialized player names."""
    agent_name = extract_agent_name(player_name)
    base = agent_name.replace("(CoT)", "").replace("(IO)", "")
    return base.strip()


def normalize_filter(values: list[str] | None) -> set[str]:
    """Normalize optional CLI filter lists to lowercase sets."""
    if not values:
        return set()
    return {
        value.strip().lower() for value in values if value and value.strip()
    }


def sort_mechanisms(mechanisms: Iterable[str]) -> list[str]:
    """Sort mechanisms in the preferred predefined order.

    Unknown mechanisms fall back to alphabetical order after the known ones.
    """
    preferred_order = [
        "NoMechanism",
        "Repetition",
        "ReputationFirstOrder",
        "Reputation",
        "Similarity",
        "Disarmament",
        "Mediation",
        "Contracting",
    ]
    order_map = {name.lower(): i for i, name in enumerate(preferred_order)}

    def sort_key(mech: str) -> tuple[int, int | str, str]:
        base_mech = mech.split(" (")[0]
        mech_lower = base_mech.lower()
        if mech_lower in order_map:
            return (0, order_map[mech_lower], mech)
        return (1, mech, "")

    return sorted(mechanisms, key=sort_key)


def sort_models(models: Iterable[str]) -> list[str]:
    """Sort models in the preferred order."""

    def sort_key(model: str) -> tuple[int, str]:
        model_lower = model.lower()
        if "claude" in model_lower:
            return (0, model)
        if "gemini" in model_lower and "(cot)" in model_lower:
            return (1, model)
        if "gemini" in model_lower:
            return (2, model)
        if "gpt-5" in model_lower:
            return (3, model)
        if "gpt-4" in model_lower:
            return (4, model)
        if "gpt-oss" in model_lower:
            return (5, model)
        if "qwen" in model_lower:
            return (6, model)
        return (7, model)

    return sorted(models, key=sort_key)
