"""Shared utilities for similarity mechanisms and elicitation."""

import numpy as np

from benchmarks.registry import resolve_framing_words
from src.mechanisms.prompts import (
    SIMILARITY_CONSTRUCT_FRAMING,
    SIMILARITY_CONSTRUCT_FRAMING_MULTI,
    SIMILARITY_CUSTOM_FRAMING,
    SIMILARITY_DOMAIN_FRAMING,
    SIMILARITY_DOMAIN_FRAMING_MULTI,
    SIMILARITY_PERCENTAGE_FRAMING,
    SIMILARITY_PERCENTAGE_FRAMING_MULTI,
    SIMILARITY_PERCENTAGE_UPDATED_FRAMING,
    SIMILARITY_PERCENTAGE_UPDATED_FRAMING_MULTI,
    SIMILARITY_VAGUE_FRAMING,
)


def _display_pct(similarity_pct: int, difference_framing: bool | str) -> int:
    """Flip the percentage when showing difference/dissimilarity framing."""
    flip = bool(difference_framing) and difference_framing != "similar"
    return (100 - similarity_pct) if flip else similarity_pct


def build_similarity_framing(
    similarity_pct: int,
    prompt_mode: str = "percentage_updated",
    domain: str = "",
    custom_prompt: str = "",
    num_other_players: int = 1,
    difference_framing: bool | str = False,
) -> str:
    """Build the similarity framing string for a given percentage and mode.

    Args:
        similarity_pct: The raw similarity percentage. When difference_framing is
            "different"/"dissimilar", it is flipped to (100 - similarity_pct) for
            display (e.g. 70% similar → 30% different).
        prompt_mode: One of "percentage", "percentage_updated", "domain", "vague", "custom".
        domain: Domain string (only used when prompt_mode="domain").
        custom_prompt: Custom prompt template (only used when prompt_mode="custom").
            May contain {similarity_pct} placeholder.
        num_other_players: Number of other players in the game (for multiplayer framing).
        difference_framing: False/"similar" (default), True/"different", or "dissimilar".
            Controls whether the percentage is presented as similarity or difference.

    Returns:
        Formatted similarity framing string.
    """
    multi = num_other_players > 1
    measure_word, relation_word = resolve_framing_words(difference_framing)
    pct = _display_pct(similarity_pct, difference_framing)
    if prompt_mode == "percentage":
        if multi:
            return SIMILARITY_PERCENTAGE_FRAMING_MULTI.format(
                similarity_pct=pct,
                num_other_players=num_other_players,
                measure_word=measure_word,
            )
        return SIMILARITY_PERCENTAGE_FRAMING.format(
            similarity_pct=pct, measure_word=measure_word
        )
    elif prompt_mode == "percentage_updated":
        if multi:
            return SIMILARITY_PERCENTAGE_UPDATED_FRAMING_MULTI.format(
                similarity_pct=pct,
                num_other_players=num_other_players,
                measure_word=measure_word,
                relation_word=relation_word,
            )
        return SIMILARITY_PERCENTAGE_UPDATED_FRAMING.format(
            similarity_pct=pct,
            measure_word=measure_word,
            relation_word=relation_word,
        )
    elif prompt_mode == "domain":
        if multi:
            return SIMILARITY_DOMAIN_FRAMING_MULTI.format(
                similarity_pct=pct,
                domain=domain,
                num_other_players=num_other_players,
                measure_word=measure_word,
            )
        return SIMILARITY_DOMAIN_FRAMING.format(
            similarity_pct=pct,
            domain=domain,
            measure_word=measure_word,
        )
    elif prompt_mode == "vague":
        return SIMILARITY_VAGUE_FRAMING
    elif prompt_mode == "construct":
        if multi:
            return SIMILARITY_CONSTRUCT_FRAMING_MULTI.format(
                num_other_players=num_other_players
            )
        return SIMILARITY_CONSTRUCT_FRAMING
    elif prompt_mode == "custom":
        formatted_custom = custom_prompt.format(similarity_pct=pct)
        return SIMILARITY_CUSTOM_FRAMING.format(custom_text=formatted_custom)
    else:
        raise ValueError(f"Unknown prompt_mode: {prompt_mode!r}")


def js_divergence(p: dict, q: dict) -> float:
    """Compute Jensen-Shannon divergence between two action distributions.

    Args:
        p, q: Dicts mapping action keys (str or int) to percentage points (summing to 100).

    Returns:
        JS divergence in [0, 1]. 0 means identical distributions.
    """
    all_keys = sorted(set(p.keys()) | set(q.keys()))
    p_arr = np.array([p.get(k, 0) for k in all_keys], dtype=float)
    q_arr = np.array([q.get(k, 0) for k in all_keys], dtype=float)

    p_sum = p_arr.sum()
    q_sum = q_arr.sum()
    if p_sum == 0 or q_sum == 0:
        return 1.0
    p_arr /= p_sum
    q_arr /= q_sum

    m = 0.5 * (p_arr + q_arr)
    eps = 1e-12
    kl_pm = np.sum(p_arr * np.log((p_arr + eps) / (m + eps)))
    kl_qm = np.sum(q_arr * np.log((q_arr + eps) / (m + eps)))

    return float(0.5 * kl_pm + 0.5 * kl_qm)
