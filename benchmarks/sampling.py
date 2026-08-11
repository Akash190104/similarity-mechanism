"""Stratified sampling utilities for benchmarks."""

import random
from collections import defaultdict
from typing import Callable, Union


def stratified_sample(
    items: list[dict],
    category_key: Union[str, Callable[[dict], str]],
    max_items: int,
    seed: int = 42,
) -> list[dict]:
    """Sample items equally across subcategories.

    If len(items) <= max_items, returns all items unchanged.
    Otherwise, allocates slots equally across categories, randomly
    sampling within each. Categories with fewer items than their
    allocation contribute all items; surplus slots are redistributed.

    Args:
        items: Full list of question dicts.
        category_key: Either a string key to look up in each dict,
            or a callable that extracts the category string from a dict.
        max_items: Target number of items to select.
        seed: Random seed for reproducible sampling.

    Returns:
        Stratified subset of items, length <= max_items.
    """
    if len(items) <= max_items:
        return items

    get_cat = (
        (lambda item: item[category_key])
        if isinstance(category_key, str)
        else category_key
    )

    # Group items by category, preserving original indices
    groups: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for idx, item in enumerate(items):
        groups[get_cat(item)].append((idx, item))

    sorted_cats = sorted(groups.keys())
    num_cats = len(sorted_cats)

    # Two-pass allocation to handle categories with fewer items than their share
    remaining_slots = max_items
    allocation: dict[str, int] = {}
    unfilled_cats = list(sorted_cats)

    while remaining_slots > 0 and unfilled_cats:
        per_cat = remaining_slots // len(unfilled_cats)
        extra = remaining_slots % len(unfilled_cats)

        next_unfilled = []
        slots_used = 0

        for i, cat in enumerate(unfilled_cats):
            cat_alloc = per_cat + (1 if i < extra else 0)
            cat_size = len(groups[cat]) - allocation.get(cat, 0)

            if cat_size <= cat_alloc:
                # Category can't fill its allocation — take all remaining
                allocation[cat] = len(groups[cat])
                slots_used += cat_size
            else:
                allocation[cat] = allocation.get(cat, 0) + cat_alloc
                slots_used += cat_alloc
                next_unfilled.append(cat)

        remaining_slots -= slots_used
        unfilled_cats = next_unfilled

        # Break if no progress was made (all cats exhausted)
        if slots_used == 0:
            break

    # Sample within each category using the seed
    rng = random.Random(seed)
    selected: list[tuple[int, dict]] = []

    for cat in sorted_cats:
        cat_items = groups[cat]
        n = allocation.get(cat, 0)
        if n >= len(cat_items):
            selected.extend(cat_items)
        else:
            selected.extend(rng.sample(cat_items, n))

    # Sort by original index to preserve order
    selected.sort(key=lambda x: x[0])
    return [item for _, item in selected]
