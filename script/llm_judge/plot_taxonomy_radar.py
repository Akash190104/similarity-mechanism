#!/usr/bin/env python3
"""
Radar (polar) plots of taxonomy prevalence from an LLM judge run.

Reads ``<judge-dir>/normalized/normalized.jsonl`` (output of
``normalize_justification_labels.py``) and writes radar plots into
``<judge-dir>/figures/``:

- ``radar_overall.png`` — one polygon: per-category prevalence across all rows.
- ``radar_by_model.png`` — one polygon per agent model.
- ``radar_by_player.png`` — one polygon per player slot (P1 vs P2).
- ``radar_by_mechanism.png`` — one polygon per mechanism (skipped if only one).

Prevalence = share of rows for which a category is True. Works with both the
new per-category schema (``classification_category_assignments`` dict) and
the legacy schema (``classification_labels_normalized`` list).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.llm_judge.config import COOPERATION_TAXONOMY  # noqa: E402

CATEGORIES: list[str] = list(COOPERATION_TAXONOMY["categories"].keys())

LEGACY_OTHERS_KEY = "Other"
NEW_OTHERS_KEY = "Others"


def iter_normalized_rows(path: Path) -> Iterable[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def row_assignments(row: dict[str, Any]) -> dict[str, bool] | None:
    """Return per-category booleans for a row, or None if unusable.

    Excludes rows whose label is 'Failed classification'.
    """
    new = row.get("classification_category_assignments")
    if isinstance(new, dict) and new:
        if any(not isinstance(v, bool) for v in new.values()):
            return None
        return {cat: bool(new.get(cat, False)) for cat in CATEGORIES}

    labels = row.get("classification_labels_normalized")
    if isinstance(labels, list):
        if "Failed classification" in labels:
            return None
        label_set = set(labels)
        return {
            cat: (
                cat in label_set
                or (cat == NEW_OTHERS_KEY and LEGACY_OTHERS_KEY in label_set)
            )
            for cat in CATEGORIES
        }

    return None


def prevalence(rows: list[dict[str, bool]]) -> np.ndarray:
    """Return per-category prevalence as a numpy array of length len(CATEGORIES)."""
    if not rows:
        return np.zeros(len(CATEGORIES))
    counts = np.array(
        [[1.0 if row[cat] else 0.0 for cat in CATEGORIES] for row in rows]
    )
    return counts.mean(axis=0)


def short_label(name: str, width: int = 18) -> str:
    """Wrap long category names onto multiple lines for the radar axis."""
    words = name.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


def make_radar_axes(fig: plt.Figure) -> plt.Axes:
    ax = fig.add_subplot(111, projection="polar")
    n = len(CATEGORIES)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(theta)
    ax.set_xticklabels([short_label(c) for c in CATEGORIES], fontsize=8)
    ax.set_rlabel_position(180 / n)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=7)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.4)
    return ax


def close_polygon(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(CATEGORIES)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    theta = np.concatenate([theta, theta[:1]])
    values = np.concatenate([values, values[:1]])
    return theta, values


def plot_overall(
    rows: list[dict[str, bool]],
    out_path: Path,
    title: str,
) -> Path:
    fig = plt.figure(figsize=(9, 9))
    ax = make_radar_axes(fig)
    prev = prevalence(rows)
    theta, values = close_polygon(prev)
    ax.plot(theta, values, linewidth=2.0, color="#1f77b4")
    ax.fill(theta, values, alpha=0.25, color="#1f77b4")
    ax.set_title(f"{title}\nN = {len(rows)} reasoning traces", pad=24, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_grouped(
    grouped: dict[str, list[dict[str, bool]]],
    out_path: Path,
    title: str,
    legend_label: str,
) -> Path:
    fig = plt.figure(figsize=(11, 9))
    ax = make_radar_axes(fig)

    cmap = plt.get_cmap("tab10")
    sorted_groups = sorted(grouped.items(), key=lambda kv: kv[0])
    for i, (group, rows) in enumerate(sorted_groups):
        prev = prevalence(rows)
        theta, values = close_polygon(prev)
        color = cmap(i % cmap.N)
        ax.plot(
            theta,
            values,
            linewidth=2.0,
            color=color,
            label=f"{group} (N={len(rows)})",
        )
        ax.fill(theta, values, alpha=0.12, color=color)

    ax.legend(
        title=legend_label,
        loc="upper right",
        bbox_to_anchor=(1.30, 1.10),
        fontsize=8,
        title_fontsize=9,
    )
    ax.set_title(title, pad=24, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "judge_dir",
        type=Path,
        help="Path to judge directory, e.g. outputs/.../judge/<output-name>",
    )
    parser.add_argument(
        "--figures-subdir",
        type=str,
        default="figures",
        help="Subdir of judge_dir to write plots into (default: figures).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    judge_dir = args.judge_dir.expanduser().resolve()
    normalized_path = judge_dir / "normalized" / "normalized.jsonl"
    if not normalized_path.exists():
        raise FileNotFoundError(
            f"Normalized JSONL not found: {normalized_path}. "
            "Run normalize_justification_labels.py first."
        )

    figures_dir = judge_dir / args.figures_subdir
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, bool]] = []
    by_model: dict[str, list[dict[str, bool]]] = defaultdict(list)
    by_player: dict[str, list[dict[str, bool]]] = defaultdict(list)
    by_mechanism: dict[str, list[dict[str, bool]]] = defaultdict(list)
    skipped = 0

    for raw_row in iter_normalized_rows(normalized_path):
        assignments = row_assignments(raw_row)
        if assignments is None:
            skipped += 1
            continue
        all_rows.append(assignments)

        model = str(raw_row.get("model", "unknown"))
        by_model[model].append(assignments)

        player = str(raw_row.get("player", "unknown"))
        slot = "P1" if "#P1" in player else "P2" if "#P2" in player else "P?"
        by_player[slot].append(assignments)

        mechanism = str(raw_row.get("mechanism", "unknown"))
        by_mechanism[mechanism].append(assignments)

    if not all_rows:
        raise RuntimeError(
            "No usable rows found in normalized JSONL "
            "(all rows had failed classification or missing fields)."
        )

    written: list[Path] = []

    written.append(
        plot_overall(
            all_rows,
            figures_dir / "radar_overall.png",
            title="Per-category prevalence (overall)",
        )
    )

    if len(by_model) >= 2:
        written.append(
            plot_grouped(
                by_model,
                figures_dir / "radar_by_model.png",
                title="Per-category prevalence by agent model",
                legend_label="Model",
            )
        )

    if len(by_player) >= 2:
        written.append(
            plot_grouped(
                by_player,
                figures_dir / "radar_by_player.png",
                title="Per-category prevalence by player position",
                legend_label="Player",
            )
        )

    if len(by_mechanism) >= 2:
        written.append(
            plot_grouped(
                by_mechanism,
                figures_dir / "radar_by_mechanism.png",
                title="Per-category prevalence by mechanism",
                legend_label="Mechanism",
            )
        )

    print(f"Rows used: {len(all_rows)} | skipped (failed classification): {skipped}")
    print(f"Figures dir: {figures_dir}")
    for path in written:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
