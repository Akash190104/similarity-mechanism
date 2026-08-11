#!/usr/bin/env python3
"""Export canonical taxonomy dataset (counts + share) from normalized JSONL."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

NORMALIZED_SUBDIR = "normalized"
DATASET_SUBDIR = "dataset"
DATASET_COUNTS = "taxonomy_by_game_mechanism_model_player.counts.csv"
DATASET_SHARES = "taxonomy_by_game_mechanism_model_player.share_pct.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build canonical taxonomy datasets (counts + share) from "
            "normalized JSONL."
        )
    )
    parser.add_argument(
        "judge_dir",
        type=Path,
        help=(
            "Path to the judge directory containing "
            "normalized/normalized.jsonl, e.g. "
            "outputs/2026/04/17/22:43/judge/gpt4o-mini."
        ),
    )
    return parser.parse_args()


def unique_labels(value: Any) -> list[str]:
    if isinstance(value, list):
        labels = [x for x in value if isinstance(x, str) and x.strip()]
    elif isinstance(value, str):
        labels = [x.strip() for x in value.split(",") if x.strip()]
    else:
        labels = []
    seen: set[str] = set()
    out: list[str] = []
    for label in labels:
        if label not in seen:
            seen.add(label)
            out.append(label)
    return out


def pct(count: int, denom: int) -> float:
    return 0.0 if denom == 0 else (100.0 * count / denom)


def export_dataset(normalized_jsonl: Path, output_dir: Path) -> None:
    grouped_counts: dict[tuple[str, str, str, str], Counter[str]] = defaultdict(
        Counter
    )
    row_counts: Counter[tuple[str, str, str, str]] = Counter()
    labels_seen: set[str] = set()

    with normalized_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            labels = unique_labels(
                row.get("classification_labels_normalized")
                or row.get("classification_labels")
            )
            if not labels:
                continue
            game = str(row.get("game", "Unknown"))
            mechanism = str(row.get("mechanism", "Unknown"))
            model = str(row.get("model", "Unknown"))
            player = str(row.get("player", "Unknown"))
            key = (game, mechanism, model, player)
            row_counts[key] += 1
            for label in labels:
                grouped_counts[key][label] += 1
                labels_seen.add(label)

    if not grouped_counts:
        raise RuntimeError("No labeled rows found in normalized JSONL.")

    counts_path = output_dir / DATASET_COUNTS
    shares_path = output_dir / DATASET_SHARES

    with (
        counts_path.open("w", encoding="utf-8", newline="") as f_counts,
        shares_path.open("w", encoding="utf-8", newline="") as f_shares,
    ):
        counts_writer = csv.writer(f_counts)
        counts_writer.writerow(
            ["game", "mechanism", "model", "player", "label", "count"]
        )
        shares_writer = csv.writer(f_shares)
        shares_writer.writerow(
            [
                "game",
                "mechanism",
                "model",
                "player",
                "label",
                "share_pct",
                "group_count",
            ]
        )

        for key in sorted(grouped_counts.keys()):
            game, mechanism, model, player = key
            denom = row_counts[key]
            for label, count in grouped_counts[key].most_common():
                counts_writer.writerow(
                    [game, mechanism, model, player, label, count]
                )
                share_value = pct(count, denom)
                shares_writer.writerow(
                    [
                        game,
                        mechanism,
                        model,
                        player,
                        label,
                        f"{share_value:.4f}",
                        denom,
                    ]
                )

    print(
        f"Wrote {counts_path.name} and {shares_path.name} "
        f"(labels={len(labels_seen)}, groups={len(grouped_counts)})"
    )


def main() -> None:
    args = parse_args()
    judge_dir = args.judge_dir.expanduser().resolve()
    normalized_dir = judge_dir / NORMALIZED_SUBDIR
    normalized_jsonl = (normalized_dir / "normalized.jsonl").resolve()
    if not normalized_jsonl.exists():
        raise FileNotFoundError(f"Missing normalized JSONL: {normalized_jsonl}")

    output_dir = judge_dir / DATASET_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    export_dataset(normalized_jsonl, output_dir)


if __name__ == "__main__":
    main()
