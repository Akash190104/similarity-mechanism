"""Shared helpers for llm_judge plotting/report scripts.

In this repo, judge artifacts live next to the experiment run directory that
produced the raw reasoning traces (``<run-dir>/judge/<output-name>/…``), not in
a global output folder. All path helpers take an explicit ``judge_dir`` —
typically ``<run-dir>/judge/<output-name>`` — and derive subpaths from there.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

RAW_SUBDIR = "raw"
NORMALIZED_SUBDIR = "normalized"
DATASET_SUBDIR = "dataset"
FIGURES_SUBDIR = "figures"
REPORTS_SUBDIR = "reports"

NORMALIZED_JSON_FILENAME = "normalized.jsonl"
DATASET_SHARE_FILENAME = "taxonomy_by_game_mechanism_model_player.share_pct.csv"
DATASET_COUNTS_FILENAME = "taxonomy_by_game_mechanism_model_player.counts.csv"
MECHANISM_DIFFERENCES_FILENAME = "mechanism_label_differences.csv"


def validate_output_name(value: str) -> str:
    """Ensure the provided judge output name is a simple slug."""
    candidate = value.strip()
    if not candidate:
        raise ValueError("output_name cannot be empty.")
    if Path(candidate).name != candidate:
        raise ValueError("output_name must not include directory separators.")
    return candidate


def _ensure_exists(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path.resolve()


def _first_existing(description: str, candidates: Iterable[Path]) -> Path:
    tried: list[Path] = []
    for path in candidates:
        tried.append(path)
        if path.exists():
            return path.resolve()
    tried_list = "\n".join(f"- {p}" for p in tried)
    raise FileNotFoundError(f"Missing {description}. Looked in:\n{tried_list}")


def judge_dir_for_run(run_dir: Path, output_name: str) -> Path:
    """Return ``<run_dir>/judge/<output_name>`` without requiring it to exist."""
    return (Path(run_dir) / "judge" / validate_output_name(output_name)).resolve()


def raw_jsonl_path(judge_dir: Path) -> Path:
    """Return ``<judge_dir>/raw/judgement.jsonl``."""
    return (Path(judge_dir) / RAW_SUBDIR / "judgement.jsonl").resolve()


def raw_summary_path(judge_dir: Path) -> Path:
    """Return ``<judge_dir>/raw/judgement_summary.json``."""
    return (Path(judge_dir) / RAW_SUBDIR / "judgement_summary.json").resolve()


def normalized_jsonl_path(judge_dir: Path) -> Path:
    """Return ``<judge_dir>/normalized/normalized.jsonl`` (must exist)."""
    return _ensure_exists(
        Path(judge_dir) / NORMALIZED_SUBDIR / NORMALIZED_JSON_FILENAME,
        f"normalized JSONL under {judge_dir}",
    )


def dataset_share_csv_path(judge_dir: Path) -> Path:
    """Return canonical taxonomy share CSV for a judge run."""
    return _ensure_exists(
        Path(judge_dir) / DATASET_SUBDIR / DATASET_SHARE_FILENAME,
        f"dataset share CSV under {judge_dir}",
    )


def mechanism_differences_csv_path(judge_dir: Path) -> Path:
    """Locate mechanism_label_differences.csv for a judge run."""
    candidates = [
        Path(judge_dir) / MECHANISM_DIFFERENCES_FILENAME,
        Path(judge_dir) / FIGURES_SUBDIR / MECHANISM_DIFFERENCES_FILENAME,
        Path(judge_dir) / REPORTS_SUBDIR / MECHANISM_DIFFERENCES_FILENAME,
    ]
    return _first_existing(
        f"mechanism_label_differences.csv under {judge_dir}", candidates
    )


def prepare_figure_subdir(judge_dir: Path, relative: str) -> Path:
    """Ensure ``<judge_dir>/figures/<relative>/`` exists and return it."""
    path = Path(judge_dir) / FIGURES_SUBDIR / relative
    path.mkdir(parents=True, exist_ok=True)
    return path
