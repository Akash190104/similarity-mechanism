#!/usr/bin/env python3
"""Unified plotting CLI for the similarity project.

Consolidates the per-plot scripts under ``script/plot_*.py`` into a single
entry point with a ``--plot <name>`` style subparser.

Available subcommands (one ``plot_<name>`` function per subcommand):

  battle_of_the_sexes  Battle-of-the-Sexes tournament: payoff, coordination,
                       concession, joint-payoff lines, plus a 2x2 dashboard.
  benchmark_sweep      Per-model heatmaps of cooperation rate from a benchmark
                       sweep (benchmarks x similarity %), with combined view.
  combined_heatmap     Merge several elicitation_sweep.json files into a
                       single (similarity x model) cooperation heatmap.
  cooperation_with_sem Cooperation-rate vs similarity with SEM bands; falls
                       back to per-player bar plot for non-sweep records.
  defection_rate       Gemini-vs-Random comparison: similarity scores,
                       cooperation/defection bar groups, scatter, overview,
                       text table.
  fixed_point          Fixed-point similarity plots: per-pair f(s) curves,
                       heatmap of fixed points, multi-pair overlay.
  heatmaps             Mode x benchmark heatmaps (similarity / cooperation /
                       interleaved) from a comparison.json file.
  newcomb              Newcomb benchmark: capabilities/EDT/CDT bars, running
                       accuracy, EDT-vs-CDT scatter, agreement strip.
  similarity_heatmap   Heatmaps from similarity-sweep or benchmark-sweep
                       results JSONs (auto-detected by schema).
  similarity_tournament Two-phase similarity tournament: told-vs-actual,
                       elicitation+validation payoffs, payoff comparison,
                       per-pair cooperation rate.
  trust_game_sweep     Trust-game (asymmetric-game) sweep: per-matchup payoff
                       curves and aggregate per-model payoff.

Usage:
    python script/plot.py <subcommand> [args]
    python script/plot.py battle_of_the_sexes path/to/results.json
    python script/plot.py cooperation_with_sem --run-dir outputs/.../12:34
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import stdev
from typing import Any, Iterable

# ── Third-party ───────────────────────────────────────────────────────────────
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch

# ── Project paths ─────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from config import PROJECT_DIR as _CONFIG_PROJECT_DIR  # noqa: E402

# Use the value from config.py to stay consistent with the rest of the code base.
PROJECT_DIR = _CONFIG_PROJECT_DIR


# =============================================================================
#  Shared helpers (extracted from the original per-plot scripts)
# =============================================================================

# Regex that strips trailing seat suffix (e.g. "#P0", "#P1") from agent ids.
_SEAT_SUFFIX_RE = re.compile(r"#P\d+$")


def load_json(path: Path | str) -> dict:
    """Read a JSON file and return the decoded object."""
    return json.loads(Path(path).read_text())


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (treated as a directory) if missing and return it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def short_model_name(agent: str) -> str:
    """Strip provider prefix and parenthesised type tag.

    e.g. ``google/gemini-3-flash-preview(CoT)`` -> ``gemini-3-flash-preview``.
    """
    return agent.split("/")[-1].split("(")[0]


def strip_seat(agent: str) -> str:
    """Drop the trailing ``#P<N>`` suffix (group by model rather than seat)."""
    return _SEAT_SUFFIX_RE.sub("", agent)


def short_pair_name(agent: str) -> str:
    """Variant used by the tournament plots: drop seat *and* (CoT)/(IO) tag."""
    name = strip_seat(agent)
    name = re.sub(r"\(.*?\)$", "", name)
    return name.split("/")[-1]


def safe_filename(name: str) -> str:
    """Make a string safe to use as a path segment."""
    return name.replace("/", "_").replace(" ", "_")


def save_fig(fig: plt.Figure, *paths: Path, dpi: int = 300, close: bool = True) -> None:
    """Save ``fig`` to one or more paths and (optionally) close it."""
    for p in paths:
        p = Path(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {p}")
    if close:
        plt.close(fig)


def tab10_colors(n: int) -> np.ndarray:
    """Return ``n`` distinct colours from the tab10 colormap."""
    return plt.cm.tab10(np.linspace(0, 1, max(n, 1)))


def mean_or_zero(values: list) -> float:
    return float(np.mean(values)) if values else 0.0


def sem(values: list) -> float:
    """Standard error of the mean, returning 0 for n < 2."""
    return float(np.std(values) / np.sqrt(len(values))) if len(values) > 1 else 0.0


def mean_sem(values: list[int]) -> tuple[float, float, int]:
    """Return ``(mean, sem, n)`` for a list of indicator/value samples."""
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = sum(values) / n
    if n < 2:
        return mean, float("nan"), n
    return mean, stdev(values) / math.sqrt(n), n


def write_csv(rows: list[dict], path: Path) -> None:
    """Write a list of homogeneous dict rows as CSV (or empty file if no rows)."""
    if not rows:
        Path(path).write_text("")
        return
    rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
#  battle_of_the_sexes
# =============================================================================

_BOS_STYLES = {
    "self":  dict(linestyle="-",  marker="o"),
    "cross": dict(linestyle="--", marker="s", alpha=0.7),
}


def plot_battle_of_the_sexes(args: argparse.Namespace) -> None:
    """Plot Battle of the Sexes tournament results (payoff, coord., concession, joint)."""
    results_path: Path = args.results_path
    results = load_json(results_path)

    output_dir = results_path.parent
    plots_dir = ensure_dir(PROJECT_DIR / "plots" / "battle_of_the_sexes")

    percentages = sorted(int(p) for p in results["results_by_percentage"].keys())

    all_models: set[str] = set()
    for pct_results in results["results_by_percentage"].values():
        for m in pct_results["matches"]:
            all_models.add(strip_seat(m["agent1_name"]))
            all_models.add(strip_seat(m["agent2_name"]))
    agents = sorted(all_models)

    PLAY = ("self", "cross")

    payoffs      = {pt: {a: defaultdict(list) for a in agents} for pt in PLAY}
    concession   = {pt: {a: defaultdict(list) for a in agents} for pt in PLAY}
    coordination = {pt: {a: defaultdict(list) for a in agents} for pt in PLAY}
    joint_payoff = {pt: {a: defaultdict(list) for a in agents} for pt in PLAY}

    for pct in percentages:
        pct_key = str(pct)
        if pct_key not in results["results_by_percentage"]:
            continue
        for m in results["results_by_percentage"][pct_key]["matches"]:
            m1 = strip_seat(m["agent1_name"])
            m2 = strip_seat(m["agent2_name"])
            a1, a2 = m["agent1_action"], m["agent2_action"]
            pt = "self" if m1 == m2 else "cross"
            jp = m["agent1_payoff"] + m["agent2_payoff"]
            coordinated = int(a1 == a2)

            for agent in (m1, m2):
                payoffs[pt][agent][pct].append(
                    m["agent1_payoff"] if agent == m1 else m["agent2_payoff"]
                )
                joint_payoff[pt][agent][pct].append(jp)
                coordination[pt][agent][pct].append(coordinated)

            concession[pt][m1][pct].append(1 if a1 == "A1" else 0)
            concession[pt][m2][pct].append(1 if a2 == "A0" else 0)

    colors = tab10_colors(len(agents))
    seed = results.get("seed", "")

    def _four_lines(ax, data, scale: float = 1.0, use_errorbar: bool = False) -> None:
        for idx, agent in enumerate(agents):
            name = short_model_name(agent)
            for pt in PLAY:
                vals = [mean_or_zero(data[pt][agent][p]) * scale for p in percentages]
                kw = dict(color=colors[idx], linewidth=2, markersize=6,
                          label=f"{name} ({pt})", **_BOS_STYLES[pt])
                if use_errorbar:
                    errs = [sem(data[pt][agent][p]) * scale for p in percentages]
                    ax.errorbar(percentages, vals, yerr=errs, capsize=3, **kw)
                else:
                    ax.plot(percentages, vals, **kw)

    # Plot 1: Payoff
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    _four_lines(ax1, payoffs, use_errorbar=True)
    ax1.set_xlabel("Similarity Percentage (%)", fontsize=12)
    ax1.set_ylabel("Average Payoff", fontsize=12)
    ax1.set_title("Battle of the Sexes — Payoff vs Similarity", fontsize=14, fontweight="bold")
    ax1.legend(loc="best", fontsize=9); ax1.grid(True, alpha=0.3); ax1.set_xlim(-5, 105)
    save_fig(fig1, output_dir / "bos_payoffs.png", plots_dir / f"bos_payoffs_{seed}.png")

    # Plot 2: Coordination rate
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    _four_lines(ax2, coordination, scale=100)
    ax2.set_xlabel("Similarity Percentage (%)", fontsize=12)
    ax2.set_ylabel("Coordination Rate (%)", fontsize=12)
    ax2.set_title(
        "Battle of the Sexes — Coordination Rate vs Similarity\n"
        "(fraction where both players chose the same action)",
        fontsize=13, fontweight="bold",
    )
    ax2.legend(loc="best", fontsize=9)
    ax2.set_ylim(-5, 105); ax2.set_xlim(-5, 105); ax2.grid(True, alpha=0.3)
    save_fig(fig2, output_dir / "bos_coordination.png", plots_dir / f"bos_coordination_{seed}.png")

    # Plot 3: Concession rate
    fig3, ax3 = plt.subplots(figsize=(12, 7))
    _four_lines(ax3, concession, scale=100)
    ax3.set_xlabel("Similarity Percentage (%)", fontsize=12)
    ax3.set_ylabel("Concession Rate (%)", fontsize=12)
    ax3.set_title(
        "Battle of the Sexes — Concession Rate vs Similarity\n"
        "(fraction where a player chose the OTHER player's preferred action)",
        fontsize=13, fontweight="bold",
    )
    ax3.legend(loc="best", fontsize=9)
    ax3.set_ylim(-5, 105); ax3.set_xlim(-5, 105); ax3.grid(True, alpha=0.3)
    save_fig(fig3, output_dir / "bos_concession.png", plots_dir / f"bos_concession_{seed}.png")

    # Plot 4: Joint payoff
    fig4, ax4 = plt.subplots(figsize=(12, 7))
    _four_lines(ax4, joint_payoff, use_errorbar=True)
    ax4.axhline(5.0, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Max (full coord.)")
    ax4.set_xlabel("Similarity Percentage (%)", fontsize=12)
    ax4.set_ylabel("Average Joint Payoff", fontsize=12)
    ax4.set_title(
        "Battle of the Sexes — Joint Payoff vs Similarity\n"
        "(coordination = 5, miscoordination = 0)",
        fontsize=13, fontweight="bold",
    )
    ax4.legend(loc="best", fontsize=9)
    ax4.set_xlim(-5, 105); ax4.set_ylim(-0.5, 5.5); ax4.grid(True, alpha=0.3)
    save_fig(fig4, output_dir / "bos_joint_payoff.png", plots_dir / f"bos_joint_payoff_{seed}.png")

    # Dashboard (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(
        "Battle of the Sexes — Similarity Tournament Dashboard",
        fontsize=16, fontweight="bold", y=0.99,
    )

    _four_lines(axes[0, 0], payoffs)
    axes[0, 0].set_xlabel("Similarity %"); axes[0, 0].set_ylabel("Avg Payoff")
    axes[0, 0].set_title("Payoff"); axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3); axes[0, 0].set_xlim(-5, 105)

    _four_lines(axes[0, 1], joint_payoff)
    axes[0, 1].axhline(5.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    axes[0, 1].set_xlabel("Similarity %"); axes[0, 1].set_ylabel("Joint Payoff")
    axes[0, 1].set_title("Joint Payoff"); axes[0, 1].legend(fontsize=8)
    axes[0, 1].set_xlim(-5, 105); axes[0, 1].set_ylim(-0.5, 5.5); axes[0, 1].grid(True, alpha=0.3)

    _four_lines(axes[1, 0], coordination, scale=100)
    axes[1, 0].set_xlabel("Similarity %"); axes[1, 0].set_ylabel("Coord. Rate (%)")
    axes[1, 0].set_title("Coordination Rate"); axes[1, 0].legend(fontsize=8)
    axes[1, 0].set_ylim(-5, 105); axes[1, 0].set_xlim(-5, 105); axes[1, 0].grid(True, alpha=0.3)

    _four_lines(axes[1, 1], concession, scale=100)
    axes[1, 1].set_xlabel("Similarity %"); axes[1, 1].set_ylabel("Concession Rate (%)")
    axes[1, 1].set_title("Concession Rate"); axes[1, 1].legend(fontsize=8)
    axes[1, 1].set_ylim(-5, 105); axes[1, 1].set_xlim(-5, 105); axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_fig(fig, output_dir / "bos_dashboard.png", plots_dir / f"bos_dashboard_{seed}.png")

    print(f"\nAll plots saved to:")
    print(f"  - {output_dir}/")
    print(f"  - {plots_dir}/")


# =============================================================================
#  benchmark_sweep
# =============================================================================

_BENCHMARK_ORDER = [
    "random_coin_toss_alt",
    "random_die_roll_alt",
    "cabin",
    "trait",
    "hle",
    "ggb",
    "dilemmas",
    "moral_choice",
    "newcomb",
    "similarity_game",
]

_BENCHMARK_DISPLAY_NAMES = {
    "random_coin_toss_alt": "Random Coin",
    "random_die_roll_alt": "Random Die",
    "cabin": "CABIN",
    "trait": "TRAIT",
    "hle": "HLE",
    "ggb": "GGB",
    "dilemmas": "DDilemma",
    "moral_choice": "Moral",
    "newcomb": "Newcomb",
    "similarity_game": "Similarity",
}

# Order used in the side-by-side combined heatmap (left → right).
_MODEL_ORDER_DISPLAY = ["Gemini", "GPT", "Claude", "DeepSeek", "Gemma"]
_MODEL_DISPLAY_NAMES = {
    "gemini-3-flash-preview": "Gemini",
    "gpt-5.4-mini": "GPT",
    "claude-haiku-4.5": "Claude",
    "deepseek-v4-pro": "DeepSeek",
    "gemma-4-31b-it": "Gemma",
}


def display_model_name(agent: str) -> str:
    """Map a full agent string to its display name (e.g. ``GPT``); falls back
    to ``short_model_name`` when no canonical mapping exists."""
    short = short_model_name(agent)
    return _MODEL_DISPLAY_NAMES.get(short, short)


def _bench_build_matrix(model_rates, benchmarks, percentages, lookup_pcts=None):
    if lookup_pcts is None:
        lookup_pcts = percentages
    matrix = np.full((len(benchmarks), len(percentages)), np.nan)
    for i, bench in enumerate(benchmarks):
        if bench not in model_rates:
            continue
        for j, pct in enumerate(lookup_pcts):
            entry = model_rates[bench].get(str(pct), {})
            # The cooperative action goes by different names per game (PD →
            # COOPERATE, StagHunt → STAG, PublicGoods → CONTRIBUTE, ...).
            # Match against any token in ``_COOP_TOKENS`` that's present.
            coop = next(
                (entry[k] for k in entry if k.upper() in _COOP_TOKENS),
                entry.get("COOPERATE", entry.get("Cooperate", 0.0)),
            )
            matrix[i, j] = coop * 100
    return matrix


def _bench_build_sem_matrix(results_by_benchmark, model_id, benchmarks,
                            percentages, lookup_pcts=None):
    """SEM (in % units) of cooperation rate for one model, per (benchmark, pct).

    Each move's contribution is the agent's *continuous* cooperate-probability
    (``mix_probs["A0"] / 100``) when available, falling back to the binary
    sampled action otherwise. SEM is computed *within each cell* — so cells
    where both seats agree show SEM = 0, and cells with disagreement show
    larger values bounded by ``|x1 − x2| / 2`` for n=2.
    """
    if lookup_pcts is None:
        lookup_pcts = percentages
    matrix = np.full((len(benchmarks), len(percentages)), np.nan)
    for i, bench in enumerate(benchmarks):
        bench_data = results_by_benchmark.get(bench, {})
        for j, pct in enumerate(lookup_pcts):
            trials = bench_data.get(str(pct)) or bench_data.get(pct) or []
            samples: list[float] = []
            for trial in trials:
                for round_moves in trial:
                    for move in round_moves:
                        if _SEAT_SUFFIX_RE.sub("", move["player"]) != model_id:
                            continue
                        mp = move.get("mix_probs")
                        if mp:
                            samples.append(mp.get("A0", 0) / 100.0)
                        else:
                            samples.append(1.0 if move["action"] == "A0" else 0.0)
            _, sem_v, n = mean_sem(samples)
            if n >= 2 and not math.isnan(sem_v):
                matrix[i, j] = 100.0 * sem_v
    return matrix


def _bench_plot_single(matrix, benchmarks, percentages, model_name, out_path,
                       x_label="Similarity Percentage", sem_matrix=None):
    # Transpose so rows = similarity %, cols = benchmarks; flip rows so 100%
    # is at the top of the y-axis.
    plot_mat = np.flipud(matrix.T)
    plot_sem = np.flipud(sem_matrix.T) if sem_matrix is not None else None
    pct_order = list(reversed(percentages))

    fig, ax = plt.subplots(figsize=(max(10, len(benchmarks) * 1.1),
                                    max(6, len(percentages) * 0.55)))
    im = ax.imshow(plot_mat, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    display = [_BENCHMARK_DISPLAY_NAMES.get(b, b) for b in benchmarks]
    ax.set_xticks(range(len(benchmarks)))
    ax.set_xticklabels(display, fontsize=10, rotation=30, ha="right")
    ax.set_xlabel("Benchmark", fontsize=12, fontweight="bold")

    ax.set_yticks(range(len(pct_order)))
    ax.set_yticklabels([f"{p}%" for p in pct_order], fontsize=10,
                       fontstyle="italic")
    ax.set_ylabel(x_label, fontsize=12, fontweight="bold")

    for i in range(plot_mat.shape[0]):
        for j in range(plot_mat.shape[1]):
            val = plot_mat[i, j]
            if np.isnan(val):
                continue
            color = "white" if val < 30 or val > 70 else "black"
            if plot_sem is not None and not np.isnan(plot_sem[i, j]):
                ax.text(j, i, f"{val:.1f}\n\u00b1 {plot_sem[i, j]:.1f}",
                        ha="center", va="center", fontsize=8,
                        fontweight="bold", color=color)
            else:
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Cooperation Rate (%)", fontsize=11)

    ax.set_title(f"Spoofed Benchmark \u2014 {display_model_name(model_name)}",
                 fontsize=14, fontweight="bold", pad=10)
    fig.tight_layout()
    save_fig(fig, out_path, dpi=150)


def plot_benchmark_sweep(args: argparse.Namespace) -> None:
    """Per-model heatmaps of cooperation rate from a benchmark sweep run."""
    data = load_json(args.results)
    out_dir = ensure_dir(Path(args.output_dir) if args.output_dir else Path(args.results).parent)

    rates = data["action_rates_by_sweep"]
    available = set(data["benchmarks"])
    percentages = data["percentages"]
    results_by_benchmark = data.get("results_by_benchmark") or {}

    # Resolve framing: explicit --label/--flip > JSON > sibling config.json
    if "difference_framing" in data:
        df = data["difference_framing"]
    else:
        df = False
        cfg_path = Path(args.results).parent / "config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            df = (
                cfg.get("mechanism", {})
                .get("kwargs", {})
                .get("difference_framing", False)
            )
    if df is True:
        detected_key = "different"
    elif df is False or df == "similar":
        detected_key = "similar"
    else:
        detected_key = str(df)
    default_labels = {
        "similar": "Similarity",
        "different": "Difference",
        "dissimilar": "Dissimilarity",
    }
    label = args.label or default_labels.get(detected_key, "Similarity")
    if args.flip == "auto":
        flip = detected_key != "similar"
    else:
        flip = args.flip == "yes"

    lookup_pcts = [100 - p for p in percentages] if flip else percentages

    benchmarks = [b for b in _BENCHMARK_ORDER if b in available]
    for b in data["benchmarks"]:
        if b not in benchmarks:
            benchmarks.append(b)

    # Order models for the combined image: canonical Gemini\u2192Gemma sequence
    # first, then any unrecognised models appended alphabetically.
    short_to_full = {short_model_name(m): m for m in rates.keys()}
    canonical_short_order = [s for s, d in _MODEL_DISPLAY_NAMES.items()
                             if d in _MODEL_ORDER_DISPLAY]
    canonical_short_order.sort(key=lambda s: _MODEL_ORDER_DISPLAY.index(_MODEL_DISPLAY_NAMES[s]))
    models = [short_to_full[s] for s in canonical_short_order if s in short_to_full]
    for s, full in sorted(short_to_full.items()):
        if full not in models:
            models.append(full)
    x_label = f"{label} Percentage"

    for model in models:
        matrix = _bench_build_matrix(rates[model], benchmarks, percentages, lookup_pcts)
        sem_matrix = (
            _bench_build_sem_matrix(results_by_benchmark, model, benchmarks,
                                    percentages, lookup_pcts)
            if results_by_benchmark
            else None
        )
        safe = safe_filename(short_model_name(model))
        _bench_plot_single(matrix, benchmarks, percentages, model,
                           out_dir / f"sweep_heatmap_{safe}.png",
                           x_label=x_label, sem_matrix=sem_matrix)

    if len(models) >= 2:
        # Each subplot now has similarity on Y, benchmarks on X.
        panel_w = max(6, len(benchmarks) * 0.7)
        panel_h = max(8, len(percentages) * 0.85)
        fig, axes = plt.subplots(
            1, len(models),
            figsize=(panel_w * len(models), panel_h),
        )
        if len(models) == 1:
            axes = [axes]

        pct_order = list(reversed(percentages))
        display_benches = [_BENCHMARK_DISPLAY_NAMES.get(b, b) for b in benchmarks]

        for ax, model in zip(axes, models):
            matrix = _bench_build_matrix(rates[model], benchmarks, percentages, lookup_pcts)
            sem_matrix = (
                _bench_build_sem_matrix(results_by_benchmark, model, benchmarks,
                                        percentages, lookup_pcts)
                if results_by_benchmark
                else None
            )
            plot_mat = np.flipud(matrix.T)
            plot_sem = np.flipud(sem_matrix.T) if sem_matrix is not None else None

            im = ax.imshow(plot_mat, cmap="RdYlGn", aspect="auto",
                           vmin=0, vmax=100)
            ax.set_xticks(range(len(benchmarks)))
            ax.set_xticklabels(display_benches, fontsize=9,
                               rotation=30, ha="right")
            ax.set_xlabel("Benchmark", fontsize=11, fontweight="bold")

            ax.set_yticks(range(len(pct_order)))
            ax.set_yticklabels([f"{p}%" for p in pct_order], fontsize=9,
                               fontstyle="italic")
            ax.set_ylabel(x_label, fontsize=11, fontweight="bold")
            ax.set_title(f"Spoofed Benchmark \u2014 {display_model_name(model)}",
                         fontsize=13, fontweight="bold")

            for i in range(plot_mat.shape[0]):
                for j in range(plot_mat.shape[1]):
                    val = plot_mat[i, j]
                    if np.isnan(val):
                        continue
                    color = "white" if val < 30 or val > 70 else "black"
                    if plot_sem is not None and not np.isnan(plot_sem[i, j]):
                        ax.text(j, i, f"{val:.1f}\n\u00b1 {plot_sem[i, j]:.1f}",
                                ha="center", va="center", fontsize=7,
                                fontweight="bold", color=color)
                    else:
                        ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                                fontsize=8, fontweight="bold", color=color)

            plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

        fig.suptitle(f"Spoofed Benchmark \u2014 {label}",
                     fontsize=15, fontweight="bold", y=1.01)
        fig.tight_layout()
        save_fig(fig, out_dir / "sweep_heatmap_combined.png", dpi=150)


# =============================================================================
#  combined_heatmap
# =============================================================================

def _combined_load_and_merge(paths: list[str]):
    merged: dict = defaultdict(lambda: defaultdict(list))
    percentages = None
    for p in paths:
        data = load_json(p)
        pcts = data["percentages"]
        if percentages is None:
            percentages = pcts
        for pct_str, entries in data["results_by_percentage"].items():
            pct = int(pct_str)
            for e in entries:
                model = short_model_name(e["agent_name"])
                coop = e.get("action_distribution", {}).get("A0", 0)
                merged[model][pct].append(coop)
    return merged, sorted(percentages or [])


def plot_combined_heatmap(args: argparse.Namespace) -> None:
    """Merge multiple elicitation_sweep.json files into one cooperation heatmap."""
    merged, percentages = _combined_load_and_merge(args.inputs)

    if args.model_order:
        models = [m.strip() for m in args.model_order.split(",")]
    else:
        models = sorted(merged.keys())

    matrix = []
    for pct in percentages:
        row = [np.mean(merged[m].get(pct, [np.nan])) for m in models]
        matrix.append(row)
    matrix = np.flipud(np.array(matrix))
    pcts_rev = percentages[::-1]

    fig, ax = plt.subplots(figsize=(max(10, len(models) * 2), max(6, len(percentages) * 0.55)))
    sns.heatmap(
        matrix,
        xticklabels=models,
        yticklabels=[f"{p}%" for p in pcts_rev],
        cmap="RdYlGn",
        vmin=0, vmax=100,
        annot=True, fmt=".1f",
        cbar_kws={"label": "Cooperation Rate (%)"},
        ax=ax, linewidths=0.5, linecolor="gray",
    )
    ax.set_xlabel("Model", fontsize=13, fontweight="bold")
    ax.set_ylabel("Told Similarity (%)", fontsize=13, fontweight="bold")
    ax.set_title("Cooperation Rate by Similarity Level — Combined", fontsize=14, fontweight="bold", pad=20)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)

    out = Path(args.out)
    save_fig(fig, out)
    print(f"Models: {models}")
    print(f"N per (model, pct) cell:")
    for m in models:
        counts = [len(merged[m].get(p, [])) for p in percentages]
        print(f"  {m:35s} {counts}")


# =============================================================================
#  cooperation_with_sem
# =============================================================================

_COOP_GAMES = {
    "PrisonersDilemma",
    "PublicGoods",
    "StagHunt",
    "TrustGame",
    "TravellersDilemma",
    "Chicken",
}

# Fallback (non-sweep) detector tokens.
_COOP_TOKENS = ("COOPERATE", "CONTRIBUTE", "STAG", "GIVE", "SAFE", "HIGH")

_COOP_PLOT_SUBDIR = "cooperation_plots"


def _coop_y_label(game: str) -> str:
    return "Cooperation rate" if game in _COOP_GAMES else "P(A0)"


def _coop_player_name(move: dict) -> str:
    return move.get("player") or move.get("name") or "unknown"


def _coop_is_coop_action(action: str, *, sweep_format: bool) -> int:
    if sweep_format:
        return 1 if action == "A0" else 0
    s = action.upper()
    return 1 if any(tok in s for tok in _COOP_TOKENS) else 0


def _coop_indicator(move: dict, *, sweep_format: bool) -> float:
    """Continuous cooperation probability for one move (0-1).

    Uses the agent's reported ``mix_probs["A0"] / 100`` when available — the
    upstream mean is computed from these continuous values, so SEM should be
    too. Falls back to the binary sampled action when ``mix_probs`` isn't
    present. Handles both ``mix_probs`` schemas: ``{"A0": 60, "A1": 40}`` and
    the raw-index variant ``{"0": 60, "1": 40}`` used by benchmark-sweep
    records.
    """
    mp = move.get("mix_probs")
    if mp:
        if "A0" in mp:
            return mp["A0"] / 100.0
        if "0" in mp:
            return mp["0"] / 100.0
        if 0 in mp:
            return mp[0] / 100.0
    return float(_coop_is_coop_action(str(move.get("action", "")), sweep_format=sweep_format))


def _coop_flatten_trial_moves(trial: list) -> Iterable[dict]:
    """Yield every move dict from a trial (depth-2 or depth-3 nesting)."""
    for entry in trial:
        if isinstance(entry, dict):
            yield entry
            continue
        if not isinstance(entry, list):
            continue
        if entry and isinstance(entry[0], dict):
            for move in entry:
                if isinstance(move, dict):
                    yield move
        else:
            for matchup in entry:
                if not isinstance(matchup, list):
                    continue
                for move in matchup:
                    if isinstance(move, dict):
                        yield move


def _coop_summarize_pct_block(by_pct: dict, percentages: list, *,
                              benchmark: str | None = None) -> list[dict]:
    rows: list[dict] = []
    for pct in percentages:
        per_model: dict[str, list[float]] = {}
        for trial in by_pct.get(str(pct), []):
            for move in _coop_flatten_trial_moves(trial):
                model = strip_seat(_coop_player_name(move))
                per_model.setdefault(model, []).append(
                    _coop_indicator(move, sweep_format=True)
                )

        for model, samples in sorted(per_model.items()):
            mean, sem_v, n = mean_sem(samples)
            row = {
                "similarity_pct": int(pct),
                "model": model,
                "n": n,
                "mean_coop": mean,
                "sem_coop": sem_v,
                "_samples": samples,
            }
            if benchmark is not None:
                row["benchmark"] = benchmark
            rows.append(row)
    return rows


def _coop_summarize_sweep(sweep: dict) -> list[dict]:
    return _coop_summarize_pct_block(
        sweep.get("results_by_percentage", {}),
        sweep.get("percentages", []),
    )


def _coop_summarize_benchmark_sweep(sweep: dict) -> list[dict]:
    rows: list[dict] = []
    percentages = sweep.get("percentages", [])
    for benchmark, by_pct in (sweep.get("results_by_benchmark") or {}).items():
        rows.extend(_coop_summarize_pct_block(by_pct, percentages, benchmark=benchmark))
    return rows


def _coop_pooled_sem_by_model_pct(rows: list[dict]) -> dict[tuple[str, int], float]:
    """SEM at each (model, pct), pooling samples across benchmarks (Option A)."""
    pooled: dict[tuple[str, int], list[float]] = {}
    for r in rows:
        samples = r.get("_samples")
        if not samples:
            continue
        key = (r["model"], r["similarity_pct"])
        pooled.setdefault(key, []).extend(samples)
    out: dict[tuple[str, int], float] = {}
    for key, samples in pooled.items():
        _, sem, n = mean_sem(samples)
        if n >= 2 and not math.isnan(sem):
            out[key] = sem
    return out


def _coop_aggregate_across_benchmarks(rows: list[dict]) -> list[dict]:
    pooled: dict[tuple[int, str], list[float]] = {}
    for r in rows:
        n = r["n"]
        if n == 0 or r["mean_coop"] != r["mean_coop"]:
            continue
        key = (r["similarity_pct"], r["model"])
        bucket = pooled.setdefault(key, [])
        if r.get("_samples"):
            bucket.extend(r["_samples"])
        else:
            coop_count = round(r["mean_coop"] * n)
            bucket.extend([1.0] * coop_count + [0.0] * (n - coop_count))

    out: list[dict] = []
    for (pct, model), samples in sorted(pooled.items()):
        mean, sem_v, n = mean_sem(samples)
        out.append(
            {
                "similarity_pct": pct,
                "model": model,
                "n": n,
                "mean_coop": mean,
                "sem_coop": sem_v,
            }
        )
    return out


_DEFAULT_PD_PAYOFFS = {"CC": [3, 3], "CD": [0, 5], "DC": [5, 0], "DD": [1, 1]}


def _payoff_tag(config: dict) -> str | None:
    """Detect a non-default PD payoff matrix and return a short label.

    Returns "Nx" for a uniform scale of the default matrix (CC=3, DC=5, DD=1)
    or "CC -> N" for the CC=N, DC=N+1, DD=1 family. None for the default or
    anything we can't recognise.
    """
    pm = (config.get("game") or {}).get("kwargs", {}).get("payoff_matrix")
    if not pm:
        return None
    cc, cd, dc, dd = pm.get("CC"), pm.get("CD"), pm.get("DC"), pm.get("DD")
    if [cc, cd, dc, dd] == [_DEFAULT_PD_PAYOFFS[k] for k in ("CC", "CD", "DC", "DD")]:
        return None
    if (cc and dd and len(cc) == 2 and len(dd) == 2
            and dd[0] > 0 and cc == [2 * dd[0], 2 * dd[0]]
            and cd == [0, 3 * dd[0]] and dc == [3 * dd[0], 0]
            and dd[0] == dd[1]):
        return f"{dd[0]}x"
    if (cc and len(cc) == 2 and cc[0] == cc[1]
            and dc == [cc[0] + 1, 0] and cd == [0, cc[0] + 1] and dd == [1, 1]):
        return f"CC -> {cc[0]}"
    return None


def _coop_plot_by_similarity(rows: list[dict], game: str, out_path: Path, *,
                             subtitle: str | None = None,
                             title: str | None = None) -> None:
    if not rows:
        print(f"No sweep rows to plot for {out_path}")
        return

    by_model: dict[str, list[dict]] = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)

    # Order models in the legend using the canonical Gemini→Gemma sequence,
    # appending any unrecognised ones alphabetically.
    canonical = []
    for short, display in _MODEL_DISPLAY_NAMES.items():
        if display not in _MODEL_ORDER_DISPLAY:
            continue
        for full_model in by_model:
            if short_model_name(full_model) == short:
                canonical.append((full_model, display))
                break
    canonical.sort(key=lambda pair: _MODEL_ORDER_DISPLAY.index(pair[1]))
    seen = {full for full, _ in canonical}
    for full in sorted(by_model):
        if full not in seen:
            canonical.append((full, display_model_name(full)))

    fig, ax = plt.subplots(figsize=(10, 6))
    palette = plt.get_cmap("tab10")

    for i, (model, label) in enumerate(canonical):
        model_rows = sorted(by_model[model], key=lambda r: r["similarity_pct"])
        xs = np.array([r["similarity_pct"] for r in model_rows], dtype=float)
        ys = np.array([r["mean_coop"] for r in model_rows], dtype=float)
        sems = np.array(
            [0.0 if r["sem_coop"] != r["sem_coop"] else r["sem_coop"] for r in model_rows],
            dtype=float,
        )
        color = palette(i % 10)
        ax.plot(xs, ys, marker="o", linewidth=2, color=color, label=label)
        ax.fill_between(xs, ys - sems, ys + sems, color=color, alpha=0.2)
        ax.errorbar(xs, ys, yerr=sems, fmt="none", ecolor=color, capsize=3, alpha=0.8)

    ax.set_xlabel("Similarity (%)")
    ax.set_ylabel(_coop_y_label(game))
    ax.set_xticks(sorted({r["similarity_pct"] for r in rows}))
    ax.set_ylim(0, 1.05)
    if title is not None:
        ax.set_title(title, fontsize=12)
    elif subtitle:
        ax.set_title(subtitle, fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=9, loc="best", frameon=False)
    fig.tight_layout()
    save_fig(fig, out_path, dpi=200)


def _coop_load_records(path: Path) -> list[list[dict]]:
    matchups: list[list[dict]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if isinstance(entry, list) and entry and isinstance(entry[0], list):
                matchups.extend(entry)
            elif isinstance(entry, list):
                matchups.append(entry)
    return matchups


def _coop_summarize_records(matchups: list[list[dict]]) -> list[dict]:
    per_model: dict[str, list[int]] = {}
    for matchup in matchups:
        for move in matchup:
            model = strip_seat(_coop_player_name(move))
            indicator = _coop_is_coop_action(str(move.get("action", "")), sweep_format=False)
            per_model.setdefault(model, []).append(indicator)

    rows: list[dict] = []
    for model, indicators in sorted(per_model.items()):
        mean, sem_v, n = mean_sem(indicators)
        rows.append(
            {
                "model": model,
                "mean_cooperation_rate": mean,
                "sem_cooperation_rate": sem_v,
                "n_matchups": n,
            }
        )
    return rows


def _coop_plot_per_player(rows: list[dict], game: str, out_path: Path) -> None:
    if not rows:
        return
    players = [r["model"] for r in rows]
    means = [r["mean_cooperation_rate"] for r in rows]
    sems = [
        0.0 if r["sem_cooperation_rate"] != r["sem_cooperation_rate"] else r["sem_cooperation_rate"]
        for r in rows
    ]
    ns = [r["n_matchups"] for r in rows]

    fig_width = max(8, 1.4 * len(players))
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    x = np.arange(len(players))
    bars = ax.bar(
        x,
        means,
        yerr=sems,
        color="#355070",
        alpha=0.9,
        capsize=4,
        edgecolor="white",
        error_kw={"elinewidth": 1.2, "ecolor": "#2F3437"},
    )
    for bar, m, n in zip(bars, means, ns):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(bar.get_height() + 0.02, 0.98),
            f"{m:.2f}\n(n={n})",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(players, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(_coop_y_label(game))
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"Per-player {_coop_y_label(game).lower()} — {game}\n(error bars = SEM across matchups)",
        fontsize=12,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_fig(fig, out_path, dpi=200)


def plot_cooperation_with_sem(args: argparse.Namespace) -> None:
    """Cooperation rate vs similarity with SEM error bands (or per-player fallback)."""
    run_dir: Path = args.run_dir
    if not run_dir.exists():
        print(f"ERROR: {run_dir} does not exist")
        sys.exit(1)

    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    game = (config.get("game") or {}).get("type", "Unknown")
    payoff_tag = _payoff_tag(config)
    payoff_title = f"Cooperation Rate — Modified Payoffs ({payoff_tag})" if payoff_tag else None

    sweep_path = run_dir / "sweep_results_by_percentage.json"
    bench_sweep_path = run_dir / "benchmark_sweep_results.json"
    records_path = run_dir / "records.jsonl"

    out_dir = ensure_dir(run_dir / _COOP_PLOT_SUBDIR)
    outputs: dict[str, Path] = {}

    if bench_sweep_path.exists():
        sweep = load_json(bench_sweep_path)
        rows = _coop_summarize_benchmark_sweep(sweep)

        csv_path = out_dir / "cooperation_by_similarity.csv"
        write_csv(rows, csv_path)
        outputs["csv"] = csv_path

        by_bench: dict[str, list[dict]] = {}
        for r in rows:
            by_bench.setdefault(r["benchmark"], []).append(r)
        for benchmark, bench_rows in sorted(by_bench.items()):
            png = out_dir / f"cooperation_by_similarity_{benchmark}.png"
            bench_label = _BENCHMARK_DISPLAY_NAMES.get(benchmark, benchmark)
            _coop_plot_by_similarity(
                bench_rows,
                game,
                png,
                title=f"Cooperation Rate — {bench_label}",
            )
            outputs[f"plot_{benchmark}"] = png

        pooled = _coop_aggregate_across_benchmarks(rows)
        combined_png = out_dir / "cooperation_by_similarity.png"
        _coop_plot_by_similarity(
            pooled,
            game,
            combined_png,
            title="Cooperation Rate — Pooled",
        )
        outputs["plot"] = combined_png
    elif sweep_path.exists():
        sweep = load_json(sweep_path)
        rows = _coop_summarize_sweep(sweep)
        csv_path = out_dir / "cooperation_by_similarity.csv"
        png_path = out_dir / "cooperation_by_similarity.png"
        write_csv(rows, csv_path)
        _coop_plot_by_similarity(rows, game, png_path, title=payoff_title)
        outputs["csv"] = csv_path
        outputs["plot"] = png_path
    elif records_path.exists():
        matchups = _coop_load_records(records_path)
        rows = _coop_summarize_records(matchups)
        csv_path = out_dir / "cooperation_summary.csv"
        png_path = out_dir / "cooperation_with_sem.png"
        write_csv(rows, csv_path)
        _coop_plot_per_player(rows, game, png_path)
        outputs["csv"] = csv_path
        outputs["plot"] = png_path
    else:
        raise FileNotFoundError(
            f"Neither sweep_results nor records.jsonl in {run_dir}"
        )

    for name, path in outputs.items():
        print(f"  {name}: {path}")


# =============================================================================
#  cooperation_heatmap
# =============================================================================

_COOP_HM_PROMPT_HEAD_RE = re.compile(r"^===== Prompt \[ID: (\w+)\]", re.M)
# Matches all three framings: "X% similar to", "X% different from", "X% dissimilar to".
_COOP_HM_PCT_RE = re.compile(
    r"decision-making is (\d+)% (similar to|different from|dissimilar to) yours"
)


def _coop_hm_framing(config: dict) -> str:
    """Return one of 'similar', 'different', 'dissimilar' from config.json."""
    df = (config.get("mechanism") or {}).get("kwargs", {}).get("difference_framing", False)
    if df is True:
        return "different"
    if df is False or df == "similar":
        return "similar"
    return str(df)


def _coop_hm_continuous_prob(move: dict) -> float | None:
    """Return P(A0) for one move using mix_probs when available, else 0/1
    from the sampled action name (matching ``_COOP_TOKENS``)."""
    mp = move.get("mix_probs")
    if mp:
        try:
            return float(mp.get("A0", 0)) / 100.0
        except (TypeError, ValueError):
            return None
    action = (move.get("action") or "").upper()
    if not action:
        return None
    return 1.0 if any(tok in action for tok in _COOP_TOKENS) else 0.0


def _coop_hm_short_model(player: str) -> str:
    return short_model_name(strip_seat(player))


def _coop_hm_samples_from_sweep_json(data: dict) -> dict[str, dict[int, list[float]]]:
    out: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for pct_key, trials in data.get("results_by_percentage", {}).items():
        try:
            pct = int(pct_key)
        except (TypeError, ValueError):
            continue
        for trial in trials:
            for round_moves in trial:
                for mv in round_moves:
                    p = _coop_hm_continuous_prob(mv)
                    if p is None:
                        continue
                    out[_coop_hm_short_model(mv["player"])][pct].append(p)
    return out


def _coop_hm_samples_from_records(run_dir: Path) -> dict[str, dict[int, list[float]]]:
    log_path = run_dir / "game_log.txt"
    rec_path = run_dir / "records.jsonl"
    if not log_path.exists() or not rec_path.exists():
        return {}
    log = log_path.read_text()
    blocks = re.split(r"(?=^===== Prompt \[ID: \w+\])", log, flags=re.M)
    tid_to_pct: dict[str, int] = {}
    for blk in blocks:
        h = _COOP_HM_PROMPT_HEAD_RE.search(blk)
        m = _COOP_HM_PCT_RE.search(blk)
        if h and m:
            # m.group(1) is the displayed % (already framing-correct).
            tid_to_pct[h.group(1)] = int(m.group(1))
    out: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    with rec_path.open() as f:
        for line in f:
            try:
                moves = json.loads(line)
            except json.JSONDecodeError:
                continue
            for mv in moves:
                pct = tid_to_pct.get(mv.get("trace_id"))
                if pct is None:
                    continue
                p = _coop_hm_continuous_prob(mv)
                if p is None:
                    continue
                out[_coop_hm_short_model(mv["player"])][pct].append(p)
    return out


def _coop_hm_load_samples(run_dir: Path) -> dict[str, dict[int, list[float]]]:
    """Prefer mix_probs in sweep JSON when present; fall back to records.jsonl join."""
    sweep_path = run_dir / "sweep_results_by_percentage.json"
    samples = {}
    if sweep_path.exists():
        data = load_json(sweep_path)
        samples = _coop_hm_samples_from_sweep_json(data)
        has_mp = any(
            mv.get("mix_probs")
            for trials in data.get("results_by_percentage", {}).values()
            for trial in trials
            for round_moves in trial
            for mv in round_moves
        )
        if has_mp and samples:
            return samples
    return _coop_hm_samples_from_records(run_dir)


def plot_cooperation_heatmap(args: argparse.Namespace) -> None:
    """PD-style mean ± SEM heatmap (model × similarity %) from a basic sweep run."""
    run_dir: Path = args.run_dir
    if not run_dir.exists():
        print(f"ERROR: {run_dir} does not exist")
        sys.exit(1)
    config = load_json(run_dir / "config.json") if (run_dir / "config.json").exists() else {}
    game = (config.get("game") or {}).get("type", "Unknown")
    framing = _coop_hm_framing(config)

    samples = _coop_hm_load_samples(run_dir)
    if not samples:
        print(f"ERROR: no cooperation samples extracted from {run_dir}")
        sys.exit(1)

    # For non-similar framings, mechanism announces the *flipped* pct (raw 70 → "30% different").
    # records JSON stores the raw pct; game_log.txt has the displayed pct. Always plot the
    # displayed (treatment) value the model actually saw.
    percentages = sorted({p for d in samples.values() for p in d})
    if framing != "similar":
        records_samples = _coop_hm_samples_from_records(run_dir)
        if records_samples:
            # records-based path already keys on the announced (displayed) pct.
            samples = records_samples
            percentages = sorted({p for d in samples.values() for p in d})
        else:
            # Sweep-JSON-only path: keys are raw mechanism pcts. Flip them so the
            # x-axis matches what the model actually saw.
            samples = {m: {100 - k: v for k, v in d.items()} for m, d in samples.items()}
            percentages = sorted({p for d in samples.values() for p in d})
    models = sorted(samples.keys(), key=_model_display_sort_key)
    model_labels = [_model_display_label(m) for m in models]

    # Rows = percentages (y-axis), columns = models (x-axis).
    mean_mat = np.full((len(percentages), len(models)), np.nan)
    sem_mat = np.full_like(mean_mat, np.nan)
    for j, m in enumerate(models):
        for i, p in enumerate(percentages):
            vals = samples[m].get(p, [])
            mu, se, _ = mean_sem(vals)
            mean_mat[i, j] = 100 * mu if mu == mu else np.nan
            sem_mat[i, j] = 100 * se if se == se else np.nan

    # Similar framing: high pct at top (visually = most similar). For
    # difference/dissimilar framings, flip so 0% (most similar) is at top.
    if framing == "similar":
        row_pcts = list(reversed(percentages))
        display_mat = mean_mat[::-1, :]
        display_sem = sem_mat[::-1, :]
    else:
        row_pcts = list(percentages)
        display_mat = mean_mat
        display_sem = sem_mat

    fig, ax = plt.subplots(figsize=(9, 11))
    im = ax.imshow(display_mat, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(model_labels, fontsize=11, rotation=30, ha="right")
    ax.set_xlabel("Model", fontsize=12, fontweight="bold")

    ax.set_yticks(range(len(row_pcts)))
    ax.set_yticklabels([f"{p}%" for p in row_pcts], fontsize=10)
    y_label = {
        "similar": "Similarity Percentage",
        "different": "Difference Percentage",
        "dissimilar": "Dissimilarity Percentage",
    }.get(framing, "Similarity Percentage")
    ax.set_ylabel(y_label, fontsize=12, fontweight="bold")

    for i in range(len(row_pcts)):
        for j in range(len(models)):
            val = display_mat[i, j]
            if np.isnan(val):
                continue
            color = "white" if val < 30 or val > 70 else "black"
            se = display_sem[i, j]
            if not np.isnan(se):
                ax.text(j, i, f"{val:.0f}\n±{se:.0f}",
                        ha="center", va="center", fontsize=9,
                        fontweight="bold", color=color)
            else:
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Cooperation Rate (%)", fontsize=11)

    payoff_tag = _payoff_tag(config)
    if payoff_tag:
        title = f"Cooperation Rate — Modified Payoffs ({payoff_tag})"
    else:
        framing_word = {"different": "difference"}.get(framing, framing)
        title = f'Cooperation Rate ("{framing_word}" framing)'
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    fig.tight_layout()
    out_dir = ensure_dir(run_dir / _COOP_PLOT_SUBDIR)
    save_fig(fig, out_dir / "cooperation_heatmap.png", dpi=150)


def _coop_hm_render_panel(ax, samples, percentages, *, panel_title):
    models = sorted(samples.keys(), key=_model_display_sort_key)
    model_labels = [_model_display_label(m) for m in models]
    mean_mat = np.full((len(percentages), len(models)), np.nan)
    sem_mat = np.full_like(mean_mat, np.nan)
    for j, m in enumerate(models):
        for i, p in enumerate(percentages):
            vals = samples[m].get(p, [])
            mu, se, _ = mean_sem(vals)
            mean_mat[i, j] = 100 * mu if mu == mu else np.nan
            sem_mat[i, j] = 100 * se if se == se else np.nan
    row_pcts = list(reversed(percentages))
    display_mat = mean_mat[::-1, :]
    display_sem = sem_mat[::-1, :]
    im = ax.imshow(display_mat, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(model_labels, fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(len(row_pcts)))
    ax.set_yticklabels([f"{p}%" for p in row_pcts], fontsize=8)
    ax.set_title(panel_title, fontsize=12, fontweight="bold", pad=8)
    for i in range(len(row_pcts)):
        for j in range(len(models)):
            val = display_mat[i, j]
            if np.isnan(val):
                continue
            color = "white" if val < 30 or val > 70 else "black"
            se = display_sem[i, j]
            if not np.isnan(se):
                ax.text(j, i, f"{val:.0f}\n±{se:.0f}",
                        ha="center", va="center", fontsize=7,
                        fontweight="bold", color=color)
            else:
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=8, fontweight="bold", color=color)
    return im


def plot_payoff_grid(args: argparse.Namespace) -> None:
    """Two 1x2 dashboards of payoff-variation cooperation heatmaps."""
    base = Path(args.base_dir)
    if not base.exists():
        print(f"ERROR: {base} not found")
        sys.exit(1)
    groups = [
        ("scale", [
            ("scale3",    "Updated Payoffs (3x)"),
            ("scale10",   "Updated Payoffs (10x)"),
        ]),
        ("prop", [
            ("prop5_6",   "Updated Payoffs (CC -> 5)"),
            ("prop10_11", "Updated Payoffs (CC -> 10)"),
        ]),
    ]
    for group_name, panels in groups:
        fig, axes = plt.subplots(1, 2, figsize=(14, 8))
        im = None
        for ax, (subdir, title) in zip(axes, panels):
            run_dir = base / subdir
            samples = _coop_hm_load_samples(run_dir)
            if not samples:
                print(f"  warning: no samples in {run_dir}; skipping panel")
                ax.set_visible(False)
                continue
            percentages = sorted({p for d in samples.values() for p in d})
            im = _coop_hm_render_panel(ax, samples, percentages, panel_title=title)
        axes[0].set_ylabel("Similarity Percentage", fontsize=10, fontweight="bold")
        for ax in axes:
            ax.set_xlabel("Model", fontsize=10, fontweight="bold")
        fig.suptitle("Cooperation Rate — Updated Payoff Variations",
                     fontsize=15, fontweight="bold", y=0.995)
        fig.tight_layout(rect=[0, 0, 0.92, 0.96])
        if im is not None:
            cbar_ax = fig.add_axes([0.93, 0.12, 0.018, 0.78])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label("Cooperation Rate (%)", fontsize=11)
        out_path = base / f"payoff_variations_grid_{group_name}.png"
        save_fig(fig, out_path, dpi=150)


# =============================================================================
#  defection_rate
# =============================================================================

_DEF_MODE_STYLE = {
    "objective":   {"color": "#2196F3", "label": "Objective (kappa/EMD)",       "marker": "o"},
    "decision":    {"color": "#FF5722", "label": "Subjective — decision",        "marker": "s"},
    "explanation": {"color": "#9C27B0", "label": "Subjective — explanation",     "marker": "^"},
    "both":        {"color": "#4CAF50", "label": "Subjective — both",            "marker": "D"},
}


def _def_val(comparison: dict, mode: str, bench: str, key: str) -> float | None:
    if mode == "objective":
        return comparison.get("objective", {}).get(bench, {}).get(key)
    return comparison.get("subjective", {}).get(mode, {}).get(bench, {}).get(key)


def _def_present_modes(comparison: dict) -> list[str]:
    modes: list[str] = []
    if comparison.get("objective"):
        modes.append("objective")
    for m in ["decision", "explanation", "both"]:
        if comparison.get("subjective", {}).get(m):
            modes.append(m)
    return modes


def _def_bar_positions(n_modes: int, n_benchmarks: int, width: float = 0.18):
    x = np.arange(n_benchmarks)
    total_w = n_modes * width
    offsets = np.linspace(-total_w / 2 + width / 2, total_w / 2 - width / 2, n_modes)
    return x, offsets


def _def_plot_similarity_scores(comparison: dict, save_dir: Path) -> None:
    benchmarks = comparison["benchmarks"]
    modes = _def_present_modes(comparison)
    x, offsets = _def_bar_positions(len(modes), len(benchmarks))

    fig, ax = plt.subplots(figsize=(max(12, len(benchmarks) * 1.6), 6))

    for i, mode in enumerate(modes):
        vals = [_def_val(comparison, mode, b, "similarity") or 0 for b in benchmarks]
        style = _DEF_MODE_STYLE[mode]
        bars = ax.bar(x + offsets[i], vals, width=0.17,
                      color=style["color"], label=style["label"], edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v > 2:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=6.5, fontweight="bold",
                        color=style["color"])

    ax.axhline(50, linestyle="--", color="#757575", linewidth=0.9, label="Chance (50%)")
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=35, ha="right", fontsize=10)
    ax.set_ylabel("Similarity Score (%)", fontsize=12)
    ax.set_ylim(0, 115)
    ax.set_title(
        "Gemini vs Random — Similarity Scores per Benchmark\n"
        "(Objective = algorithmic kappa/EMD; Subjective = Gemini self-reports after seeing Random's answers)",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, save_dir / "similarity_scores.png", dpi=150)


def _def_plot_cooperation_rates(comparison: dict, save_dir: Path) -> None:
    benchmarks = comparison["benchmarks"]
    modes = _def_present_modes(comparison)
    x, offsets = _def_bar_positions(len(modes), len(benchmarks))

    fig, ax = plt.subplots(figsize=(max(12, len(benchmarks) * 1.6), 6))

    for i, mode in enumerate(modes):
        vals = [(_def_val(comparison, mode, b, "cooperation_rate") or 0) * 100
                for b in benchmarks]
        style = _DEF_MODE_STYLE[mode]
        bars = ax.bar(x + offsets[i], vals, width=0.17,
                      color=style["color"], label=style["label"], edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v > 3:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{v:.0f}%", ha="center", va="bottom", fontsize=6.5, fontweight="bold",
                        color=style["color"])

    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=35, ha="right", fontsize=10)
    ax.set_ylabel("Gemini Cooperation Rate (%)", fontsize=12)
    ax.set_ylim(0, 115)
    ax.set_title(
        "Gemini Cooperation Rate per Benchmark × Similarity Mode\n"
        "(Prisoners Dilemma, 10 trials each, vs Random Agent)",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, save_dir / "cooperation_rates.png", dpi=150)


def _def_plot_similarity_vs_coop(comparison: dict, save_dir: Path) -> None:
    benchmarks = comparison["benchmarks"]
    modes = _def_present_modes(comparison)

    fig, ax = plt.subplots(figsize=(8, 6))
    legend_handles = []

    for mode in modes:
        style = _DEF_MODE_STYLE[mode]
        xs, ys, labels = [], [], []
        for bench in benchmarks:
            sim_v  = _def_val(comparison, mode, bench, "similarity")
            coop_v = _def_val(comparison, mode, bench, "cooperation_rate")
            if sim_v is not None and coop_v is not None:
                xs.append(sim_v)
                ys.append(coop_v * 100)
                labels.append(bench)
        if xs:
            ax.scatter(xs, ys, color=style["color"], marker=style["marker"],
                       s=70, zorder=3, label=style["label"])
            for lbl, xi, yi in zip(labels, xs, ys):
                ax.annotate(lbl, (xi, yi), textcoords="offset points",
                            xytext=(5, 4), fontsize=7, color=style["color"])
            legend_handles.append(
                mpatches.Patch(color=style["color"], label=style["label"])
            )

    ax.axvline(50, linestyle="--", color="#757575", linewidth=0.8)
    ax.text(51, ax.get_ylim()[0] + 2 if ax.get_ylim()[0] < 98 else 2,
            "chance", fontsize=8, color="#757575")
    ax.set_xlabel("Similarity Score (%)", fontsize=12)
    ax.set_ylabel("Gemini Cooperation Rate (%)", fontsize=12)
    ax.set_title("Similarity Score → Gemini Cooperation Rate\n(each point = one benchmark)", fontsize=11)
    ax.legend(handles=legend_handles, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, save_dir / "similarity_vs_coop.png", dpi=150)


def _def_plot_overview(comparison: dict, save_dir: Path) -> None:
    benchmarks = comparison["benchmarks"]
    modes = _def_present_modes(comparison)
    x, offsets = _def_bar_positions(len(modes), len(benchmarks))

    fig, (ax_sim, ax_coop) = plt.subplots(2, 1, figsize=(max(13, len(benchmarks) * 1.7), 11))
    fig.suptitle(
        "Gemini vs Random Agent — All Benchmarks, All Similarity Modes\n"
        f"(Prisoners Dilemma · 10 trials · {len(benchmarks)} benchmarks · {len(modes)} modes)",
        fontsize=13, fontweight="bold",
    )

    for ax, key, ylabel, title, ylim in [
        (ax_sim,  "similarity",       "Similarity Score (%)",            "Similarity Scores",         115),
        (ax_coop, "cooperation_rate", "Gemini Cooperation Rate (%)", "Gemini Cooperation Rates", 115),
    ]:
        for i, mode in enumerate(modes):
            mult = 100 if key == "cooperation_rate" else 1
            vals = [(_def_val(comparison, mode, b, key) or 0) * mult for b in benchmarks]
            style = _DEF_MODE_STYLE[mode]
            ax.bar(x + offsets[i], vals, width=0.17,
                   color=style["color"], label=style["label"], edgecolor="white", linewidth=0.5)

        if key == "similarity":
            ax.axhline(50, linestyle="--", color="#757575", linewidth=0.8, label="Chance (50%)")
        ax.set_xticks(x)
        ax.set_xticklabels(benchmarks, rotation=35, ha="right", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_ylim(0, ylim)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_fig(fig, save_dir / "overview.png", dpi=150)


def _def_write_table(comparison: dict, save_dir: Path) -> None:
    benchmarks = comparison["benchmarks"]
    modes = _def_present_modes(comparison)

    col = 12
    header_labels = [_DEF_MODE_STYLE[m]["label"][:col] for m in modes]

    lines: list[str] = []

    for section, key, fmt in [
        ("SIMILARITY SCORES",          "similarity",       lambda v: f"{v:.1f}%" if v else "N/A"),
        ("GEMINI COOPERATION RATE",     "cooperation_rate", lambda v: f"{v:.1%}"  if v else "N/A"),
        ("GEMINI DEFECTION RATE",       "defection_rate",   lambda v: f"{v:.1%}"  if v else "N/A"),
    ]:
        lines.append(f"\n{section}")
        lines.append("=" * (22 + col * len(modes)))
        lines.append(f"{'Benchmark':<22}" + "".join(f"{h:>{col}}" for h in header_labels))
        lines.append("-" * (22 + col * len(modes)))
        for bench in benchmarks:
            row = f"  {bench:<20}"
            for mode in modes:
                v = _def_val(comparison, mode, bench, key)
                row += f"{fmt(v):>{col}}"
            lines.append(row)

    table_str = "\n".join(lines)
    print(table_str)

    path = save_dir / "table.txt"
    path.write_text(table_str)
    print(f"\nTable saved to: {path}")


def plot_defection_rate(args: argparse.Namespace) -> None:
    """Plot Gemini cooperation rates and similarity scores across benchmarks/modes."""
    comparison_path = Path(args.comparison)
    if not comparison_path.exists():
        print(f"ERROR: {comparison_path} not found")
        sys.exit(1)

    comparison = load_json(comparison_path)
    save_dir = ensure_dir(Path(args.save_dir) if args.save_dir else comparison_path.parent)

    modes = _def_present_modes(comparison)
    print(f"\nLoaded: {len(comparison.get('benchmarks', []))} benchmarks, modes: {modes}")

    _def_plot_similarity_scores(comparison, save_dir)
    _def_plot_cooperation_rates(comparison, save_dir)
    _def_plot_similarity_vs_coop(comparison, save_dir)
    _def_plot_overview(comparison, save_dir)
    _def_write_table(comparison, save_dir)

    print(f"\nAll plots saved to: {save_dir}")


# =============================================================================
#  fixed_point
# =============================================================================

def _fp_simplify_name(name: str) -> str:
    """Shorten agent name for plot labels."""
    return name.split("/")[-1].split("(")[0].rstrip("#P1").rstrip("#P0")


def _fp_pair_label(pair: dict) -> str:
    a = _fp_simplify_name(pair["agent_a"])
    b = _fp_simplify_name(pair["agent_b"])
    if a == b:
        return f"{a} (self)"
    return f"{a} vs {b}"


def _fp_plot_curve(pair: dict, output_dir: Path, seed: int) -> None:
    sweep = pair["sweep"]
    pcts = sweep["percentages"]
    means = sweep["rates_mean"]
    stds = sweep["rates_std"]
    metric = pair.get("metric", "agreement")
    metric_label = "Action Agreement Rate" if metric != "cooperation" else "Cooperation Rate"

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot([0, 100], [0, 100], "r--", linewidth=1.5, alpha=0.7, label="y = x (identity)")
    ax.plot(pcts, means, "b-o", linewidth=2, markersize=6, label=f"f(s) = {metric_label.lower()}")
    lower = [m - s for m, s in zip(means, stds)]
    upper = [m + s for m, s in zip(means, stds)]
    ax.fill_between(pcts, lower, upper, alpha=0.2, color="blue")

    fp = pair.get("primary_fixed_point")
    if fp is not None:
        ax.axhline(y=fp, color="green", linestyle=":", alpha=0.5, linewidth=1)
        ax.axvline(x=fp, color="green", linestyle=":", alpha=0.5, linewidth=1)
        ax.plot(fp, fp, "*", color="green", markersize=20, zorder=5,
                label=f"Grounded similarity: {fp:.1f}%")

    validation = pair.get("validation")
    if validation and validation.get("rate_mean") is not None:
        ax.plot(
            validation["similarity_pct"],
            validation["rate_mean"],
            "D",
            color="orange",
            markersize=10,
            zorder=5,
            label=f"Validation: {validation['rate_mean']:.1f}% @ {validation['similarity_pct']}%",
        )

    label = _fp_pair_label(pair)
    ax.set_xlabel("Told Similarity (%)", fontsize=12)
    ax.set_ylabel(f"{metric_label} (%)", fontsize=12)
    ax.set_title(f"Fixed-Point Analysis: {label}", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)

    safe_label = label.replace(" ", "_").replace("(", "").replace(")", "")
    path = output_dir / f"fixed_point_{safe_label}_{seed}.png"
    save_fig(fig, path)


def _fp_plot_heatmap(pairs: list[dict], output_dir: Path, seed: int) -> None:
    agents: list[str] = []
    for p in pairs:
        for key in ("agent_a", "agent_b"):
            if p[key] not in agents:
                agents.append(p[key])

    n = len(agents)
    matrix = np.full((n, n), np.nan)
    annot = [[" "] * n for _ in range(n)]

    for p in pairs:
        i = agents.index(p["agent_a"])
        j = agents.index(p["agent_b"])
        fp = p.get("primary_fixed_point")
        if fp is not None:
            matrix[i][j] = fp
            matrix[j][i] = fp
            annot[i][j] = f"{fp:.1f}"
            annot[j][i] = f"{fp:.1f}"
        else:
            annot[i][j] = "N/A"
            annot[j][i] = "N/A"

    labels = [_fp_simplify_name(a) for a in agents]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
        annot=annot,
        fmt="",
        cbar_kws={"label": "Fixed-Point Similarity (%)"},
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
    )
    ax.set_title("Fixed-Point Similarity Matrix", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    save_fig(fig, output_dir / f"fixed_point_heatmap_{seed}.png")


def _fp_plot_overlay(pairs: list[dict], output_dir: Path, seed: int) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot([0, 100], [0, 100], "r--", linewidth=1.5, alpha=0.7, label="y = x")

    colors = tab10_colors(len(pairs))

    for idx, pair in enumerate(pairs):
        sweep = pair["sweep"]
        pcts = sweep["percentages"]
        means = sweep["rates_mean"]
        stds = sweep["rates_std"]
        label = _fp_pair_label(pair)

        ax.plot(pcts, means, "-o", color=colors[idx], linewidth=2,
                markersize=5, label=label)
        lower = [m - s for m, s in zip(means, stds)]
        upper = [m + s for m, s in zip(means, stds)]
        ax.fill_between(pcts, lower, upper, alpha=0.15, color=colors[idx])

        fp = pair.get("primary_fixed_point")
        if fp is not None:
            ax.plot(fp, fp, "*", color=colors[idx], markersize=16, zorder=5)

    ax.set_xlabel("Told Similarity (%)", fontsize=12)
    metric = pairs[0].get("metric", "agreement") if pairs else "agreement"
    metric_label = "Action Agreement Rate" if metric != "cooperation" else "Cooperation Rate"
    ax.set_ylabel(f"{metric_label} (%)", fontsize=12)
    ax.set_title("Fixed-Point Analysis: All Pairs", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    save_fig(fig, output_dir / f"fixed_point_overlay_{seed}.png")


def plot_fixed_point(args: argparse.Namespace) -> None:
    """Plot fixed-point similarity search results (per-pair curves, heatmap, overlay)."""
    results = load_json(args.results_json)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.results_json).parent

    plots_dir = ensure_dir(PROJECT_DIR / "plots" / "similarity")

    seed = results.get("seed", 0)
    pairs = results["pairs"]

    print("\nGenerating fixed-point plots...")

    for pair in pairs:
        _fp_plot_curve(pair, plots_dir, seed)

    if len(pairs) > 1:
        _fp_plot_heatmap(pairs, plots_dir, seed)

    _fp_plot_overlay(pairs, plots_dir, seed)

    if output_dir != plots_dir:
        ensure_dir(output_dir)
        for pair in pairs:
            _fp_plot_curve(pair, output_dir, seed)
        if len(pairs) > 1:
            _fp_plot_heatmap(pairs, output_dir, seed)
        _fp_plot_overlay(pairs, output_dir, seed)


# =============================================================================
#  heatmaps
# =============================================================================

def plot_heatmaps(args: argparse.Namespace) -> None:
    """Heatmaps of similarity scores and cooperation rates across modes x benchmarks."""
    data = load_json(args.comparison)
    out_dir = ensure_dir(Path(args.output_dir) if args.output_dir else Path(args.comparison).parent)

    benchmarks = data["benchmarks"]
    modes = ["objective", "decision", "explanation", "both"]
    mode_labels = ["Objective", "Subj. Decision", "Subj. Explanation", "Subj. Both"]

    sim_matrix = np.full((len(benchmarks), len(modes)), np.nan)
    coop_matrix = np.full((len(benchmarks), len(modes)), np.nan)

    for j, mode in enumerate(modes):
        for i, bench in enumerate(benchmarks):
            if mode == "objective":
                entry = data.get("objective", {}).get(bench, {})
            else:
                entry = data.get("subjective", {}).get(mode, {}).get(bench, {})

            s = entry.get("similarity")
            c = entry.get("cooperation_rate")
            if s is not None:
                sim_matrix[i, j] = s
            if c is not None:
                coop_matrix[i, j] = c * 100

    short_names = [b.replace("random_coin_toss_alt", "coin_toss")
                    .replace("random_die_roll_alt", "die_roll")
                   for b in benchmarks]

    def _plot_one(matrix, title, filename, fmt=".1f", vmin=0, vmax=100, cmap="RdYlGn"):
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

        ax.set_xticks(range(len(modes)))
        ax.set_xticklabels(mode_labels, rotation=25, ha="right", fontsize=11)
        ax.set_yticks(range(len(benchmarks)))
        ax.set_yticklabels(short_names, fontsize=11)

        for i in range(len(benchmarks)):
            for j in range(len(modes)):
                val = matrix[i, j]
                if np.isnan(val):
                    text = "N/A"
                    color = "gray"
                else:
                    text = f"{val:{fmt}}%"
                    color = "white" if val < 25 or val > 75 else "black"
                ax.text(j, i, text, ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color)

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Percentage", fontsize=11)

        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        fig.tight_layout()
        save_fig(fig, out_dir / filename, dpi=150)

    _plot_one(sim_matrix, "Similarity Scores: Gemini vs Random",
              "heatmap_similarity.png", fmt=".1f")
    _plot_one(coop_matrix, "Gemini Cooperation Rate: by Similarity Mode",
              "heatmap_cooperation.png", fmt=".2f")

    # Interleaved heatmap: sim | coop side-by-side per mode
    n_bench = len(benchmarks)
    n_cols = len(modes) * 2
    interleaved = np.full((n_bench, n_cols), np.nan)

    for j, mode in enumerate(modes):
        for i in range(n_bench):
            interleaved[i, j * 2] = sim_matrix[i, j]
            interleaved[i, j * 2 + 1] = coop_matrix[i, j]

    col_labels = []
    for label in mode_labels:
        col_labels.append(f"{label}\nSimilarity")
        col_labels.append(f"{label}\nCoop Rate")

    fig, ax = plt.subplots(figsize=(16, 7))
    im = ax.imshow(interleaved, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=0, ha="center", fontsize=9)
    ax.set_yticks(range(n_bench))
    ax.set_yticklabels(short_names, fontsize=11)

    for sep in range(1, len(modes)):
        ax.axvline(x=sep * 2 - 0.5, color="white", linewidth=3)

    for i in range(n_bench):
        for j in range(n_cols):
            val = interleaved[i, j]
            if np.isnan(val):
                text = "N/A"
                color = "gray"
            else:
                text = f"{val:.2f}%"
                color = "white" if val < 25 or val > 75 else "black"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=8, fontweight="bold", color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("Percentage", fontsize=11)

    ax.set_title("Gemini vs Random — Similarity & Cooperation Rate (Interleaved)",
                 fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()
    save_fig(fig, out_dir / "heatmap_interleaved.png", dpi=150)


# =============================================================================
#  newcomb
# =============================================================================

def plot_newcomb(args: argparse.Namespace) -> None:
    """Plot Newcomb benchmark results: capabilities, EDT/CDT, agreement strip."""
    results_path: Path = args.results_path
    if not results_path.exists():
        print(f"Error: {results_path} not found")
        sys.exit(1)

    data = load_json(results_path)

    output_dir = results_path.parent
    agents = data["agents"]
    short_names = [short_model_name(a) for a in agents]

    cap_acc = []
    edt_align = []
    cdt_align = []
    for agent in agents:
        scores = data["results"][agent]["scores"]
        cap_acc.append(scores.get("capabilities_accuracy", 0) * 100)
        edt_align.append(scores.get("edt_alignment", 0) * 100)
        cdt_align.append(scores.get("cdt_alignment", 0) * 100)

    # Plot 1: Grouped bar chart — capabilities + EDT/CDT
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(agents))
    width = 0.25

    bars1 = ax1.bar(x - width, cap_acc, width, label="Capabilities Accuracy", color="#4C72B0")
    bars2 = ax1.bar(x, edt_align, width, label="EDT Alignment", color="#DD8452")
    bars3 = ax1.bar(x + width, cdt_align, width, label="CDT Alignment", color="#55A868")

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax1.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 4), textcoords="offset points",
                         ha="center", va="bottom", fontsize=9)

    ax1.set_ylabel("Score (%)", fontsize=12)
    ax1.set_title("Newcomb Benchmark: Model Comparison", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, rotation=30, ha="right")
    ax1.set_ylim(0, 110)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3, axis="y")

    fig1.tight_layout()
    save_fig(fig1, output_dir / "newcomb_scores.png")

    # Plot 2: Per-question running accuracy
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    colors = tab10_colors(len(agents))

    for idx, agent in enumerate(agents):
        raw = data["results"][agent]["raw_responses"]
        cap_responses = [r for r in raw if not r["attitude_q"]]
        if not cap_responses:
            continue
        running = np.cumsum([1 if r.get("correct") else 0 for r in cap_responses])
        running_acc = running / np.arange(1, len(running) + 1) * 100
        ax2.plot(running_acc, label=short_model_name(agent), color=colors[idx],
                 linewidth=1.5, alpha=0.85)

    ax2.set_xlabel("Question #", fontsize=12)
    ax2.set_ylabel("Running Accuracy (%)", fontsize=12)
    ax2.set_title("Capabilities: Running Accuracy Over Questions", fontsize=14, fontweight="bold")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    fig2.tight_layout()
    save_fig(fig2, output_dir / "newcomb_running_accuracy.png")

    # Plot 3: EDT vs CDT scatter
    fig3, ax3 = plt.subplots(figsize=(7, 7))
    for idx, agent in enumerate(agents):
        scores = data["results"][agent]["scores"]
        ex = scores.get("edt_alignment", 0) * 100
        cy = scores.get("cdt_alignment", 0) * 100
        ax3.scatter(ex, cy, s=200, color=colors[idx], zorder=5, edgecolors="black", linewidth=0.8)
        ax3.annotate(short_model_name(agent), (ex, cy), textcoords="offset points",
                     xytext=(8, 8), fontsize=10)

    lim = max(max(edt_align), max(cdt_align), 50) + 15
    ax3.plot([0, lim], [0, lim], "k--", alpha=0.3, label="EDT = CDT line")
    ax3.set_xlabel("EDT Alignment (%)", fontsize=12)
    ax3.set_ylabel("CDT Alignment (%)", fontsize=12)
    ax3.set_title("Decision Theory Alignment", fontsize=14, fontweight="bold")
    ax3.set_xlim(0, lim)
    ax3.set_ylim(0, lim)
    ax3.set_aspect("equal")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    fig3.tight_layout()
    save_fig(fig3, output_dir / "newcomb_edt_vs_cdt.png")

    # Plot 4: Per-question agreement heatmap
    if len(agents) >= 2:
        fig4, ax4 = plt.subplots(figsize=(10, 4))

        raw_a = data["results"][agents[0]]["raw_responses"]
        raw_b = data["results"][agents[1]]["raw_responses"]

        qids_a = {r["qid"]: r for r in raw_a}
        qids_b = {r["qid"]: r for r in raw_b}
        common_qids = sorted(set(qids_a.keys()) & set(qids_b.keys()))

        agreements = [1 if qids_a[q]["chosen_index"] == qids_b[q]["chosen_index"] else 0
                      for q in common_qids]

        agree_arr = np.array(agreements).reshape(1, -1)
        ax4.imshow(agree_arr, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                   interpolation="nearest")
        ax4.set_yticks([])
        ax4.set_xlabel("Question index", fontsize=12)
        ax4.set_title(
            f"Per-Question Agreement: {short_model_name(agents[0])} vs {short_model_name(agents[1])}  "
            f"({sum(agreements)}/{len(agreements)} = {sum(agreements)/len(agreements)*100:.1f}%)",
            fontsize=13, fontweight="bold",
        )

        ax4.legend(
            handles=[Patch(facecolor="#2ca02c", label="Agree"),
                     Patch(facecolor="#d62728", label="Disagree")],
            loc="upper right", fontsize=10,
        )

        fig4.tight_layout()
        save_fig(fig4, output_dir / "newcomb_agreement.png")

    if "similarity_matrix" in data:
        print("\nSimilarity matrix:")
        for pair, sim_v in data["similarity_matrix"].items():
            print(f"  {pair}: {sim_v:.1f}%")

    print(f"\nAll plots saved to {output_dir}/")


# =============================================================================
#  similarity_heatmap
# =============================================================================

def _sh_benchmark_sweep_heatmap(results: dict, results_path: Path) -> None:
    seed = results.get("seed", 42)
    percentages = sorted(results.get("percentages", list(range(0, 101, 10))))
    benchmarks = list(results["results_by_benchmark"].keys())

    model_data: dict[str, dict[str, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for bench_name, bench_data in results["results_by_benchmark"].items():
        rbp = bench_data.get("results_by_percentage", bench_data)
        for pct_str, entries in rbp.items():
            pct = int(pct_str)
            items = entries if isinstance(entries, list) else entries.get("single_results", entries)
            for result in items:
                agent_name = result["agent_name"]
                model_name = short_model_name(agent_name)
                coop_rate = result.get(
                    "cooperation_rate",
                    result.get("action_distribution", {}).get("A0", 0),
                )
                model_data[model_name][bench_name][pct].append(coop_rate)

    models = sorted(model_data.keys())
    plots_dir = ensure_dir(PROJECT_DIR / "plots" / "similarity")

    for model in models:
        matrix = []
        for bench in benchmarks:
            row = []
            for pct in percentages:
                vals = model_data[model][bench].get(pct, [])
                row.append(np.mean(vals) if vals else np.nan)
            matrix.append(row)

        matrix = np.array(matrix)

        fig, ax = plt.subplots(
            figsize=(max(10, len(percentages) * 0.9), max(6, len(benchmarks) * 0.6))
        )

        sns.heatmap(
            matrix,
            xticklabels=[f"{p}%" for p in percentages],
            yticklabels=benchmarks,
            cmap="RdYlGn",
            vmin=0,
            vmax=100,
            annot=True,
            fmt=".1f",
            cbar_kws={"label": "Cooperation Rate (%)"},
            ax=ax,
            linewidths=0.5,
            linecolor="gray",
        )

        ax.set_xlabel("Similarity Percentage", fontsize=13, fontweight="bold")
        ax.set_ylabel("Benchmark", fontsize=13, fontweight="bold")
        ax.set_title(
            f"Cooperation Rate — {model}",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        safe_model = safe_filename(model)
        save_fig(
            fig,
            plots_dir / f"benchmark_sweep_{safe_model}_{seed}.png",
            results_path.parent / f"benchmark_sweep_{safe_model}.png",
        )

    print(f"\n{len(models)} model heatmaps generated.")


def _sh_similarity_sweep_heatmap(results: dict, results_path: Path) -> None:
    seed = results.get("seed", 42)
    percentages = sorted(results.get("percentages", list(range(0, 101, 10))))

    data_dict: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))

    for pct_str, pct_results in results["results_by_percentage"].items():
        pct = int(pct_str)
        items = (
            pct_results if isinstance(pct_results, list)
            else pct_results.get("single_results", [])
        )
        for result in items:
            agent_name = result["agent_name"]
            model_name = short_model_name(agent_name)
            coop_rate = result.get("action_distribution", {}).get("A0", 0)
            data_dict[model_name][pct].append(coop_rate)

    models = sorted(data_dict.keys())

    matrix = []
    for pct in percentages:
        row = []
        for model in models:
            vals = data_dict[model].get(pct, [])
            row.append(np.mean(vals) if vals else np.nan)
        matrix.append(row)

    matrix = np.flipud(np.array(matrix))
    pcts_reversed = percentages[::-1]

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.heatmap(
        matrix,
        xticklabels=models,
        yticklabels=pcts_reversed,
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
        annot=True,
        fmt=".1f",
        cbar_kws={"label": "Cooperation Rate (%)"},
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
    )

    ax.set_xlabel("Model", fontsize=13, fontweight="bold")
    ax.set_ylabel("Similarity Percentage (%)", fontsize=13, fontweight="bold")
    ax.set_title("Cooperation Rate", fontsize=14, fontweight="bold", pad=20)

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plots_dir = ensure_dir(PROJECT_DIR / "plots" / "similarity")
    save_fig(
        fig,
        plots_dir / f"similarity_heatmap_{seed}.png",
        results_path.parent / "similarity_heatmap.png",
    )


def plot_similarity_heatmap(args: argparse.Namespace) -> None:
    """Heatmap from similarity-sweep or benchmark-sweep results JSONs (auto-detected)."""
    results_path = Path(args.results_path)
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        sys.exit(1)

    results = load_json(results_path)

    if "results_by_benchmark" in results:
        _sh_benchmark_sweep_heatmap(results, results_path)
    elif "results_by_percentage" in results:
        _sh_similarity_sweep_heatmap(results, results_path)
    else:
        print("Error: Unrecognized results format.")
        sys.exit(1)


# =============================================================================
#  similarity_tournament
# =============================================================================

def _st_pair_label(a1: str, a2: str) -> str:
    s1, s2 = short_pair_name(a1), short_pair_name(a2)
    if s1 == s2:
        return f"{s1} (self)"
    return f"{s1} vs {s2}"


def _st_parse_per_level(results: dict) -> dict[tuple[str, str, int], float]:
    source = results.get("per_level_similarity", results.get("computed_similarity", {}))
    out: dict[tuple[str, str, int], float] = {}
    for key, val in source.items():
        m = re.match(r"^(.+?) vs (.+?) @ told (\d+)%$", key)
        if m:
            out[(m.group(1), m.group(2), int(m.group(3)))] = val
    return out


def _st_parse_final_similarity(results: dict) -> dict[tuple[str, str], float]:
    cs = results.get("computed_similarity", {})
    out: dict[tuple[str, str], float] = {}
    for key, val in cs.items():
        if "@ told" in key:
            continue
        m = re.match(r"^(.+?) vs (.+?)$", key)
        if m:
            out[(m.group(1), m.group(2))] = val
    return out


def plot_similarity_tournament(args: argparse.Namespace) -> None:
    """Two-phase similarity tournament: told-vs-actual, payoffs, comparison, coop."""
    results_path: Path = args.results_path
    results = load_json(results_path)

    output_dir = results_path.parent
    plots_dir = ensure_dir(PROJECT_DIR / "plots" / "similarity")
    seed = results.get("seed", 0)

    per_level = _st_parse_per_level(results)
    final_sim = _st_parse_final_similarity(results)

    if not final_sim and per_level:
        pair_vals: dict[tuple[str, str], list[float]] = defaultdict(list)
        for (a1, a2, _pct), s in per_level.items():
            pair_vals[(a1, a2)].append(s)
        final_sim = {p: float(np.mean(v)) for p, v in pair_vals.items()}

    elicitation_by_pct = results.get("elicitation_results_by_percentage", {})
    validation_results = results.get("validation_results", [])

    pairs = sorted(per_level.keys(), key=lambda x: (x[0], x[1], x[2]))
    unique_pairs = sorted(set((a1, a2) for a1, a2, _ in pairs))

    colors = tab10_colors(len(unique_pairs))

    # Plot 1: Told vs actual similarity
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    ax1.plot([0, 100], [0, 100], "k--", alpha=0.3, linewidth=1, label="y = x")

    for idx, (a1, a2) in enumerate(unique_pairs):
        told_pcts = sorted([p for (x1, x2, p) in per_level if x1 == a1 and x2 == a2])
        actual_sims = [per_level[(a1, a2, p)] for p in told_pcts]

        label = _st_pair_label(a1, a2)
        ax1.plot(
            told_pcts,
            actual_sims,
            marker="o",
            color=colors[idx],
            linewidth=2,
            markersize=6,
            label=label,
        )

        if (a1, a2) in final_sim:
            avg = final_sim[(a1, a2)]
            ax1.axhline(y=avg, color=colors[idx], linestyle=":",
                        alpha=0.5, linewidth=1)
            ax1.annotate(
                f"avg={avg:.1f}%",
                xy=(102, avg),
                fontsize=8,
                color=colors[idx],
                va="center",
            )

    ax1.set_xlabel("Told Similarity (%)", fontsize=12)
    ax1.set_ylabel("Actual Distribution Similarity (%)", fontsize=12)
    ax1.set_title("Told vs Actual Similarity (JS Divergence)", fontsize=14, fontweight="bold")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-5, 115)
    ax1.set_ylim(-5, 105)

    save_fig(
        fig1,
        output_dir / "told_vs_actual_similarity.png",
        plots_dir / f"told_vs_actual_similarity_{seed}.png",
    )

    # Plot 2: Per-pair elicitation + validation payoffs
    pair_elicit: dict[str, dict[str, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for pct_str, pct_data in elicitation_by_pct.items():
        pct = int(pct_str) if pct_str.isdigit() else pct_data.get("similarity_pct", 0)
        for match in pct_data.get("matches", []):
            label = _st_pair_label(match["agent1_name"], match["agent2_name"])
            pair_elicit[label][short_pair_name(match["agent1_name"])][pct].append(match["agent1_payoff"])
            pair_elicit[label][short_pair_name(match["agent2_name"])][pct].append(match["agent2_payoff"])

    pair_valid: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    pair_valid_sim: dict[str, int] = {}
    for match in validation_results:
        label = _st_pair_label(match["agent1_name"], match["agent2_name"])
        pair_valid[label][short_pair_name(match["agent1_name"])].append(match["agent1_payoff"])
        pair_valid[label][short_pair_name(match["agent2_name"])].append(match["agent2_payoff"])
        pair_valid_sim[label] = match["similarity_pct"]

    pair_labels = sorted(pair_elicit.keys())
    n_pairs = len(pair_labels)
    ncols = min(n_pairs, 3) if n_pairs else 1
    nrows = (n_pairs + ncols - 1) // ncols if n_pairs else 1

    fig2, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
    pair_colors = ["#1f77b4", "#d62728"]

    for idx, pair_label in enumerate(pair_labels):
        ax = axes[idx // ncols][idx % ncols]
        agent_payoffs = pair_elicit[pair_label]
        agent_names = sorted(agent_payoffs.keys())

        for a_idx, agent in enumerate(agent_names):
            color = pair_colors[a_idx % len(pair_colors)]
            pcts_sorted = sorted(agent_payoffs[agent].keys())
            avg_payoffs = [np.mean(agent_payoffs[agent][p]) for p in pcts_sorted]

            ax.plot(
                pcts_sorted,
                avg_payoffs,
                marker="o",
                color=color,
                linewidth=2,
                markersize=4,
                label=f"{agent} (elicitation)",
            )

            if pair_label in pair_valid and agent in pair_valid[pair_label]:
                val_avg = np.mean(pair_valid[pair_label][agent])
                val_sim = pair_valid_sim.get(pair_label, 50)
                ax.plot(
                    val_sim,
                    val_avg,
                    marker="*",
                    color=color,
                    markersize=20,
                    markeredgecolor="black",
                    markeredgewidth=1,
                    zorder=10,
                    label=f"{agent} (validation)",
                )

        if pair_label in pair_valid_sim:
            vs = pair_valid_sim[pair_label]
            ax.axvline(x=vs, color="gray", linestyle="--", alpha=0.5, linewidth=1)
            ax.set_title(f"{pair_label}\ncomputed similarity = {vs}%", fontsize=11, fontweight="bold")
        else:
            ax.set_title(pair_label, fontsize=11, fontweight="bold")

        ax.set_xlabel("Told Similarity (%)", fontsize=10)
        ax.set_ylabel("Avg Payoff", fontsize=10)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-5, 105)

    for idx in range(n_pairs, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig2.suptitle(
        "Elicitation Payoffs + Validation at Computed Similarity",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig2.tight_layout()

    save_fig(
        fig2,
        output_dir / "elicitation_vs_validation_payoffs.png",
        plots_dir / f"elicitation_vs_validation_payoffs_{seed}.png",
    )

    # Plot 3: Per-pair payoff comparison
    pair_elicit_avg: dict[tuple[str, str], tuple[list, list]] = {}
    pair_valid_avg: dict[tuple[str, str], tuple[list, list]] = {}

    for pct_str, pct_data in elicitation_by_pct.items():
        for match in pct_data.get("matches", []):
            pair = (match["agent1_name"], match["agent2_name"])
            if pair not in pair_elicit_avg:
                pair_elicit_avg[pair] = ([], [])
            pair_elicit_avg[pair][0].append(match["agent1_payoff"])
            pair_elicit_avg[pair][1].append(match["agent2_payoff"])

    for match in validation_results:
        pair = (match["agent1_name"], match["agent2_name"])
        if pair not in pair_valid_avg:
            pair_valid_avg[pair] = ([], [])
        pair_valid_avg[pair][0].append(match["agent1_payoff"])
        pair_valid_avg[pair][1].append(match["agent2_payoff"])

    fig3, ax3 = plt.subplots(figsize=(10, 6))

    bar_labels: list[str] = []
    elicit_means: list[float] = []
    valid_means: list[float] = []
    computed_sims: list[float | None] = []

    for pair in sorted(pair_elicit_avg.keys()):
        a1, a2 = pair
        label = _st_pair_label(a1, a2)
        e_avg = np.mean(pair_elicit_avg[pair][0] + pair_elicit_avg[pair][1])
        v_avg = np.mean(pair_valid_avg[pair][0] + pair_valid_avg[pair][1]) if pair in pair_valid_avg else 0

        bar_labels.append(label)
        elicit_means.append(e_avg)
        valid_means.append(v_avg)
        computed_sims.append(final_sim.get(pair, None))

    x = np.arange(len(bar_labels))
    width = 0.35

    ax3.bar(x - width / 2, elicit_means, width, label="Elicitation (all levels)",
            color="#4C72B0", alpha=0.85)
    ax3.bar(x + width / 2, valid_means, width, label="Validation (computed sim)",
            color="#DD8452", alpha=0.85)

    for i, sim_v in enumerate(computed_sims):
        if sim_v is not None:
            ax3.annotate(
                f"sim={sim_v:.0f}%",
                xy=(x[i], max(elicit_means[i], valid_means[i]) + 0.05),
                ha="center",
                fontsize=9,
                fontstyle="italic",
                color="#555",
            )

    ax3.set_xlabel("Agent Pair", fontsize=12)
    ax3.set_ylabel("Average Payoff", fontsize=12)
    ax3.set_title("Elicitation vs Validation Payoffs by Pair", fontsize=14, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(bar_labels, rotation=20, ha="right", fontsize=9)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis="y")

    save_fig(
        fig3,
        output_dir / "pair_elicitation_vs_validation.png",
        plots_dir / f"pair_elicitation_vs_validation_{seed}.png",
    )

    # Plot 4: Cooperation rate vs told similarity (per pair)
    pair_coop: dict[str, dict[str, dict[int, list[int]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for pct_str, pct_data in elicitation_by_pct.items():
        pct = int(pct_str) if pct_str.isdigit() else pct_data.get("similarity_pct", 0)
        for match in pct_data.get("matches", []):
            label = _st_pair_label(match["agent1_name"], match["agent2_name"])
            pair_coop[label][short_pair_name(match["agent1_name"])][pct].append(
                1 if match["agent1_action"] == "A0" else 0
            )
            pair_coop[label][short_pair_name(match["agent2_name"])][pct].append(
                1 if match["agent2_action"] == "A0" else 0
            )

    pair_coop_valid: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for match in validation_results:
        label = _st_pair_label(match["agent1_name"], match["agent2_name"])
        pair_coop_valid[label][short_pair_name(match["agent1_name"])].append(
            1 if match["agent1_action"] == "A0" else 0
        )
        pair_coop_valid[label][short_pair_name(match["agent2_name"])].append(
            1 if match["agent2_action"] == "A0" else 0
        )

    coop_pair_labels = sorted(pair_coop.keys())
    n_coop = len(coop_pair_labels)
    ncols4 = min(n_coop, 3) if n_coop else 1
    nrows4 = (n_coop + ncols4 - 1) // ncols4 if n_coop else 1

    fig4, axes4 = plt.subplots(nrows4, ncols4, figsize=(6 * ncols4, 5 * nrows4), squeeze=False)

    for idx, plabel in enumerate(coop_pair_labels):
        ax = axes4[idx // ncols4][idx % ncols4]
        agent_data = pair_coop[plabel]
        agent_names = sorted(agent_data.keys())

        for a_idx, agent in enumerate(agent_names):
            color = pair_colors[a_idx % len(pair_colors)]
            pcts_sorted = sorted(agent_data[agent].keys())
            coop_rates = [np.mean(agent_data[agent][p]) * 100 for p in pcts_sorted]

            ax.plot(
                pcts_sorted,
                coop_rates,
                marker="o",
                color=color,
                linewidth=2,
                markersize=4,
                label=f"{agent} (elicitation)",
            )

            if plabel in pair_coop_valid and agent in pair_coop_valid[plabel]:
                val_coop = np.mean(pair_coop_valid[plabel][agent]) * 100
                val_sim = pair_valid_sim.get(plabel, 50)
                ax.plot(
                    val_sim,
                    val_coop,
                    marker="*",
                    color=color,
                    markersize=20,
                    markeredgecolor="black",
                    markeredgewidth=1,
                    zorder=10,
                    label=f"{agent} (validation)",
                )

        if plabel in pair_valid_sim:
            vs = pair_valid_sim[plabel]
            ax.axvline(x=vs, color="gray", linestyle="--", alpha=0.5, linewidth=1)
            ax.set_title(f"{plabel}\ncomputed similarity = {vs}%", fontsize=11, fontweight="bold")
        else:
            ax.set_title(plabel, fontsize=11, fontweight="bold")

        ax.set_xlabel("Told Similarity (%)", fontsize=10)
        ax.set_ylabel("Cooperation Rate (%)", fontsize=10)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)

    for idx in range(n_coop, nrows4 * ncols4):
        axes4[idx // ncols4][idx % ncols4].set_visible(False)

    fig4.suptitle("Cooperation Rate vs Told Similarity", fontsize=14, fontweight="bold", y=1.02)
    fig4.tight_layout()

    save_fig(
        fig4,
        output_dir / "cooperation_rate.png",
        plots_dir / f"cooperation_rate_{seed}.png",
    )

    print(f"\nAll plots saved to:\n  - {output_dir}/\n  - {plots_dir}/")


# =============================================================================
#  trust_game_sweep
# =============================================================================

def _tg_short_name(full_name: str) -> str:
    """Shorten 'google/gemini-3-flash-preview(CoT)#P1' → 'gemini-3-flash#P1'."""
    m = re.match(r"(?:.*/)?([^(/]+)\(.*?\)(#P\d+)", full_name)
    if m:
        return m.group(1) + m.group(2)
    return full_name


def _tg_model_name(full_name: str) -> str:
    """Extract just the model: drops parenthetical and seat suffix."""
    m = re.match(r"(?:.*/)?([^(]+)\(", full_name)
    if m:
        return m.group(1)
    return full_name


def _tg_extract_payoffs(data: dict):
    percentages = sorted(data["percentages"])
    matchup_payoffs: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    model_payoffs: dict = defaultdict(lambda: defaultdict(list))

    for pct_str, trials in data["results_by_percentage"].items():
        pct = int(pct_str)
        for trial in trials:
            for round_moves in trial:
                players_in_round = sorted(round_moves, key=lambda m: m["player"])
                matchup_key = " vs ".join(_tg_short_name(m["player"]) for m in players_in_round)

                for move in round_moves:
                    name = _tg_short_name(move["player"])
                    model = _tg_model_name(move["player"])
                    matchup_payoffs[matchup_key][pct][name].append(move["points"])
                    model_payoffs[model][pct].append(move["points"])

    return percentages, matchup_payoffs, model_payoffs


def _tg_plot_matchup_payoffs(percentages, matchup_payoffs, output_dir: Path) -> None:
    matchups = sorted(matchup_payoffs.keys())
    n = len(matchups)
    cols = min(n, 2) if n else 1
    rows = (n + cols - 1) // cols if n else 1

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), squeeze=False)

    for idx, matchup_key in enumerate(matchups):
        ax = axes[idx // cols][idx % cols]
        pct_data = matchup_payoffs[matchup_key]

        player_names: set[str] = set()
        for pct_players in pct_data.values():
            player_names.update(pct_players.keys())
        player_names = sorted(player_names)

        for pname in player_names:
            means: list[float] = []
            stds: list[float] = []
            xs: list[int] = []
            for pct in percentages:
                pts_list = pct_data.get(pct, {}).get(pname, [])
                if pts_list:
                    means.append(np.mean(pts_list))
                    stds.append(np.std(pts_list) if len(pts_list) > 1 else 0)
                    xs.append(pct)
            means = np.array(means)
            stds = np.array(stds)
            ax.plot(xs, means, marker="o", markersize=4, label=pname)
            if np.any(stds > 0):
                ax.fill_between(xs, means - stds, means + stds, alpha=0.15)

        ax.set_xlabel("Similarity %")
        ax.set_ylabel("Average Payoff")
        ax.set_title(matchup_key, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle("Payoffs by Matchup × Similarity Level", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, output_dir / "trust_game_matchup_payoffs.png", dpi=200)


def _tg_plot_model_payoffs(percentages, model_payoffs, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for model in sorted(model_payoffs.keys()):
        pct_data = model_payoffs[model]
        means: list[float] = []
        xs: list[int] = []
        for pct in percentages:
            pts = pct_data.get(pct, [])
            if pts:
                means.append(np.mean(pts))
                xs.append(pct)
        ax.plot(xs, means, marker="o", markersize=5, linewidth=2, label=model)

    ax.set_xlabel("Similarity %", fontsize=12)
    ax.set_ylabel("Average Payoff", fontsize=12)
    ax.set_title("Aggregate Model Payoffs vs Similarity", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_fig(fig, output_dir / "trust_game_model_payoffs.png", dpi=200)


def plot_trust_game_sweep(args: argparse.Namespace) -> None:
    """Trust-game (or any asymmetric game) sweep plots: per-matchup + aggregate-by-model."""
    path = Path(args.results_path)
    if not path.exists():
        print(f"Error: {path} not found")
        sys.exit(1)

    data = load_json(path)
    percentages, matchup_payoffs, model_payoffs = _tg_extract_payoffs(data)

    output_dir = path.parent
    _tg_plot_matchup_payoffs(percentages, matchup_payoffs, output_dir)
    _tg_plot_model_payoffs(percentages, model_payoffs, output_dir)


# =============================================================================
#  travellers_sweep
# =============================================================================

_TD_CLAIM_PALETTE = ("#d62728", "#fdae61", "#a6d96a", "#1a9850", "#006837", "#f4a582", "#92c5de")

_MODEL_DISPLAY = (
    ("gemini",   "Gemini"),
    ("gpt",      "GPT"),
    ("claude",   "Claude"),
    ("deepseek", "DeepSeek"),
    ("gemma",    "Gemma"),
)


def _model_display_label(model: str) -> str:
    m = model.lower()
    for key, label in _MODEL_DISPLAY:
        if key in m:
            return label
    return model


def _model_display_sort_key(model: str) -> tuple:
    m = model.lower()
    for i, (key, _) in enumerate(_MODEL_DISPLAY):
        if key in m:
            return (0, i)
    return (1, m)


def _td_claim_values(config: dict) -> tuple[int, ...]:
    g = (config.get("game") or {}).get("kwargs") or {}
    min_claim = int(g.get("min_claim", 2))
    num = int(g.get("num_actions", 4))
    spacing = int(g.get("claim_spacing", 1))
    return tuple(min_claim + i * spacing for i in range(num))


def _td_token_to_claim(token: str, claim_values: tuple[int, ...]) -> int | None:
    if not token or not token.startswith("A"):
        return None
    try:
        idx = int(token[1:])
    except ValueError:
        return None
    return claim_values[idx] if 0 <= idx < len(claim_values) else None


def _td_expected_claim(move: dict, claim_values: tuple[int, ...]) -> float | None:
    mp = move.get("mix_probs")
    if mp:
        total = 0.0
        weight = 0.0
        for tok, pct in mp.items():
            claim = _td_token_to_claim(tok, claim_values)
            if claim is None:
                continue
            total += (pct / 100.0) * claim
            weight += pct / 100.0
        if weight > 0:
            return total / weight
    claim = _td_token_to_claim(move.get("action", ""), claim_values)
    return float(claim) if claim is not None else None


def _td_dist_per_move(move: dict, claim_values: tuple[int, ...]) -> dict[int, float] | None:
    mp = move.get("mix_probs")
    if mp:
        out = {c: 0.0 for c in claim_values}
        total = 0.0
        for tok, pct in mp.items():
            claim = _td_token_to_claim(tok, claim_values)
            if claim is not None and claim in out:
                out[claim] += pct / 100.0
                total += pct / 100.0
        if total > 0:
            return {c: v / total for c, v in out.items()}
    sampled = _td_token_to_claim(move.get("action", ""), claim_values)
    if sampled is None:
        return None
    return {c: (1.0 if c == sampled else 0.0) for c in claim_values}


def _td_collect(results_by_pct: dict, claim_values: tuple[int, ...]) -> dict:
    """Bucket per-move expected claims and per-claim mass by (model, pct)."""
    seat_re = re.compile(r"#P\d+$")
    expected: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    dist: dict[str, dict[int, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    for pct_key, trials in results_by_pct.items():
        try:
            pct = int(pct_key)
        except (TypeError, ValueError):
            continue
        for trial in trials:
            for round_moves in trial:
                for mv in round_moves:
                    model = seat_re.sub("", mv["player"])
                    short = short_model_name(model)
                    ec = _td_expected_claim(mv, claim_values)
                    if ec is not None:
                        expected[short][pct].append(ec)
                    dm = _td_dist_per_move(mv, claim_values)
                    if dm is not None:
                        for c, p in dm.items():
                            dist[short][pct][c].append(p)
    return {"expected": expected, "dist": dist}


def _td_plot_heatmap(expected, percentages, claim_values, out_path) -> None:
    models = sorted(expected.keys())
    if not models:
        print(f"  warning: no models in expected; skipping {out_path}")
        return
    mean_mat = np.full((len(models), len(percentages)), np.nan)
    sem_mat = np.full((len(models), len(percentages)), np.nan)
    for i, m in enumerate(models):
        for j, p in enumerate(percentages):
            vals = expected[m].get(p, [])
            mu, se, _ = mean_sem(vals)
            mean_mat[i, j] = mu
            sem_mat[i, j] = se if se == se else 0.0
    fig, ax = plt.subplots(
        figsize=(0.9 * len(percentages) + 3, 0.7 * len(models) + 2.5)
    )
    vmin, vmax = float(claim_values[0]), float(claim_values[-1])
    im = ax.imshow(mean_mat, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(percentages)), [f"{p}%" for p in percentages])
    ax.set_yticks(range(len(models)), models)
    ax.set_xlabel("Similarity %")
    ax.set_title(
        f"Mean claim per (model, similarity %) — TD claims {claim_values[0]}–{claim_values[-1]}"
        "\n(cells show mean ± SEM across self-play moves)"
    )
    for i in range(len(models)):
        for j in range(len(percentages)):
            mu = mean_mat[i, j]
            if np.isnan(mu):
                continue
            color = "white" if (mu - vmin) / (vmax - vmin) < 0.55 else "black"
            ax.text(j, i, f"{mu:.2f}\n± {sem_mat[i, j]:.2f}",
                    ha="center", va="center", fontsize=8, color=color)
    fig.colorbar(im, ax=ax, label=f"E[claim] (range {vmin:.0f}–{vmax:.0f})")
    fig.tight_layout()
    save_fig(fig, out_path)


def _td_plot_stacked(dist, percentages, claim_values, out_path) -> None:
    models = sorted(dist.keys(), key=_model_display_sort_key)
    if not models:
        print(f"  warning: no models in dist; skipping {out_path}")
        return
    palette = _TD_CLAIM_PALETTE[: len(claim_values)]
    fig, axes = plt.subplots(len(models), 1, figsize=(8, 2.6 * len(models)),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    y = np.arange(len(percentages))
    for ax, model in zip(axes, models):
        lefts = np.zeros(len(percentages))
        for c, color in zip(claim_values, palette):
            widths = np.zeros(len(percentages))
            for j, p in enumerate(percentages):
                samples = dist[model].get(p, {}).get(c, [])
                widths[j] = float(np.mean(samples)) if samples else 0.0
            ax.barh(y, widths, left=lefts, color=color, label=f"claim={c}",
                    edgecolor="white", linewidth=0.4)
            lefts += widths
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_yticks(y, [f"{p}%" for p in percentages])
        ax.set_ylabel(_model_display_label(model), fontsize=10)
        ax.invert_yaxis()
    axes[-1].set_xlabel("Claim fraction")
    axes[0].set_title(
        f"Claim distribution per (model, similarity %) — TD claims {claim_values[0]}–{claim_values[-1]}"
    )
    handles = [Patch(color=c, label=f"claim={v}") for c, v in zip(palette, claim_values)]
    axes[0].legend(handles=handles, ncol=len(claim_values), loc="lower center",
                   bbox_to_anchor=(0.5, 1.18), frameon=False, fontsize=9)
    fig.tight_layout()
    save_fig(fig, out_path)


def _td_plot_lines(expected, percentages, claim_values, out_path) -> None:
    models = sorted(expected.keys())
    if not models:
        print(f"  warning: no models in expected; skipping {out_path}")
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = tab10_colors(len(models))
    x = np.array(percentages, dtype=float)
    for model, color in zip(models, colors):
        means = np.full(len(percentages), np.nan)
        sems = np.zeros(len(percentages))
        for j, p in enumerate(percentages):
            vals = expected[model].get(p, [])
            mu, se, _ = mean_sem(vals)
            means[j] = mu
            sems[j] = se if se == se else 0.0
        ax.plot(x, means, marker="o", color=color, label=model, linewidth=2)
        ax.fill_between(x, means - sems, means + sems, color=color, alpha=0.18)
    ax.set_xlabel("Similarity %")
    ax.set_ylabel(f"Mean claim (range {claim_values[0]}–{claim_values[-1]})")
    ax.set_ylim(claim_values[0] - 0.05, claim_values[-1] + 0.05)
    ax.set_title("Mean claim vs similarity (Traveller's Dilemma)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    save_fig(fig, out_path)


def plot_travellers_sweep(args: argparse.Namespace) -> None:
    """Mean-claim heatmap, claim distribution, and mean-claim line overlay
    for Traveller's Dilemma similarity-sweep results."""
    results_path = Path(args.results_path)
    if not results_path.exists():
        print(f"ERROR: {results_path} not found")
        sys.exit(1)
    run_dir = results_path.parent
    config_path = run_dir / "config.json"
    config = load_json(config_path) if config_path.exists() else {}
    if (config.get("game") or {}).get("type") not in (None, "TravellersDilemma"):
        print(f"WARNING: game type is {(config.get('game') or {}).get('type')}, "
              f"not TravellersDilemma — proceeding anyway")
    claim_values = _td_claim_values(config)
    data = load_json(results_path)
    percentages = sorted(int(p) for p in data.get("percentages", []))
    if not percentages:
        percentages = sorted(int(p) for p in data.get("results_by_percentage", {}).keys())
    results_by_pct = data.get("results_by_percentage", {})
    buckets = _td_collect(results_by_pct, claim_values)
    out_dir = ensure_dir(run_dir / "travellers_plots")

    _td_plot_heatmap(buckets["expected"], percentages, claim_values,
                     out_dir / "mean_claim_heatmap.png")
    _td_plot_stacked(buckets["dist"], percentages, claim_values,
                     out_dir / "claim_distribution.png")
    _td_plot_lines(buckets["expected"], percentages, claim_values,
                   out_dir / "mean_claim_lines.png")

    rows: list[dict] = []
    for model, by_pct in sorted(buckets["expected"].items()):
        for p in percentages:
            vals = by_pct.get(p, [])
            mu, se, n = mean_sem(vals)
            rows.append({"model": model, "similarity_pct": p,
                         "mean_claim": round(mu, 4) if mu == mu else "",
                         "sem": round(se, 4) if se == se else "",
                         "n_moves": n})
    write_csv(rows, out_dir / "mean_claim_summary.csv")
    print(f"  csv: {out_dir / 'mean_claim_summary.csv'}")


DISPLAY_NAME_TOKENS = {
    "gemini": "Gemini",
    "gpt": "GPT",
    "claude": "Claude",
    "deepseek": "DeepSeek",
    "gemma": "Gemma",
}

DISPLAY_ORDER = ["Gemini", "GPT", "Claude", "DeepSeek", "Gemma"]

BENCH_DISPLAY = {
    "newcomb": "Newcomb",
    "trait": "Trait",
    "moral_choice": "Moral",
    "hle": "HLE",
}


def _display_name(short: str) -> str:
    s = short.lower()
    for tok, display in DISPLAY_NAME_TOKENS.items():
        if tok in s:
            return display
    return short


_BIMATRIX_SLOT = re.compile(r"\(CoT\)(?:#P\d+)?$|:nitro")


def _bimatrix_short(name: str) -> str:
    return _BIMATRIX_SLOT.sub("", name).split("/")[-1]


def _build_bimatrix_data(bench: str, cond: str | None, source_dirs: list[Path]):
    """Read records.jsonl from source_dirs for one (bench, cond) and return
    (labels, cell_p1, cell_p2, cell_joint, cell_text, sim_matrix, rows).

    When ``cond`` is None, looks at ``<src>/<bench>/`` instead of
    ``<src>/<bench>__<cond>/`` (used for exogenous tournaments).

    Returns None if no data found.
    """
    per_pair: dict[tuple[str, str], list[tuple[str, str, float, float]]] = (
        defaultdict(list)
    )
    sim_matrix: dict[tuple[str, str], int] = {}

    for src in source_dirs:
        sub = src / (f"{bench}__{cond}" if cond is not None else bench)
        recs = sub / "records.jsonl"
        if not recs.exists():
            continue
        with open(recs) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if not (isinstance(rec, list) and len(rec) >= 2):
                    continue
                m1, m2 = rec[0], rec[1]
                a = _bimatrix_short(m1["player"])
                b = _bimatrix_short(m2["player"])
                act1 = str(m1["action"]).split(".")[-1]
                act2 = str(m2["action"]).split(".")[-1]
                p1 = float(m1.get("points") or 0.0)
                p2 = float(m2.get("points") or 0.0)
                per_pair[(a, b)].append((act1, act2, p1, p2))

        sim_path = sub / "subjective_similarity_matrix.json"
        if sim_path.exists() and not sim_matrix:
            cached = load_json(sim_path)
            short_matrix = cached.get("matrix_judge_to_target", {})
            short_names = sorted({k.split(" vs ")[0] for k in short_matrix})
            for a in short_names:
                for b in short_names:
                    if a == b:
                        sim_matrix[(a, b)] = int(round(float(
                            short_matrix.get(f"{a} vs {a}", 0)
                        )))
                    else:
                        sa = float(short_matrix.get(f"{a} vs {b}", 0))
                        sb = float(short_matrix.get(f"{b} vs {a}", 0))
                        sim_matrix[(a, b)] = int(round((sa + sb) / 2))

    if not per_pair:
        return None

    all_short = {m for pair in per_pair for m in pair}
    short_to_display = {s: _display_name(s) for s in all_short}
    display_to_short: dict[str, str] = {}
    for s, d in short_to_display.items():
        display_to_short.setdefault(d, s)
    ordered_display = [d for d in DISPLAY_ORDER if d in display_to_short]
    ordered_display += sorted(d for d in display_to_short
                              if d not in DISPLAY_ORDER)
    models = [display_to_short[d] for d in ordered_display]
    labels = ordered_display
    n = len(models)

    rows: list[dict] = []
    cell_p1 = np.full((n, n), np.nan)
    cell_p2 = np.full((n, n), np.nan)
    cell_joint = np.full((n, n), np.nan)
    cell_text = [["" for _ in range(n)] for _ in range(n)]
    for i, a in enumerate(models):
        for j, b in enumerate(models):
            obs = per_pair.get((a, b), [])
            if not obs:
                cell_text[i][j] = "—"
                continue
            p1 = sum(o[2] for o in obs) / len(obs)
            p2 = sum(o[3] for o in obs) / len(obs)
            cell_p1[i, j] = p1
            cell_p2[i, j] = p2
            cell_joint[i, j] = (p1 + p2) / 2
            counts = {"CC": 0, "CD": 0, "DC": 0, "DD": 0}
            for a1, a2, _, _ in obs:
                k = ("C" if a1 == "COOPERATE" else "D") + (
                    "C" if a2 == "COOPERATE" else "D"
                )
                counts[k] += 1
            sim = sim_matrix.get((a, b))
            cell_text[i][j] = f"{p1:.1f}/{p2:.1f}"
            rows.append({
                "p1_model": a, "p2_model": b,
                "n_trials": len(obs),
                "similarity_pct": sim if sim is not None else "",
                "p1_mean_payoff": round(p1, 4),
                "p2_mean_payoff": round(p2, 4),
                "joint_mean_payoff": round((p1 + p2) / 2, 4),
                "CC": counts["CC"], "CD": counts["CD"],
                "DC": counts["DC"], "DD": counts["DD"],
            })

    return labels, cell_p1, cell_p2, cell_joint, cell_text, sim_matrix, rows


def _build_similarity_matrix(short_dict: dict[str, float]):
    """Convert a ``{"<a> vs <b>": value}`` dict into (labels, 2d array)
    using the standard display order. Returns None if empty."""
    if not short_dict:
        return None
    pairs = []
    for k in short_dict:
        if " vs " not in k:
            continue
        a, b = k.split(" vs ")
        pairs.append((a.strip(), b.strip()))
    all_short = sorted({s for ab in pairs for s in ab})
    short_to_display = {s: _display_name(s) for s in all_short}
    display_to_short: dict[str, str] = {}
    for s, d in short_to_display.items():
        display_to_short.setdefault(d, s)
    ordered_display = [d for d in DISPLAY_ORDER if d in display_to_short]
    ordered_display += sorted(d for d in display_to_short
                              if d not in DISPLAY_ORDER)
    models = [display_to_short[d] for d in ordered_display]
    n = len(models)
    mat = np.full((n, n), np.nan)
    for i, a in enumerate(models):
        for j, b in enumerate(models):
            v = short_dict.get(f"{a} vs {b}")
            if v is not None:
                mat[i, j] = float(v)
    return ordered_display, mat


def _render_similarity_row(benches: list[str], panels: dict, cond: str | None,
                           suptitle: str | None, out_path: Path,
                           ylabel: str = "Model A",
                           xlabel: str = "Model B") -> None:
    n_cols = len(benches)
    fig_w = 6.0 * n_cols + 1.6
    fig_h = 6.4
    fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    im = None
    vmin, vmax = 0.0, 100.0
    span = vmax - vmin
    for j, bench in enumerate(benches):
        ax = axes[0][j]
        key = (cond, bench) if cond is not None else (None, bench)
        data = panels.get(key)
        if data is None:
            ax.axis("off")
            ax.set_title(f"{BENCH_DISPLAY.get(bench, bench)}\n(no data)",
                         fontsize=12)
            continue
        labels, mat = data
        n = len(labels)
        im = ax.imshow(mat, cmap="RdYlGn", aspect="equal",
                       vmin=vmin, vmax=vmax)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=11)
        ax.set_yticklabels(labels if j == 0 else [], fontsize=11)
        if j == 0:
            ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_title(BENCH_DISPLAY.get(bench, bench), fontsize=15, pad=8)
        for r in range(n):
            for c in range(n):
                v = mat[r, c]
                if np.isnan(v):
                    ax.text(c, r, "—", ha="center", va="center",
                            fontsize=13, color="black")
                    continue
                norm = (v - vmin) / span
                color = "white" if (norm < 0.28 or norm > 0.72) else "black"
                ax.text(c, r, f"{v:.0f}", ha="center", va="center",
                        fontsize=13, fontweight="bold", color=color)
    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(),
                            fraction=0.015, pad=0.02, shrink=0.8)
        cbar.set_label("similarity (%)", fontsize=11)
    if suptitle:
        fig.suptitle(suptitle, fontsize=16, y=0.995)
    save_fig(fig, out_path)
    print(f"  png: {out_path}")


def plot_similarity_matrix_exogenous(args: argparse.Namespace) -> None:
    """1×N row of objective (algorithmic) similarity matrices, one panel per
    benchmark, read from a per-benchmark objective_similarity_matrix.json."""
    benches = args.benches
    out_path = Path(args.output_path)
    ensure_dir(out_path.parent)
    src = Path(args.source_path)
    if not src.exists():
        print(f"  not found: {src}")
        return
    data = load_json(src)
    per_bench = data.get("per_benchmark", {})

    panels: dict[tuple[str | None, str], tuple] = {}
    for bench in benches:
        section = per_bench.get(bench)
        if not section:
            print(f"  no objective sim for {bench} in {src}")
            continue
        result = _build_similarity_matrix(section)
        if result is not None:
            panels[(None, bench)] = result

    if not panels:
        print("  no data anywhere — nothing to plot")
        return

    _render_similarity_row(
        benches, panels, None,
        "Exogenous Similarity",
        out_path,
    )


def plot_similarity_matrix_endogenous(args: argparse.Namespace) -> None:
    """Per-condition row plots of subjective (endogenous) similarity matrices.
    Reads ``<source>/<bench>__<cond>/subjective_similarity_matrix.json``."""
    benches = args.benches
    conditions = args.conditions
    source_dirs = [Path(s) for s in args.source_dirs]
    out_path = Path(args.output_path)
    ensure_dir(out_path.parent)

    panels: dict[tuple[str | None, str], tuple] = {}
    for cond in conditions:
        for bench in benches:
            mat_dict: dict[str, float] = {}
            for src in source_dirs:
                p = src / f"{bench}__{cond}" / "subjective_similarity_matrix.json"
                if p.exists():
                    j = load_json(p)
                    mat_dict = j.get("matrix_judge_to_target", {})
                    break
            result = _build_similarity_matrix(mat_dict)
            if result is None:
                print(f"  no endo sim: {bench}__{cond}")
                continue
            panels[(cond, bench)] = result

    if not panels:
        print("  no data anywhere — nothing to plot")
        return

    stem = out_path.stem
    for cond in conditions:
        sub_path = out_path.with_name(f"{stem}_{cond}{out_path.suffix}")
        _render_similarity_row(
            benches, panels, cond,
            f"Endogenous Similarity {cond}",
            sub_path,
            ylabel="Judging Model",
            xlabel="Target Model",
        )


def _build_coop_data(bench: str, cond: str | None, source_dirs: list[Path]):
    """Like ``_build_bimatrix_data`` but produces cooperation-rate matrices.

    Returns (labels, cell_coop_p1, cell_coop_p2, cell_text) where
    cell_coop_pK ∈ [0, 1] is the per-cell P_K cooperation rate. Returns
    None if no data found.
    """
    result = _build_bimatrix_data(bench, cond, source_dirs)
    if result is None:
        return None
    labels, _p1, _p2, _joint, _text, _sim, rows = result
    n = len(labels)
    short_to_display = {row["p1_model"]: _display_name(row["p1_model"])
                        for row in rows}
    short_to_display.update(
        {row["p2_model"]: _display_name(row["p2_model"]) for row in rows}
    )
    display_to_pos = {d: i for i, d in enumerate(labels)}

    cell_coop_p1 = np.full((n, n), np.nan)
    cell_coop_p2 = np.full((n, n), np.nan)
    cell_text = [["—" for _ in range(n)] for _ in range(n)]
    for row in rows:
        i = display_to_pos[short_to_display[row["p1_model"]]]
        j = display_to_pos[short_to_display[row["p2_model"]]]
        n_trials = row["n_trials"]
        if n_trials == 0:
            continue
        p1_coop = (row["CC"] + row["CD"]) / n_trials
        p2_coop = (row["CC"] + row["DC"]) / n_trials
        cell_coop_p1[i, j] = p1_coop
        cell_coop_p2[i, j] = p2_coop
        cell_text[i][j] = f"{p1_coop:.0%}/{p2_coop:.0%}".replace("%", "")
    return labels, cell_coop_p1, cell_coop_p2, cell_text


def _render_coop_row(benches: list[str], panels: dict, cond: str | None,
                     suptitle: str | None, out_path: Path) -> None:
    n_cols = len(benches)
    fig_w = 6.0 * n_cols + 1.6
    fig_h = 6.4
    fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    im = None
    vmin, vmax = 0.0, 1.0
    span = vmax - vmin
    for j, bench in enumerate(benches):
        ax = axes[0][j]
        key = (cond, bench) if cond is not None else (None, bench)
        data = panels.get(key)
        if data is None:
            ax.axis("off")
            ax.set_title(f"{BENCH_DISPLAY.get(bench, bench)}\n(no data)",
                         fontsize=12)
            continue
        labels, cell_coop_p1, _coop_p2, cell_text = data
        n = len(labels)
        im = ax.imshow(cell_coop_p1, cmap="RdYlGn", aspect="equal",
                       vmin=vmin, vmax=vmax)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=11)
        ax.set_yticklabels(labels if j == 0 else [], fontsize=11)
        if j == 0:
            ax.set_ylabel("Model A", fontsize=12)
        ax.set_xlabel("Model B", fontsize=12)
        ax.set_title(BENCH_DISPLAY.get(bench, bench), fontsize=15, pad=8)
        for r in range(n):
            for c in range(n):
                v = cell_coop_p1[r, c]
                if np.isnan(v):
                    color = "black"
                else:
                    norm = (v - vmin) / span
                    color = "white" if (norm < 0.28 or norm > 0.72) else "black"
                ax.text(c, r, cell_text[r][c], ha="center", va="center",
                        fontsize=13, fontweight="bold", color=color)
    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(),
                            fraction=0.015, pad=0.02, shrink=0.8)
        cbar.set_label("Model A cooperation rate", fontsize=11)
    if suptitle:
        fig.suptitle(suptitle, fontsize=16, y=0.995)
    save_fig(fig, out_path)
    print(f"  png: {out_path}")


def plot_cooperation_bimatrix_grid(args: argparse.Namespace) -> None:
    """Per-condition cooperation-rate row plots (mirrors
    ``payoff_bimatrix_grid --per-condition``)."""
    benches = args.benches
    conditions = args.conditions
    source_dirs = [Path(s) for s in args.source_dirs]
    out_path = Path(args.output_path)
    ensure_dir(out_path.parent)

    panels: dict[tuple[str | None, str], tuple] = {}
    for cond in conditions:
        for bench in benches:
            data = _build_coop_data(bench, cond, source_dirs)
            if data is not None:
                panels[(cond, bench)] = data
            else:
                print(f"  no data: {bench}__{cond}")

    if not panels:
        print("  no data anywhere — nothing to plot")
        return

    stem = out_path.stem
    for cond in conditions:
        sub_path = out_path.with_name(f"{stem}_{cond}{out_path.suffix}")
        _render_coop_row(
            benches, panels, cond,
            f"Cooperation Rates — Endogenous Similarity "
            f"(Judging model: {cond})",
            sub_path,
        )


def plot_cooperation_bimatrix_exogenous(args: argparse.Namespace) -> None:
    """1×N cooperation-rate row for the exogenous tournament."""
    benches = args.benches
    source_dirs = [Path(s) for s in args.source_dirs]
    out_path = Path(args.output_path)
    ensure_dir(out_path.parent)

    panels: dict[tuple[str | None, str], tuple] = {}
    for bench in benches:
        data = _build_coop_data(bench, None, source_dirs)
        if data is not None:
            panels[(None, bench)] = data
        else:
            print(f"  no data: {bench}")

    if not panels:
        print("  no data anywhere — nothing to plot")
        return

    _render_coop_row(
        benches, panels, None,
        "Cooperation Rates — Exogenous Similarity",
        out_path,
    )


def plot_payoff_bimatrix(args: argparse.Namespace) -> None:
    """N×N bimatrix of mean payoffs per ordered (P1, P2) model pair.

    Reads ``records.jsonl`` from one or more source dirs structured as
    ``<source>/<bench>__<condition>/records.jsonl``. Trials across the source
    dirs are merged. Renders one figure + one CSV per condition.
    """
    bench = args.bench
    conditions = args.conditions
    source_dirs = [Path(s) for s in args.source_dirs]
    out_dir = ensure_dir(Path(args.output_dir))

    for cond in conditions:
        result = _build_bimatrix_data(bench, cond, source_dirs)
        if result is None:
            print(f"  no data for {bench}__{cond} — skipped")
            continue
        labels, _cell_p1, _cell_p2, cell_joint, cell_text, _sim, rows = result
        n = len(labels)

        fig, ax = plt.subplots(figsize=(1.4 * n + 1.5, 1.4 * n + 1.0))
        ax.imshow(cell_joint, cmap="RdYlGn", aspect="equal",
                  vmin=np.nanmin(cell_joint), vmax=np.nanmax(cell_joint))
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=11)
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_xlabel("Player 2", fontsize=12)
        ax.set_ylabel("Player 1", fontsize=12)
        ax.set_title(f"{bench} — {cond}", fontsize=13)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, cell_text[i][j], ha="center", va="center",
                        fontsize=12, fontweight="bold", color="black")
        fig.tight_layout()

        out_png = out_dir / f"payoff_bimatrix_{bench}__{cond}.png"
        save_fig(fig, out_png)
        print(f"  png: {out_png}")

        out_csv = out_dir / f"payoff_bimatrix_{bench}__{cond}.csv"
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"  csv: {out_csv}")


def _gather_panels(conditions: list[str], benches: list[str],
                   source_dirs: list[Path]):
    panels: dict[tuple[str, str], tuple] = {}
    color_mats: list[np.ndarray] = []
    for cond in conditions:
        for bench in benches:
            result = _build_bimatrix_data(bench, cond, source_dirs)
            if result is None:
                print(f"  no data: {bench}__{cond}")
                continue
            labels, cell_p1, _p2, _cell_joint, cell_text, _sim, _rows = result
            panels[(cond, bench)] = (labels, cell_p1, cell_text)
            color_mats.append(cell_p1)
    return panels, color_mats


def _render_bimatrix_row(benches: list[str], panels: dict, vmin: float,
                         vmax: float, cond: str | None, suptitle: str | None,
                         out_path: Path) -> None:
    n_cols = len(benches)
    fig_w = 6.0 * n_cols + 1.6
    fig_h = 6.4
    fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    im = None
    span = max(vmax - vmin, 1e-9)
    for j, bench in enumerate(benches):
        ax = axes[0][j]
        key = (cond, bench) if cond is not None else (None, bench)
        data = panels.get(key)
        if data is None:
            ax.axis("off")
            ax.set_title(f"{BENCH_DISPLAY.get(bench, bench)}\n(no data)",
                         fontsize=12)
            continue
        labels, cell_p1, cell_text = data
        n = len(labels)
        im = ax.imshow(cell_p1, cmap="RdYlGn", aspect="equal",
                       vmin=vmin, vmax=vmax)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=11)
        ax.set_yticklabels(labels if j == 0 else [], fontsize=11)
        if j == 0:
            ax.set_ylabel("Model A", fontsize=12)
        ax.set_xlabel("Model B", fontsize=12)
        ax.set_title(BENCH_DISPLAY.get(bench, bench), fontsize=15, pad=8)
        for r in range(n):
            for c in range(n):
                v = cell_p1[r, c]
                if np.isnan(v):
                    color = "black"
                else:
                    norm = (v - vmin) / span
                    color = "white" if (norm < 0.28 or norm > 0.72) else "black"
                ax.text(c, r, cell_text[r][c], ha="center", va="center",
                        fontsize=13, fontweight="bold", color=color)
    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(),
                            fraction=0.015, pad=0.02, shrink=0.8)
        cbar.set_label("Player 1 payoff (1=NE, 2=mutual coop)", fontsize=11)
    if suptitle:
        fig.suptitle(suptitle, fontsize=16, y=0.995)
    save_fig(fig, out_path)
    print(f"  png: {out_path}")


def plot_payoff_bimatrix_grid(args: argparse.Namespace) -> None:
    """Combined grid: rows = conditions, cols = benchmarks. With
    ``--per-condition`` set, emit one PNG per condition (1×4 row each)
    instead of a single combined grid."""
    benches = args.benches
    conditions = args.conditions
    source_dirs = [Path(s) for s in args.source_dirs]
    out_path = Path(args.output_path)
    ensure_dir(out_path.parent)

    panels, _color_mats = _gather_panels(conditions, benches, source_dirs)
    if not panels:
        print("  no data anywhere — nothing to plot")
        return

    vmin, vmax = 0.0, 3.0

    if args.per_condition:
        stem = out_path.stem
        for cond in conditions:
            sub_path = out_path.with_name(f"{stem}_{cond}{out_path.suffix}")
            _render_bimatrix_row(
                benches, panels, vmin, vmax, cond,
                f"Match-Up Payoffs — Endogenous Similarity "
                f"(Judging model: {cond})",
                sub_path,
            )
        return

    n_rows = len(conditions)
    n_cols = len(benches)
    fig_w = 3.0 * n_cols + 0.6
    fig_h = 3.0 * n_rows + 0.6
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                             squeeze=False)
    im = None
    for i, cond in enumerate(conditions):
        for j, bench in enumerate(benches):
            ax = axes[i][j]
            data = panels.get((cond, bench))
            if data is None:
                ax.axis("off")
                ax.set_title(f"{bench} — {cond}\n(no data)", fontsize=10)
                continue
            labels, cell_p1, cell_text = data
            n = len(labels)
            im = ax.imshow(cell_p1, cmap="RdYlGn", aspect="equal",
                           vmin=vmin, vmax=vmax)
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(labels if i == n_rows - 1 else [],
                               rotation=30, ha="right", fontsize=8)
            ax.set_yticklabels(labels if j == 0 else [], fontsize=8)
            if i == n_rows - 1:
                ax.set_xlabel("Player 2", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{cond}\n\nPlayer 1", fontsize=10)
            if i == 0:
                ax.set_title(bench, fontsize=12)
            for r in range(n):
                for c in range(n):
                    ax.text(c, r, cell_text[r][c], ha="center", va="center",
                            fontsize=6.5, fontweight="bold", color="black")
    if im is not None:
        cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02,
                            shrink=0.85)
        cbar.set_label("mean joint payoff", fontsize=10)
    fig.suptitle("PD payoff bimatrices — endogenous similarity",
                 fontsize=14, y=0.995)
    save_fig(fig, out_path)
    print(f"  png: {out_path}")


def plot_payoff_bimatrix_exogenous(args: argparse.Namespace) -> None:
    """1×N row of payoff bimatrices for the exogenous (algorithmic) similarity
    tournament. Reads ``<source-dir>/<bench>/records.jsonl``."""
    benches = args.benches
    source_dirs = [Path(s) for s in args.source_dirs]
    out_path = Path(args.output_path)
    ensure_dir(out_path.parent)

    panels: dict[tuple[str | None, str], tuple] = {}
    for bench in benches:
        result = _build_bimatrix_data(bench, None, source_dirs)
        if result is None:
            print(f"  no data: {bench}")
            continue
        labels, cell_p1, _p2, _cell_joint, cell_text, _sim, _rows = result
        panels[(None, bench)] = (labels, cell_p1, cell_text)

    if not panels:
        print("  no data anywhere — nothing to plot")
        return

    _render_bimatrix_row(
        benches, panels, 0.0, 3.0, None,
        "Match-Up Payoffs — Exogenous Similarity",
        out_path,
    )


def _avg_payoff_from_records(records_paths: list[Path]) -> tuple[float, int]:
    total = 0.0
    n = 0
    for path in records_paths:
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if not (isinstance(rec, list) and len(rec) >= 2):
                    continue
                for m in rec[:2]:
                    pts = m.get("points")
                    if pts is None:
                        continue
                    total += float(pts)
                    n += 1
    return (total / n if n else float("nan")), n


def plot_payoff_summary_table(args: argparse.Namespace) -> None:
    """Summary heatmap-table: rows = benchmarks, cols = similarity methods.

    Columns: Exogenous, Endogenous (both), Endogenous (decision),
    Endogenous (explanation). Each cell shows the average per-player payoff
    pooled across all matches under that (benchmark, method)."""
    benches = args.benches
    out_path = Path(args.output_path)
    ensure_dir(out_path.parent)

    exo_root = Path(args.exo_dir) if args.exo_dir else None
    endo_dirs = [Path(s) for s in args.endo_dirs]

    methods: list[tuple[str, str]] = [("Exogenous", "exo")]
    for cond in ("both", "decision", "explanation"):
        methods.append((f"Endogenous ({cond})", cond))

    n_rows = len(benches)
    n_cols = len(methods)
    matrix = np.full((n_rows, n_cols), np.nan)
    counts = np.zeros((n_rows, n_cols), dtype=int)

    for i, bench in enumerate(benches):
        for j, (_, key) in enumerate(methods):
            if key == "exo":
                paths = ([exo_root / bench / "records.jsonl"]
                         if exo_root else [])
            else:
                paths = [d / f"{bench}__{key}" / "records.jsonl"
                         for d in endo_dirs]
            avg, n = _avg_payoff_from_records(paths)
            matrix[i, j] = avg
            counts[i, j] = n

    fig_w = 1.6 * n_cols + 2.2
    fig_h = 0.95 * n_rows + 2.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    vmin, vmax = 0.0, 3.0
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto",
                   vmin=vmin, vmax=vmax)

    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels([m[0] for m in methods], rotation=20, ha="right",
                       fontsize=11)
    ax.set_yticklabels([BENCH_DISPLAY.get(b, b) for b in benches], fontsize=12)

    span = max(vmax - vmin, 1e-9)
    for i in range(n_rows):
        for j in range(n_cols):
            v = matrix[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=14,
                        color="black")
                continue
            norm = (v - vmin) / span
            color = "white" if (norm < 0.28 or norm > 0.72) else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=14, fontweight="bold", color=color)

    ax.set_title("Average Per-Player Payoff by Similarity Method",
                 fontsize=14, pad=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("avg payoff (1=NE, 2=mutual coop)", fontsize=10)
    fig.tight_layout()
    save_fig(fig, out_path)
    print(f"  png: {out_path}")

    csv_path = out_path.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["benchmark"] + [m[0] for m in methods])
        writer.writerow([""] + [f"n={counts[0, j]}" for j in range(n_cols)])
        for i, bench in enumerate(benches):
            row = [BENCH_DISPLAY.get(bench, bench)]
            for j in range(n_cols):
                v = matrix[i, j]
                row.append(f"{v:.4f}" if not np.isnan(v) else "")
            writer.writerow(row)
    print(f"  csv: {csv_path}")


# =============================================================================
#  CLI / dispatch
# =============================================================================

# Maps subcommand name -> (handler, parser-builder).
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plot.py",
        description="Unified plotting CLI for the similarity project.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="plot", required=True, metavar="<plot>")

    # ── battle_of_the_sexes ────────────────────────────────────────────────
    p = sub.add_parser("battle_of_the_sexes", help="Battle of the Sexes tournament plots.")
    p.add_argument("results_path", type=Path, help="Path to similarity_tournament_results.json")
    p.set_defaults(func=plot_battle_of_the_sexes)

    # ── benchmark_sweep ────────────────────────────────────────────────────
    p = sub.add_parser("benchmark_sweep",
                       help="Per-model heatmaps from benchmark_sweep_results.json.")
    p.add_argument("--results", type=str, required=True,
                   help="Path to benchmark_sweep_results.json")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--label", type=str, default=None,
                   help="X-axis label prefix. Auto-detected from results when omitted "
                        "(Similarity / Difference / Dissimilarity).")
    p.add_argument("--flip", choices=["auto", "yes", "no"], default="auto",
                   help="Flip the x-axis (JSON is keyed by raw similarity %%, so "
                        "difference/dissimilar runs need a flip to match the axis "
                        "the agent was shown). 'auto' reads difference_framing from "
                        "the results file.")
    p.set_defaults(func=plot_benchmark_sweep)

    # ── combined_heatmap ───────────────────────────────────────────────────
    p = sub.add_parser("combined_heatmap",
                       help="Combine multiple elicitation_sweep.json into one heatmap.")
    p.add_argument("inputs", nargs="+", help="elicitation_sweep.json paths")
    p.add_argument("--out", default="plots/similarity/combined_heatmap.png")
    p.add_argument("--model-order", default=None,
                   help="Comma-separated model short names in the order to display")
    p.set_defaults(func=plot_combined_heatmap)

    # ── cooperation_heatmap ────────────────────────────────────────────────
    p = sub.add_parser(
        "cooperation_heatmap",
        help="PD-style mean ± SEM heatmap (model × similarity %) from a basic sweep run.",
    )
    p.add_argument("--run-dir", type=Path, required=True, help="Experiment output directory")
    p.set_defaults(func=plot_cooperation_heatmap)

    # ── cooperation_with_sem ───────────────────────────────────────────────
    p = sub.add_parser(
        "cooperation_with_sem",
        help="Cooperation rate vs similarity with SEM (or per-player fallback).",
    )
    p.add_argument("--run-dir", type=Path, required=True, help="Experiment output directory")
    p.set_defaults(func=plot_cooperation_with_sem)

    # ── payoff_grid ────────────────────────────────────────────────────────
    p = sub.add_parser(
        "payoff_grid",
        help="2x2 dashboard of payoff-variation cooperation heatmaps "
             "(scale3, scale10, prop5_6, prop10_11).",
    )
    p.add_argument("--base-dir", type=Path,
                   default=Path("outputs/pd_payoff_variations"),
                   help="Directory containing scale3/scale10/prop5_6/prop10_11 subdirs.")
    p.set_defaults(func=plot_payoff_grid)

    # ── defection_rate ─────────────────────────────────────────────────────
    p = sub.add_parser(
        "defection_rate",
        help="Plot Gemini cooperation/similarity bars + scatter + table.",
    )
    p.add_argument("--comparison", type=str, required=True,
                   help="Path to comparison.json from run_random_vs_gemini.py")
    p.add_argument("--save-dir", type=str, default=None,
                   help="Where to save plots (default: same dir as comparison.json)")
    p.set_defaults(func=plot_defection_rate)

    # ── fixed_point ────────────────────────────────────────────────────────
    p = sub.add_parser("fixed_point", help="Fixed-point similarity plots.")
    p.add_argument("results_json", type=str, help="Path to fixed_point_results.json")
    p.add_argument("--output-dir", type=str, default=None)
    p.set_defaults(func=plot_fixed_point)

    # ── heatmaps ───────────────────────────────────────────────────────────
    p = sub.add_parser(
        "heatmaps",
        help="Mode x benchmark heatmaps from a comparison.json file.",
    )
    p.add_argument("--comparison", type=str, required=True)
    p.add_argument("--output-dir", type=str, default=None)
    p.set_defaults(func=plot_heatmaps)

    # ── newcomb ────────────────────────────────────────────────────────────
    p = sub.add_parser("newcomb", help="Newcomb benchmark plots.")
    p.add_argument("results_path", type=Path, help="Path to benchmark_results.json")
    p.set_defaults(func=plot_newcomb)

    # ── similarity_heatmap ─────────────────────────────────────────────────
    p = sub.add_parser(
        "similarity_heatmap",
        help="Heatmap from similarity-sweep or benchmark-sweep results JSONs.",
    )
    p.add_argument(
        "results_path",
        type=str,
        help="Path to similarity_sweep_results.json or benchmark_sweep_results.json",
    )
    p.set_defaults(func=plot_similarity_heatmap)

    # ── similarity_tournament ──────────────────────────────────────────────
    p = sub.add_parser("similarity_tournament", help="Two-phase similarity tournament plots.")
    p.add_argument("results_path", type=Path, help="Path to similarity_tournament_results.json")
    p.set_defaults(func=plot_similarity_tournament)

    # ── travellers_sweep ───────────────────────────────────────────────────
    p = sub.add_parser(
        "travellers_sweep",
        help="Mean-claim heatmap + claim distribution for TD similarity sweeps.",
    )
    p.add_argument("results_path", type=str,
                   help="Path to sweep_results_by_percentage.json")
    p.set_defaults(func=plot_travellers_sweep)

    # ── trust_game_sweep ───────────────────────────────────────────────────
    p = sub.add_parser(
        "trust_game_sweep",
        help="Trust-game (asymmetric-game) sweep payoff plots.",
    )
    p.add_argument("results_path", type=str, help="Path to sweep_results_by_percentage.json")
    p.set_defaults(func=plot_trust_game_sweep)

    # ── payoff_bimatrix ────────────────────────────────────────────────────
    p = sub.add_parser(
        "payoff_bimatrix",
        help="N×N payoff bimatrix per condition from subjective tournament records.jsonl.",
    )
    p.add_argument("--source-dirs", nargs="+", required=True,
                   help="One or more dirs containing <bench>__<condition>/records.jsonl. "
                        "Trials are merged across dirs.")
    p.add_argument("--bench", default="newcomb",
                   help="Benchmark name used as the first part of the subdir.")
    p.add_argument("--conditions", nargs="+",
                   default=["decision", "explanation", "both"],
                   help="Conditions (judging modes) to plot.")
    p.add_argument("--output-dir", required=True,
                   help="Directory for output PNGs + CSVs.")
    p.set_defaults(func=plot_payoff_bimatrix)

    # ── payoff_bimatrix_grid ───────────────────────────────────────────────
    p = sub.add_parser(
        "payoff_bimatrix_grid",
        help="Combined grid: rows=conditions, cols=benchmarks. One PNG.",
    )
    p.add_argument("--source-dirs", nargs="+", required=True,
                   help="One or more dirs containing <bench>__<condition>/records.jsonl.")
    p.add_argument("--benches", nargs="+",
                   default=["newcomb", "trait", "moral_choice", "hle"],
                   help="Benchmark column order (left to right).")
    p.add_argument("--conditions", nargs="+",
                   default=["decision", "explanation", "both"],
                   help="Condition row order (top to bottom).")
    p.add_argument("--output-path", required=True,
                   help="Output PNG path for the combined grid (used as a "
                        "stem when --per-condition is set).")
    p.add_argument("--per-condition", action="store_true",
                   help="Emit one PNG per condition (1×N benches each) "
                        "instead of a single combined grid.")
    p.set_defaults(func=plot_payoff_bimatrix_grid)

    # ── payoff_summary_table ───────────────────────────────────────────────
    p = sub.add_parser(
        "payoff_summary_table",
        help="Heatmap-table: rows=benchmarks, cols=similarity methods "
             "(Exogenous + 3 endogenous variants). Cell = avg payoff.",
    )
    p.add_argument("--benches", nargs="+",
                   default=["newcomb", "trait", "moral_choice", "hle"])
    p.add_argument("--exo-dir", default="outputs/objective_sim_tournament",
                   help="Directory containing <bench>/records.jsonl for the "
                        "exogenous (algorithmic) tournament.")
    p.add_argument("--endo-dirs", nargs="+", required=True,
                   help="Endogenous source dirs holding "
                        "<bench>__<condition>/records.jsonl. Trials merged.")
    p.add_argument("--output-path", required=True,
                   help="Output PNG path (CSV written alongside).")
    p.set_defaults(func=plot_payoff_summary_table)

    # ── payoff_bimatrix_exogenous ──────────────────────────────────────────
    p = sub.add_parser(
        "payoff_bimatrix_exogenous",
        help="1×N row of payoff bimatrices for the exogenous tournament.",
    )
    p.add_argument("--source-dirs", nargs="+", required=True,
                   help="One or more dirs containing <bench>/records.jsonl.")
    p.add_argument("--benches", nargs="+",
                   default=["trait", "hle", "moral_choice", "newcomb"],
                   help="Benchmark column order (left to right).")
    p.add_argument("--output-path", required=True,
                   help="Output PNG path.")
    p.set_defaults(func=plot_payoff_bimatrix_exogenous)

    # ── cooperation_bimatrix_grid ──────────────────────────────────────────
    p = sub.add_parser(
        "cooperation_bimatrix_grid",
        help="Per-condition cooperation-rate row plots (Model A coop "
             "rate, P1/P2 in cell text).",
    )
    p.add_argument("--source-dirs", nargs="+", required=True)
    p.add_argument("--benches", nargs="+",
                   default=["trait", "hle", "moral_choice", "newcomb"])
    p.add_argument("--conditions", nargs="+",
                   default=["decision", "explanation", "both"])
    p.add_argument("--output-path", required=True,
                   help="Stem path; one PNG per condition is written.")
    p.set_defaults(func=plot_cooperation_bimatrix_grid)

    # ── cooperation_bimatrix_exogenous ─────────────────────────────────────
    p = sub.add_parser(
        "cooperation_bimatrix_exogenous",
        help="1×N cooperation-rate row for the exogenous tournament.",
    )
    p.add_argument("--source-dirs", nargs="+", required=True)
    p.add_argument("--benches", nargs="+",
                   default=["trait", "hle", "moral_choice", "newcomb"])
    p.add_argument("--output-path", required=True)
    p.set_defaults(func=plot_cooperation_bimatrix_exogenous)

    # ── similarity_matrix_exogenous ────────────────────────────────────────
    p = sub.add_parser(
        "similarity_matrix_exogenous",
        help="1×N row of objective similarity matrices (one panel per bench).",
    )
    p.add_argument("--source-path", required=True,
                   help="Path to objective_similarity_matrix.json with a "
                        "per_benchmark section.")
    p.add_argument("--benches", nargs="+",
                   default=["trait", "hle", "moral_choice", "newcomb"])
    p.add_argument("--output-path", required=True)
    p.set_defaults(func=plot_similarity_matrix_exogenous)

    # ── similarity_matrix_endogenous ───────────────────────────────────────
    p = sub.add_parser(
        "similarity_matrix_endogenous",
        help="Per-condition row plots of subjective similarity matrices.",
    )
    p.add_argument("--source-dirs", nargs="+", required=True,
                   help="One or more dirs containing "
                        "<bench>__<condition>/subjective_similarity_matrix.json.")
    p.add_argument("--benches", nargs="+",
                   default=["trait", "hle", "moral_choice", "newcomb"])
    p.add_argument("--conditions", nargs="+",
                   default=["decision", "explanation", "both"])
    p.add_argument("--output-path", required=True,
                   help="Stem path; one PNG per condition is written.")
    p.set_defaults(func=plot_similarity_matrix_endogenous)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
