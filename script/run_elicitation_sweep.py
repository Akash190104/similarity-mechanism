#!/usr/bin/env python3
"""Run single-agent similarity elicitation across similarity levels and plot results.

Prompts each agent at every similarity level (0%, 10%, ..., 100%) for their
strategy distribution, without playing against anyone. Plots cooperation rate
vs told similarity.

Usage:
    # With GPT-5.4
    python script/run_elicitation_sweep.py \
        --agents configs/agents/gpt54.yaml \
        --game PrisonersDilemma \
        --trials 10 \
        --seed 42

    # Dry run with test agents
    python script/run_elicitation_sweep.py \
        --agents configs/agents/test_agents_3.yaml \
        --game PrisonersDilemma \
        --trials 2 \
        --seed 42
"""

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import yaml

from src.logger_manager import LOGGER
from src.mechanisms.similarity_elicitation import SimilarityElicitation
from src.registry.game_registry import GAME_REGISTRY

# Default PD payoff matrix
DEFAULT_PD_PAYOFF = {"CC": [2, 2], "CD": [0, 3], "DC": [3, 0], "DD": [1, 1]}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def load_agents_yaml(path: str) -> list[dict]:
    config_path = Path("configs") / path
    if not config_path.exists():
        config_path = Path(path)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_agents(agent_cfgs: list[dict]):
    """Create agents using the original sync agent factory (no Inspect needed)."""
    from src.registry.agent_registry import create_agent

    agents = []
    for cfg in agent_cfgs:
        agent_config = copy.deepcopy(cfg)
        agent_config["player_id"] = 1
        agents.append(create_agent(agent_config))
    return agents


def run_sweep(agent_cfgs, game, percentages, trials, prompt_mode, max_workers=1):
    """Run elicitation at each similarity level for each agent.

    Creates a fresh agent per thread to avoid sharing Inspect's async
    model/httpx client across event loops.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # We need agent names for display — create one set to get names
    temp_agents = create_agents(agent_cfgs)
    agent_names = [a.name for a in temp_agents]
    del temp_agents

    # Build all tasks: (pct, agent_cfg_idx, trial)
    all_tasks = [
        (pct, cfg_idx, trial)
        for pct in percentages
        for cfg_idx in range(len(agent_cfgs))
        for trial in range(1, trials + 1)
    ]
    total = len(all_tasks)
    print(f"\n[Sweep] {len(percentages)} levels x {len(agent_cfgs)} agents x {trials} trials = {total} calls, {max_workers} workers")

    results_by_pct = {str(pct): [] for pct in percentages}

    def run_one(task):
        pct, cfg_idx, trial = task
        from src.registry.agent_registry import create_agent
        cfg = copy.deepcopy(agent_cfgs[cfg_idx])
        cfg["player_id"] = 1
        agent = create_agent(cfg)

        elicitation = SimilarityElicitation(
            base_game=game,
            similarity_pct=pct,
            prompt_mode=prompt_mode,
        )
        result = elicitation.elicit_single(agent)
        return {
            "agent_name": result.agent_name,
            "similarity_pct": pct,
            "action_distribution": result.action_distribution,
            "trial": trial,
            "trace_id": result.trace_id,
        }

    from tqdm import tqdm
    if max_workers <= 1:
        for task in tqdm(all_tasks, desc="Elicitation sweep"):
            entry = run_one(task)
            results_by_pct[str(entry["similarity_pct"])].append(entry)
            coop = entry["action_distribution"].get("A0", 0)
            tqdm.write(f"  {entry['agent_name']} sim={entry['similarity_pct']}% trial {entry['trial']}: A0(coop)={coop}%")
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(run_one, t): t for t in all_tasks}
            for fut in tqdm(as_completed(futures), total=total, desc="Elicitation sweep"):
                entry = fut.result()
                results_by_pct[str(entry["similarity_pct"])].append(entry)
                coop = entry["action_distribution"].get("A0", 0)
                tqdm.write(f"  {entry['agent_name']} sim={entry['similarity_pct']}% trial {entry['trial']}: A0(coop)={coop}%")

    return results_by_pct


def run_benchmark_sweep(agent_cfgs, game, percentages, trials, benchmark_keys, max_workers=1):
    """Run elicitation with spoofed benchmark framing at each benchmark × similarity level.

    Returns: {benchmark: {pct_str: [entries]}}
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from benchmarks.registry import BENCHMARK_INFO, build_benchmark_prompt_with_context
    from src.mechanisms.prompts import SIMILARITY_MECHANISM_PROMPT_SINGLE
    from src.mechanisms.similarity_utils import build_similarity_framing

    # Build all tasks: (benchmark_key, pct, cfg_idx, trial)
    all_tasks = [
        (bk, pct, cfg_idx, trial)
        for bk in benchmark_keys
        for pct in percentages
        for cfg_idx in range(len(agent_cfgs))
        for trial in range(1, trials + 1)
    ]
    total = len(all_tasks)
    print(
        f"\n[Benchmark Sweep] {len(benchmark_keys)} benchmarks x "
        f"{len(percentages)} levels x {len(agent_cfgs)} agents x "
        f"{trials} trials = {total} calls, {max_workers} workers"
    )

    results_by_benchmark = {
        bk: {str(pct): [] for pct in percentages} for bk in benchmark_keys
    }

    def run_one(task):
        bk, pct, cfg_idx, trial = task
        cfg = copy.deepcopy(agent_cfgs[cfg_idx])
        cfg["player_id"] = 1
        provider = cfg.get("llm", {}).get("provider", "")
        if provider == "TestInstance":
            from src.registry.agent_registry import create_agent
            agent = create_agent(cfg)
        else:
            from inspect_similarity.agents.inspect_agent import create_inspect_agent
            agent = create_inspect_agent(cfg)

        # Build spoofed benchmark prompt
        bench_prompt = build_benchmark_prompt_with_context(
            bk, all_benchmarks=benchmark_keys,
        )
        framing = build_similarity_framing(
            similarity_pct=pct,
            prompt_mode="custom",
            custom_prompt=bench_prompt,
            num_other_players=game.num_players - 1,
        )
        extra_info = SIMILARITY_MECHANISM_PROMPT_SINGLE.format(
            similarity_framing=framing,
        )

        trace_id, mix_probs = game.prompt_player_mix_probs(
            agent, extra_info=extra_info
        )

        action_dist = {}
        for idx, prob in mix_probs.items():
            token = game.action_class.from_index(idx).to_token()
            action_dist[token] = prob

        return {
            "benchmark": bk,
            "agent_name": agent.name,
            "similarity_pct": pct,
            "action_distribution": action_dist,
            "trial": trial,
            "trace_id": trace_id,
        }

    from tqdm import tqdm
    if max_workers <= 1:
        for task in tqdm(all_tasks, desc="Benchmark sweep"):
            entry = run_one(task)
            results_by_benchmark[entry["benchmark"]][str(entry["similarity_pct"])].append(entry)
            coop = entry["action_distribution"].get("A0", 0)
            tqdm.write(f"  {entry['agent_name']} {entry['benchmark']}@{entry['similarity_pct']}% trial {entry['trial']}: A0={coop}%")
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(run_one, t): t for t in all_tasks}
            for fut in tqdm(as_completed(futures), total=total, desc="Benchmark sweep"):
                entry = fut.result()
                results_by_benchmark[entry["benchmark"]][str(entry["similarity_pct"])].append(entry)
                coop = entry["action_distribution"].get("A0", 0)
                tqdm.write(f"  {entry['agent_name']} {entry['benchmark']}@{entry['similarity_pct']}% trial {entry['trial']}: A0={coop}%")

    return results_by_benchmark


def plot_benchmark_heatmap(results_by_benchmark, percentages, output_dir):
    """Plot per-model heatmap: rows=benchmarks, cols=similarity %, cells=cooperation rate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from collections import defaultdict

    benchmark_keys = list(results_by_benchmark.keys())

    # Collect: {model: {benchmark: {pct: [coop_rates]}}}
    model_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for bk, pct_data in results_by_benchmark.items():
        for pct_str, entries in pct_data.items():
            pct = int(pct_str)
            for entry in entries:
                name = entry["agent_name"].split("/")[-1].split("(")[0]
                coop = entry.get("action_distribution", {}).get("A0", 0)
                model_data[name][bk][pct].append(coop)

    models = sorted(model_data.keys())

    for model in models:
        matrix = []
        for bk in benchmark_keys:
            row = [np.mean(model_data[model][bk].get(p, [np.nan])) for p in percentages]
            matrix.append(row)
        matrix = np.array(matrix)

        fig, ax = plt.subplots(
            figsize=(max(10, len(percentages) * 0.9), max(6, len(benchmark_keys) * 0.6))
        )
        sns.heatmap(
            matrix,
            xticklabels=[f"{p}%" for p in percentages],
            yticklabels=benchmark_keys,
            cmap="RdYlGn",
            vmin=0, vmax=100,
            annot=True, fmt=".1f",
            cbar_kws={"label": "Cooperation Rate (%)"},
            ax=ax,
            linewidths=0.5, linecolor="gray",
        )
        ax.set_xlabel("Told Similarity (%)", fontsize=13, fontweight="bold")
        ax.set_ylabel("Benchmark", fontsize=13, fontweight="bold")
        ax.set_title(
            f"Cooperation Rate — {model}\n(Spoofed Benchmark Framing)",
            fontsize=14, fontweight="bold", pad=20,
        )
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        safe = model.replace("/", "_").replace(" ", "_")
        out_path = output_dir / f"benchmark_sweep_{safe}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"\nHeatmap saved to {out_path}")


def plot_heatmap(results_by_pct, output_dir):
    """Plot heatmap: rows=similarity %, columns=models, cells=mean cooperation rate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from collections import defaultdict

    # Collect: {model: {pct: [coop_rates]}}
    data_dict = defaultdict(lambda: defaultdict(list))
    for pct_str, entries in results_by_pct.items():
        pct = int(pct_str)
        for entry in entries:
            agent_name = entry["agent_name"]
            model_name = agent_name.split("/")[-1].split("(")[0]
            coop = entry.get("action_distribution", {}).get("A0", 0)
            data_dict[model_name][pct].append(coop)

    models = sorted(data_dict.keys())
    percentages = sorted({int(k) for k in results_by_pct.keys()})

    # Build matrix: rows=similarity % (high→low), cols=models
    matrix = []
    for pct in percentages:
        row = [np.mean(data_dict[m].get(pct, [np.nan])) for m in models]
        matrix.append(row)

    matrix = np.flipud(np.array(matrix))
    pcts_reversed = percentages[::-1]

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 2.5), max(6, len(percentages) * 0.55)))

    sns.heatmap(
        matrix,
        xticklabels=models,
        yticklabels=[f"{p}%" for p in pcts_reversed],
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
    ax.set_ylabel("Told Similarity (%)", fontsize=13, fontweight="bold")
    ax.set_title("Cooperation Rate by Similarity Level", fontsize=14, fontweight="bold", pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    out_path = output_dir / "cooperation_heatmap.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nHeatmap saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Elicitation sweep: cooperation rate vs told similarity"
    )
    parser.add_argument(
        "--agents", type=str, required=True,
        help="Path to agents config YAML (relative to configs/)",
    )
    parser.add_argument(
        "--game", type=str, default="PrisonersDilemma",
        help="Game type (default: PrisonersDilemma)",
    )
    parser.add_argument(
        "--game-kwargs", type=str, default=None,
        help='Game kwargs as JSON (default: standard PD payoff matrix)',
    )
    parser.add_argument(
        "--trials", type=int, default=10,
        help="Number of trials per agent per similarity level (default: 10)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--increment", type=int, default=10,
        help="Similarity level increment (default: 10)",
    )
    parser.add_argument(
        "--prompt-mode", type=str, default="percentage_updated",
        help="Prompt mode (default: percentage_updated)",
    )
    parser.add_argument(
        "--benchmarks", type=str, default=None,
        help=(
            "Comma-separated benchmark names for spoofed benchmark sweep. "
            "When set, runs benchmarks x similarity levels with benchmark-specific framing. "
            "Use 'all' for all benchmarks. (e.g. newcomb,cabin,ggb,dilemmas,moral_choice,trait)"
        ),
    )
    parser.add_argument(
        "--max-workers", type=int, default=1,
        help="Max parallel LLM calls (default: 1, sequential)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Custom output directory",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    # Disable tqdm mp lock (same issue as Inspect tasks)
    import tqdm.std as _tqdm_std
    if not hasattr(_tqdm_std.TqdmDefaultWriteLock, "_mp_patched"):
        @classmethod
        def _noop(cls):
            cls.mp_lock = None
        _tqdm_std.TqdmDefaultWriteLock.create_mp_lock = _noop
        _tqdm_std.TqdmDefaultWriteLock._mp_patched = True

    if args.output_dir:
        LOGGER.set_log_dir(Path(args.output_dir))

    # Create game
    game_class = GAME_REGISTRY[args.game]
    if args.game_kwargs:
        game_kwargs = json.loads(args.game_kwargs)
    elif args.game == "PrisonersDilemma":
        game_kwargs = {"payoff_matrix": DEFAULT_PD_PAYOFF}
    else:
        game_kwargs = {}
    game = game_class(**game_kwargs)

    # Load agent configs
    agent_cfgs = load_agents_yaml(args.agents)
    temp_agents = create_agents(agent_cfgs)
    print(f"Agents: {[a.name for a in temp_agents]}")
    agent_names = [a.name for a in temp_agents]
    del temp_agents

    # Similarity levels
    percentages = list(range(0, 101, args.increment))

    if args.benchmarks:
        # Benchmark sweep mode
        from benchmarks.registry import BENCHMARK_INFO, BENCHMARKS_EXCLUDE_FROM_ALL

        if args.benchmarks == "all":
            benchmark_keys = [k for k in BENCHMARK_INFO if k not in BENCHMARKS_EXCLUDE_FROM_ALL]
        else:
            benchmark_keys = [b.strip() for b in args.benchmarks.split(",")]

        results_by_benchmark = run_benchmark_sweep(
            agent_cfgs, game, percentages, args.trials, benchmark_keys,
            max_workers=args.max_workers,
        )

        output = {
            "seed": args.seed,
            "game": args.game,
            "trials": args.trials,
            "percentages": percentages,
            "benchmarks": benchmark_keys,
            "agents": agent_names,
            "results_by_benchmark": {
                bk: pct_data for bk, pct_data in results_by_benchmark.items()
            },
        }
        LOGGER.log_record(output, "benchmark_sweep_results.json")

        plot_benchmark_heatmap(results_by_benchmark, percentages, LOGGER.log_dir)
    else:
        # Simple similarity sweep mode
        results_by_pct = run_sweep(
            agent_cfgs, game, percentages, args.trials, args.prompt_mode,
            max_workers=args.max_workers,
        )

        output = {
            "seed": args.seed,
            "game": args.game,
            "trials": args.trials,
            "percentages": percentages,
            "prompt_mode": args.prompt_mode,
            "agents": agent_names,
            "results_by_percentage": results_by_pct,
        }
        LOGGER.log_record(output, "elicitation_sweep.json")

        plot_heatmap(results_by_pct, LOGGER.log_dir)

    print(f"\nResults saved to {LOGGER.log_dir}")


if __name__ == "__main__":
    main()
