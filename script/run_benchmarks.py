#!/usr/bin/env python3
"""Entry point for running benchmarks and computing pairwise similarity.

Usage:
    python script/run_benchmarks.py --agents configs/agents/cheap_llms_3.yaml
    python script/run_benchmarks.py --agents configs/agents/cheap_llms_3.yaml --benchmark newcomb,gpqa,cabin --max-items 20
    python script/run_benchmarks.py --agents configs/agents/cheap_llms_3.yaml --benchmark hle --max-items 10

Outputs:
    Writes to `outputs/<yyyy>/<mm>/<dd>/<HH:MM:SS>/` (or `--output-dir`):
    `benchmark_results_<name>.json` per benchmark, `benchmark_results_all.json`
    when multiple benchmarks are requested, or `benchmark_results.json` for a
    single run; plots are written alongside (e.g. for `newcomb`).
"""

import argparse
import copy
import csv
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Semaphore

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from benchmarks.base import Benchmark, BenchmarkResult
from src.logger_manager import LOGGER
from src.registry.agent_registry import create_agent

ALL_BENCHMARKS = [
    "newcomb", "hle", "gpqa", "dilemmas", "moral_choice", "multi_tp",
    "cabin", "ggb", "trait", "random_die_roll", "random_die_roll_alt",
    "random_coin_toss", "random_coin_toss_alt", "similarity_game",
]


def set_seed(seed: int = 42) -> None:
    if torch is not None:
        torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_agents_config(path: str) -> list[dict]:
    """Load agents config from YAML file."""
    config_path = Path("configs") / path
    if not config_path.exists():
        config_path = Path(path)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_model_workers(spec: str | None) -> dict[str, int]:
    """Parse `--model-workers` CLI string into a {model_id: cap} dict.

    Format: comma-separated `model_id=int` pairs, e.g.
        deepseek/deepseek-v4-pro=80,google/gemma-4-31b-it=80
    """
    if not spec:
        return {}
    out: dict[str, int] = {}
    for part in spec.split(","):
        if not part.strip():
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = int(v.strip())
    return out


def wrap_agent_with_semaphore(agent, cap: int):
    """Throttle ``agent.invoke`` so only ``cap`` calls are in flight at once.

    Mirrors the per-model semaphore pattern in ``Similarity._build_model_semaphores``.
    Stamps the cap onto ``agent._sem_cap`` for downstream introspection.
    """
    sem = Semaphore(cap)
    real_invoke = agent.invoke

    def throttled_invoke(messages):
        with sem:
            return real_invoke(messages)

    agent.invoke = throttled_invoke
    agent._sem_cap = cap
    return agent


def create_benchmark(name: str, args: argparse.Namespace, agent_cfgs: list[dict]) -> Benchmark:
    """Create a benchmark instance by name with the appropriate kwargs."""
    if name == "hle":
        from benchmarks.hle import HLE

        all_test = all(
            cfg.get("llm", {}).get("provider") == "TestInstance" for cfg in agent_cfgs
        )
        judge_model = args.judge_model or ("test-judge" if all_test else "google/gemini-3.1-flash-lite-preview")
        judge_provider = args.judge_provider or ("TestInstance" if all_test else "OpenRouter")
        categories = args.categories.split(",") if args.categories else None

        return HLE(
            max_items=args.max_items,
            max_workers=args.max_workers,
            categories=categories,
            judge_model=judge_model,
            judge_provider=judge_provider,
            store_traces=args.store_traces,
        )
    elif name == "gpqa":
        from benchmarks.gpqa import GPQADiamond

        return GPQADiamond(
            max_items=args.max_items,
            max_workers=args.max_workers,
        )
    elif name == "dilemmas":
        from benchmarks.daily_dilemmas import DailyDilemmas

        return DailyDilemmas(
            max_items=args.max_items,
            max_workers=args.max_workers,
            store_traces=args.store_traces,
        )
    elif name == "moral_choice":
        from benchmarks.moral_choice import MoralChoice

        return MoralChoice(
            ambiguity=args.ambiguity,
            max_items=args.max_items,
            max_workers=args.max_workers,
            store_traces=args.store_traces,
        )
    elif name == "multi_tp":
        from benchmarks.multi_tp import MultiTP

        return MultiTP(
            lang=args.lang,
            max_items=args.max_items,
            max_workers=args.max_workers,
            seed=args.seed,
        )
    elif name == "cabin":
        from benchmarks.cabin import CABIN

        return CABIN(
            max_items=args.max_items,
            max_workers=args.max_workers,
            store_traces=args.store_traces,
        )
    elif name == "ggb":
        from benchmarks.ggb import GGB

        return GGB(
            statement_type=args.statement_type,
            max_items=args.max_items,
            max_workers=args.max_workers,
            store_traces=args.store_traces,
        )
    elif name in ("random_die_roll", "random_die_roll_alt"):
        from benchmarks.random_games import RandomDieRoll

        return RandomDieRoll(
            num_trials=args.max_items or 10,
        )
    elif name in ("random_coin_toss", "random_coin_toss_alt"):
        from benchmarks.random_games import RandomCoinToss

        # Cap coin toss at 50 trials regardless of --max-items.
        # Anthropic's content filter rejects 100+ H/T sequences as suspected
        # repetition spam, leaving claude with empty data otherwise.
        return RandomCoinToss(
            num_trials=min(args.max_items or 10, 50),
        )
    elif name == "trait":
        from benchmarks.trait import TRAIT

        traits = args.traits.split(",") if args.traits else None

        return TRAIT(
            traits=traits,
            max_items=args.max_items,
            max_workers=args.max_workers,
            store_traces=args.store_traces,
        )
    elif name == "similarity_game":
        from benchmarks.similarity_game import SimilarityGame
        from src.config_loader import ConfigLoader
        from src.registry.game_registry import GAME_REGISTRY

        if not args.game_config:
            raise ValueError("--game-config is required for similarity_game benchmark")

        loader = ConfigLoader()
        game_config = loader.load_main_config(args.game_config)
        game_class = GAME_REGISTRY[game_config["game"]["type"]]
        game = game_class(**game_config["game"].get("kwargs", {}))

        return SimilarityGame(
            base_game=game,
            trials=args.trials,
            max_items=args.max_items,
            max_workers=args.max_workers,
        )
    elif name == "newcomb":
        from benchmarks.newcomb import Newcomb

        return Newcomb(
            max_items=args.max_items,
            question_types=args.question_types,
            max_workers=args.max_workers,
            store_traces=args.store_traces,
        )
    else:
        raise ValueError(
            f"Unknown benchmark: {name!r}. Available: {', '.join(ALL_BENCHMARKS)}"
        )


def build_objective_output(
    bench_name: str,
    bench: Benchmark,
    agent_results: dict[str, BenchmarkResult],
    agents: list,
    args: argparse.Namespace,
) -> dict:
    """Compute pairwise objective similarity from already-collected per-agent
    benchmark results, then return the serialized output dict.

    ``agent_results`` is ``{agent.name: BenchmarkResult}``. Agents with no
    result (e.g. failed) are skipped.
    """
    agent_names = [a.name for a in agents if a.name in agent_results]

    matrix: dict[tuple[str, str], float] = {}
    for i, a in enumerate(agent_names):
        for j, b in enumerate(agent_names):
            if i < j:
                sim = bench.compute_similarity(agent_results[a], agent_results[b])
                matrix[(a, b)] = sim
                matrix[(b, a)] = sim
            elif i == j:
                matrix[(a, b)] = 100.0

    print(f"\n{'=' * 60}")
    print(f"PAIRWISE SIMILARITY MATRIX ({bench_name})")
    print(f"{'=' * 60}")
    header = "".ljust(40) + "  ".join(n[:15].ljust(15) for n in agent_names)
    print(f"\n{header}")
    for a in agent_names:
        row = a[:40].ljust(40)
        for b in agent_names:
            sim = matrix.get((a, b), 0.0)
            row += f"{sim:6.1f}".ljust(17)
        print(row)

    return {
        "benchmark": bench_name,
        "max_items": args.max_items,
        "agents": agent_names,
        "results": {
            name: result.serialize() for name, result in agent_results.items()
        },
        "similarity_matrix": {
            f"{a} vs {b}": sim
            for (a, b), sim in sorted(matrix.items())
            if a <= b
        },
    }


def _short_name(agent_full_name: str) -> str:
    """Strip provider prefix and (CoT)/(IO) suffix; e.g.
    'anthropic/claude-haiku-4.5(CoT)#P1' -> 'claude-haiku-4.5'."""
    s = agent_full_name.split("/")[-1].split("(")[0]
    # also strip any trailing #P\d
    return s.split("#")[0]


def write_summary_tables(all_outputs: dict[str, dict], run_dir: Path) -> None:
    """Write `objective_similarity_summary.md` (per-benchmark + overall 5x5
    tables) and `objective_similarity_long.csv` (one row per
    (benchmark, model_a, model_b))."""
    if not all_outputs:
        return

    # Use the first benchmark's agent list as the canonical row/col order,
    # but project to short model names for table headers.
    first = next(iter(all_outputs.values()))
    full_names = first["agents"]
    short_names = [_short_name(n) for n in full_names]
    short_by_full = dict(zip(full_names, short_names))

    def matrix_for(output: dict) -> dict[tuple[str, str], float]:
        m: dict[tuple[str, str], float] = {}
        for k, v in output["similarity_matrix"].items():
            a, b = k.split(" vs ")
            m[(short_by_full.get(a, a), short_by_full.get(b, b))] = float(v)
            m[(short_by_full.get(b, b), short_by_full.get(a, a))] = float(v)
        for s in short_names:
            m[(s, s)] = 100.0
        return m

    md_lines: list[str] = []
    md_lines.append("# Objective similarity — pairwise per benchmark\n")
    md_lines.append(
        "Each table is the per-benchmark objective similarity matrix "
        "(0–100) computed by the benchmark's own scoring method "
        "(MCQ exact-match, QWK for Likert, JSD for similarity_game, etc.).\n"
    )

    csv_rows: list[dict] = []
    accum: dict[tuple[str, str], list[float]] = {}

    for bench_name, output in all_outputs.items():
        m = matrix_for(output)
        md_lines.append(f"\n## {bench_name}\n")
        md_lines.append("| | " + " | ".join(short_names) + " |")
        md_lines.append("|---|" + "|".join("---:" for _ in short_names) + "|")
        for r in short_names:
            cells = [f"{m.get((r, c), float('nan')):.1f}" for c in short_names]
            md_lines.append(f"| **{r}** | " + " | ".join(cells) + " |")
        for r in short_names:
            for c in short_names:
                if r >= c:
                    continue
                val = m.get((r, c))
                if val is None:
                    continue
                csv_rows.append({"benchmark": bench_name, "model_a": r,
                                 "model_b": c, "similarity": round(val, 4)})
                accum.setdefault((r, c), []).append(val)

    # Overall mean across benchmarks
    md_lines.append("\n## Mean across all benchmarks (unweighted)\n")
    md_lines.append("| | " + " | ".join(short_names) + " |")
    md_lines.append("|---|" + "|".join("---:" for _ in short_names) + "|")
    for r in short_names:
        cells = []
        for c in short_names:
            if r == c:
                cells.append("100.0")
            else:
                key = (r, c) if (r, c) in accum else (c, r)
                vals = accum.get(key, [])
                cells.append(f"{(sum(vals)/len(vals)):.1f}" if vals else "—")
        md_lines.append(f"| **{r}** | " + " | ".join(cells) + " |")

    md_path = run_dir / "objective_similarity_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"  summary table:    {md_path}")

    csv_path = run_dir / "objective_similarity_long.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["benchmark", "model_a", "model_b", "similarity"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"  similarity csv:   {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run benchmarks and compute pairwise similarity"
    )
    parser.add_argument(
        "--agents",
        type=str,
        required=True,
        help="Path to agents config YAML",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="newcomb",
        help=(
            "Comma-separated benchmark names to run "
            f"(available: {', '.join(ALL_BENCHMARKS)}). "
            "Use 'all' to run all benchmarks. Default: newcomb"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory",
    )
    parser.add_argument(
        "--question-types",
        type=str,
        default="all",
        choices=["all", "capabilities", "attitudes"],
        help="Which question types to include (newcomb only)",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Limit number of questions (for cheaper runs)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Max concurrent LLM calls per agent (default: one per question). "
             "Ignored when --workers-per-model / --model-workers are set.",
    )
    parser.add_argument(
        "--workers-per-model",
        type=int,
        default=None,
        help="Default per-agent concurrency cap. Each agent's actual in-flight "
             "calls are limited by a per-model semaphore at this size.",
    )
    parser.add_argument(
        "--model-workers",
        type=str,
        default=None,
        help="Per-model overrides as 'agent_type=int' pairs, comma-separated. "
             "agent_type matches the agents config, e.g. "
             "'deepseek/deepseek-v4-pro=80,google/gemma-4-31b-it=80'.",
    )
    parser.add_argument(
        "--store-traces",
        action="store_true",
        default=True,
        help="Save full LLM response (reasoning + answer) per question. "
             "On by default; useful for later subjective-similarity passes.",
    )
    parser.add_argument(
        "--no-store-traces",
        dest="store_traces",
        action="store_false",
        help="Disable trace capture (smaller output JSON).",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma-separated HLE categories to include (e.g. Math,Physics)",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="Language code for multi_tp benchmark (default: en)",
    )
    parser.add_argument(
        "--ambiguity",
        type=str,
        default="all",
        choices=["all", "high", "low"],
        help="Ambiguity level for moral_choice benchmark (default: all)",
    )
    parser.add_argument(
        "--statement-type",
        type=str,
        default="all",
        choices=["all", "IH", "IB"],
        help="Statement type for ggb benchmark: all, IH (Instrumental Harm), IB (Impartial Beneficence)",
    )
    parser.add_argument(
        "--traits",
        type=str,
        default=None,
        help="Comma-separated traits for trait benchmark (e.g. Extraversion,Neuroticism)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Judge model for HLE short-answer scoring (default: anthropic/claude-haiku-4-5-20251001)",
    )
    parser.add_argument(
        "--judge-provider",
        type=str,
        default=None,
        help="Judge provider for HLE (default: OpenRouter, use TestInstance for dry runs)",
    )
    parser.add_argument(
        "--game-config",
        type=str,
        default=None,
        help="Game config for similarity_game benchmark (e.g. 'main/similarity_testing.yaml')",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Trials per similarity level for similarity_game benchmark (default: 1)",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    if args.output_dir:
        LOGGER.set_log_dir(Path(args.output_dir))

    # Parse benchmark names
    # gpqa, multi_tp excluded (require special setup); originals excluded in favor of alt versions
    EXCLUDE_FROM_ALL = {"gpqa", "multi_tp", "random_die_roll", "random_coin_toss"}
    if args.benchmark == "all":
        benchmark_names = [b for b in ALL_BENCHMARKS if b not in EXCLUDE_FROM_ALL]
    else:
        benchmark_names = [b.strip() for b in args.benchmark.split(",")]
        for b in benchmark_names:
            if b not in ALL_BENCHMARKS:
                parser.error(
                    f"Unknown benchmark: {b!r}. "
                    f"Available: {', '.join(ALL_BENCHMARKS)}"
                )

    # Load agents. No game seating, but use unique player_ids so duplicate
    # agent types (e.g. two RandomAgents) don't collide on `name`.
    agent_cfgs = load_agents_config(args.agents)
    agents = []
    for i, cfg in enumerate(agent_cfgs):
        agent_config = copy.deepcopy(cfg)
        agent_config["player_id"] = i + 1
        agents.append(create_agent(agent_config))

    # Per-model semaphore wrapping. ``workers_per_model`` is the default cap;
    # ``model_workers`` overrides it for specific agent_types. Match against
    # both the full ``agent_type`` (e.g. 'deepseek/deepseek-v4-pro(CoT)') and
    # the suffix-stripped form (e.g. 'deepseek/deepseek-v4-pro') so users can
    # specify either in --model-workers.
    mw_overrides = parse_model_workers(args.model_workers)
    default_cap = (
        args.workers_per_model
        if args.workers_per_model is not None
        else (args.max_workers or 30)
    )

    def _strip_suffix(agent_type: str) -> str:
        idx = agent_type.rfind("(")
        return agent_type[:idx] if idx >= 0 else agent_type

    for a in agents:
        cap = (
            mw_overrides.get(a.agent_type)
            or mw_overrides.get(_strip_suffix(a.agent_type))
            or default_cap
        )
        wrap_agent_with_semaphore(a, cap)
    # Inner pool size for each Benchmark's ``run_tasks`` — must be >= the
    # largest per-agent cap so the semaphore is the actual limiter.
    inner_pool_size = max(a._sem_cap for a in agents)
    args.max_workers = inner_pool_size
    print(
        f"\nConcurrency: per-agent caps = "
        f"{ {a.name: a._sem_cap for a in agents} }; "
        f"inner pool = {inner_pool_size}"
    )

    # Build all benchmarks once so all 5 agents see the same stratified
    # questions (stratified_sample uses seed=42 deterministically).
    benches: dict[str, Benchmark] = {}
    for bench_name in benchmark_names:
        benches[bench_name] = create_benchmark(bench_name, args, agent_cfgs)
        print(
            f"  built {bench_name:<22} "
            f"({len(benches[bench_name].questions)} questions loaded)"
        )

    # Run one pipeline per agent in parallel. Each pipeline walks through all
    # benchmarks sequentially for that agent; fast agents finish all 10
    # benchmarks early without head-of-line blocking the slow tail.
    print(f"\n{'=' * 60}")
    print(f"DISPATCHING {len(agents)} AGENT PIPELINES × {len(benchmark_names)} BENCHMARKS")
    print(f"{'=' * 60}")

    results_by_bench: dict[str, dict[str, BenchmarkResult]] = {
        b: {} for b in benchmark_names
    }

    cell_dir = LOGGER.log_dir / "per_cell"
    cell_dir.mkdir(parents=True, exist_ok=True)

    def agent_pipeline(agent) -> dict[str, BenchmarkResult]:
        import json
        import re as _re
        out: dict[str, BenchmarkResult] = {}
        safe = _re.sub(r"[^\w\-.]", "_", agent.name)
        for bench_name in benchmark_names:
            print(f"  [{agent.name}] starting {bench_name}", flush=True)
            try:
                br = benches[bench_name].run(agent)
            except Exception as exc:
                print(f"  [{agent.name}] FAILED on {bench_name}: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                continue
            out[bench_name] = br
            # Write per-cell snapshot immediately so a crash later doesn't lose
            # this completed (agent, benchmark) result.
            try:
                (cell_dir / f"{safe}__{bench_name}.json").write_text(
                    json.dumps(br.serialize(), indent=2)
                )
            except Exception as exc:
                print(f"  [{agent.name}] WARN: per-cell write failed for "
                      f"{bench_name}: {type(exc).__name__}: {exc}", flush=True)
            print(f"  [{agent.name}] finished {bench_name}: {br.scores}", flush=True)
        return out

    with ThreadPoolExecutor(max_workers=len(agents)) as ex:
        fut_to_agent = {ex.submit(agent_pipeline, a): a for a in agents}
        for fut in as_completed(fut_to_agent):
            a = fut_to_agent[fut]
            try:
                per_bench = fut.result()
            except Exception as exc:
                print(f"  [{a.name}] PIPELINE FAILED: {type(exc).__name__}: {exc}")
                continue
            for bench_name, br in per_bench.items():
                results_by_bench[bench_name][a.name] = br

    # Compute objective similarity per benchmark and persist per-benchmark JSON.
    all_outputs: dict[str, dict] = {}
    for bench_name in benchmark_names:
        output = build_objective_output(
            bench_name, benches[bench_name], results_by_bench[bench_name],
            agents, args,
        )
        all_outputs[bench_name] = output
        LOGGER.log_record(output, f"benchmark_results_{bench_name}.json")

    # Save combined results if multiple benchmarks.
    if len(benchmark_names) > 1:
        LOGGER.log_record(all_outputs, "benchmark_results_all.json")
        print(f"\n{'=' * 60}")
        print(f"ALL BENCHMARKS COMPLETE ({len(benchmark_names)} benchmarks)")
        print(f"{'=' * 60}")
    else:
        LOGGER.log_record(all_outputs[benchmark_names[0]], "benchmark_results.json")

    # Markdown summary + flat CSV.
    write_summary_tables(all_outputs, LOGGER.log_dir)

    print(f"\nResults saved to {LOGGER.log_dir}")

    # Generate plots (newcomb only)
    if "newcomb" in benchmark_names and len(benchmark_names) == 1:
        from script.plot import plot_newcomb

        results_path = LOGGER.log_dir / "benchmark_results.json"
        print("\nGenerating plots...")
        plot_newcomb(argparse.Namespace(results_path=results_path))


if __name__ == "__main__":
    main()
