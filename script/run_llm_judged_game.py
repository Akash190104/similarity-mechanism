#!/usr/bin/env python3
"""Run LLM-judged similarity pipeline: benchmark → judge → game.

Each model judges how similar two models are based on their benchmark
responses, then fresh instances play a game with that judged similarity.

Usage:
    # Run full pipeline (benchmark + judge + game):
    python script/run_llm_judged_game.py \
        --agents configs/agents/two_models.yaml \
        --game-config games/prisoners_dilemma \
        --trials 10 --max-workers 16

    # Skip benchmark, use existing results:
    python script/run_llm_judged_game.py \
        --agents configs/agents/two_models.yaml \
        --game-config games/prisoners_dilemma \
        --benchmark-results outputs/2026/02/26/13:18/benchmark_results.json \
        --trials 10 --max-workers 16
"""

import argparse
import copy
import json
import random
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from benchmarks.base import BenchmarkResult
from benchmarks.llm_judge import LLMJudge, QAPair
from benchmarks.newcomb import DATA_PATH, Newcomb
from src.config_loader import ConfigLoader
from src.logger_manager import LOGGER
from src.mechanisms.prompts import (
    SIMILARITY_MECHANISM_PROMPT_SINGLE,
    SIMILARITY_PERCENTAGE_UPDATED_FRAMING,
)
from src.registry.agent_registry import create_agent
from src.registry.game_registry import GAME_REGISTRY
from src.utils.concurrency import run_tasks


def set_seed(seed: int = 42) -> None:
    if torch is not None:
        torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_agents_config(path: str) -> list[dict]:
    config_path = Path("configs") / path
    if not config_path.exists():
        config_path = Path(path)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_existing_benchmark(path: Path) -> tuple[dict[str, BenchmarkResult], dict[tuple[str, str], float], list[str]]:
    """Load benchmark results from a previous run's JSON file.

    Returns (benchmark_results, mechanical_similarity, agent_names).
    """
    with open(path, "r") as f:
        data = json.load(f)

    # Load question text lookup from the dataset (old results may lack question_text)
    with open(DATA_PATH, "r") as f:
        questions_by_qid = {q["qid"]: q for q in json.load(f)}

    agent_names = data["agents"]
    benchmark_results: dict[str, BenchmarkResult] = {}

    for name in agent_names:
        r = data["results"][name]
        raw_responses = r["raw_responses"]

        # Backfill question_text, chosen_answer_text, answer_options if missing (old format)
        for entry in raw_responses:
            q = questions_by_qid.get(entry["qid"], {})
            if "question_text" not in entry:
                entry["question_text"] = q.get("question_text", "")
            if "chosen_answer_text" not in entry:
                # Can't reconstruct exact shuffled answer, use letter as fallback
                entry["chosen_answer_text"] = entry.get("chosen_letter", "?")
            if "answer_options" not in entry:
                # Backfill from dataset (original order, not shuffled)
                answers = q.get("permissible_answers", [])
                if answers:
                    entry["answer_options"] = [
                        f"{chr(65 + i)}) {ans}" for i, ans in enumerate(answers)
                    ]

        benchmark_results[name] = BenchmarkResult(
            agent_name=r["agent_name"],
            benchmark_name=r["benchmark_name"],
            scores=r["scores"],
            raw_responses=raw_responses,
            timestamp=r.get("timestamp", ""),
        )

    # Reconstruct mechanical similarity from the matrix or recompute
    mechanical_sim: dict[tuple[str, str], float] = {}
    if "similarity_matrix" in data:
        for pair_str, sim in data["similarity_matrix"].items():
            parts = pair_str.split(" vs ")
            if len(parts) == 2:
                mechanical_sim[(parts[0], parts[1])] = sim

    return benchmark_results, mechanical_sim, agent_names


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-judged similarity: benchmark → judge → game"
    )
    parser.add_argument("--agents", type=str, required=True, help="Agents config YAML")
    parser.add_argument("--game-config", type=str, required=True, help="Game config path (e.g. games/prisoners_dilemma)")
    parser.add_argument("--benchmark-results", type=str, default=None, help="Path to existing benchmark_results.json (skips Phase 1)")
    parser.add_argument("--judge-mode", type=str, default="answers_only", choices=["answers_only", "reasoning_trace"], help="What the judge sees")
    parser.add_argument("--trials", type=int, default=10, help="Game trials per pair")
    parser.add_argument("--max-benchmark-items", type=int, default=None, help="Limit benchmark questions")
    parser.add_argument("--max-judge-questions", type=int, default=None, help="Limit questions shown to judge")
    parser.add_argument("--chunk-size", type=int, default=None, help="Split judge dossier into chunks of this size (avoids context window limits)")
    parser.add_argument("--prompt-mode", type=str, default="percentage_updated", help="Similarity framing style for the game")
    parser.add_argument("--max-workers", type=int, default=None, help="Max concurrent LLM calls")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")

    args = parser.parse_args()
    set_seed(args.seed)

    if args.output_dir:
        LOGGER.set_log_dir(Path(args.output_dir))

    # ── Load agent configs ──
    agent_cfgs = load_agents_config(args.agents)

    # ── Load game config ──
    loader = ConfigLoader()
    game_config = loader.load_component(args.game_config, "game")
    game_class = GAME_REGISTRY[game_config["type"]]
    game_kwargs = game_config.get("kwargs", {})

    # ══════════════════════════════════════════════════════════════════════
    # Phase 1: BENCHMARK (or load existing)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("PHASE 1: BENCHMARK")
    print(f"{'=' * 60}")

    if args.benchmark_results:
        print(f"  Loading existing results from {args.benchmark_results}")
        benchmark_results, mechanical_sim, agent_names = _load_existing_benchmark(
            Path(args.benchmark_results)
        )
        # Create stub agents for name/config pairing (player_id=1)
        bench_agents = []
        for cfg in agent_cfgs:
            bench_agents.append(create_agent({**copy.deepcopy(cfg), "player_id": 1}))

        # Verify agent names match
        loaded_names = set(agent_names)
        config_names = {a.name for a in bench_agents}
        if not config_names.issubset(loaded_names):
            print(f"  WARNING: Agent config names {config_names} don't match loaded {loaded_names}")
            print(f"  Using loaded agent names.")

        for name, r in benchmark_results.items():
            scores = {k: v for k, v in r.scores.items() if k != "answer_vector"}
            print(f"  {name}: {scores}")
        for (a, b), sim in mechanical_sim.items():
            print(f"  Mechanical similarity {a} vs {b}: {sim:.1f}%")
    else:
        store_traces = args.judge_mode == "reasoning_trace"
        bench = Newcomb(
            max_items=args.max_benchmark_items,
            store_traces=store_traces,
            max_workers=args.max_workers,
        )
        print(f"  {len(bench.questions)} questions, store_traces={store_traces}")

        bench_agents = []
        for cfg in agent_cfgs:
            bench_agents.append(create_agent({**copy.deepcopy(cfg), "player_id": 1}))

        benchmark_results = {}
        for agent in bench_agents:
            print(f"\n  Running benchmark on {agent.name}...")
            result = bench.run(agent)
            benchmark_results[agent.name] = result
            scores = {k: v for k, v in result.scores.items() if k != "answer_vector"}
            print(f"  Scores: {scores}")

        mechanical_sim = {}
        for a, b in combinations(bench_agents, 2):
            sim = bench.compute_similarity(
                benchmark_results[a.name], benchmark_results[b.name]
            )
            mechanical_sim[(a.name, b.name)] = sim
            print(f"\n  Mechanical similarity {a.name} vs {b.name}: {sim:.1f}%")

    # ══════════════════════════════════════════════════════════════════════
    # Phase 2: JUDGE
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("PHASE 2: LLM JUDGING")
    print(f"{'=' * 60}")
    if args.chunk_size:
        print(f"  Chunked mode: {args.chunk_size} questions per chunk")

    judge = LLMJudge(
        mode=args.judge_mode,
        max_questions=args.max_judge_questions,
        chunk_size=args.chunk_size,
    )

    judgments = judge.judge_all_pairs(
        agents=bench_agents,
        agent_configs=agent_cfgs,
        benchmark_results=benchmark_results,
        max_workers=args.max_workers,
    )

    for (a_name, b_name), pair_j in judgments.items():
        print(f"\n  Pair: {a_name} vs {b_name}")
        j_a = pair_j.judgment_by_a
        j_b = pair_j.judgment_by_b
        print(f"    Judge {a_name}: similarity = {j_a.similarity_score:.0f}%", end="")
        if j_a.chunk_scores:
            print(f"  (chunks: {[f'{s:.0f}' for s in j_a.chunk_scores]})", end="")
        print()
        print(f"    Judge {b_name}: similarity = {j_b.similarity_score:.0f}%", end="")
        if j_b.chunk_scores:
            print(f"  (chunks: {[f'{s:.0f}' for s in j_b.chunk_scores]})", end="")
        print()
        print(f"    Mechanical:    similarity = {mechanical_sim[(a_name, b_name)]:.1f}%")

    # ══════════════════════════════════════════════════════════════════════
    # Phase 3: PLAY
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print(f"PHASE 3: GAME ({game_config['type']}, {args.trials} trials per pair)")
    print(f"{'=' * 60}")

    game_results: list[dict] = []

    for (a_cfg, b_cfg), (a_bench, b_bench) in zip(
        combinations(agent_cfgs, 2), combinations(bench_agents, 2)
    ):
        pair_key = (a_bench.name, b_bench.name)
        pair_j = judgments[pair_key]
        sim_a = pair_j.judgment_by_a.similarity_score
        sim_b = pair_j.judgment_by_b.similarity_score

        print(f"\n  {a_bench.name} (sim={sim_a:.0f}%) vs {b_bench.name} (sim={sim_b:.0f}%)")

        # Build per-player similarity prompts
        framing_a = SIMILARITY_PERCENTAGE_UPDATED_FRAMING.format(
            similarity_pct=round(sim_a)
        )
        framing_b = SIMILARITY_PERCENTAGE_UPDATED_FRAMING.format(
            similarity_pct=round(sim_b)
        )
        prompt_a = SIMILARITY_MECHANISM_PROMPT_SINGLE.format(
            similarity_framing=framing_a
        )
        prompt_b = SIMILARITY_MECHANISM_PROMPT_SINGLE.format(
            similarity_framing=framing_b
        )

        def _play_trial(trial_idx: int) -> dict:
            game = game_class(**game_kwargs)
            player_a = create_agent({**copy.deepcopy(a_cfg), "player_id": 0})
            player_b = create_agent({**copy.deepcopy(b_cfg), "player_id": 1})

            moves = game.play(
                additional_info=[prompt_a, prompt_b],
                players=[player_a, player_b],
            )

            return {
                "agent_a": a_bench.name,
                "agent_b": b_bench.name,
                "sim_a_judged": sim_a,
                "sim_b_judged": sim_b,
                "sim_mechanical": mechanical_sim[pair_key],
                "trial": trial_idx + 1,
                "agent_a_action": moves[0].action.to_token(),
                "agent_b_action": moves[1].action.to_token(),
                "agent_a_payoff": moves[0].points,
                "agent_b_payoff": moves[1].points,
            }

        trial_results = run_tasks(
            list(range(args.trials)),
            _play_trial,
            max_workers=args.max_workers,
            desc=f"  trials [{a_bench.name.split('/')[-1][:15]} vs {b_bench.name.split('/')[-1][:15]}]",
        )
        game_results.extend(trial_results)

        # Print summary for this pair
        a_payoffs = [r["agent_a_payoff"] for r in trial_results]
        b_payoffs = [r["agent_b_payoff"] for r in trial_results]
        a_actions = [r["agent_a_action"] for r in trial_results]
        b_actions = [r["agent_b_action"] for r in trial_results]
        print(f"    {a_bench.name}: avg payoff={np.mean(a_payoffs):.2f}, actions={dict(zip(*np.unique(a_actions, return_counts=True)))}")
        print(f"    {b_bench.name}: avg payoff={np.mean(b_payoffs):.2f}, actions={dict(zip(*np.unique(b_actions, return_counts=True)))}")

    # ══════════════════════════════════════════════════════════════════════
    # Save results
    # ══════════════════════════════════════════════════════════════════════
    output = {
        "config": {
            "agents": args.agents,
            "game": game_config["type"],
            "judge_mode": args.judge_mode,
            "trials": args.trials,
            "max_benchmark_items": args.max_benchmark_items,
            "max_judge_questions": args.max_judge_questions,
            "chunk_size": args.chunk_size,
            "prompt_mode": args.prompt_mode,
            "seed": args.seed,
        },
        "benchmark_scores": {
            name: {k: v for k, v in r.scores.items() if k != "answer_vector"}
            for name, r in benchmark_results.items()
        },
        "mechanical_similarity": {
            f"{a} vs {b}": sim for (a, b), sim in mechanical_sim.items()
        },
        "judgments": {
            f"{a} vs {b}": {
                "judge_a_score": j.judgment_by_a.similarity_score,
                "judge_b_score": j.judgment_by_b.similarity_score,
                "judge_a_trace_id": j.judgment_by_a.trace_id,
                "judge_b_trace_id": j.judgment_by_b.trace_id,
                **({"judge_a_chunk_scores": j.judgment_by_a.chunk_scores} if j.judgment_by_a.chunk_scores else {}),
                **({"judge_b_chunk_scores": j.judgment_by_b.chunk_scores} if j.judgment_by_b.chunk_scores else {}),
            }
            for (a, b), j in judgments.items()
        },
        "game_results": game_results,
    }

    LOGGER.log_record(output, "llm_judged_game_results.json")
    results_path = LOGGER.log_dir / "llm_judged_game_results.json"
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
