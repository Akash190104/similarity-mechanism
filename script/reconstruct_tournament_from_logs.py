#!/usr/bin/env python3
"""Reconstruct similarity-tournament results from raw game logs and create plots.

Parses an existing experiment output directory's `records.jsonl` and
`game_log.txt`, recovers the similarity percentage associated with each
trace_id, regroups matches by similarity level, and re-emits the
tournament-results JSON used by the plotting code.

Usage:
    python script/reconstruct_tournament_from_logs.py outputs/2025/01/15/12:00:00

Outputs:
    Writes into the supplied output directory:
    `similarity_tournament_results_reconstructed.json`, plus the standard
    similarity-tournament plot PNGs from `plot_similarity_tournament`.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from script.plot import plot_similarity_tournament


def extract_similarity_from_log(log_path: Path) -> dict[str, int]:
    """Extract similarity percentage for each trace_id from game log."""

    trace_to_similarity = {}

    with open(log_path, 'r') as f:
        content = f.read()

    # Split content by prompt headers
    prompts = re.split(r'(===== Prompt \[ID: [a-f0-9]+\])', content)

    # Process prompts in pairs: header, content, header, content, ...
    for i in range(1, len(prompts), 2):
        if i + 1 >= len(prompts):
            break

        header = prompts[i]
        prompt_content = prompts[i + 1]

        # Extract trace_id from header
        trace_match = re.search(r'ID: ([a-f0-9]+)', header)
        if not trace_match:
            continue

        trace_id = trace_match.group(1)

        # Look for similarity percentage in the prompt content
        # Patterns: "0% similar", "10% similar", etc.
        sim_match = re.search(r'(\d+)%\s*similar', prompt_content, re.IGNORECASE)

        if sim_match:
            similarity_pct = int(sim_match.group(1))
            trace_to_similarity[trace_id] = similarity_pct

    return trace_to_similarity


def reconstruct_tournament_results(output_dir: Path) -> dict:
    """Reconstruct tournament results from logs."""

    records_path = output_dir / "records.jsonl"
    log_path = output_dir / "game_log.txt"

    print(f"Reading records from: {records_path}")
    print(f"Reading log from: {log_path}")

    # Extract similarity percentages from log
    trace_to_similarity = extract_similarity_from_log(log_path)
    print(f"Found similarity info for {len(trace_to_similarity)} trace IDs")

    # Read all game records
    games = []
    with open(records_path, 'r') as f:
        for line in f:
            game_record = json.loads(line.strip())
            games.append(game_record)

    print(f"Found {len(games)} game records")

    # Organize by similarity percentage
    results_by_pct = defaultdict(lambda: {"matches": []})

    for game_record in games:
        # Each record is a list of 2 players
        if len(game_record) != 2:
            continue

        player1, player2 = game_record

        # Get similarity from trace_id
        trace_id = player1.get("trace_id")
        if not trace_id or trace_id not in trace_to_similarity:
            continue

        similarity_pct = trace_to_similarity[trace_id]

        # Extract agent names (remove #P0, #P1 suffixes)
        agent1_name = re.sub(r'#P\d+$', '', player1["player"])
        agent2_name = re.sub(r'#P\d+$', '', player2["player"])

        match_result = {
            "agent1_name": agent1_name,
            "agent2_name": agent2_name,
            "agent1_action": player1["action"],
            "agent2_action": player2["action"],
            "agent1_payoff": player1["points"],
            "agent2_payoff": player2["points"],
            "similarity_pct": similarity_pct,
            "trial": len(results_by_pct[similarity_pct]["matches"]) + 1,
        }

        results_by_pct[similarity_pct]["matches"].append(match_result)
        results_by_pct[similarity_pct]["similarity_pct"] = similarity_pct

    # Create final structure
    percentages = sorted(results_by_pct.keys())

    tournament_results = {
        "seed": 42,  # Default
        "trials": max(len(results_by_pct[pct]["matches"]) for pct in percentages) if percentages else 0,
        "percentages": percentages,
        "results_by_percentage": {str(pct): results_by_pct[pct] for pct in percentages},
    }

    # Print summary
    print(f"\nReconstructed tournament data:")
    print(f"  Similarity levels: {percentages}")
    for pct in percentages:
        n_matches = len(results_by_pct[pct]["matches"])
        print(f"  {pct}%: {n_matches} matches")

    return tournament_results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Reconstruct tournament from game logs")
    parser.add_argument("output_dir", type=Path, help="Path to output directory with game logs")
    args = parser.parse_args()

    # Reconstruct results
    results = reconstruct_tournament_results(args.output_dir)

    # Save reconstructed results
    results_path = args.output_dir / "similarity_tournament_results_reconstructed.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved reconstructed results to: {results_path}")

    # Generate plots
    print("\nGenerating plots...")
    plot_similarity_tournament(argparse.Namespace(results_path=results_path))

    print("\nDone!")


if __name__ == "__main__":
    main()
