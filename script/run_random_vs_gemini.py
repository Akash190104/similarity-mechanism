#!/usr/bin/env python3
"""Run Gemini vs Random: objective similarity (all benchmarks) + subjective (all 3 modes × all benchmarks).

Experiment matrix
-----------------
  Objective  : 9 benchmarks × 10 trials = 90 games
               (benchmark answer agreement computed algorithmically)
  Subjective  decision    : 9 benchmarks × 10 trials = 90 games
               (Gemini sees Random's answers, rates similarity)
  Subjective  explanation : 9 benchmarks × 10 trials = 90 games
               (Gemini sees Random's reasoning traces, rates similarity)
  Subjective  both        : 9 benchmarks × 10 trials = 90 games
               (Gemini sees Random's answers + reasoning traces, rates similarity)

Total: 4 modes × 9 benchmarks × 10 trials = 360 games

Only Gemini vs Random runs (pinned player positions in the agent config).
RandomAgent cooperation rates are ignored; only Gemini's rates are extracted.

Usage
-----
  python script/run_random_vs_gemini.py --seed 42
  python script/run_random_vs_gemini.py --seed 42 --mode objective
  python script/run_random_vs_gemini.py --seed 42 --benchmarks newcomb,cabin,ggb
  python script/run_random_vs_gemini.py --seed 42 --gemini-cache outputs/random_vs_gemini/old
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ALL_BENCHMARKS = [
    "newcomb",
    "dilemmas",
    "moral_choice",
    "cabin",
    "ggb",
    "trait",
    "hle",
    "random_coin_toss_alt",
    "random_die_roll_alt",
]

SUBJECTIVE_MODES = ["decision", "explanation", "both"]

SUBJECTIVE_CONFIGS = {
    "decision":    "main/random_vs_gemini_subjective_decision.yaml",
    "explanation": "main/random_vs_gemini_subjective_explanation.yaml",
    "both":        "main/random_vs_gemini_subjective_both.yaml",
}

OBJECTIVE_CONFIG = "main/random_vs_gemini_objective.yaml"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_experiment(config: str, seed: int, output_dir: Path,
                    max_items: int | None = None, benchmark: str | None = None) -> None:
    cmd = [
        sys.executable, str(ROOT / "script" / "run_experiment.py"),
        "--config", config,
        "--seed", str(seed),
        "--output-dir", str(output_dir),
    ]
    if max_items is not None:
        cmd += ["--max-items", str(max_items)]
    if benchmark is not None:
        cmd += ["--benchmark", benchmark]
    print(f"\n{'='*60}\nRunning: {' '.join(cmd)}\n{'='*60}")
    subprocess.run(cmd, check=True)


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _gemini_key(rates: dict) -> str | None:
    for k in rates:
        if "gemini" in k.lower():
            return k
    return None


def _extract_cooperation_rate(out_dir: Path) -> float | None:
    p = out_dir / "action_rates.json"
    if not p.exists():
        return None
    rates = _load_json(p)
    key = _gemini_key(rates)
    if key is None:
        return None
    r = rates[key]
    coop = r.get("COOPERATE", r.get("Cooperate", r.get("A0")))
    # If agent only defected, COOPERATE key is absent — that means 0
    if coop is None and ("DEFECT" in r or "Defect" in r):
        return 0.0
    return coop


def _extract_defection_rate(out_dir: Path) -> float | None:
    p = out_dir / "action_rates.json"
    if not p.exists():
        return None
    rates = _load_json(p)
    key = _gemini_key(rates)
    if key is None:
        return None
    r = rates[key]
    defect = r.get("DEFECT", r.get("Defect", r.get("A1")))
    if defect is None and ("COOPERATE" in r or "Cooperate" in r):
        return 0.0
    return defect


def _extract_objective_similarity(out_dir: Path) -> float | None:
    p = out_dir / "benchmark_similarity.json"
    if not p.exists():
        return None
    data = _load_json(p)
    matrix = data.get("similarity_matrix", {})
    # Skip self-similarity entries (e.g. "A vs A": 100.0)
    for key, val in matrix.items():
        parts = key.split(" vs ")
        if len(parts) == 2 and parts[0] != parts[1]:
            return float(val)
    return None


def _extract_subjective_similarity(out_dir: Path) -> float | None:
    """Return Gemini's subjective score for Random from subjective_similarity.json."""
    p = out_dir / "subjective_similarity.json"
    if not p.exists():
        return None
    data = _load_json(p)
    asym = data.get("asymmetric_matrix", {})
    # Find the entry where Gemini is the judge
    for k, v in asym.items():
        if "gemini" in k.lower():
            return float(v)
    # Fallback: symmetric average
    sym = data.get("symmetric_averages", {})
    return float(next(iter(sym.values()))) if sym else None


def _check_gemini_cache(cache_dir: Path, benchmark: str) -> bool:
    return any(
        (cache_dir / f).exists()
        for f in [f"benchmark_results_{benchmark}.json", "benchmark_results_all.json"]
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gemini vs Random: objective + subjective (decision/explanation/both) across all benchmarks"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-items", type=int, default=None,
                        help="Max benchmark questions per agent (default: full benchmark)")
    parser.add_argument("--benchmarks", type=str, default=",".join(ALL_BENCHMARKS),
                        help="Comma-separated benchmarks to run")
    parser.add_argument(
        "--mode",
        choices=["all", "objective", "subjective", "decision", "explanation", "both"],
        default="all",
        help=(
            "all            = objective + all 3 subjective modes\n"
            "objective      = only benchmark-based similarity\n"
            "subjective     = all 3 subjective modes\n"
            "decision       = subjective decision only\n"
            "explanation    = subjective explanation only\n"
            "both           = subjective both only"
        ),
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--gemini-cache", type=str, default=None,
                        help="Dir with pre-computed Gemini benchmark results (advisory)")
    parser.add_argument("--plot", action="store_true", default=True)
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    args = parser.parse_args()

    root_out = (Path(args.output_dir) if args.output_dir
                else ROOT / "outputs" / "random_vs_gemini")
    root_out.mkdir(parents=True, exist_ok=True)

    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]

    run_objective = args.mode in ("all", "objective")
    if args.mode == "subjective":
        run_subjective_modes = list(SUBJECTIVE_MODES)
    elif args.mode in SUBJECTIVE_MODES:
        run_subjective_modes = [args.mode]
    elif args.mode == "all":
        run_subjective_modes = list(SUBJECTIVE_MODES)
    else:
        run_subjective_modes = []

    # ── Cache notice ─────────────────────────────────────────────────────────
    if args.gemini_cache:
        cache_dir = Path(args.gemini_cache)
        if cache_dir.exists():
            found   = [b for b in benchmarks if _check_gemini_cache(cache_dir, b)]
            missing = [b for b in benchmarks if b not in found]
            print(f"\n[Cache] Gemini results at: {cache_dir}")
            print(f"        Cached : {found or 'none'}")
            print(f"        Missing: {missing or 'none'}")
        else:
            print(f"\n[Cache] --gemini-cache not found: {cache_dir}. Running fresh.")

    total_runs = (len(benchmarks) if run_objective else 0) + \
                 len(run_subjective_modes) * len(benchmarks)
    print(f"\nTotal experiments to run: {total_runs} "
          f"({len(benchmarks)} benchmarks × "
          f"{'objective + ' if run_objective else ''}"
          f"{len(run_subjective_modes)} subjective mode(s))")

    # ── Objective ─────────────────────────────────────────────────────────────
    # Benchmark results are automatically cached at data/benchmark_cache/
    # so the first run computes them, and all subsequent modes reuse them.
    objective_results: dict[str, dict] = {}

    if run_objective:
        print(f"\n{'#'*60}")
        print(f"OBJECTIVE — {len(benchmarks)} benchmarks × 10 trials each")
        print(f"{'#'*60}")
        for bench in benchmarks:
            out_dir = root_out / "objective" / bench
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n── {bench} ──")
            _run_experiment(OBJECTIVE_CONFIG, args.seed, out_dir,
                            max_items=args.max_items, benchmark=bench)
            objective_results[bench] = {
                "similarity":       _extract_objective_similarity(out_dir),
                "cooperation_rate": _extract_cooperation_rate(out_dir),
                "defection_rate":   _extract_defection_rate(out_dir),
                "output_dir":       str(out_dir),
            }
            r = objective_results[bench]
            print(f"  sim={r['similarity']}, coop={r['cooperation_rate']}, defect={r['defection_rate']}")

    # ── Subjective (decision / explanation / both) ────────────────────────────
    # subjective_results[mode][benchmark] = {similarity, cooperation_rate, ...}
    subjective_results: dict[str, dict[str, dict]] = {}

    for mode in run_subjective_modes:
        print(f"\n{'#'*60}")
        print(f"SUBJECTIVE ({mode.upper()}) — {len(benchmarks)} benchmarks × 10 trials each")
        print("Gemini sees Random's answers/traces and self-reports similarity")
        print(f"{'#'*60}")
        subjective_results[mode] = {}
        for bench in benchmarks:
            out_dir = root_out / "subjective" / mode / bench
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n── {bench} ──")
            _run_experiment(SUBJECTIVE_CONFIGS[mode], args.seed, out_dir,
                            max_items=args.max_items, benchmark=bench)
            subjective_results[mode][bench] = {
                "similarity":       _extract_subjective_similarity(out_dir),
                "cooperation_rate": _extract_cooperation_rate(out_dir),
                "defection_rate":   _extract_defection_rate(out_dir),
                "output_dir":       str(out_dir),
            }
            r = subjective_results[mode][bench]
            print(f"  sim={r['similarity']}, coop={r['cooperation_rate']}, defect={r['defection_rate']}")

    # ── Summary table ─────────────────────────────────────────────────────────
    modes_present = (["objective"] if objective_results else []) + \
                    [f"subj_{m}" for m in run_subjective_modes]
    col_w = 13
    header = f"{'Benchmark':<22}" + "".join(f"{m:>{col_w}}" for m in modes_present)
    print(f"\n{'='*80}\nSIMILARITY SCORES\n{'='*80}")
    print(header)
    print("-" * len(header))
    for bench in benchmarks:
        row = f"  {bench:<20}"
        if objective_results:
            s = objective_results.get(bench, {}).get("similarity")
            row += f"{(f'{s:.1f}%' if s is not None else 'N/A'):>{col_w}}"
        for m in run_subjective_modes:
            s = subjective_results.get(m, {}).get(bench, {}).get("similarity")
            row += f"{(f'{s:.1f}%' if s is not None else 'N/A'):>{col_w}}"
        print(row)

    print(f"\n{'='*80}\nGEMINI COOPERATION RATE\n{'='*80}")
    print(header)
    print("-" * len(header))
    for bench in benchmarks:
        row = f"  {bench:<20}"
        if objective_results:
            c = objective_results.get(bench, {}).get("cooperation_rate")
            row += f"{(f'{c:.1%}' if c is not None else 'N/A'):>{col_w}}"
        for m in run_subjective_modes:
            c = subjective_results.get(m, {}).get(bench, {}).get("cooperation_rate")
            row += f"{(f'{c:.1%}' if c is not None else 'N/A'):>{col_w}}"
        print(row)
    print("=" * 80)

    # ── Save comparison JSON (merge with existing if present) ───────────────
    comp_path = root_out / "comparison.json"
    if comp_path.exists():
        existing = _load_json(comp_path)
    else:
        existing = {"objective": {}, "subjective": {}}

    # Merge objective results
    merged_objective = existing.get("objective", {})
    merged_objective.update(objective_results)

    # Merge subjective results (per mode, per benchmark)
    merged_subjective = existing.get("subjective", {})
    for mode, bench_results in subjective_results.items():
        if mode not in merged_subjective:
            merged_subjective[mode] = {}
        merged_subjective[mode].update(bench_results)

    # Merge benchmark list
    all_benchmarks = list(dict.fromkeys(
        existing.get("benchmarks", []) + benchmarks
    ))

    comparison = {
        "seed": args.seed,
        "max_items": args.max_items,
        "benchmarks": all_benchmarks,
        "objective": merged_objective,
        "subjective": merged_subjective,
    }
    comp_path.write_text(json.dumps(comparison, indent=2))
    print(f"\nComparison saved to: {comp_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    if args.plot and (objective_results or subjective_results):
        plot_cmd = [
            sys.executable,
            str(ROOT / "script" / "plot_defection_rate.py"),
            "--comparison", str(comp_path),
        ]
        print(f"\nGenerating plots: {' '.join(plot_cmd)}")
        subprocess.run(plot_cmd, check=False)


if __name__ == "__main__":
    main()
