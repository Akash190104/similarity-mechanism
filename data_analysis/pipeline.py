from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import stdev
from typing import Any, Dict, Iterable, List, Tuple

from .aggregate import write_csv, write_json
from .compute import (
    compute_agent_metrics,
    compute_conditional_cooperation,
    compute_disarmament_metrics,
    compute_mediation_metrics,
    compute_pairwise_metrics,
    compute_repetition_run_metrics,
    compute_reputation_metrics,
    compute_round_trajectory,
)
from .disarmament import generate_disarmament_report
from .io import find_runs
from .models import RunData
from .parsers import parse_run


def _build_baseline_map(runs: List[RunData]) -> Dict[str, float]:
    baseline: Dict[str, float] = {}
    for run in runs:
        if run.mechanism == "NoMechanism":
            for agent, payoff in run.expected_payoffs.items():
                baseline[agent] = payoff
    return baseline


def _group_rows(rows: Iterable[Dict[str, Any]], run_dir: str) -> List[Dict[str, Any]]:
    return [row for row in rows if row.get("run_dir") == run_dir]


def _nanmean(values: List[float]) -> float:
    filtered = [v for v in values if isinstance(v, (int, float)) and not (v != v)]
    return sum(filtered) / len(filtered) if filtered else float("nan")


def _clean_floats(values: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for v in values:
        if isinstance(v, (int, float)) and not (v != v):
            out.append(float(v))
    return out


def _mean_sem(values: List[float]) -> Tuple[float, float]:
    """Sample mean and standard error. SEM is NaN when n < 2."""
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, float("nan")
    return mean, stdev(values) / math.sqrt(len(values))


def _aggregate_mechanism_rows(
    mechanism_rows: List[Dict[str, Any]],
    agent_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Aggregate per-run mechanism rows into per-(game, mechanism) summaries with SEM.

    SEM is computed across runs when there are >=2 runs for a (game, mechanism);
    otherwise it falls back to across-agents within the single run, which captures
    between-agent heterogeneity rather than seed/sampling variance.
    """
    runs_by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in mechanism_rows:
        runs_by_key[(row.get("game") or "", row.get("mechanism") or "")].append(row)

    agents_by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in agent_rows:
        agents_by_key[(row.get("game") or "", row.get("mechanism") or "")].append(row)

    out: List[Dict[str, Any]] = []
    for key, runs in sorted(runs_by_key.items()):
        game, mechanism = key
        n_runs = len(runs)

        coop_runs = _clean_floats(r.get("avg_cooperation_rate") for r in runs)
        payoff_runs = _clean_floats(r.get("avg_expected_payoff") for r in runs)
        delta_runs = _clean_floats(r.get("avg_delta_expected_payoff") for r in runs)
        delegate_runs = _clean_floats(r.get("avg_delegate_rate") for r in runs)

        if n_runs >= 2:
            mean_coop, sem_coop = _mean_sem(coop_runs)
            mean_payoff, sem_payoff = _mean_sem(payoff_runs)
            mean_delta, sem_delta = _mean_sem(delta_runs)
            mean_delegate, sem_delegate = _mean_sem(delegate_runs)
            sem_source = "across_runs"
        else:
            agent_subset = agents_by_key.get(key, [])
            coop_agents = _clean_floats(r.get("coop_rate") for r in agent_subset)
            payoff_agents = _clean_floats(r.get("expected_payoff") for r in agent_subset)
            delta_agents = _clean_floats(r.get("delta_expected_payoff") for r in agent_subset)
            delegate_agents = _clean_floats(r.get("delegate_rate") for r in agent_subset)

            mean_coop, sem_coop = _mean_sem(coop_agents)
            mean_payoff, sem_payoff = _mean_sem(payoff_agents)
            mean_delta, sem_delta = _mean_sem(delta_agents)
            mean_delegate, sem_delegate = _mean_sem(delegate_agents)
            sem_source = "across_agents"

        out.append(
            {
                "game": game,
                "mechanism": mechanism,
                "n_runs": n_runs,
                "mean_cooperation_rate": mean_coop,
                "sem_cooperation_rate": sem_coop,
                "mean_expected_payoff": mean_payoff,
                "sem_expected_payoff": sem_payoff,
                "mean_delta_expected_payoff": mean_delta,
                "sem_delta_expected_payoff": sem_delta,
                "mean_delegate_rate": mean_delegate,
                "sem_delegate_rate": sem_delegate,
                "sem_source": sem_source,
            }
        )
    return out


def _build_mechanism_rows(run: RunData, agent_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    expected = [row["expected_payoff"] for row in agent_rows if not isinstance(row["expected_payoff"], str)]
    delta = [row.get("delta_expected_payoff") for row in agent_rows if row.get("delta_expected_payoff") == row.get("delta_expected_payoff")]
    coop_rates = [row["coop_rate"] for row in agent_rows]
    delegate_rates = [row.get("delegate_rate", 0.0) for row in agent_rows]
    return {
        "run_dir": str(run.run_dir),
        "game": run.game,
        "mechanism": run.mechanism,
        "agent_count": len(agent_rows),
        "avg_expected_payoff": _nanmean(expected),
        "avg_delta_expected_payoff": _nanmean(delta),
        "avg_cooperation_rate": _nanmean(coop_rates),
        "avg_delegate_rate": _nanmean(delegate_rates),
    }


def _run_paths_for_dir(directory: Path) -> "RunPaths | None":
    from .io import RunPaths

    config_path = directory / "config.json"
    payoffs_path = directory / "payoffs.json"
    history_paths = sorted(directory.glob("*.jsonl"))
    if not (config_path.exists() or payoffs_path.exists() or history_paths):
        return None
    return RunPaths(
        root=directory,
        config_path=config_path if config_path.exists() else None,
        payoffs_path=payoffs_path if payoffs_path.exists() else None,
        history_paths=history_paths,
    )


def _discover_runs(root: Path) -> List[RunData]:
    pending = [root]
    parsed: List[RunData] = []
    seen_dirs: set[Path] = set()
    while pending:
        current = pending.pop()
        if current in seen_dirs:
            continue
        seen_dirs.add(current)

        # Treat the directory itself as a candidate run when it has artefacts.
        self_run = _run_paths_for_dir(current)
        if self_run is not None and self_run.config_path is not None:
            parsed.append(parse_run(self_run))
            continue

        run_paths = [r for r in find_runs(current) if r.config_path or r.payoffs_path or r.history_paths]
        if run_paths:
            parsed.extend(parse_run(r) for r in run_paths)
        else:
            for child in current.iterdir():
                if child.is_dir():
                    pending.append(child)
    return parsed


def analyze(root: Path, out_dir: Path, *, disarmament: bool = False) -> Dict[str, Path]:
    runs = _discover_runs(root)
    baseline = _build_baseline_map(runs)

    agent_rows: List[Dict[str, Any]] = []
    pairwise_rows: List[Dict[str, Any]] = []
    trajectory_rows: List[Dict[str, Any]] = []
    conditional_rows: List[Dict[str, Any]] = []
    repetition_rows: List[Dict[str, Any]] = []
    disarmament_rows: List[Dict[str, Any]] = []
    reputation_rows: List[Dict[str, Any]] = []
    mediation_mediator_rows: List[Dict[str, Any]] = []
    mediation_agent_rows: List[Dict[str, Any]] = []
    mechanism_rows: List[Dict[str, Any]] = []

    for run in runs:
        agent_metrics = compute_agent_metrics(run, baseline)
        agent_rows.extend(agent_metrics)
        pairwise_rows.extend(compute_pairwise_metrics(run))
        trajectory_rows.extend(compute_round_trajectory(run))
        conditional_rows.extend(compute_conditional_cooperation(run))

        repetition_metric = compute_repetition_run_metrics(run)
        if repetition_metric:
            repetition_rows.append(repetition_metric)

        disarmament_rows.extend(compute_disarmament_metrics(run))
        reputation_rows.extend(compute_reputation_metrics(run))
        mediator_rows, med_agents = compute_mediation_metrics(run)
        mediation_mediator_rows.extend(mediator_rows)
        mediation_agent_rows.extend(med_agents)

        mechanism_rows.append(_build_mechanism_rows(run, agent_metrics))

    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: Dict[str, Path] = {}
    outputs["agent_metrics"] = _write_csv(agent_rows, out_dir / "agent_metrics.csv")
    outputs["pairwise_metrics"] = _write_csv(pairwise_rows, out_dir / "pairwise_metrics.csv")
    outputs["round_trajectory"] = _write_csv(trajectory_rows, out_dir / "round_trajectory.csv")
    outputs["conditional_cooperation"] = _write_csv(conditional_rows, out_dir / "conditional_cooperation.csv")
    outputs["repetition_metrics"] = _write_csv(repetition_rows, out_dir / "repetition_metrics.csv")
    outputs["disarmament_metrics"] = _write_csv(disarmament_rows, out_dir / "disarmament_metrics.csv")
    outputs["reputation_metrics"] = _write_csv(reputation_rows, out_dir / "reputation_metrics.csv")
    outputs["mediator_metrics"] = _write_csv(mediation_mediator_rows, out_dir / "mediation_mediator_metrics.csv")
    outputs["mediation_agent_metrics"] = _write_csv(mediation_agent_rows, out_dir / "mediation_agent_metrics.csv")
    outputs["mechanism_summary"] = _write_csv(mechanism_rows, out_dir / "mechanism_summary.csv")

    aggregated_rows = _aggregate_mechanism_rows(mechanism_rows, agent_rows)
    outputs["mechanism_summary_aggregated"] = _write_csv(
        aggregated_rows, out_dir / "mechanism_summary_aggregated.csv"
    )

    summary_json = out_dir / "summaries.json"
    summary_payload = [
        {
            "run_dir": str(run.run_dir),
            "game": run.game,
            "mechanism": run.mechanism,
            "agents": run.agents,
            "expected_payoffs": run.expected_payoffs,
        }
        for run in runs
    ]
    write_json(summary_payload, summary_json)
    outputs["summaries"] = summary_json

    # Generate plots unless explicitly disabled. Some environments (e.g. headless
    # CI or sandboxed runners without GUI backends) cannot render Matplotlib
    # figures, so allow skipping via an environment flag.
    if os.environ.get("SKIP_PLOTS") not in {"1", "true", "True"}:
        from .plots import (
            plot_agent_metrics,
            plot_mechanism_summary,
            plot_pairwise_heatmaps,
        )

        plot_mechanism_summary(out_dir / "mechanism_summary.csv", out_dir)
        plot_agent_metrics(out_dir / "agent_metrics.csv", out_dir)
        plot_pairwise_heatmaps(out_dir / "pairwise_metrics.csv", out_dir)
        outputs["figures"] = out_dir / "figures"

    if disarmament:
        disarm_outputs = generate_disarmament_report(
            runs=runs,
            out_dir=out_dir,
            agent_metrics=agent_rows,
        )
        outputs.update(disarm_outputs)

    return outputs


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> Path:
    write_csv(rows, path)
    return path
