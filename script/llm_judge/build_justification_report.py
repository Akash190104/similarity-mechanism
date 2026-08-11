#!/usr/bin/env python3
"""
Build a bundled justification-judge report directory.

Given a judge directory (``<run-dir>/judge/<output-name>``) that already has
``normalized/normalized.jsonl``, this script writes:
- A markdown report with key tables
- CSV matrices for taxonomy prevalence across mechanisms
- Optional copies of source artifacts into the same output directory
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from scipy.stats import chi2_contingency

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.llm_judge.analysis_utils import sort_mechanisms, sort_models  # noqa: E402

RAW_OUTPUT_SUBDIR = "raw"
RAW_JSON_FILENAME = "judgement.jsonl"
RAW_SUMMARY_FILENAME = "judgement_summary.json"
NORMALIZED_SUBDIR = "normalized"
NORMALIZED_JSON_FILENAME = "normalized.jsonl"
NORMALIZED_SUMMARY_FILENAME = "normalized.summary.json"
NORMALIZED_LABEL_MAP_FILENAME = "normalized.label_map.json"
DATASET_SUBDIR = "dataset"
DATASET_COUNTS_FILENAME = "taxonomy_by_game_mechanism_model_player.counts.csv"
DATASET_SHARE_FILENAME = "taxonomy_by_game_mechanism_model_player.share_pct.csv"
REPORTS_SUBDIR = "reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a bundled report directory from normalized LLM judge "
            "outputs."
        )
    )
    parser.add_argument(
        "judge_dir",
        type=Path,
        help=(
            "Path to the judge directory produced by "
            "run_justification_judge.py, e.g. "
            "outputs/2026/04/17/22:43/judge/gpt4o-mini."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory to write the bundle to. Defaults to "
            "<judge_dir>/reports/."
        ),
    )
    parser.add_argument(
        "--top-k-report-labels",
        type=int,
        default=8,
        help=(
            "Number of top global labels to show in compact markdown matrices "
            "(full CSV matrices always include all labels)."
        ),
    )
    parser.add_argument(
        "--no-copy-inputs",
        action="store_true",
        help="Do not copy source input artifacts into output directory.",
    )
    args = parser.parse_args()
    args.judge_dir = args.judge_dir.expanduser().resolve()
    if args.output_dir is not None:
        args.output_dir = args.output_dir.expanduser().resolve()
    return args


def _first_existing_path(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path.resolve()
    return None


def discover_artifact_paths(judge_dir: Path) -> dict[str, Path | None]:
    """Infer standard artifact locations within a judge directory."""
    normalized_dir = judge_dir / NORMALIZED_SUBDIR
    raw_dir = judge_dir / RAW_OUTPUT_SUBDIR
    dataset_dir = judge_dir / DATASET_SUBDIR

    normalized_candidates = [
        normalized_dir / NORMALIZED_JSON_FILENAME,
        judge_dir / NORMALIZED_JSON_FILENAME,
    ]
    normalized_jsonl = _first_existing_path(normalized_candidates)
    if normalized_jsonl is None:
        searched = "\n".join(f"- {path}" for path in normalized_candidates)
        raise FileNotFoundError(
            f"Could not locate normalized JSONL under {judge_dir}. "
            f"Looked in:\n{searched}"
        )

    normalized_summary_candidates = [
        normalized_dir / NORMALIZED_SUMMARY_FILENAME,
        judge_dir / NORMALIZED_SUMMARY_FILENAME,
    ]
    label_map_candidates = [
        normalized_dir / NORMALIZED_LABEL_MAP_FILENAME,
        judge_dir / NORMALIZED_LABEL_MAP_FILENAME,
    ]
    raw_json_candidates = [
        raw_dir / RAW_JSON_FILENAME,
        judge_dir / RAW_JSON_FILENAME,
    ]
    raw_summary_candidates = [
        raw_dir / RAW_SUMMARY_FILENAME,
        judge_dir / RAW_SUMMARY_FILENAME,
    ]
    dataset_counts_candidates = [
        dataset_dir / DATASET_COUNTS_FILENAME,
        judge_dir / DATASET_COUNTS_FILENAME,
    ]
    dataset_share_candidates = [
        dataset_dir / DATASET_SHARE_FILENAME,
        judge_dir / DATASET_SHARE_FILENAME,
    ]

    return {
        "judge_dir": judge_dir,
        "normalized_jsonl": normalized_jsonl,
        "normalized_summary_json": _first_existing_path(
            normalized_summary_candidates
        ),
        "label_map_json": _first_existing_path(label_map_candidates),
        "raw_jsonl": _first_existing_path(raw_json_candidates),
        "raw_summary_json": _first_existing_path(raw_summary_candidates),
        "dataset_counts_csv": _first_existing_path(dataset_counts_candidates),
        "dataset_share_csv": _first_existing_path(dataset_share_candidates),
    }


def split_mechanism_model_key(group_key: str) -> tuple[str, str]:
    if "|||" in group_key:
        return tuple(group_key.split("|||", 1))  # type: ignore[return-value]
    return group_key, "Unknown"


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def unique_labels(value: Any) -> list[str]:
    if isinstance(value, list):
        labels = [x for x in value if isinstance(x, str) and x.strip()]
    elif isinstance(value, str):
        labels = [x.strip() for x in value.split(",") if x.strip()]
    else:
        labels = []
    out: list[str] = []
    seen: set[str] = set()
    for label in labels:
        if label not in seen:
            seen.add(label)
            out.append(label)
    return out


def pct(count: int, denom: int) -> float:
    return 0.0 if denom == 0 else (100.0 * count / denom)


def format_pct(value: float) -> str:
    return f"{value:.2f}"


def md_escape(text: str) -> str:
    return text.replace("|", "\\|")


def md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    escaped_headers = [md_escape(h) for h in headers]
    out = []
    out.append("| " + " | ".join(escaped_headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(md_escape(cell) for cell in row) + " |")
    return "\n".join(out)


def top_k_from_counter(counter: Counter[str], k: int) -> list[str]:
    return [label for label, _ in counter.most_common(k)]


def compute_mechanism_label_differences(
    by_mechanism: dict[str, Counter[str]],
    mechanism_rows: Counter[str],
    labels: Sequence[str],
    total_rows: int,
) -> tuple[list[dict[str, Any]], str | None]:
    mechanisms = sort_mechanisms(by_mechanism.keys())
    if not mechanisms or total_rows <= 0:
        return ([], "Statistical test section skipped (no mechanism rows).")
    if len(mechanisms) < 2:
        return (
            [],
            "Statistical test section skipped (need ≥2 mechanisms; got "
            f"{len(mechanisms)}).",
        )

    rows: list[dict[str, Any]] = []
    for label in labels:
        table = []
        for mechanism in mechanisms:
            present = by_mechanism[mechanism].get(label, 0)
            absent = mechanism_rows[mechanism] - present
            table.append([present, absent])

        chi2_stat, p_value, _, _ = chi2_contingency(table)
        k = min(len(mechanisms) - 1, 1)
        cramers_v = math.sqrt((chi2_stat / total_rows) / k) if k > 0 else 0.0

        per_mechanism_share = {
            mechanism: (
                by_mechanism[mechanism].get(label, 0)
                / mechanism_rows[mechanism]
                if mechanism_rows[mechanism] > 0
                else 0.0
            )
            for mechanism in mechanisms
        }
        per_mechanism_count = {
            mechanism: by_mechanism[mechanism].get(label, 0)
            for mechanism in mechanisms
        }
        max_mechanism = max(per_mechanism_share, key=per_mechanism_share.get)
        min_mechanism = min(per_mechanism_share, key=per_mechanism_share.get)
        range_pp = 100.0 * (
            per_mechanism_share[max_mechanism]
            - per_mechanism_share[min_mechanism]
        )

        rows.append(
            {
                "label": label,
                "cramers_v": cramers_v,
                "p_value": float(p_value),
                "max_mechanism": max_mechanism,
                "max_share": per_mechanism_share[max_mechanism],
                "min_mechanism": min_mechanism,
                "min_share": per_mechanism_share[min_mechanism],
                "range_pp": range_pp,
                "per_mechanism_share": per_mechanism_share,
                "per_mechanism_count": per_mechanism_count,
            }
        )

    rows_by_p = sorted(rows, key=lambda x: x["p_value"])
    m = len(rows_by_p)
    if m == 0:
        return (rows_by_p, None)
    for i, row in enumerate(rows_by_p, start=1):
        row["p_bh"] = min(1.0, row["p_value"] * m / i)
    for i in range(m - 2, -1, -1):
        rows_by_p[i]["p_bh"] = min(
            rows_by_p[i]["p_bh"], rows_by_p[i + 1]["p_bh"]
        )

    rows_by_effect = sorted(
        rows_by_p, key=lambda x: x["cramers_v"], reverse=True
    )
    return (rows_by_effect, None)


def write_mechanism_differences_csv(
    out_path: Path, rows: Sequence[dict[str, Any]]
) -> None:
    mechanisms: list[str] = []
    if rows:
        mechanisms = sort_mechanisms(
            rows[0].get("per_mechanism_share", {}).keys()
        )

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = [
            "label",
            "cramers_v",
            "p_value",
            "p_bh",
            "max_mechanism",
            "max_share_pct",
            "min_mechanism",
            "min_share_pct",
            "range_pp",
        ]
        header.extend([f"share_pct__{mechanism}" for mechanism in mechanisms])
        header.extend([f"count__{mechanism}" for mechanism in mechanisms])
        writer.writerow(header)
        for row in rows:
            values = [
                row["label"],
                f"{row['cramers_v']:.6f}",
                f"{row['p_value']:.6e}",
                f"{row.get('p_bh', 1.0):.6e}",
                row["max_mechanism"],
                f"{100.0 * row['max_share']:.2f}",
                row["min_mechanism"],
                f"{100.0 * row['min_share']:.2f}",
                f"{row['range_pp']:.2f}",
            ]
            per_mechanism_share = row.get("per_mechanism_share", {})
            per_mechanism_count = row.get("per_mechanism_count", {})
            values.extend(
                [
                    f"{100.0 * per_mechanism_share.get(mechanism, 0.0):.2f}"
                    for mechanism in mechanisms
                ]
            )
            values.extend(
                [
                    str(per_mechanism_count.get(mechanism, 0))
                    for mechanism in mechanisms
                ]
            )
            writer.writerow(values)


def build_report_markdown(
    *,
    normalized_jsonl: Path,
    output_dir: Path,
    total_rows: int,
    mean_confidence: float | None,
    overall_counts: Counter[str],
    mechanism_rows: Counter[str],
    model_rows: Counter[str],
    by_mechanism: dict[str, Counter[str]],
    by_model: dict[str, Counter[str]],
    by_mechanism_model: dict[str, Counter[str]],
    mechanism_model_rows: Counter[str],
    top_report_labels: Sequence[str],
    mechanism_diff_rows: Sequence[dict[str, Any]],
    mechanism_diff_warning: str | None,
) -> str:
    now = datetime.now(timezone.utc).isoformat()
    mechanisms_sorted = sort_mechanisms(by_mechanism.keys())
    models_sorted = sort_models(by_model.keys())
    lines: list[str] = []
    lines.append("# Justification Judge Report")
    lines.append("")
    lines.append(f"- Generated (UTC): `{now}`")
    lines.append(f"- Source normalized JSONL: `{normalized_jsonl}`")
    lines.append(f"- Rows: `{total_rows}`")
    if mean_confidence is not None:
        lines.append(
            f"- Mean classification confidence: `{mean_confidence:.4f}`"
        )
    lines.append("")

    overall_rows = []
    for label, count in overall_counts.most_common():
        overall_rows.append(
            [label, str(count), format_pct(pct(count, total_rows))]
        )
    lines.append("## Overall Label Prevalence")
    lines.append("")
    lines.append(
        md_table(["label", "count", "share_of_rows_pct"], overall_rows)
    )
    lines.append("")

    lines.append("## Canonical Dataset")
    lines.append("")
    lines.append(
        "- `taxonomy_by_game_mechanism_model_player.counts.csv` (raw counts) and "
        "`taxonomy_by_game_mechanism_model_player.share_pct.csv` (percentages) "
        "cover every `(game, mechanism, model, player, label)` combination. "
        "Generate them with `export_taxonomy_dataset.py`."
    )
    lines.append("")

    lines.append("## Taxonomy by Mechanism (Top 5 per Mechanism)")
    lines.append("")
    mech_rows: list[list[str]] = []
    for mechanism in mechanisms_sorted:
        denom = mechanism_rows.get(mechanism, 0)
        top = by_mechanism[mechanism].most_common(5)
        top_fmt = ", ".join(
            f"{label} ({format_pct(pct(count, denom))}%)"
            for label, count in top
        )
        mech_rows.append([mechanism, str(denom), top_fmt])
    lines.append(md_table(["mechanism", "row_count", "top_labels"], mech_rows))
    lines.append("")

    lines.append("## Taxonomy by Model (Top 5 per Model)")
    lines.append("")
    model_rows_md: list[list[str]] = []
    for model in models_sorted:
        denom = model_rows.get(model, 0)
        top = by_model[model].most_common(5)
        top_fmt = ", ".join(
            f"{label} ({format_pct(pct(count, denom))}%)"
            for label, count in top
        )
        model_rows_md.append([model, str(denom), top_fmt])
    lines.append(md_table(["model", "row_count", "top_labels"], model_rows_md))
    lines.append("")

    lines.append("## Taxonomy by Mechanism and Model (Compact Matrix)")
    lines.append("")
    lines.append(
        "Rows are `(mechanism | model)` groups. Columns are top global taxonomy labels."
    )
    lines.append("")
    headers = ["mechanism | model", "row_count", *top_report_labels]
    compact_rows: list[list[str]] = []

    def group_sort_key(k: str) -> tuple[int, int]:
        mech, mod = split_mechanism_model_key(k)
        m_idx = (
            mechanisms_sorted.index(mech) if mech in mechanisms_sorted else 999
        )
        mod_idx = models_sorted.index(mod) if mod in models_sorted else 999
        return (m_idx, mod_idx)

    for group_key in sorted(by_mechanism_model.keys(), key=group_sort_key):
        mechanism, model = split_mechanism_model_key(group_key)
        group_name = f"{mechanism} | {model}"
        denom = mechanism_model_rows.get(group_key, 0)
        row = [group_name, str(denom)]
        counts = by_mechanism_model[group_key]
        for label in top_report_labels:
            row.append(format_pct(pct(counts.get(label, 0), denom)))
        compact_rows.append(row)
    lines.append(md_table(headers, compact_rows))
    lines.append("")

    lines.append("## Mechanism Differences (Statistical Test)")
    lines.append("")
    lines.append(
        "Per-label chi-square tests across mechanisms with Benjamini-Hochberg "
        "correction (`p_bh`) and Cramer's V effect size."
    )
    lines.append("")
    if mechanism_diff_warning is not None:
        lines.append(f"- {mechanism_diff_warning}")
        lines.append("")
    else:
        lines.append(
            "- Test setup: one chi-square test per taxonomy label on binary label "
            f"presence across the {len(by_mechanism)} mechanisms "
            f"(`N={total_rows}` rows; `{len(mechanism_diff_rows)}` labels tested)."
        )
        lines.append(
            "- Multiple-comparison correction: Benjamini-Hochberg FDR across all labels."
        )
        significant = [
            row for row in mechanism_diff_rows if row.get("p_bh", 1.0) < 0.05
        ]
        if significant:
            strongest = significant[0]
            lines.append(
                "- Result summary: "
                f"`{len(significant)}/{len(mechanism_diff_rows)}` labels are "
                "significant at `p_bh < 0.05`; strongest effect is "
                f"`{strongest['label']}` (`V={strongest['cramers_v']:.3f}`, "
                f"range `{strongest['range_pp']:.1f}` percentage points)."
            )
        lines.append(
            "- Full per-label statistics (all labels, raw `p`, corrected `p_bh`, "
            "effect sizes, per-mechanism shares and counts): "
            "`mechanism_label_differences.csv`."
        )
        lines.append("")

        top_sig = significant[:10]
        if top_sig:
            sig_rows: list[list[str]] = []
            for row in top_sig:
                row_values = [
                    row["label"],
                    f"{row['cramers_v']:.3f}",
                    f"{row['max_mechanism']} ({100.0 * row['max_share']:.1f}%)",
                    f"{row['min_mechanism']} ({100.0 * row['min_share']:.1f}%)",
                    f"{row['range_pp']:.1f}",
                    f"{row.get('p_bh', 1.0):.2e}",
                ]
                per_mechanism_share = row.get("per_mechanism_share", {})
                row_values.extend(
                    [
                        f"{100.0 * per_mechanism_share.get(mechanism, 0.0):.1f}"
                        for mechanism in mechanisms_sorted
                    ]
                )
                sig_rows.append(row_values)
            lines.append(
                md_table(
                    [
                        "label",
                        "effect_size_v",
                        "max_mechanism_share",
                        "min_mechanism_share",
                        "range_pp",
                        "p_bh",
                        *[
                            f"{mechanism}_pct"
                            for mechanism in mechanisms_sorted
                        ],
                    ],
                    sig_rows,
                )
            )
            lines.append("")

        nonsig = [
            row["label"]
            for row in mechanism_diff_rows
            if row.get("p_bh", 1.0) >= 0.05
        ]
        if nonsig:
            lines.append(
                "- Non-significant after correction: "
                + ", ".join(sorted(nonsig))
            )
            lines.append("")

    lines.append("## Artifact Files")
    lines.append("")
    artifact_files = [
        "README.md",
        "taxonomy_by_game_mechanism_model_player.counts.csv",
        "taxonomy_by_game_mechanism_model_player.share_pct.csv",
        "metadata.json",
    ]
    if mechanism_diff_rows:
        artifact_files.append("mechanism_label_differences.csv")
    for name in artifact_files:
        lines.append(f"- `{output_dir / name}`")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    judge_dir = args.judge_dir
    inferred_paths = discover_artifact_paths(judge_dir)

    normalized_jsonl = inferred_paths["normalized_jsonl"]
    if normalized_jsonl is None:
        raise FileNotFoundError(
            f"Missing normalized JSONL under {judge_dir}."
        )
    raw_jsonl = inferred_paths.get("raw_jsonl")
    raw_summary_json = inferred_paths.get("raw_summary_json")
    normalized_summary_json = inferred_paths.get("normalized_summary_json")
    label_map_json = inferred_paths.get("label_map_json")
    dataset_counts_csv = inferred_paths.get("dataset_counts_csv")
    dataset_share_csv = inferred_paths.get("dataset_share_csv")

    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    else:
        output_dir = (judge_dir / REPORTS_SUBDIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    confidence_sum = 0.0
    confidence_n = 0

    overall_counts: Counter[str] = Counter()
    mechanism_rows: Counter[str] = Counter()
    model_rows: Counter[str] = Counter()
    mechanism_model_rows: Counter[str] = Counter()

    by_mechanism: dict[str, Counter[str]] = defaultdict(Counter)
    by_model: dict[str, Counter[str]] = defaultdict(Counter)
    by_mechanism_model: dict[str, Counter[str]] = defaultdict(Counter)

    for row in read_jsonl(normalized_jsonl):
        total_rows += 1
        mechanism = str(row.get("mechanism", "Unknown"))
        model = str(row.get("model", "Unknown"))
        mechanism_key = mechanism
        model_key = model
        mm_key = f"{mechanism}|||{model}"

        mechanism_rows[mechanism_key] += 1
        model_rows[model_key] += 1
        mechanism_model_rows[mm_key] += 1

        conf = row.get("classification_confidence")
        if isinstance(conf, (float, int)):
            confidence_sum += float(conf)
            confidence_n += 1

        labels = unique_labels(row.get("classification_labels_normalized"))
        for label in labels:
            overall_counts[label] += 1
            by_mechanism[mechanism_key][label] += 1
            by_model[model_key][label] += 1
            by_mechanism_model[mm_key][label] += 1

    observed_labels = [label for label, _ in overall_counts.most_common()]
    mechanism_diff_rows, mechanism_diff_warning = (
        compute_mechanism_label_differences(
            by_mechanism=by_mechanism,
            mechanism_rows=mechanism_rows,
            labels=observed_labels,
            total_rows=total_rows,
        )
    )
    if mechanism_diff_rows:
        write_mechanism_differences_csv(
            output_dir / "mechanism_label_differences.csv",
            mechanism_diff_rows,
        )

    top_report_labels = top_k_from_counter(
        overall_counts, args.top_k_report_labels
    )
    mean_confidence = confidence_sum / confidence_n if confidence_n else None
    report_md = build_report_markdown(
        normalized_jsonl=normalized_jsonl,
        output_dir=output_dir,
        total_rows=total_rows,
        mean_confidence=mean_confidence,
        overall_counts=overall_counts,
        mechanism_rows=mechanism_rows,
        model_rows=model_rows,
        by_mechanism=by_mechanism,
        by_model=by_model,
        by_mechanism_model=by_mechanism_model,
        mechanism_model_rows=mechanism_model_rows,
        top_report_labels=top_report_labels,
        mechanism_diff_rows=mechanism_diff_rows,
        mechanism_diff_warning=mechanism_diff_warning,
    )

    with open(output_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "judge_dir": str(judge_dir),
        "source": {
            "normalized_jsonl": str(normalized_jsonl),
            "raw_jsonl": str(raw_jsonl) if raw_jsonl else None,
            "raw_summary_json": (
                str(raw_summary_json) if raw_summary_json else None
            ),
            "normalized_summary_json": (
                str(normalized_summary_json)
                if normalized_summary_json
                else None
            ),
            "label_map_json": str(label_map_json) if label_map_json else None,
        },
        "dataset": {
            "counts_csv": (
                str(dataset_counts_csv) if dataset_counts_csv else None
            ),
            "share_csv": str(dataset_share_csv) if dataset_share_csv else None,
        },
        "stats": {
            "rows": total_rows,
            "mean_confidence": mean_confidence,
            "unique_labels": len(overall_counts),
        },
        "top_report_labels": top_report_labels,
        "mechanism_difference_stats": {
            "rows": len(mechanism_diff_rows),
            "warning": mechanism_diff_warning,
            "csv": (
                str(output_dir / "mechanism_label_differences.csv")
                if mechanism_diff_rows
                else None
            ),
        },
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        f.write("\n")

    if not args.no_copy_inputs:
        for src in [
            raw_jsonl,
            raw_summary_json,
            normalized_jsonl,
            normalized_summary_json,
            label_map_json,
            dataset_counts_csv,
            dataset_share_csv,
        ]:
            if src is None or not src.exists():
                continue
            dst = output_dir / src.name
            if src.resolve() == dst.resolve():
                continue
            shutil.copy2(src, dst)

    print(f"Report bundle written to: {output_dir}")
    print(f"Report markdown: {output_dir / 'README.md'}")


if __name__ == "__main__":
    main()
