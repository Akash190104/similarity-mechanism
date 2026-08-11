#!/usr/bin/env python3
"""
Run post-hoc LLM judging on similarity-repo reasoning traces.

This adapter links run artifacts (``records.jsonl``, ``game_log.txt``,
``config.json``) in ``outputs/<date>/<time>/`` to the bundled
:mod:`src.llm_judge` classifier and writes one JSONL row per judged
decision into ``<run-dir>/judge/<output-name>/raw/``.

Records shape in this repo: each line of ``records.jsonl`` is a JSON array of
move dicts (one array per round), each element has ``player``, ``action``,
``points``, ``trace_id``, and optional ``mix_probs``. The ``game`` and
``mechanism`` fields are read once from the run's sibling ``config.json``.

Example:
    python script/llm_judge/run_justification_judge.py \
        outputs/2026/04/17/22:43 \
        --provider OpenRouter \
        --model-name openai/gpt-4o-mini \
        --output-name gpt4o-mini \
        --max-workers 20
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
import textwrap
import threading
from collections import Counter
from concurrent.futures import (
    ALL_COMPLETED,
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    wait,
)
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.agents.client_api_llm import ClientAPILLM  # noqa: E402
from src.llm_judge import judge as llm_judge_module  # noqa: E402
from src.llm_judge import taxonomy as llm_taxonomy_module  # noqa: E402
from src.llm_judge.analysis_utils import (  # noqa: E402
    detect_agent_type,
    extract_model_name,
    extract_player_id,
    normalize_filter,
)
from src.llm_judge.config import COOPERATION_TAXONOMY  # noqa: E402
from src.llm_judge.plotting_utils import judge_dir_for_run  # noqa: E402

SKIP_SCAN_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "configs",
    "node_modules",
    "slurm",
    "venv",
    "judge",
}

TRACE_BLOCK_RE = re.compile(
    r"===== Prompt \[ID: (?P<trace>[^\]]+)\] \[[^\]]+\] =====\n"
    r"agent: (?P<prompt_agent>[^\n]+)\n"
    r"prompt:\n(?P<prompt>.*?)\n"
    r"===== Response \[ID: (?P=trace)\] \[[^\]]+\] =====\n"
    r"agent: (?P<response_agent>[^\n]+)\n"
    r"response:\n(?P<response>.*?)(?=\n===== Prompt \[ID: |\Z)",
    re.S,
)

BUILTIN_TAXONOMY_LABEL = "builtin:cooperation_taxonomy"
RAW_OUTPUT_SUBDIR = "raw"
RAW_JSON_FILENAME = "judgement.jsonl"
RAW_SUMMARY_FILENAME = "judgement_summary.json"
SCHEMA_VERSION = "per_category_v1"

TAXONOMY_KEYS: list[str] = list(COOPERATION_TAXONOMY["categories"].keys())


def utc_now_iso() -> str:
    """Return UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def is_experiment_dir(path: Path) -> bool:
    """Check whether a directory looks like a similarity-repo run folder."""
    return (
        (path / "config.json").exists()
        and (path / "records.jsonl").exists()
        and (path / "game_log.txt").exists()
    )


def discover_experiment_dirs(input_paths: list[Path]) -> list[Path]:
    """Discover run folders from one or more roots.

    Supports both direct run paths and parent dirs containing nested runs
    (e.g. ``outputs/2026/04/17``).
    """
    discovered: dict[str, Path] = {}

    for raw_path in input_paths:
        path = raw_path.resolve()
        if not path.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {path}")

        if is_experiment_dir(path):
            discovered[str(path)] = path
            continue

        for records_path in path.rglob("records.jsonl"):
            run_dir = records_path.parent
            if any(part in SKIP_SCAN_DIRS for part in run_dir.parts):
                continue
            if is_experiment_dir(run_dir):
                resolved = run_dir.resolve()
                discovered[str(resolved)] = resolved

    return sorted(discovered.values())


def split_labels(justification: str) -> list[str]:
    """Parse comma-separated justification labels (legacy schema fallback)."""
    if not isinstance(justification, str):
        return ["Other"]
    labels = [piece.strip() for piece in justification.split(",")]
    labels = [label for label in labels if label]
    return labels or ["Other"]


def derive_labels_from_assignments(
    judged: dict[str, Any],
    taxonomy_keys: list[str] = TAXONOMY_KEYS,
) -> list[str]:
    """Derive ordered list of true categories from a per-category judge result.

    Returns ``["Failed classification"]`` when the assignments dict is missing,
    empty, or malformed (non-bool values). Returns ``["Others"]`` when every
    category is False (the rubric's catch-all). Otherwise returns the taxonomy-
    ordered list of category names whose value is True.
    """
    assignments = judged.get("category_assignments")
    if not isinstance(assignments, dict) or not assignments:
        return ["Failed classification"]

    labels: list[str] = []
    for key in taxonomy_keys:
        value = assignments.get(key)
        if value is True:
            labels.append(key)
        elif value is False:
            continue
        else:
            return ["Failed classification"]

    return labels or ["Others"]


def build_output_record(
    *,
    event: dict[str, Any],
    taxonomy_identifier: str,
    save_response_text: bool,
    dry_run: bool,
    judged: dict[str, Any] | None = None,
    judge_provider: str | None = None,
    judge_model: str | None = None,
    judge_input_chars: int | None = None,
) -> dict[str, Any]:
    """Build one output row from extracted event (+ optional judge result)."""
    record = dict(event)
    if not save_response_text:
        record.pop("response_text", None)

    record["processed_at_utc"] = utc_now_iso()
    record["taxonomy_path"] = taxonomy_identifier
    record["dry_run"] = dry_run

    if dry_run:
        record["classification_explanation"] = None
        record["classification_confidence"] = None
        record["classification_justification"] = None
        record["classification_labels"] = []
        record["classification_category_assignments"] = None
        record["classification_schema_version"] = SCHEMA_VERSION
        return record

    if judged is None:
        judged = {
            "Reasoning_behind_classification": "Error in analysis: empty result",
            "Confidence": 0.0,
            "category_assignments": {},
            "justification_type": "Failed classification",
        }

    labels = derive_labels_from_assignments(judged)
    justification = ", ".join(labels)
    assignments = judged.get("category_assignments")
    if not isinstance(assignments, dict):
        assignments = {}

    confidence_raw = judged.get("Confidence", 0.0)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.0

    record["judge_provider"] = judge_provider
    record["judge_model"] = judge_model
    if judge_input_chars is not None:
        record["judge_input_chars"] = judge_input_chars
    record["classification_explanation"] = str(
        judged.get("Reasoning_behind_classification", "")
    )
    record["classification_confidence"] = confidence
    record["classification_justification"] = justification
    record["classification_labels"] = labels
    record["classification_category_assignments"] = assignments
    record["classification_schema_version"] = SCHEMA_VERSION
    return record


def parse_game_log(path: Path) -> dict[str, dict[str, str]]:
    """Parse game log into ``trace_id -> {response_agent, response}`` mapping."""
    content = path.read_text(encoding="utf-8", errors="replace")
    trace_map: dict[str, dict[str, str]] = {}

    for match in TRACE_BLOCK_RE.finditer(content):
        trace_id = match.group("trace")
        trace_map[trace_id] = {
            "response_agent": match.group("response_agent").strip(),
            "response": match.group("response").strip(),
        }

    return trace_map


def iter_action_nodes_from_records(records_path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(line_number, move_dict)`` pairs from ``records.jsonl``.

    Each line is a JSON value — typically a list of move dicts (one entry per
    round), but a single dict or nested list is tolerated. Move dicts have
    ``player``, ``action``, and ``trace_id`` keys.
    """

    def _walk(obj: Any) -> Iterator[dict[str, Any]]:
        if isinstance(obj, dict):
            if (
                "player" in obj
                and "action" in obj
                and "trace_id" in obj
                and isinstance(obj["trace_id"], str)
            ):
                yield dict(obj)
                return
            for value in obj.values():
                yield from _walk(value)
            return
        if isinstance(obj, list):
            for item in obj:
                yield from _walk(item)

    with open(records_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            for node in _walk(payload):
                yield line_number, node


def load_completed_trace_ids(output_path: Path) -> set[str]:
    """Load already-judged trace IDs from existing output JSONL."""
    completed: set[str] = set()
    if not output_path.exists():
        return completed

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            trace_id = row.get("trace_id")
            if isinstance(trace_id, str):
                completed.add(trace_id)
    return completed


def build_judge_input(event: dict[str, Any]) -> str:
    """Build text payload sent to the external LLM judge."""
    template = textwrap.dedent("""\
        Game: ${game}
        Model response to classify:
        ${response_text}""")
    return string.Template(template).substitute(
        game=event["game"],
        response_text=event["response_text"],
    )


def should_keep_event(
    event: dict[str, Any],
    args: argparse.Namespace,
    mechanism_filter: set[str],
    game_filter: set[str],
) -> tuple[bool, str | None]:
    """Apply all event-level filters and return ``(keep, reason_if_dropped)``."""
    if mechanism_filter and event["mechanism"].lower() not in mechanism_filter:
        return False, "filter_mechanism"
    if game_filter and event["game"].lower() not in game_filter:
        return False, "filter_game"

    agent_type = event["agent_type"]
    if args.agent_type == "cot" and agent_type != "CoT":
        return False, "filter_agent_type"
    if args.agent_type == "io" and agent_type != "IO":
        return False, "filter_agent_type"

    response_text = event.get("response_text", "")
    if not response_text:
        return False, "missing_response"
    if len(response_text.strip()) < args.min_response_chars:
        return False, "filter_response_length"

    return True, None


def summarize_output(output_path: Path) -> dict[str, Any]:
    """Build high-level summary statistics from output JSONL."""
    summary: dict[str, Any] = {
        "generated_at_utc": utc_now_iso(),
        "output_file": str(output_path.resolve()),
    }
    if not output_path.exists():
        summary["rows"] = 0
        summary["unique_trace_ids"] = 0
        summary["mean_confidence"] = None
        summary["label_counts"] = {}
        summary["mechanism_counts"] = {}
        summary["game_counts"] = {}
        summary["model_counts"] = {}
        return summary

    label_counts: Counter[str] = Counter()
    mechanism_counts: Counter[str] = Counter()
    game_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    unique_trace_ids: set[str] = set()
    confidence_sum = 0.0
    confidence_n = 0
    rows = 0

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows += 1

            trace_id = row.get("trace_id")
            if isinstance(trace_id, str):
                unique_trace_ids.add(trace_id)

            mechanism = row.get("mechanism")
            if isinstance(mechanism, str):
                mechanism_counts[mechanism] += 1

            game = row.get("game")
            if isinstance(game, str):
                game_counts[game] += 1

            model = row.get("model")
            if isinstance(model, str):
                model_counts[model] += 1

            confidence = row.get("classification_confidence")
            if isinstance(confidence, (int, float)):
                confidence_sum += float(confidence)
                confidence_n += 1

            labels = row.get("classification_labels")
            if isinstance(labels, list):
                for label in labels:
                    if isinstance(label, str) and label.strip():
                        label_counts[label.strip()] += 1
            else:
                raw = row.get("classification_justification", "Other")
                if isinstance(raw, str):
                    for label in split_labels(raw):
                        label_counts[label] += 1

    summary["rows"] = rows
    summary["unique_trace_ids"] = len(unique_trace_ids)
    summary["mean_confidence"] = (
        confidence_sum / confidence_n if confidence_n else None
    )
    summary["label_counts"] = dict(label_counts.most_common())
    summary["mechanism_counts"] = dict(mechanism_counts.most_common())
    summary["game_counts"] = dict(game_counts.most_common())
    summary["model_counts"] = dict(model_counts.most_common())
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Apply llm_judge classifications to similarity-repo decision "
            "justifications."
        )
    )
    parser.add_argument(
        "input_paths",
        type=Path,
        nargs="+",
        help=(
            "Run directory or parent directory/directories containing "
            "similarity-repo runs (e.g. outputs/2026/04/17/22:43, or "
            "outputs/2026/04/17 to batch-scan)."
        ),
    )
    parser.add_argument(
        "--provider",
        choices=["OpenAI", "OpenRouter", "Gemini"],
        default="OpenRouter",
        help="Judge API provider.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Provider model/deployment (e.g., openai/gpt-4o-mini).",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        required=True,
        help=(
            "Slug used to name the judge subdirectory under each run. "
            "Raw files are written to <run-dir>/judge/<output-name>/raw/."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Judge model temperature.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=900,
        help="Max completion tokens for judge responses.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file instead of resumable append.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call judge API; only extract filtered candidate rows.",
    )
    parser.add_argument(
        "--agent-type",
        choices=["cot", "io", "all"],
        default="cot",
        help="Filter by agent type. Default is cot for justification quality.",
    )
    parser.add_argument(
        "--min-response-chars",
        type=int,
        default=80,
        help="Minimum response length after trimming.",
    )
    parser.add_argument(
        "--mechanisms",
        nargs="+",
        default=None,
        help="Optional mechanism allowlist (case-insensitive).",
    )
    parser.add_argument(
        "--games",
        nargs="+",
        default=None,
        help="Optional game allowlist (case-insensitive).",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional cap on number of newly written rows.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=20,
        help="Flush output every N written rows.",
    )
    parser.add_argument(
        "--save-response-text",
        action="store_true",
        help="Persist full response text into output rows.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=20,
        help=(
            "Number of parallel judge workers for API calls (default: 20). "
            "Raise to ~50 for faster throughput on OpenRouter if you don't "
            "see judge_errors spike in the summary."
        ),
    )
    args = parser.parse_args()
    args.input_paths = [
        path.expanduser().resolve() for path in args.input_paths
    ]
    return args


def main() -> None:
    """Coordinate filtering, judging, and output writing for similarity runs."""
    args = parse_args()
    if args.max_workers < 1:
        raise ValueError("--max-workers must be >= 1")

    run_dirs = discover_experiment_dirs(args.input_paths)
    if not run_dirs:
        raise RuntimeError(
            "No run folders found. Expected directories containing "
            "config.json, records.jsonl, and game_log.txt."
        )

    print(f"Discovered {len(run_dirs)} run folder(s).")

    mechanism_filter = normalize_filter(args.mechanisms)
    game_filter = normalize_filter(args.games)

    taxonomy_label = BUILTIN_TAXONOMY_LABEL
    taxonomy_spec = {"data": COOPERATION_TAXONOMY}

    print(f"Taxonomy: {taxonomy_label}")

    # Build a shared judge probe + factory so workers each get their own client.
    judge = None
    judge_factory = None
    thread_local = threading.local()
    judge_provider: str | None = None
    judge_model: str | None = None
    if not args.dry_run:
        taxonomy_mod = llm_taxonomy_module
        judge_mod = llm_judge_module

        def _make_judge() -> tuple[Any, ClientAPILLM]:
            api_client = ClientAPILLM(
                provider=args.provider,
                model_name=args.model_name,
            )
            if "path" in taxonomy_spec:
                taxonomy = taxonomy_mod.Taxonomy.from_json_file(
                    taxonomy_spec["path"]
                )
            else:
                taxonomy = taxonomy_mod.Taxonomy.from_dict(
                    deepcopy(taxonomy_spec["data"])
                )
            local_judge = judge_mod.LLMJudge(
                api_client=api_client,
                taxonomy=taxonomy,
                temperature=args.temperature,
            )
            return local_judge, api_client

        probe_judge, probe_client = _make_judge()
        if args.max_workers == 1:
            judge = probe_judge
        judge_factory = _make_judge
        judge_provider = args.provider.lower()
        judge_model = getattr(probe_client, "model_name", None)

    # Process each run independently — judge output is written per-run.
    for run_dir in run_dirs:
        config_path = run_dir / "config.json"
        records_path = run_dir / "records.jsonl"
        game_log_path = run_dir / "game_log.txt"

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        mechanism = config.get("mechanism", {}).get("type", "Unknown")
        game = config.get("game", {}).get("type", "Unknown")

        judge_dir = judge_dir_for_run(run_dir, args.output_name)
        raw_dir = judge_dir / RAW_OUTPUT_SUBDIR
        raw_dir.mkdir(parents=True, exist_ok=True)
        output_path = raw_dir / RAW_JSON_FILENAME
        summary_path = raw_dir / RAW_SUMMARY_FILENAME
        print(
            f"\n[{run_dir.name}] {game} × {mechanism} — writing to {output_path}"
        )

        completed_trace_ids: set[str] = set()
        if output_path.exists() and not args.overwrite:
            completed_trace_ids = load_completed_trace_ids(output_path)
            print(
                f"  Resume: {len(completed_trace_ids)} trace IDs already "
                f"judged."
            )

        file_mode = "w" if args.overwrite else "a"

        trace_map = parse_game_log(game_log_path)
        counters: Counter[str] = Counter()
        seen_trace_ids: set[str] = set(completed_trace_ids)
        wrote_rows = 0

        def classify_event_with_judge(
            event: dict[str, Any],
        ) -> tuple[dict[str, Any], int]:
            """Classify one event using a thread-local judge instance."""
            if judge is not None:
                local_judge = judge
            else:
                local_judge = getattr(thread_local, "judge", None)
                if local_judge is None:
                    assert judge_factory is not None
                    local_judge, _local_client = judge_factory()
                    thread_local.judge = local_judge

            judge_input = build_judge_input(event)
            judged = local_judge.classify_text(
                judge_input,
                max_tokens=args.max_tokens,
            )
            return judged, len(judge_input)

        progress = tqdm(
            desc=f"{run_dir.name}: written rows", unit="row", leave=False
        )
        with (
            open(output_path, file_mode, encoding="utf-8") as out_file,
            ThreadPoolExecutor(max_workers=args.max_workers) as executor,
        ):
            stop_early = False
            inflight: dict[Future[tuple[dict[str, Any], int]], dict[str, Any]] = {}
            queue_limit = max(1, args.max_workers * 4)

            def write_record(record: dict[str, Any]) -> None:
                nonlocal wrote_rows
                json.dump(record, out_file, ensure_ascii=False)
                out_file.write("\n")
                wrote_rows += 1
                progress.update(1)
                if wrote_rows % args.flush_every == 0:
                    out_file.flush()

            def drain_futures(return_when: Any) -> None:
                if not inflight:
                    return
                done, _pending = wait(
                    set(inflight.keys()),
                    return_when=return_when,
                )
                for fut in done:
                    event = inflight.pop(fut)
                    try:
                        judged, judge_input_chars = fut.result()
                    except Exception as exc:
                        counters["judge_errors"] += 1
                        judged = {
                            "Reasoning_behind_classification": (
                                f"Error in analysis: {exc}"
                            ),
                            "Confidence": 0.0,
                            "category_assignments": {},
                            "justification_type": "Failed classification",
                        }
                        judge_input_chars = len(build_judge_input(event))
                    record = build_output_record(
                        event=event,
                        taxonomy_identifier=taxonomy_label,
                        save_response_text=args.save_response_text,
                        dry_run=False,
                        judged=judged,
                        judge_provider=judge_provider,
                        judge_model=judge_model,
                        judge_input_chars=judge_input_chars,
                    )
                    counters["judged_rows"] += 1
                    write_record(record)

            for line_number, node in iter_action_nodes_from_records(records_path):
                counters["action_nodes_seen"] += 1

                trace_id = node["trace_id"]
                if trace_id in seen_trace_ids:
                    counters["skip_duplicate_or_resume"] += 1
                    continue

                player = node["player"]
                try:
                    agent_type = detect_agent_type(player)
                except ValueError:
                    counters["skip_unknown_agent_type"] += 1
                    continue

                response_blob = trace_map.get(trace_id)
                response_text = (
                    response_blob["response"].strip() if response_blob else ""
                )

                event = {
                    "trace_id": trace_id,
                    "run_dir": str(run_dir),
                    "run_name": run_dir.name,
                    "batch_name": run_dir.parent.name,
                    "record_line": line_number,
                    "game": game,
                    "mechanism": mechanism,
                    "player": player,
                    "player_id": extract_player_id(player),
                    "agent_type": agent_type,
                    "model": extract_model_name(player),
                    "action": str(node.get("action")),
                    "points": node.get("points"),
                    "mediated": bool(node.get("mediated", False)),
                    "response_chars": len(response_text),
                    "response_text": response_text,
                }

                keep, drop_reason = should_keep_event(
                    event=event,
                    args=args,
                    mechanism_filter=mechanism_filter,
                    game_filter=game_filter,
                )
                if not keep:
                    if drop_reason is not None:
                        counters[drop_reason] += 1
                    continue

                if (
                    args.max_items is not None
                    and wrote_rows >= args.max_items
                ):
                    stop_early = True
                    break

                seen_trace_ids.add(trace_id)
                if args.dry_run:
                    record = build_output_record(
                        event=event,
                        taxonomy_identifier=taxonomy_label,
                        save_response_text=args.save_response_text,
                        dry_run=True,
                    )
                    counters["dry_run_rows"] += 1
                    write_record(record)
                else:
                    if args.max_workers > 1:
                        fut = executor.submit(
                            classify_event_with_judge, event
                        )
                        inflight[fut] = event
                        counters["submitted_for_judging"] += 1
                        if len(inflight) >= queue_limit:
                            drain_futures(FIRST_COMPLETED)
                    else:
                        judged, judge_input_chars = classify_event_with_judge(
                            event
                        )
                        record = build_output_record(
                            event=event,
                            taxonomy_identifier=taxonomy_label,
                            save_response_text=args.save_response_text,
                            dry_run=False,
                            judged=judged,
                            judge_provider=judge_provider,
                            judge_model=judge_model,
                            judge_input_chars=judge_input_chars,
                        )
                        counters["judged_rows"] += 1
                        write_record(record)

            if inflight:
                drain_futures(ALL_COMPLETED)
            out_file.flush()

        progress.close()

        summary = summarize_output(output_path)
        summary["taxonomy_source"] = taxonomy_label
        summary["run_dir"] = str(run_dir)
        summary["game"] = game
        summary["mechanism"] = mechanism
        summary["processing_counters"] = dict(counters)
        summary["max_items"] = args.max_items
        summary["agent_type_filter"] = args.agent_type
        summary["min_response_chars"] = args.min_response_chars
        summary["dry_run"] = args.dry_run
        summary["max_workers"] = args.max_workers

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            f.write("\n")

        print(f"  Wrote {wrote_rows} rows to {output_path}")
        print(f"  Summary: {summary_path}")
        if counters:
            print("  Counters:")
            for key, value in sorted(counters.items()):
                print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
