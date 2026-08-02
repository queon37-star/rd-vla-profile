#!/usr/bin/env python3
"""Run ACTUAL_WARM task-level OOF evaluation for raw-MSE Coda activation."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.coda_activation_oof import (  # noqa: E402
    DEFAULT_BETAS,
    DEFAULT_MIN_ACTIVATION_DUE_SAMPLES,
    evaluate_coda_activation_oof,
    format_pareto_table,
)
from scripts.latent_only_metric_evaluator import (  # noqa: E402
    load_fold_assignment,
    load_jsonl_records,
    parse_trace_predictions,
)


DEFAULT_TRACE_DIR = REPO_ROOT / "benchmark_results/latent_only/calibration_cb93a8b"
DEFAULT_FOLD_MANIFEST = (
    REPO_ROOT / "experiments/robot/libero/manifests/libero_spatial_task_oof_5fold_v1.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    traces = parser.add_mutually_exclusive_group()
    traces.add_argument("--trace", action="append", type=Path)
    traces.add_argument(
        "--trace-dir",
        type=Path,
        default=DEFAULT_TRACE_DIR,
        help="directory containing task*/steps.jsonl calibration traces",
    )
    parser.add_argument("--fold-manifest", type=Path, default=DEFAULT_FOLD_MANIFEST)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--beta",
        action="append",
        type=float,
        help="activation miss budget; repeat to override the default grid",
    )
    parser.add_argument(
        "--min-activation-due-samples",
        type=int,
        default=DEFAULT_MIN_ACTIVATION_DUE_SAMPLES,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _resolve_trace_paths(args: argparse.Namespace) -> list[Path]:
    if args.trace:
        return list(args.trace)
    paths = sorted(args.trace_dir.glob("task*/steps.jsonl"))
    if not paths:
        raise FileNotFoundError(
            f"No task*/steps.jsonl traces found under: {args.trace_dir}"
        )
    return paths


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite output: {args.output}")
    trace_paths = _resolve_trace_paths(args)
    predictions = parse_trace_predictions(load_jsonl_records(trace_paths))
    fold_manifest, assignment = load_fold_assignment(
        args.fold_manifest, {item["task_id"] for item in predictions}
    )
    result = evaluate_coda_activation_oof(
        predictions,
        assignment,
        betas=DEFAULT_BETAS if args.beta is None else args.beta,
        min_activation_due_samples=args.min_activation_due_samples,
    )
    result["inputs"] = {
        "trace_files": [str(path.resolve()) for path in trace_paths],
        "fold_manifest": str(args.fold_manifest.resolve()),
        "fold_manifest_name": fold_manifest.get("name"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    baseline = result["coda_every_iteration_reference"]
    print(
        "Coda every iteration: "
        f"calls={baseline['baseline_total_coda_calls']} "
        f"mean_calls={baseline['mean_coda_calls_per_prediction']:.4f}"
    )
    print("Dynamic activation Pareto frontier:")
    print(format_pareto_table(result))
    selection = result["selection"]
    if selection["selected_schedule"] is None:
        print(
            "No dynamic schedule satisfies both mean delta_K <= 0.1 "
            "and p95 delta_K <= 1; no winner selected."
        )
    else:
        winner = selection["selected_schedule"]
        print(
            f"Selected beta={winner['beta']:.17g} with "
            f"Coda-call reduction={winner['coda_call_reduction']:.4%}."
        )
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
