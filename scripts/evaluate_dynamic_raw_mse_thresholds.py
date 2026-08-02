#!/usr/bin/env python3
"""Evaluate iteration-conditioned raw-MSE thresholds with task-level OOF replay."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.dynamic_raw_mse_oof import (  # noqa: E402
    DEFAULT_ALPHAS,
    DEFAULT_MIN_NEGATIVE_SAMPLES,
    evaluate_dynamic_raw_mse_oof,
    format_pareto_table,
)
from scripts.latent_only_metric_evaluator import (  # noqa: E402
    load_fold_assignment,
    load_jsonl_records,
    parse_trace_predictions,
)


DEFAULT_FOLD_MANIFEST = (
    REPO_ROOT / "experiments/robot/libero/manifests/libero_spatial_task_oof_5fold_v1.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    traces = parser.add_mutually_exclusive_group(required=True)
    traces.add_argument("--trace", action="append", type=Path)
    traces.add_argument(
        "--trace-dir",
        type=Path,
        help="directory containing task*/steps.jsonl calibration traces",
    )
    parser.add_argument("--fold-manifest", type=Path, default=DEFAULT_FOLD_MANIFEST)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--alpha",
        action="append",
        type=float,
        help="false-positive budget; repeat to override the default grid",
    )
    parser.add_argument(
        "--min-negative-samples",
        type=int,
        default=DEFAULT_MIN_NEGATIVE_SAMPLES,
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
    result = evaluate_dynamic_raw_mse_oof(
        predictions,
        assignment,
        alphas=DEFAULT_ALPHAS if args.alpha is None else args.alpha,
        min_negative_samples=args.min_negative_samples,
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
    fixed = result["fixed_raw_mse_reference"]["oof_stopping"]
    print(
        "Fixed raw_mse reference: "
        f"false={fixed['false_convergence_count']} "
        f"capture={fixed['convergence_capture']:.4%} "
        f"mean_delta_K={fixed['mean_delta_K']:.4f}"
    )
    print("Dynamic raw_mse Pareto frontier:")
    print(format_pareto_table(result))
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
