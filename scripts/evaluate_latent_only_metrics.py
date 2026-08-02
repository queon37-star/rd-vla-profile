#!/usr/bin/env python3
"""Compare latent-only metrics using task-level out-of-fold trace evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.latent_only_metric_evaluator import (  # noqa: E402
    evaluate_oof,
    load_fold_assignment,
    load_jsonl_records,
    parse_trace_predictions,
)


DEFAULT_FOLD_MANIFEST = (
    REPO_ROOT / "experiments/robot/libero/manifests/libero_spatial_task_oof_5fold_v1.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="append", required=True, type=Path)
    parser.add_argument("--fold-manifest", type=Path, default=DEFAULT_FOLD_MANIFEST)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-iter", type=int, default=2)
    parser.add_argument("--capture-target", type=float, default=0.995)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite output: {args.output}")
    predictions = parse_trace_predictions(load_jsonl_records(args.trace))
    fold_manifest, assignment = load_fold_assignment(
        args.fold_manifest, {item["task_id"] for item in predictions}
    )
    result = evaluate_oof(
        predictions,
        assignment,
        min_iter=args.min_iter,
        capture_target=args.capture_target,
    )
    result["inputs"] = {
        "trace_files": [str(path.resolve()) for path in args.trace],
        "fold_manifest": str(args.fold_manifest.resolve()),
        "fold_manifest_name": fold_manifest.get("name"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Nominal best metric: {result['nominal_best_metric']}")
    print(f"Runtime defaults modified: {result['runtime_defaults_modified']}")
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
