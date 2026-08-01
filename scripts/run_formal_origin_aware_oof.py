#!/usr/bin/env python3
"""Validate the formal calibration set and run frozen multi-scenario OOF replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.formal_origin_aware_oof_lib import run_formal_cost_sensitivity_oof  # noqa: E402
from scripts.origin_aware_calibration_lib import validate_calibration_run  # noqa: E402
from scripts.origin_aware_replay_lib import parse_shadow_predictions  # noqa: E402
from scripts.replay_origin_aware_shadow import load_json_records  # noqa: E402


DEFAULT_INITIAL_STATE_MANIFEST = (
    REPO_ROOT / "experiments/robot/libero/manifests/libero_spatial_official_50_v1.json"
)
DEFAULT_FOLD_MANIFEST = (
    REPO_ROOT / "experiments/robot/libero/manifests/libero_spatial_task_oof_5fold_v1.json"
)
DEFAULT_COST_MANIFEST = (
    REPO_ROOT / "experiments/robot/libero/manifests/origin_aware_oof_cost_sensitivity_v1.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument(
        "--initial-state-manifest", type=Path, default=DEFAULT_INITIAL_STATE_MANIFEST
    )
    parser.add_argument("--fold-manifest", type=Path, default=DEFAULT_FOLD_MANIFEST)
    parser.add_argument("--cost-manifest", type=Path, default=DEFAULT_COST_MANIFEST)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-seed", type=int, default=7)
    parser.add_argument("--top-n", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite formal OOF report: {args.output}")

    calibration_validation = validate_calibration_run(
        str(args.run_root), str(args.initial_state_manifest), base_seed=args.base_seed
    )
    if calibration_validation.get("complete_10_task_gate") is not True:
        raise ValueError("formal OOF requires the complete ten-task calibration gate")

    trace_paths = [args.run_root / f"task{task_id}" / "steps.jsonl" for task_id in range(10)]
    records = []
    for path in trace_paths:
        records.extend(load_json_records(path))
    predictions = parse_shadow_predictions(records)
    expected_count = calibration_validation["totals"]["predictions"]
    if len(predictions) != expected_count:
        raise ValueError(
            f"calibration/replay prediction count mismatch: {expected_count} != {len(predictions)}"
        )

    fold_manifest = _load_json_object(args.fold_manifest)
    cost_manifest = _load_json_object(args.cost_manifest)
    report = run_formal_cost_sensitivity_oof(
        predictions,
        fold_manifest,
        cost_manifest,
        top_n=args.top_n,
    )
    report["calibration_validation"] = calibration_validation
    report["inputs"] = {
        "run_root": str(args.run_root.resolve()),
        "trace_files": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in trace_paths
        ],
        "initial_state_manifest": {
            "path": str(args.initial_state_manifest.resolve()),
            "sha256": sha256_file(args.initial_state_manifest),
        },
        "fold_manifest": {
            "path": str(args.fold_manifest.resolve()),
            "sha256": sha256_file(args.fold_manifest),
        },
        "cost_manifest": {
            "path": str(args.cost_manifest.resolve()),
            "sha256": sha256_file(args.cost_manifest),
        },
    }
    report["prediction_count"] = len(predictions)
    report["task_ids"] = sorted({prediction.task_id for prediction in predictions})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Validated calibration predictions: {len(predictions)}")
    print(f"Safety-passing families: {report['passing_family_count']}")
    print(f"GPU microbenchmark shortlist: {report['microbenchmark_shortlist_count']}")
    print(
        "Best tested candidate-favorable linear improvement: "
        f"{100 * report['tested_candidate_favorable_best_improvement']:.3f}%"
    )
    print(f"Linear 5% gate met: {report['linear_model_5pct_gate_met']}")
    print(f"Online screening allowed: {report['online_screening_allowed']}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
