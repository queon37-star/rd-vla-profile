#!/usr/bin/env python3
"""Audit strict action-boundary reference definitions offline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.boundary_latent_oof_lib import (  # noqa: E402
    EXPECTED_TRAJECTORY_BUNDLE_SHA256,
    PRIMARY_THRESHOLD,
    build_boundary_reference_audit,
    load_fold_assignment,
    load_frozen_trajectory_bundle,
    require_outputs_absent,
    source_git_commit,
    write_csv,
    write_json,
)

DEFAULT_BUNDLE_DIR = REPO_ROOT / "benchmark_results/preconvergence_trigger/seed7/action_latent_audit_v1/bundle"
DEFAULT_FOLD_MANIFEST = REPO_ROOT / "experiments/robot/libero/manifests/libero_spatial_task_oof_5fold_v1.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmark_results/preconvergence_trigger/seed7/boundary_latent_oof_v1/reference_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--fold-manifest", type=Path, default=DEFAULT_FOLD_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-bundle-sha256", default=EXPECTED_TRAJECTORY_BUNDLE_SHA256)
    parser.add_argument("--threshold", type=float, default=PRIMARY_THRESHOLD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_paths = [
        args.output_dir / "boundary_reference_audit.json",
        args.output_dir / "boundary_reference_by_task.csv",
    ]
    require_outputs_absent(output_paths)
    manifest, records = load_frozen_trajectory_bundle(
        args.bundle_dir, expected_sha256=args.expected_bundle_sha256
    )
    _, _, fold_hash = load_fold_assignment(
        args.fold_manifest, {int(record["task_id"]) for record in records}
    )
    report, task_rows = build_boundary_reference_audit(
        records,
        source_bundle_sha256=manifest["output_bundle_sha256"],
        fold_manifest_sha256=fold_hash,
        git_commit=source_git_commit(REPO_ROOT),
        threshold=args.threshold,
    )
    write_json(output_paths[0], report)
    write_csv(output_paths[1], task_rows)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "actual_warm_predictions": report["primary_actual_warm"]["prediction_count"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
