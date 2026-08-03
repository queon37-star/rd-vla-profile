#!/usr/bin/env python3
"""Build the reusable strict-boundary latent dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.boundary_latent_oof_lib import (  # noqa: E402
    BOUNDARY_WIDTH,
    EXPECTED_TRAJECTORY_BUNDLE_SHA256,
    PRIMARY_THRESHOLD,
    build_boundary_dataset_payload,
    load_fold_assignment,
    load_frozen_trajectory_bundle,
    save_boundary_dataset,
    source_git_commit,
)

DEFAULT_BUNDLE_DIR = REPO_ROOT / "benchmark_results/preconvergence_trigger/seed7/action_latent_audit_v1/bundle"
DEFAULT_FOLD_MANIFEST = REPO_ROOT / "experiments/robot/libero/manifests/libero_spatial_task_oof_5fold_v1.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmark_results/preconvergence_trigger/seed7/boundary_latent_oof_v1/dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--fold-manifest", type=Path, default=DEFAULT_FOLD_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-bundle-sha256", default=EXPECTED_TRAJECTORY_BUNDLE_SHA256)
    parser.add_argument("--threshold", type=float, default=PRIMARY_THRESHOLD)
    parser.add_argument("--boundary-width", type=int, default=BOUNDARY_WIDTH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_manifest, records = load_frozen_trajectory_bundle(
        args.bundle_dir, expected_sha256=args.expected_bundle_sha256
    )
    _, _, fold_hash = load_fold_assignment(
        args.fold_manifest, {int(record["task_id"]) for record in records}
    )
    payload, coverage = build_boundary_dataset_payload(
        records, threshold=args.threshold, boundary_width=args.boundary_width
    )
    manifest = save_boundary_dataset(
        args.output_dir,
        payload,
        coverage,
        source_bundle_path=args.bundle_dir / source_manifest["bundle_file"],
        source_bundle_sha256=source_manifest["output_bundle_sha256"],
        fold_manifest_sha256=fold_hash,
        git_commit=source_git_commit(REPO_ROOT),
    )
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "boundary_dataset_sha256": manifest["boundary_dataset_sha256"], "counts": manifest["counts_by_target_and_task"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
