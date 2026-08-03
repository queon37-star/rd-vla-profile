#!/usr/bin/env python3
"""Fit strict-boundary task-level OOF models and train-only gate thresholds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.boundary_latent_oof_lib import (  # noqa: E402
    DEFAULT_LEARNING_RATE,
    DEFAULT_SEED,
    DEFAULT_STEPS,
    DEFAULT_WEIGHT_DECAY,
    fit_boundary_oof_bundle,
    load_boundary_dataset,
    load_fold_assignment,
    save_training_bundle,
    source_git_commit,
)

DEFAULT_DATASET_DIR = REPO_ROOT / "benchmark_results/preconvergence_trigger/seed7/boundary_latent_oof_v1/dataset"
DEFAULT_FOLD_MANIFEST = REPO_ROOT / "experiments/robot/libero/manifests/libero_spatial_task_oof_5fold_v1.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmark_results/preconvergence_trigger/seed7/boundary_latent_oof_v1/training"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--fold-manifest", type=Path, default=DEFAULT_FOLD_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest, payload = load_boundary_dataset(args.dataset_dir)
    _, assignment, fold_hash = load_fold_assignment(
        args.fold_manifest,
        {int(row["task_id"]) for row in payload["scoring_trajectories"]},
    )
    if fold_hash != manifest["fold_manifest_sha256"]:
        raise ValueError("fold manifest SHA-256 differs from boundary dataset contract")
    bundle = fit_boundary_oof_bundle(
        payload,
        assignment,
        source_bundle_sha256=manifest["source_trajectory_bundle_sha256"],
        boundary_dataset_sha256=manifest["boundary_dataset_sha256"],
        fold_manifest_sha256=fold_hash,
        git_commit=source_git_commit(REPO_ROOT),
        seed=args.seed,
        steps=args.steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    summary = save_training_bundle(args.output_dir, bundle)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "training_bundle_sha256": summary["training_bundle_sha256"], "fold_count": len(summary["folds"])}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
