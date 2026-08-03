#!/usr/bin/env python3
"""Evaluate frozen strict-boundary models on outer held-out tasks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.boundary_latent_oof_lib import (  # noqa: E402
    evaluate_boundary_oof,
    load_boundary_dataset,
    load_fold_assignment,
    load_training_bundle,
    require_outputs_absent,
    write_csv,
    write_json,
    write_jsonl,
)

DEFAULT_DATASET_DIR = REPO_ROOT / "benchmark_results/preconvergence_trigger/seed7/boundary_latent_oof_v1/dataset"
DEFAULT_TRAINING_DIR = REPO_ROOT / "benchmark_results/preconvergence_trigger/seed7/boundary_latent_oof_v1/training"
DEFAULT_FOLD_MANIFEST = REPO_ROOT / "experiments/robot/libero/manifests/libero_spatial_task_oof_5fold_v1.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmark_results/preconvergence_trigger/seed7/boundary_latent_oof_v1/evaluation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--training-dir", type=Path, default=DEFAULT_TRAINING_DIR)
    parser.add_argument("--fold-manifest", type=Path, default=DEFAULT_FOLD_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_paths = [
        args.output_dir / "boundary_oof_report.json",
        args.output_dir / "boundary_oof_predictions.jsonl",
        args.output_dir / "boundary_oof_by_task.csv",
    ]
    require_outputs_absent(output_paths)
    dataset_manifest, payload = load_boundary_dataset(args.dataset_dir)
    _, training_bundle = load_training_bundle(args.training_dir)
    _, assignment, fold_hash = load_fold_assignment(
        args.fold_manifest,
        {int(row["task_id"]) for row in payload["scoring_trajectories"]},
    )
    if fold_hash != dataset_manifest["fold_manifest_sha256"] or fold_hash != training_bundle["fold_manifest_sha256"]:
        raise ValueError("fold manifest identity mismatch")
    report, predictions, task_rows = evaluate_boundary_oof(
        payload, training_bundle, assignment
    )
    write_json(output_paths[0], report)
    write_jsonl(output_paths[1], predictions)
    write_csv(output_paths[2], task_rows)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "prediction_replay_count": len(predictions), "model_result_count": len(report["models"])}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
