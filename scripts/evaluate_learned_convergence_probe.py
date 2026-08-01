#!/usr/bin/env python3
"""Evaluate learned convergence probes with scheduler-level OOF gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.learned_convergence_probe_lib import (  # noqa: E402
    canonical_json_bytes,
    compact_result_manifest,
    evaluate_oof_bundle,
    source_git_commit,
    load_dataset,
    load_fold_manifest,
    sha256_file,
)


DEFAULT_FOLD_MANIFEST = (
    REPO_ROOT / "experiments/robot/libero/manifests/libero_spatial_task_oof_5fold_v1.json"
)


DEFAULT_PARENT_RESULT_MANIFEST = (
    REPO_ROOT / "experiments/robot/libero/manifests/learned_convergence_probe_seed7_result_v1.json"
)
DEFAULT_MODEL_ARTIFACT = (
    REPO_ROOT / "experiments/robot/libero/manifests/learned_convergence_probe_seed7_model_v1.json"
)
DEFAULT_COST_MANIFEST = (
    REPO_ROOT / "experiments/robot/libero/manifests/origin_aware_oof_cost_sensitivity_v1.json"
)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--training-bundle", required=True, type=Path)
    parser.add_argument("--fold-manifest", type=Path, default=DEFAULT_FOLD_MANIFEST)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--compact-manifest", type=Path)
    parser.add_argument("--parent-result-manifest", type=Path, default=DEFAULT_PARENT_RESULT_MANIFEST)
    parser.add_argument("--existing-model-artifact", type=Path, default=DEFAULT_MODEL_ARTIFACT)
    parser.add_argument("--cost-manifest", type=Path, default=DEFAULT_COST_MANIFEST)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _check_output(path: Path | None, overwrite: bool) -> None:
    if path is not None and path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite output: {path}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    for path in (args.output, args.compact_manifest):
        _check_output(path, args.overwrite)
    dataset_manifest, records = load_dataset(args.dataset_dir)
    _, assignment = load_fold_manifest(
        args.fold_manifest, {str(record["task_id"]) for record in records}
    )
    bundle = json.loads(args.training_bundle.read_text(encoding="utf-8"))
    if bundle.get("inputs", {}).get("dataset_sha256") != dataset_manifest["dataset_sha256"]:
        raise ValueError("training bundle/dataset SHA-256 mismatch")
    if bundle.get("inputs", {}).get("fold_manifest_sha256") != sha256_file(args.fold_manifest):
        raise ValueError("training bundle/fold manifest SHA-256 mismatch")

    parent_result = json.loads(args.parent_result_manifest.read_text(encoding="utf-8"))
    model_artifact = json.loads(args.existing_model_artifact.read_text(encoding="utf-8"))
    cost_manifest = json.loads(args.cost_manifest.read_text(encoding="utf-8"))
    if parent_result.get("schema_version") != 1:
        raise ValueError("parent result manifest must be schema version 1")
    if model_artifact.get("dataset_sha256") != dataset_manifest["dataset_sha256"]:
        raise ValueError("existing model artifact/dataset SHA-256 mismatch")
    if model_artifact.get("fold_manifest_sha256") != sha256_file(args.fold_manifest):
        raise ValueError("existing model artifact/fold manifest SHA-256 mismatch")
    planning = cost_manifest.get("planning_anchors", {})
    action_compare = cost_manifest.get("action_compare_anchor", {})
    cost_anchors = {
        "recurrent_ms": planning.get("recurrent_ms"),
        "coda_ms": planning.get("decode_ms"),
        "baseline_action_compare_ms": action_compare.get("median_ms"),
        "planning_anchor_source": planning.get("source"),
        "action_compare_source": action_compare.get("source_glob"),
        "cost_manifest_path": str(args.cost_manifest),
        "cost_manifest_sha256": sha256_file(args.cost_manifest),
    }
    evaluation = evaluate_oof_bundle(
        records,
        assignment,
        bundle,
        cost_anchors=cost_anchors,
    )
    evaluation["inputs"] = {
        "dataset_sha256": dataset_manifest["dataset_sha256"],
        "training_bundle_sha256": sha256_file(args.training_bundle),
        "fold_manifest_sha256": sha256_file(args.fold_manifest),
        "parent_result_manifest_sha256": sha256_file(args.parent_result_manifest),
        "model_artifact_sha256": sha256_file(args.existing_model_artifact),
        "cost_manifest_sha256": sha256_file(args.cost_manifest),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(evaluation))

    if args.compact_manifest:
        compact = compact_result_manifest(
            evaluation,
            dataset_manifest,
            bundle,
            evaluator_source_git_commit=source_git_commit(REPO_ROOT),
            parent_result_manifest_path=args.parent_result_manifest,
            parent_result_manifest_sha256=sha256_file(args.parent_result_manifest),
            model_artifact_path=args.existing_model_artifact,
            model_artifact_sha256=sha256_file(args.existing_model_artifact),
            fold_manifest_path=args.fold_manifest,
            training_bundle_sha256=sha256_file(args.training_bundle),
        )
        args.compact_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.compact_manifest.write_bytes(canonical_json_bytes(compact))

    print(f"Selected diagnostic model: {evaluation['selected_diagnostic_model']}")
    print(f"Online integration worth investigating: {evaluation['online_integration_worth_investigating']}")
    print(f"Final decision: {evaluation['final_decision']}")
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
