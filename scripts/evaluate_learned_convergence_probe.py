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
    load_dataset,
    load_fold_manifest,
    sha256_file,
)


DEFAULT_FOLD_MANIFEST = (
    REPO_ROOT / "experiments/robot/libero/manifests/libero_spatial_task_oof_5fold_v1.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--training-bundle", required=True, type=Path)
    parser.add_argument("--fold-manifest", type=Path, default=DEFAULT_FOLD_MANIFEST)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--compact-manifest", type=Path)
    parser.add_argument("--model-artifact", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _check_output(path: Path | None, overwrite: bool) -> None:
    if path is not None and path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite output: {path}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    for path in (args.output, args.compact_manifest, args.model_artifact):
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
    evaluation = evaluate_oof_bundle(records, assignment, bundle)
    evaluation["inputs"] = {
        "dataset_sha256": dataset_manifest["dataset_sha256"],
        "training_bundle_sha256": sha256_file(args.training_bundle),
        "fold_manifest_sha256": sha256_file(args.fold_manifest),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(evaluation))

    if args.compact_manifest:
        compact = compact_result_manifest(
            evaluation,
            dataset_manifest,
            bundle,
            fold_manifest_path=args.fold_manifest,
            training_bundle_sha256=sha256_file(args.training_bundle),
        )
        args.compact_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.compact_manifest.write_bytes(canonical_json_bytes(compact))
    if args.model_artifact:
        selected = evaluation["selected_model"]
        artifact = {
            "schema_version": 1,
            "status": "diagnostic_offline_only",
            "online_integration_authorized": False,
            "selected_model": selected,
            "source_git_commit": dataset_manifest["source_git_commit"],
            "dataset_sha256": dataset_manifest["dataset_sha256"],
            "fold_manifest_sha256": sha256_file(args.fold_manifest),
            "feature_schema": bundle["feature_schema"],
            "random_seed": bundle["random_seed"],
            "dependency_versions": dataset_manifest["dependency_versions"],
            "leakage_audit": bundle["leakage_audit"],
            "oof_models_thresholds_and_normalization": bundle["models"],
            "full_data_refit_diagnostic_not_oof": bundle["models"][selected]["full_data_refit_diagnostic_not_oof"],
            "oof_gate_passed": evaluation["models"][selected]["passes_all_gates"],
            "study_conclusion": evaluation["conclusion"],
        }
        args.model_artifact.parent.mkdir(parents=True, exist_ok=True)
        args.model_artifact.write_bytes(canonical_json_bytes(artifact))
        if args.compact_manifest:
            compact["model_artifact"] = {
                "path": str(args.model_artifact),
                "sha256": sha256_file(args.model_artifact),
                "contains_all_oof_normalization_parameters": True,
            }
            args.compact_manifest.write_bytes(canonical_json_bytes(compact))

    print(f"Selected diagnostic model: {evaluation['selected_model']}")
    print(f"Online integration worth investigating: {evaluation['online_integration_worth_investigating']}")
    print(f"Conclusion: {evaluation['conclusion']}")
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
