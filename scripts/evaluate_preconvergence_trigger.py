#!/usr/bin/env python3
"""Evaluate frozen OOF preconvergence triggers with exact CONFIRM_NEXT replay."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.preconvergence_trigger_lib import (  # noqa: E402
    canonical_json_bytes,
    evaluate_oof_bundle,
    load_dataset_bundle,
    load_fold_assignment,
    sha256_file,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--training-bundle", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compact-manifest", type=Path)
    parser.add_argument("--coda-latency-ms", type=float, required=True)
    parser.add_argument("--recurrent-iteration-latency-ms", type=float, required=True)
    parser.add_argument("--gate-latency-ms", type=float, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite report: {args.output}")
    dataset_manifest, sequences = load_dataset_bundle(args.dataset_dir)
    fold_manifest, assignment = load_fold_assignment(
        args.fold_manifest, {sequence.identity.task_id for sequence in sequences}
    )
    training_bundle = torch.load(
        args.training_bundle, map_location="cpu", weights_only=True
    )
    latency = {
        "coda_latency_ms": args.coda_latency_ms,
        "recurrent_iteration_latency_ms": args.recurrent_iteration_latency_ms,
        "gate_latency_ms": args.gate_latency_ms,
    }
    report = evaluate_oof_bundle(
        sequences, assignment, training_bundle, latency=latency
    )
    report["inputs"] = {
        "dataset_sha256": dataset_manifest["dataset_sha256"],
        "training_bundle_sha256": sha256_file(args.training_bundle),
        "fold_manifest_sha256": sha256_file(args.fold_manifest),
        "fold_manifest": fold_manifest,
        "latency_assumptions_ms": latency,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(report))
    if args.compact_manifest is not None:
        if args.compact_manifest.exists():
            raise FileExistsError(
                f"refusing to overwrite compact manifest: {args.compact_manifest}"
            )
        compact_models = {}
        for name, result in report["models"].items():
            primary = dict(result["primary_actual_warm"])
            primary.pop("per_task", None)
            compact_models[name] = {
                "rank": result["rank"],
                "variant": result["variant"],
                "parameter_count": result["parameter_count"],
                "inference_flops": result["inference_flops"],
                "fold_thresholds": [
                    {
                        "fold_id": fold["fold_id"],
                        "selected_threshold": fold["selected_threshold"],
                        "selected_threshold_hex": fold["selected_threshold_hex"],
                        "selection_status": fold["selection_status"],
                    }
                    for fold in result["folds"]
                ],
                "primary_actual_warm": primary,
                "secondary_cold": result["secondary_cold"],
                "zero_overhead_projection": result["zero_overhead_projection"],
                "promotion_checks": result["promotion_checks"],
                "passes_all_promotion_checks": result[
                    "passes_all_promotion_checks"
                ],
            }
        compact = {
            "schema_version": report["schema_version"],
            "study": "one-step-ahead-coda-trigger-feasibility",
            "inputs": report["inputs"],
            "models": compact_models,
            "online_integration_implemented": False,
            "gpu_microbenchmark_required": report["gpu_microbenchmark_required"],
        }
        args.compact_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.compact_manifest.write_bytes(canonical_json_bytes(compact))
    passing = [
        name
        for name, result in report["models"].items()
        if result["passes_all_promotion_checks"]
    ]
    print(f"OOF evaluation complete; passing configurations={passing or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
