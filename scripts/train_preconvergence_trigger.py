#!/usr/bin/env python3
"""Fit task-level OOF low-rank preconvergence triggers and train-only thresholds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.preconvergence_trigger_lib import (  # noqa: E402
    DEFAULT_SEED,
    MODEL_VARIANTS,
    RANK_CANDIDATES,
    TrainingConfig,
    canonical_json_bytes,
    fit_oof_bundle,
    load_dataset_bundle,
    load_fold_assignment,
    sha256_file,
)


def _summary(bundle: Mapping[str, Any]) -> dict[str, Any]:
    models = {}
    for name, model in bundle["models"].items():
        models[name] = {
            "rank": model["rank"],
            "variant": model["variant"],
            "folds": [
                {
                    "fold_id": fold["fold_id"],
                    "training_task_ids": fold["training_task_ids"],
                    "held_out_task_ids": fold["held_out_task_ids"],
                    "training_prediction_count": fold["training_prediction_count"],
                    "model_applicable_training_prediction_count": fold[
                        "model_applicable_training_prediction_count"
                    ],
                    "history_unavailable_training_prediction_count": fold[
                        "history_unavailable_training_prediction_count"
                    ],
                    "threshold_selection": fold["threshold_selection"],
                    "parameter_count": fold["fitted_trigger"]["parameter_count"],
                    "inference_flops": fold["fitted_trigger"]["inference_flops"],
                }
                for fold in model["folds"]
            ],
        }
    return {
        "schema_version": bundle["schema_version"],
        **bundle["replay_contract"],
        "replay_contract": bundle["replay_contract"],
        "seed": bundle["seed"],
        "training_config": bundle["training_config"],
        "model_fitting_scope": bundle["model_fitting_scope"],
        "threshold_selection_scope": bundle["threshold_selection_scope"],
        "global_model_fitted": bundle["global_model_fitted"],
        "global_threshold_fitted": bundle["global_threshold_fitted"],
        "leakage_audit": bundle["leakage_audit"],
        "models": models,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--auxiliary-weight", type=float, default=0.1)
    parser.add_argument("--rank", action="append", type=int, choices=RANK_CANDIDATES)
    parser.add_argument("--variant", action="append", choices=MODEL_VARIANTS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bundle_path = args.output_dir / "training_bundle.pt"
    if bundle_path.exists():
        raise FileExistsError(f"refusing to overwrite training bundle: {bundle_path}")
    _, sequences = load_dataset_bundle(args.dataset_dir)
    _, assignment = load_fold_assignment(
        args.fold_manifest, {sequence.identity.task_id for sequence in sequences}
    )
    bundle = fit_oof_bundle(
        sequences,
        assignment,
        ranks=args.rank or RANK_CANDIDATES,
        variants=args.variant or MODEL_VARIANTS,
        config=TrainingConfig(
            seed=args.seed,
            steps=args.steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            auxiliary_weight=args.auxiliary_weight,
        ),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, bundle_path)
    summary = _summary(bundle)
    summary["training_bundle_file"] = bundle_path.name
    summary["training_bundle_sha256"] = sha256_file(bundle_path)
    (args.output_dir / "training_summary.json").write_bytes(
        canonical_json_bytes(summary)
    )
    print(
        f"Fit {len(bundle['models'])} OOF model configurations; "
        f"bundle sha256={summary['training_bundle_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
