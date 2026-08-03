#!/usr/bin/env python3
"""Fail-closed validator for optional raw preconvergence shadow shards."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.raw_preconvergence_trace import (  # noqa: E402
    load_and_validate_manifests,
)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _step_records(paths: Sequence[Path]) -> dict[tuple[int, int, int], dict]:
    records = {}
    for path in paths:
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                key = (
                    int(record["task_id"]),
                    int(record["episode_id"]),
                    int(record["prediction_step"]),
                )
                if key in records:
                    raise ValueError(f"duplicate step-log prediction identity: {key}")
                records[key] = record
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--step-log", type=Path, action="append", default=[])
    parser.add_argument("--parity-step-log", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compact-manifest", type=Path)
    parser.add_argument("--expected-state-shape", type=str)
    parser.add_argument("--expected-action-shape", type=str)
    parser.add_argument("--expected-state-dtype", type=str)
    parser.add_argument("--expected-action-dtype", type=str)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    compact, predictions = load_and_validate_manifests(args.manifest)
    raw_ids = {
        (
            item["identity"]["task_id"],
            item["identity"]["episode_id"],
            item["identity"]["prediction_id"],
        )
        for item in predictions
    }
    collection_steps = _step_records(args.step_log) if args.step_log else {}
    expected_ids = set(collection_steps) if collection_steps else raw_ids
    missing = sorted(expected_ids - raw_ids)
    unexpected = sorted(raw_ids - expected_ids)
    if missing or unexpected:
        raise ValueError(
            f"raw/step identity mismatch: missing={len(missing)}, unexpected={len(unexpected)}"
        )
    parity_fields = (
        "K_t",
        "returned_action_sha256",
        "next_warm_start_state_sha256",
        "iteration_mse",
        "conv_score_list",
        "coda_call_count",
        "stop_reason",
        "canonical_stop_reason",
        "shadow_production_snapshot",
        "cached_final_matches_returned",
        "numerical_retry_attempted",
        "numerical_retry_succeeded",
        "first_attempt_origin",
        "first_attempt_failure",
        "retry_coda_call_count",
        "shadow_tail_iteration_count",
        "rng_state_before_action_sha256",
        "rng_state_after_action_sha256",
    )
    parity_compared = 0
    if args.parity_step_log:
        if not collection_steps:
            raise ValueError("--parity-step-log requires --step-log")
        baseline_steps = _step_records(args.parity_step_log)
        if set(baseline_steps) != set(collection_steps):
            raise ValueError("collection off/on step-log identities differ")
        for key, collected in collection_steps.items():
            baseline = baseline_steps[key]
            for field in parity_fields:
                if baseline.get(field) != collected.get(field):
                    raise ValueError(f"collection off/on parity mismatch for {key} field {field}")
            parity_compared += 1

    state_contracts = sorted(
        {
            (
                tuple(item["tensors"]["states"].shape),
                str(item["tensors"]["states"].dtype),
                tuple(item["tensors"]["states"].stride()),
            )
            for item in predictions
        }
    )
    action_contracts = sorted(
        {
            (
                tuple(item["tensors"]["actions"].shape),
                str(item["tensors"]["actions"].dtype),
                tuple(item["tensors"]["actions"].stride()),
            )
            for item in predictions
        }
    )
    expected_state_shape = (
        tuple(int(value) for value in args.expected_state_shape.split(","))
        if args.expected_state_shape
        else None
    )
    expected_action_shape = (
        tuple(int(value) for value in args.expected_action_shape.split(","))
        if args.expected_action_shape
        else None
    )
    for item in predictions:
        states = item["tensors"]["states"]
        actions = item["tensors"]["actions"]
        if expected_state_shape is not None and tuple(states.shape) != expected_state_shape:
            raise ValueError(f"unexpected state shape: {tuple(states.shape)}")
        if expected_action_shape is not None and tuple(actions.shape) != expected_action_shape:
            raise ValueError(f"unexpected action shape: {tuple(actions.shape)}")
        if args.expected_state_dtype and str(states.dtype) != args.expected_state_dtype:
            raise ValueError(f"unexpected state dtype: {states.dtype}")
        if args.expected_action_dtype and str(actions.dtype) != args.expected_action_dtype:
            raise ValueError(f"unexpected action dtype: {actions.dtype}")
    action_delta_count = 0
    preconvergence_row_count = 0
    first_hit_count = 0
    for item in predictions:
        actions = item["tensors"]["actions"]
        deltas = actions[1:].float() - actions[:-1].float()
        if not bool(torch.isfinite(deltas).all().item()):
            raise ValueError("non-finite auxiliary action delta")
        action_delta_count += int(deltas.shape[0])
        baseline_k = int(item["production_terminal_k"])
        threshold = float(item["action_mse_threshold"])
        first_hit = next(
            (
                k
                for k in range(2, int(item["maximum_shadow_depth"]) + 1)
                if float(item["action_mse"][k]) < threshold
            ),
            None,
        )
        if first_hit is not None:
            first_hit_count += 1
            preconvergence_row_count += max(0, first_hit - 3)
        elif baseline_k != int(item["maximum_shadow_depth"]):
            raise ValueError("no first hit before a non-max production terminal K")

    report = {
        **compact,
        "missing_prediction_identity_count": len(missing),
        "duplicate_prediction_identity_count": 0,
        "unexpected_prediction_identity_count": len(unexpected),
        "state_tensor_contracts": [
            {"shape": list(shape), "dtype": dtype, "stride": list(stride)}
            for shape, dtype, stride in state_contracts
        ],
        "action_tensor_contracts": [
            {"shape": list(shape), "dtype": dtype, "stride": list(stride)}
            for shape, dtype, stride in action_contracts
        ],
        "first_hit_prediction_count": first_hit_count,
        "auxiliary_action_delta_count": action_delta_count,
        "eligible_preconvergence_row_count": preconvergence_row_count,
        "post_convergence_rows_included": 0,
        "validation_status": "passed",
        "production_parity_prediction_count": parity_compared,
        "production_parity_fields": list(parity_fields),
    }
    if args.output:
        _atomic_json(args.output, report)
    if args.compact_manifest:
        _atomic_json(args.compact_manifest, compact)
    print(
        "Validated raw preconvergence shadows: "
        f"predictions={len(predictions)}, shards/manifests={len(args.manifest)}, "
        f"trace_set_sha256={compact['trace_set_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
