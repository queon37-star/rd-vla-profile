#!/usr/bin/env python3
"""Replay fixed-depth action-head workloads to isolate repeated Coda cost."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prismatic.models.action_head_workload import load_action_head_workload  # noqa: E402
from scripts.origin_aware_calibration_lib import validate_calibration_run  # noqa: E402
from scripts.origin_aware_gpu_microbenchmark_lib import (  # noqa: E402
    GPUMicrobenchmarkValidationError,
    balanced_condition_order,
    load_json_object,
    sha256_file,
)
from scripts.run_origin_aware_gpu_microbenchmark import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_INITIAL_STATE_MANIFEST,
    DEFAULT_RUN_ROOT,
    _prepare_tensors,
    _validate_projector,
    captured_cold_initial_state,
    load_benchmark_modules,
    load_workload_descriptors,
)


DEFAULT_PROTOCOL = (
    REPO_ROOT
    / "experiments/robot/libero/manifests/fixed_coda_microprofile_v1.json"
)


@dataclass(frozen=True)
class FixedCodaCondition:
    condition_id: str
    kind: str
    fixed_k: int


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("schema_version") != 1:
        raise GPUMicrobenchmarkValidationError("unsupported fixed-Coda protocol version")
    if not isinstance(protocol.get("device"), str):
        raise GPUMicrobenchmarkValidationError("protocol device must be a string")
    for field in ("seed", "measurement_repeats", "warmup_rounds_per_origin"):
        value = protocol.get(field)
        if not isinstance(value, int) or value <= 0:
            raise GPUMicrobenchmarkValidationError(
                f"protocol {field} must be a positive integer"
            )
    depths = protocol.get("fixed_depths")
    if (
        not isinstance(depths, list)
        or not depths
        or any(not isinstance(value, int) or value <= 0 for value in depths)
        or len(set(depths)) != len(depths)
    ):
        raise GPUMicrobenchmarkValidationError(
            "protocol fixed_depths must be unique positive integers"
        )
    if depths != sorted(depths):
        raise GPUMicrobenchmarkValidationError("protocol fixed_depths must be sorted")
    if protocol.get("primary_scope") != "actual_warm":
        raise GPUMicrobenchmarkValidationError(
            "fixed-Coda protocol primary_scope must be actual_warm"
        )
    for field in (
        "expected_formal_workload_count",
        "expected_cold_workload_count",
        "expected_actual_warm_workload_count",
    ):
        value = protocol.get(field)
        if not isinstance(value, int) or value <= 0:
            raise GPUMicrobenchmarkValidationError(
                f"protocol {field} must be a positive integer"
            )


def _conditions(protocol: Mapping[str, Any]) -> list[FixedCodaCondition]:
    conditions: list[FixedCodaCondition] = []
    for fixed_k in protocol["fixed_depths"]:
        conditions.extend(
            [
                FixedCodaCondition(
                    condition_id=f"legacy_fixed_k{fixed_k}",
                    kind="legacy_fixed",
                    fixed_k=int(fixed_k),
                ),
                FixedCodaCondition(
                    condition_id=f"terminal_only_k{fixed_k}",
                    kind="terminal_only",
                    fixed_k=int(fixed_k),
                ),
            ]
        )
    if protocol.get("include_terminal_only_k5") is True:
        if 5 not in protocol["fixed_depths"]:
            conditions.append(
                FixedCodaCondition(
                    condition_id="terminal_only_k5",
                    kind="terminal_only",
                    fixed_k=5,
                )
            )
    return conditions


def _condition_kwargs(
    condition: FixedCodaCondition,
    incoming_warm_state: torch.Tensor | None,
) -> dict[str, Any]:
    return {
        "phase": "Inference",
        "num_iter": int(condition.fixed_k),
        "convergence_strategy": (
            None if condition.kind == "legacy_fixed" else "fixed_terminal_only"
        ),
        "kl_thresh": 0.001,
        "cos_thresh": 0.999,
        "max_iter": max(32, int(condition.fixed_k)),
        "warm_start_state": incoming_warm_state,
        "enable_warm_start": True,
        "warm_start_source": "midpoint",
        "warm_start_min_iter": 2,
        "validate_warm_start_finite": True,
        "profile_coda_cost": False,
        "use_cached_final_output": False,
        "use_latent_precheck": False,
        "latent_precheck_mode": "off",
        "latent_precheck_trace_level": "off",
        "latent_precheck_warm_thresh": None,
        "latent_precheck_max_skip_iters": 0,
        "latent_precheck_confirmation_mode": "next_iter",
        "nonfinite_policy": "legacy",
        "shadow_full_depth": False,
        "collect_preconvergence_raw_shadow": False,
        "capture_action_head_workload": False,
    }


def _actual_schedule(
    action_head,
    condition: FixedCodaCondition,
    returned_k: int,
) -> dict[str, Any]:
    debug = action_head.model.last_recurrence_debug
    if not isinstance(debug, Mapping):
        raise GPUMicrobenchmarkValidationError(
            "action head did not publish recurrence debug metadata"
        )
    if condition.kind == "legacy_fixed":
        return {
            "K_t": int(returned_k),
            "recurrent_calls": int(returned_k),
            "coda_calls": int(returned_k),
            "coda_calls_source": "legacy_fixed_implementation_contract",
            "canonical_recurrence_strategy": debug.get(
                "canonical_recurrence_strategy"
            ),
            "warm_start_state_used": bool(debug.get("warm_start_state_used")),
        }
    return {
        "K_t": int(returned_k),
        "recurrent_calls": int(returned_k),
        "coda_calls": int(debug.get("coda_call_count", -1)),
        "coda_calls_source": "runtime_debug",
        "canonical_recurrence_strategy": debug.get("canonical_recurrence_strategy"),
        "warm_start_state_used": bool(debug.get("warm_start_state_used")),
        "final_state_coda_executed": bool(debug.get("final_state_coda_executed")),
    }


def _schedule_mismatch(
    condition: FixedCodaCondition,
    actual_origin: str,
    schedule: Mapping[str, Any],
) -> dict[str, Any] | None:
    expected_strategy = (
        None if condition.kind == "legacy_fixed" else "fixed_terminal_only"
    )
    expected = {
        "K_t": int(condition.fixed_k),
        "recurrent_calls": int(condition.fixed_k),
        "coda_calls": (
            int(condition.fixed_k) if condition.kind == "legacy_fixed" else 1
        ),
        "warm_start_state_used": actual_origin == "ACTUAL_WARM",
    }
    differences = {
        field: {"expected": expected[field], "actual": schedule.get(field)}
        for field in expected
        if schedule.get(field) != expected[field]
    }
    actual_strategy = schedule.get("canonical_recurrence_strategy")
    if condition.kind == "terminal_only" and actual_strategy != expected_strategy:
        differences["canonical_recurrence_strategy"] = {
            "expected": expected_strategy,
            "actual": actual_strategy,
        }
    if condition.kind == "terminal_only" and schedule.get(
        "final_state_coda_executed"
    ) is not True:
        differences["final_state_coda_executed"] = {
            "expected": True,
            "actual": schedule.get("final_state_coda_executed"),
        }
    return differences or None


def _execute(
    action_head,
    proprio_projector,
    tensors: Mapping[str, torch.Tensor | None],
    condition: FixedCodaCondition,
):
    output, returned_k, final_score = action_head.predict_action(
        tensors["actions_hidden_states"],
        proprio=tensors["proprio_input"],
        proprio_projector=proprio_projector,
        **_condition_kwargs(condition, tensors["incoming_warm_start_state"]),
    )
    schedule = _actual_schedule(action_head, condition, int(returned_k))
    return output, int(returned_k), final_score, schedule


def _timed_execute(
    action_head,
    proprio_projector,
    tensors: Mapping[str, torch.Tensor | None],
    condition: FixedCodaCondition,
    device: torch.device,
):
    torch.cuda.synchronize(device)
    start_ns = time.perf_counter_ns()
    output, returned_k, final_score, schedule = _execute(
        action_head, proprio_projector, tensors, condition
    )
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
    if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
        raise GPUMicrobenchmarkValidationError(
            "measured action-head latency must be finite and positive"
        )
    if not bool(torch.isfinite(output).all().item()):
        raise GPUMicrobenchmarkValidationError(
            "fixed-Coda replay returned a non-finite action output"
        )
    return elapsed_ms, output, returned_k, final_score, schedule


def _run_workload(
    *,
    descriptor: Mapping[str, Any],
    block_index: int,
    action_head,
    proprio_projector,
    conditions: Sequence[FixedCodaCondition],
    device: torch.device,
    repeats: int,
    order_seed: int,
    measured: bool,
):
    payload = load_action_head_workload(
        descriptor["path"],
        expected_sha256=descriptor["sha256"],
        expected_identity=descriptor["identity"],
        expected_origin=descriptor["actual_origin"],
    )
    tensors = _prepare_tensors(payload, device)
    _validate_projector(proprio_projector, tensors, device)
    condition_by_id = {
        condition.condition_id: condition for condition in conditions
    }
    condition_ids = list(condition_by_id)
    measurements: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []

    with captured_cold_initial_state(
        action_head.model,
        tensors["selected_initial_state"],
        descriptor["actual_origin"],
    ):
        for repeat_index in range(repeats):
            order = balanced_condition_order(
                condition_ids,
                block_index=block_index * repeats,
                repeat_index=repeat_index,
                seed=order_seed,
            )
            for order_position, condition_id in enumerate(order):
                condition = condition_by_id[condition_id]
                if measured:
                    elapsed_ms, output, returned_k, final_score, schedule = (
                        _timed_execute(
                            action_head,
                            proprio_projector,
                            tensors,
                            condition,
                            device,
                        )
                    )
                else:
                    output, returned_k, final_score, schedule = _execute(
                        action_head, proprio_projector, tensors, condition
                    )
                    torch.cuda.synchronize(device)
                    elapsed_ms = None
                mismatch = _schedule_mismatch(
                    condition, descriptor["actual_origin"], schedule
                )
                if mismatch is not None:
                    mismatches.append(
                        {
                            **descriptor["identity"],
                            "actual_origin": descriptor["actual_origin"],
                            "condition_id": condition_id,
                            "repeat_index": repeat_index,
                            "differences": mismatch,
                        }
                    )
                if measured:
                    measurements.append(
                        {
                            **descriptor["identity"],
                            "actual_origin": descriptor["actual_origin"],
                            "condition_id": condition_id,
                            "condition_kind": condition.kind,
                            "fixed_k": int(condition.fixed_k),
                            "repeat_index": repeat_index,
                            "order_position": order_position,
                            "latency_ms": float(elapsed_ms),
                            "returned_k": int(returned_k),
                            "final_score": (
                                None if final_score is None else float(final_score)
                            ),
                            "output_finite": bool(torch.isfinite(output).all().item()),
                            "schedule": schedule,
                        }
                    )
    del tensors
    del payload
    return measurements, mismatches


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _identity_key(record: Mapping[str, Any]) -> tuple[int, ...]:
    return (
        int(record["task_id"]),
        int(record["episode_id"]),
        int(record["paired_trial_id"]),
        int(record["prediction_step"]),
        int(record["initial_state_id"]),
        int(record["episode_seed"]),
    )


def _within_workload_medians(
    measurements: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[tuple[int, ...], str], list[float]] = {}
    metadata: dict[tuple[tuple[int, ...], str], dict[str, Any]] = {}
    for record in measurements:
        key = (_identity_key(record), str(record["condition_id"]))
        grouped.setdefault(key, []).append(float(record["latency_ms"]))
        metadata.setdefault(
            key,
            {
                "task_id": int(record["task_id"]),
                "episode_id": int(record["episode_id"]),
                "paired_trial_id": int(record["paired_trial_id"]),
                "prediction_step": int(record["prediction_step"]),
                "initial_state_id": int(record["initial_state_id"]),
                "episode_seed": int(record["episode_seed"]),
                "actual_origin": str(record["actual_origin"]),
                "condition_id": str(record["condition_id"]),
                "condition_kind": str(record["condition_kind"]),
                "fixed_k": int(record["fixed_k"]),
            },
        )
    rows = []
    for key, values in grouped.items():
        rows.append(
            {
                **metadata[key],
                "repeat_count": len(values),
                "latency_median_ms": float(statistics.median(values)),
                "latency_mean_ms": float(statistics.fmean(values)),
                "latency_min_ms": float(min(values)),
                "latency_max_ms": float(max(values)),
            }
        )
    rows.sort(
        key=lambda row: (
            row["task_id"],
            row["episode_id"],
            row["prediction_step"],
            row["condition_id"],
        )
    )
    return rows


def _scope_rows(
    rows: Sequence[Mapping[str, Any]], scope: str
) -> list[Mapping[str, Any]]:
    if scope == "all":
        return list(rows)
    expected_origin = "ACTUAL_WARM" if scope == "actual_warm" else "COLD"
    return [row for row in rows if row["actual_origin"] == expected_origin]


def _condition_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [float(row["latency_median_ms"]) for row in rows]
    by_task: dict[int, list[float]] = {}
    for row in rows:
        by_task.setdefault(int(row["task_id"]), []).append(
            float(row["latency_median_ms"])
        )
    task_means = {
        str(task_id): float(statistics.fmean(task_values))
        for task_id, task_values in sorted(by_task.items())
    }
    return {
        "workload_count": len(values),
        "mean_ms": float(statistics.fmean(values)),
        "median_ms": float(statistics.median(values)),
        "p95_ms": float(_percentile(values, 0.95)),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
        "task_macro_mean_ms": float(statistics.fmean(task_means.values())),
        "task_means_ms": task_means,
    }


def _paired_comparison(
    rows: Sequence[Mapping[str, Any]], fixed_k: int
) -> dict[str, Any]:
    legacy_id = f"legacy_fixed_k{fixed_k}"
    terminal_id = f"terminal_only_k{fixed_k}"
    by_identity: dict[tuple[int, ...], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        if row["condition_id"] not in {legacy_id, terminal_id}:
            continue
        by_identity.setdefault(_identity_key(row), {})[str(row["condition_id"])] = row
    pairs = [
        values
        for values in by_identity.values()
        if legacy_id in values and terminal_id in values
    ]
    if not pairs:
        raise GPUMicrobenchmarkValidationError(
            f"no paired rows available for fixed K={fixed_k}"
        )
    legacy = [float(pair[legacy_id]["latency_median_ms"]) for pair in pairs]
    terminal = [float(pair[terminal_id]["latency_median_ms"]) for pair in pairs]
    deltas = [terminal_value - legacy_value for legacy_value, terminal_value in zip(legacy, terminal)]
    improvements = [
        (legacy_value - terminal_value) / legacy_value
        for legacy_value, terminal_value in zip(legacy, terminal)
    ]
    return {
        "fixed_k": int(fixed_k),
        "paired_workload_count": len(pairs),
        "legacy_mean_ms": float(statistics.fmean(legacy)),
        "terminal_only_mean_ms": float(statistics.fmean(terminal)),
        "mean_delta_ms_terminal_minus_legacy": float(statistics.fmean(deltas)),
        "median_delta_ms_terminal_minus_legacy": float(statistics.median(deltas)),
        "p95_delta_ms_terminal_minus_legacy": float(_percentile(deltas, 0.95)),
        "mean_pairwise_improvement_fraction": float(statistics.fmean(improvements)),
        "ratio_of_means_improvement_fraction": float(
            1.0 - statistics.fmean(terminal) / statistics.fmean(legacy)
        ),
        "terminal_faster_pair_count": int(sum(delta < 0 for delta in deltas)),
        "equal_pair_count": int(sum(delta == 0 for delta in deltas)),
        "terminal_slower_pair_count": int(sum(delta > 0 for delta in deltas)),
    }


def _terminal_k4_k5_comparison(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    condition_ids = {str(row["condition_id"]) for row in rows}
    if not {"terminal_only_k4", "terminal_only_k5"}.issubset(condition_ids):
        return None
    by_identity: dict[tuple[int, ...], dict[str, float]] = {}
    for row in rows:
        condition_id = str(row["condition_id"])
        if condition_id not in {"terminal_only_k4", "terminal_only_k5"}:
            continue
        by_identity.setdefault(_identity_key(row), {})[condition_id] = float(
            row["latency_median_ms"]
        )
    pairs = [
        values
        for values in by_identity.values()
        if "terminal_only_k4" in values and "terminal_only_k5" in values
    ]
    deltas = [
        pair["terminal_only_k5"] - pair["terminal_only_k4"] for pair in pairs
    ]
    return {
        "paired_workload_count": len(pairs),
        "mean_delta_ms_k5_minus_k4": float(statistics.fmean(deltas)),
        "median_delta_ms_k5_minus_k4": float(statistics.median(deltas)),
        "p95_delta_ms_k5_minus_k4": float(_percentile(deltas, 0.95)),
        "interpretation": (
            "Both arms execute one terminal Coda; the paired difference estimates "
            "one additional recurrent iteration plus minor fixed-depth control overhead."
        ),
    }


def _build_summary(
    measurements: Sequence[Mapping[str, Any]],
    conditions: Sequence[FixedCodaCondition],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    workload_rows = _within_workload_medians(measurements)
    scopes: dict[str, Any] = {}
    for scope in ("actual_warm", "cold", "all"):
        scoped_rows = _scope_rows(workload_rows, scope)
        condition_summaries = {}
        for condition in conditions:
            condition_rows = [
                row
                for row in scoped_rows
                if row["condition_id"] == condition.condition_id
            ]
            if condition_rows:
                condition_summaries[condition.condition_id] = _condition_summary(
                    condition_rows
                )
        scopes[scope] = {
            "workload_count": len(
                {_identity_key(row) for row in scoped_rows}
            ),
            "conditions": condition_summaries,
            "paired_fixed_k": {
                str(fixed_k): _paired_comparison(scoped_rows, int(fixed_k))
                for fixed_k in protocol["fixed_depths"]
            },
            "terminal_only_k4_vs_k5": _terminal_k4_k5_comparison(scoped_rows),
        }
    return {
        "primary_scope": str(protocol["primary_scope"]),
        "aggregation": (
            "Median across repeats within each captured workload; descriptive "
            "mean, median, and p95 across workload medians. Paired comparisons "
            "use the same workload and selected initial latent state."
        ),
        "workload_medians": workload_rows,
        "scopes": scopes,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--initial-state-manifest",
        type=Path,
        default=DEFAULT_INITIAL_STATE_MANIFEST,
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-seed", type=int, default=7)
    parser.add_argument("--max-workloads", type=int)
    parser.add_argument("--measurement-repeats", type=int)
    parser.add_argument("--warmup-rounds", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite fixed-Coda microprofile: {args.output}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("fixed-Coda microprofile requires CUDA")

    protocol = load_json_object(args.protocol)
    if args.measurement_repeats is not None:
        protocol = copy.deepcopy(protocol)
        protocol["measurement_repeats"] = int(args.measurement_repeats)
    if args.warmup_rounds is not None:
        protocol = copy.deepcopy(protocol)
        protocol["warmup_rounds_per_origin"] = int(args.warmup_rounds)
    _validate_protocol(protocol)
    conditions = _conditions(protocol)

    calibration_validation = validate_calibration_run(
        str(args.run_root),
        str(args.initial_state_manifest),
        base_seed=args.base_seed,
    )
    if calibration_validation.get("complete_10_task_gate") is not True:
        raise GPUMicrobenchmarkValidationError(
            "fixed-Coda microprofile requires complete ten-task calibration"
        )
    descriptors = load_workload_descriptors(args.run_root)
    formal_workload_count = len(descriptors)
    if args.max_workloads is not None:
        if args.max_workloads < 2:
            raise ValueError("--max-workloads must be at least 2")
        descriptors = descriptors[: args.max_workloads]

    device = torch.device(protocol["device"])
    torch.manual_seed(int(protocol["seed"]))
    torch.cuda.manual_seed_all(int(protocol["seed"]))
    random.seed(int(protocol["seed"]))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    action_head, proprio_projector, checkpoint_inputs = load_benchmark_modules(
        args.checkpoint, device
    )

    representative: dict[str, Mapping[str, Any]] = {}
    for descriptor in descriptors:
        representative.setdefault(descriptor["actual_origin"], descriptor)
    if set(representative) != {"COLD", "ACTUAL_WARM"}:
        raise GPUMicrobenchmarkValidationError(
            "selected workloads must include both COLD and ACTUAL_WARM origins"
        )

    all_mismatches: list[dict[str, Any]] = []
    with torch.inference_mode():
        for warmup_index, origin in enumerate(("COLD", "ACTUAL_WARM")):
            _, mismatches = _run_workload(
                descriptor=representative[origin],
                block_index=warmup_index,
                action_head=action_head,
                proprio_projector=proprio_projector,
                conditions=conditions,
                device=device,
                repeats=int(protocol["warmup_rounds_per_origin"]),
                order_seed=int(protocol["seed"]),
                measured=False,
            )
            all_mismatches.extend(mismatches)

        measurements: list[dict[str, Any]] = []
        for block_index, descriptor in enumerate(descriptors):
            block_measurements, mismatches = _run_workload(
                descriptor=descriptor,
                block_index=block_index,
                action_head=action_head,
                proprio_projector=proprio_projector,
                conditions=conditions,
                device=device,
                repeats=int(protocol["measurement_repeats"]),
                order_seed=int(protocol["seed"]),
                measured=True,
            )
            measurements.extend(block_measurements)
            all_mismatches.extend(mismatches)
            if (block_index + 1) % 20 == 0 or block_index + 1 == len(descriptors):
                print(
                    f"Measured workloads: {block_index + 1}/{len(descriptors)}",
                    flush=True,
                )

    unique_mismatches: dict[tuple[Any, ...], dict[str, Any]] = {}
    for mismatch in all_mismatches:
        key = (
            mismatch["task_id"],
            mismatch["episode_id"],
            mismatch["prediction_step"],
            mismatch["condition_id"],
        )
        unique_mismatches.setdefault(key, mismatch)
    mismatch_records = list(unique_mismatches.values())

    is_formal = (
        len(descriptors) == int(protocol["expected_formal_workload_count"])
        and formal_workload_count
        == int(protocol["expected_formal_workload_count"])
        and args.max_workloads is None
        and args.measurement_repeats is None
        and args.warmup_rounds is None
    )
    origin_counts = {
        "COLD": sum(
            descriptor["actual_origin"] == "COLD" for descriptor in descriptors
        ),
        "ACTUAL_WARM": sum(
            descriptor["actual_origin"] == "ACTUAL_WARM"
            for descriptor in descriptors
        ),
    }
    if is_formal:
        expected_counts = {
            "COLD": int(protocol["expected_cold_workload_count"]),
            "ACTUAL_WARM": int(
                protocol["expected_actual_warm_workload_count"]
            ),
        }
        if origin_counts != expected_counts:
            raise GPUMicrobenchmarkValidationError(
                f"formal origin counts mismatch: {origin_counts} != {expected_counts}"
            )
    if mismatch_records:
        raise GPUMicrobenchmarkValidationError(
            f"fixed-Coda schedule mismatches detected: {len(mismatch_records)}"
        )

    summary = _build_summary(measurements, conditions, protocol)
    torch.cuda.synchronize(device)
    device_properties = torch.cuda.get_device_properties(device)
    report = {
        "schema_version": 1,
        "formal_run": is_formal,
        "code_git_commit": _git_commit(),
        "protocol": protocol,
        "conditions": [asdict(condition) for condition in conditions],
        "inputs": {
            "run_root": str(args.run_root.resolve()),
            "protocol_manifest": {
                "path": str(args.protocol.resolve()),
                "sha256": sha256_file(args.protocol),
            },
            "initial_state_manifest": {
                "path": str(args.initial_state_manifest.resolve()),
                "sha256": sha256_file(args.initial_state_manifest),
            },
            "checkpoint": checkpoint_inputs,
        },
        "environment": {
            "torch_version": torch.__version__,
            "cuda_runtime_version": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "device_total_memory": int(device_properties.total_memory),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        },
        "calibration_validation": calibration_validation,
        "workloads": {
            "formal_available": formal_workload_count,
            "measured": len(descriptors),
            "cold": origin_counts["COLD"],
            "actual_warm": origin_counts["ACTUAL_WARM"],
        },
        "schedule_mismatches": mismatch_records,
        "measurements": measurements,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    primary = summary["scopes"][summary["primary_scope"]]
    print(f"Schedule mismatches: {len(mismatch_records)}")
    for fixed_k in protocol["fixed_depths"]:
        comparison = primary["paired_fixed_k"][str(fixed_k)]
        print(
            f"K={fixed_k}: legacy={comparison['legacy_mean_ms']:.4f} ms, "
            f"terminal-only={comparison['terminal_only_mean_ms']:.4f} ms, "
            f"ratio-of-means improvement="
            f"{100.0 * comparison['ratio_of_means_improvement_fraction']:.3f}%"
        )
    recurrent_delta = primary.get("terminal_only_k4_vs_k5")
    if recurrent_delta is not None:
        print(
            "Terminal-only K5-K4 mean delta: "
            f"{recurrent_delta['mean_delta_ms_k5_minus_k4']:.4f} ms"
        )
    print(f"Formal run: {is_formal}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
