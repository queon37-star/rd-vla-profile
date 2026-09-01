#!/usr/bin/env python3
"""Replay the complete paper prediction distribution through ActionHeadRecurrent.

This benchmark replaces the old prediction-step-1 Action-head microbenchmark.
It uses every captured policy-query workload from the two real source
trajectories:

    baseline source   -> Baseline, LDCE
    warm-start source -> Warm-start, Combined

No counterfactual observation distribution is constructed.  Baseline and LDCE
are paired on the exact baseline trajectory workloads; Warm-start and Combined
are paired on the exact warm-start trajectory workloads, including the one cold
first prediction in each episode.

Timing scope
------------
For each workload, disk I/O, selected-layer expansion, CPU->GPU transfer,
projector validation, and predictor binding occur before timing.  The measured
region is exactly:

    CUDA synchronize
    perf_counter_ns
    ActionHeadRecurrent.predict_action(...)
    CUDA synchronize

The default protocol runs five balanced repeats per condition and uses the
within-workload median.  The primary paper metric is the prediction-weighted
mean over all workload medians, matching the weighting of the existing online
policy-query latency.  Task-macro and episode-balanced means are also reported.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import prismatic.models.action_heads as action_heads_module  # noqa: E402
import scripts.run_spatial_paper_4arm as frozen  # noqa: E402
from prismatic.models.action_delta_gate import load_action_delta_gate_artifact  # noqa: E402
from scripts.origin_aware_gpu_microbenchmark_lib import (  # noqa: E402
    balanced_condition_order,
)
from scripts.paper_action_head_full_distribution_lib import (  # noqa: E402
    COMPACT_WORKLOAD_TYPE,
    append_jsonl,
    expand_compact_tensors,
    load_compact_workload,
    load_jsonl,
    sha256_file,
    validate_frozen_spatial_action_head,
)
from scripts.run_origin_aware_gpu_microbenchmark import (  # noqa: E402
    captured_cold_initial_state,
    load_benchmark_modules,
)


SOURCE_TO_ARMS = {
    "baseline": ("baseline", "ldce"),
    "warm_start": ("warm_start", "combined"),
}
ARMS = ("baseline", "warm_start", "ldce", "combined")
PROTOCOL_NAME = "paper-action-head-full-distribution-replay-v1"
DEFAULT_CAPTURE_ROOT = (
    REPO_ROOT / "benchmark_results/paper_action_head_full_distribution"
)
DEFAULT_OUTPUT = (
    DEFAULT_CAPTURE_ROOT / "replay" / "report.json"
)
DEFAULT_ARTIFACT = (
    REPO_ROOT
    / "benchmark_results/coda_anchor_feasibility/action_delta_gate_fold4"
)
DEFAULT_CHECKPOINT = REPO_ROOT / "outputs/12_24-24_24_Spatial_40k"
DEFAULT_REPEATS = 5
DEFAULT_WARMUP_ROUNDS = 2
DEFAULT_ORDER_SEED = 20260901


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _prediction_identity(record: Mapping[str, Any]) -> dict[str, int]:
    return {
        "task_id": int(record["task_id"]),
        "episode_id": int(record["episode_id"]),
        "paired_trial_id": int(record["paired_trial_id"]),
        "prediction_step": int(record["prediction_step"]),
        "initial_state_id": int(record["initial_state_id"]),
        "episode_seed": int(record["episode_seed"]),
    }


def _descriptor_key(descriptor: Mapping[str, Any]) -> tuple[str, int, int, int]:
    identity = descriptor["identity"]
    return (
        str(descriptor["source_arm"]),
        int(identity["task_id"]),
        int(identity["paired_trial_id"]),
        int(identity["prediction_step"]),
    )


def _measurement_block_key(record: Mapping[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(record["source_arm"]),
        int(record["task_id"]),
        int(record["paired_trial_id"]),
        int(record["prediction_step"]),
    )


def _load_capture_descriptors(capture_root: Path) -> list[dict[str, Any]]:
    descriptors: list[dict[str, Any]] = []
    for source_arm in SOURCE_TO_ARMS:
        arm_summary_path = capture_root / "capture" / source_arm / "capture_summary.json"
        _require(arm_summary_path.is_file(), f"missing capture summary: {arm_summary_path}")
        arm_summary = json.loads(arm_summary_path.read_text(encoding="utf-8"))
        _require(arm_summary.get("completed") is True, f"capture is incomplete for {source_arm}")
        expected_count = int(arm_summary["prediction_count"])

        arm_count = 0
        for task_id in frozen.PAPER_TASK_IDS:
            task_dir = capture_root / "capture" / source_arm / f"task{task_id}"
            task_summary_path = task_dir / "capture_summary.json"
            step_path = task_dir / "predictions.jsonl"
            _require(task_summary_path.is_file(), f"missing task capture summary: {task_summary_path}")
            _require(step_path.is_file(), f"missing task predictions: {step_path}")
            task_summary = json.loads(task_summary_path.read_text(encoding="utf-8"))
            _require(task_summary.get("completed") is True, f"capture incomplete: {source_arm}/task{task_id}")
            task_count = 0
            for record in load_jsonl(step_path):
                _require(record.get("action_head_workload_captured") is True, "prediction is missing compact workload")
                _require(record.get("action_head_workload_type") == COMPACT_WORKLOAD_TYPE, "unexpected workload type")
                identity = _prediction_identity(record)
                _require(identity["task_id"] == task_id, "task identity mismatch")
                workload_path = Path(record["action_head_workload_file"])
                if not workload_path.is_absolute():
                    workload_path = step_path.parent / workload_path
                captured_origin = str(record.get("actual_origin"))
                expected_origin = "ACTUAL_WARM" if bool(record.get("warm_start_used", False)) else "COLD"
                _require(captured_origin == expected_origin, "step-log warm/origin mismatch")
                descriptors.append(
                    {
                        "source_arm": source_arm,
                        "identity": identity,
                        "captured_origin": captured_origin,
                        "source_K_t": int(record["K_t"]),
                        "source_final_mse": (
                            None if record.get("final_mse") is None else float(record["final_mse"])
                        ),
                        "source_success": bool(record["success"]),
                        "warm_start_used": bool(record.get("warm_start_used", False)),
                        "path": workload_path.resolve(),
                        "sha256": str(record["action_head_workload_sha256"]),
                    }
                )
                task_count += 1
            _require(
                task_count == int(task_summary["prediction_count"]),
                f"{source_arm}/task{task_id}: step/capture count mismatch",
            )
            arm_count += task_count
        _require(arm_count == expected_count, f"{source_arm}: capture count mismatch")

    descriptors.sort(key=_descriptor_key)
    _require(len({_descriptor_key(item) for item in descriptors}) == len(descriptors), "duplicate workload identities")
    return descriptors


def _bind_shared_predictor(action_head, artifact_payload, *, task_id: int, device) -> None:
    action_head.clear_scalar_task_policy()
    prepared = frozen._prepare_shared_spatial_predictor(
        artifact_payload,
        device=device,
        task_id=int(task_id),
    )
    action_head.configure_action_delta_gate(prepared, deferred_scorer_backend="eager")


@contextmanager
def _deferred_validator_without_inner_coda_profiling():
    original = action_heads_module.validate_action_delta_deferred_backfill_configuration

    def wrapper(**kwargs):
        forwarded = dict(kwargs)
        forwarded["profile_coda_cost"] = True
        return original(**forwarded)

    action_heads_module.validate_action_delta_deferred_backfill_configuration = wrapper
    try:
        yield
    finally:
        action_heads_module.validate_action_delta_deferred_backfill_configuration = original


def _validate_projector(proprio_projector, tensors, device: torch.device) -> None:
    projected = proprio_projector(
        tensors["proprio_input"].reshape(1, -1).to(torch.bfloat16)
    ).unsqueeze(1)
    torch.cuda.synchronize(device)
    expected = tensors["proprio_features"]
    _require(expected is not None, "captured proprio feature is missing")
    if not torch.equal(projected, expected):
        max_error = float(torch.max(torch.abs(projected.float() - expected.float())).item())
        raise RuntimeError(
            f"captured proprio feature mismatch; max absolute error={max_error}"
        )


def _condition_kwargs(
    arm: str,
    *,
    captured_origin: str,
    incoming_warm_state: torch.Tensor | None,
) -> dict[str, Any]:
    _require(arm in ARMS, f"unknown arm: {arm}")
    use_warm = arm in {"warm_start", "combined"}
    use_ldce = arm in {"ldce", "combined"}
    if use_warm:
        if captured_origin == "ACTUAL_WARM":
            _require(incoming_warm_state is not None, f"{arm}: warm workload cache is missing")
            warm_state = incoming_warm_state
        else:
            _require(incoming_warm_state is None, f"{arm}: cold first prediction unexpectedly has a cache")
            warm_state = None
    else:
        _require(captured_origin == "COLD", f"{arm}: cold-source replay received warm-origin workload")
        _require(incoming_warm_state is None, f"{arm}: baseline source unexpectedly contains warm cache")
        warm_state = None

    return {
        "phase": "Inference",
        "num_iter": None,
        "convergence_strategy": "adjacent_action_mse",
        "kl_thresh": 0.001,
        "cos_thresh": 0.999,
        "max_iter": 32,
        "warm_start_state": warm_state,
        "enable_warm_start": use_warm,
        "warm_start_source": "midpoint",
        "warm_start_min_iter": 2,
        "validate_warm_start_finite": use_warm,
        "profile_coda_cost": False,
        "use_cached_final_output": True,
        "use_latent_precheck": False,
        "latent_precheck_mode": "off",
        "latent_precheck_trace_level": "off",
        "latent_precheck_min_iter": 2,
        "latent_precheck_force_interval": 0,
        "shadow_full_depth": False,
        "collect_preconvergence_raw_shadow": False,
        "capture_action_head_workload": False,
        "use_action_delta_gate": False,
        "collect_action_delta_gate_shadow": False,
        "use_action_delta_nonconvergence_filter": False,
        "use_action_delta_deferred_backfill_filter": use_ldce,
        "action_delta_gate_max_skip": 1,
        "action_delta_gate_min_terminal_iter": 2,
        "action_delta_gate_exact_coda_audit": False,
        "action_delta_gate_return_mode": "anchor",
        "action_delta_deferred_scorer_backend": "eager",
        "action_delta_deferred_runtime_policy": "lazy_prefix_exact",
        "action_delta_deferred_apply_to_cold": arm == "ldce",
    }


def _schedule(action_head, arm: str, K_t: int) -> dict[str, Any]:
    debug = action_head.model.last_recurrence_debug
    _require(isinstance(debug, Mapping), "action head did not publish recurrence debug metadata")
    use_ldce = arm in {"ldce", "combined"}
    if use_ldce:
        coda_calls = int(
            debug["action_delta_deferred_backfill_filter_total_exact_coda_call_count"]
        )
        eliminated = int(
            debug["action_delta_deferred_backfill_filter_truly_eliminated_coda_call_count"]
        )
        score_calls = int(
            debug["action_delta_deferred_backfill_filter_score_call_count"]
        )
        ldce_applied = bool(debug["action_delta_deferred_backfill_filter_applied"])
    else:
        coda_calls = int(K_t)
        eliminated = 0
        score_calls = 0
        ldce_applied = False
    return {
        "K_t": int(K_t),
        "coda_calls": coda_calls,
        "eliminated_coda_calls": eliminated,
        "ldce_score_calls": score_calls,
        "ldce_applied": ldce_applied,
        "actual_origin": "ACTUAL_WARM" if debug.get("warm_start_state_used") else "COLD",
        "numerical_retry_attempted": bool(debug.get("numerical_retry_attempted", False)),
    }


def _execute(
    *,
    action_head,
    proprio_projector,
    tensors,
    arm: str,
    captured_origin: str,
):
    kwargs = _condition_kwargs(
        arm,
        captured_origin=captured_origin,
        incoming_warm_state=tensors["incoming_warm_start_state"],
    )
    selected_state = tensors["selected_initial_state"]
    _require(selected_state is not None, "selected initial state is missing")
    with captured_cold_initial_state(
        action_head.model,
        selected_state,
        captured_origin,
    ):
        output, K_t, final_score = action_head.predict_action(
            tensors["actions_hidden_states"],
            proprio=tensors["proprio_input"],
            proprio_projector=proprio_projector,
            **kwargs,
        )
    return output, int(K_t), final_score, _schedule(action_head, arm, int(K_t))


def _validate_schedule(
    schedule: Mapping[str, Any],
    *,
    arm: str,
    captured_origin: str,
) -> None:
    _require(not schedule["numerical_retry_attempted"], f"{arm}: numerical retry occurred")
    _require(schedule["actual_origin"] == captured_origin, f"{arm}: replay origin mismatch")
    if arm == "ldce":
        _require(schedule["ldce_applied"], "LDCE must apply on the baseline cold source")
    elif arm == "combined":
        expected = captured_origin == "ACTUAL_WARM"
        _require(
            bool(schedule["ldce_applied"]) == expected,
            "Combined LDCE application must match warm-origin availability",
        )
    else:
        _require(not schedule["ldce_applied"], f"{arm}: LDCE unexpectedly applied")


def _timed_execute(
    *,
    action_head,
    proprio_projector,
    tensors,
    arm: str,
    captured_origin: str,
    device: torch.device,
):
    torch.cuda.synchronize(device)
    start_ns = time.perf_counter_ns()
    output, K_t, final_score, schedule = _execute(
        action_head=action_head,
        proprio_projector=proprio_projector,
        tensors=tensors,
        arm=arm,
        captured_origin=captured_origin,
    )
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
    _require(math.isfinite(elapsed_ms) and elapsed_ms > 0.0, "latency must be finite and positive")
    _require(bool(torch.isfinite(output).all().item()), "Action-head output is non-finite")
    _validate_schedule(schedule, arm=arm, captured_origin=captured_origin)
    return elapsed_ms, K_t, final_score, schedule


def _run_workload(
    *,
    descriptor: Mapping[str, Any],
    block_index: int,
    action_head,
    proprio_projector,
    device: torch.device,
    required_layers: Sequence[int],
    repeats: int,
    order_seed: int,
    measured: bool,
) -> list[dict[str, Any]]:
    source_arm = str(descriptor["source_arm"])
    conditions = SOURCE_TO_ARMS[source_arm]
    payload = load_compact_workload(
        descriptor["path"],
        expected_sha256=descriptor["sha256"],
        expected_identity=descriptor["identity"],
        expected_source_arm=source_arm,
        expected_origin=descriptor["captured_origin"],
    )
    tensors = expand_compact_tensors(
        payload,
        device=device,
        expected_layer_indices=required_layers,
    )
    _validate_projector(proprio_projector, tensors, device)

    measurements: list[dict[str, Any]] = []
    schedules: dict[str, Mapping[str, Any]] = {}
    source_condition = source_arm
    for repeat_index in range(repeats):
        order = balanced_condition_order(
            conditions,
            block_index=block_index * repeats,
            repeat_index=repeat_index,
            seed=order_seed,
        )
        for order_position, arm in enumerate(order):
            if measured:
                latency_ms, K_t, final_score, schedule = _timed_execute(
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    tensors=tensors,
                    arm=arm,
                    captured_origin=descriptor["captured_origin"],
                    device=device,
                )
            else:
                _, K_t, final_score, schedule = _execute(
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    tensors=tensors,
                    arm=arm,
                    captured_origin=descriptor["captured_origin"],
                )
                torch.cuda.synchronize(device)
                _validate_schedule(
                    schedule,
                    arm=arm,
                    captured_origin=descriptor["captured_origin"],
                )
                latency_ms = None

            prior = schedules.setdefault(arm, dict(schedule))
            _require(prior == schedule, f"{arm}: replay schedule changed across repeats")
            if arm == source_condition:
                _require(
                    int(K_t) == int(descriptor["source_K_t"]),
                    f"{source_arm}: source K replay mismatch for {descriptor['identity']}: "
                    f"captured={descriptor['source_K_t']}, replay={K_t}",
                )
            if measured:
                measurements.append(
                    {
                        "protocol": PROTOCOL_NAME,
                        "source_arm": source_arm,
                        **descriptor["identity"],
                        "captured_origin": descriptor["captured_origin"],
                        "source_K_t": int(descriptor["source_K_t"]),
                        "condition_id": arm,
                        "repeat_index": int(repeat_index),
                        "order_position": int(order_position),
                        "latency_ms": float(latency_ms),
                        "K_t": int(K_t),
                        "final_score": None if final_score is None else float(final_score),
                        "coda_calls": int(schedule["coda_calls"]),
                        "eliminated_coda_calls": int(schedule["eliminated_coda_calls"]),
                        "ldce_score_calls": int(schedule["ldce_score_calls"]),
                        "ldce_applied": bool(schedule["ldce_applied"]),
                        "actual_origin": schedule["actual_origin"],
                    }
                )

    del tensors
    del payload
    return measurements


def _validate_resume_measurements(
    records: Sequence[Mapping[str, Any]],
    *,
    repeats: int,
) -> set[tuple[str, int, int, int]]:
    grouped: dict[tuple[str, int, int, int], list[Mapping[str, Any]]] = {}
    for row in records:
        grouped.setdefault(_measurement_block_key(row), []).append(row)
    complete: set[tuple[str, int, int, int]] = set()
    for key, rows in grouped.items():
        source_arm = key[0]
        expected_conditions = SOURCE_TO_ARMS[source_arm]
        expected_rows = len(expected_conditions) * repeats
        _require(
            len(rows) == expected_rows,
            f"resume file contains a partial workload block {key}: {len(rows)}/{expected_rows}. "
            "Remove measurements.jsonl and restart replay.",
        )
        for arm in expected_conditions:
            arm_rows = [row for row in rows if row["condition_id"] == arm]
            _require(len(arm_rows) == repeats, f"resume block {key}/{arm} repeat mismatch")
            _require(
                sorted(int(row["repeat_index"]) for row in arm_rows) == list(range(repeats)),
                f"resume block {key}/{arm} repeat indices are incomplete",
            )
        complete.add(key)
    return complete


def _workload_medians(
    measurements: Sequence[Mapping[str, Any]],
    *,
    repeats: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int, int, str], list[Mapping[str, Any]]] = {}
    for row in measurements:
        key = (
            str(row["source_arm"]),
            int(row["task_id"]),
            int(row["paired_trial_id"]),
            int(row["prediction_step"]),
            str(row["condition_id"]),
        )
        grouped.setdefault(key, []).append(row)

    medians: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        _require(len(rows) == repeats, f"{key}: expected {repeats} repeats, got {len(rows)}")
        schedule_fields = (
            "K_t",
            "coda_calls",
            "eliminated_coda_calls",
            "ldce_score_calls",
            "ldce_applied",
            "actual_origin",
        )
        for field in schedule_fields:
            _require(
                all(row[field] == rows[0][field] for row in rows),
                f"{key}: {field} changed across repeats",
            )
        values = np.asarray([float(row["latency_ms"]) for row in rows], dtype=np.float64)
        medians.append(
            {
                "source_arm": key[0],
                "task_id": key[1],
                "episode_id": int(rows[0]["episode_id"]),
                "paired_trial_id": key[2],
                "prediction_step": key[3],
                "initial_state_id": int(rows[0]["initial_state_id"]),
                "episode_seed": int(rows[0]["episode_seed"]),
                "condition_id": key[4],
                "median_latency_ms": float(np.median(values)),
                "K_t": int(rows[0]["K_t"]),
                "coda_calls": int(rows[0]["coda_calls"]),
                "eliminated_coda_calls": int(rows[0]["eliminated_coda_calls"]),
                "ldce_score_calls": int(rows[0]["ldce_score_calls"]),
                "ldce_applied": bool(rows[0]["ldce_applied"]),
                "actual_origin": str(rows[0]["actual_origin"]),
            }
        )
    return medians


def _arm_aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    latency = np.asarray([float(row["median_latency_ms"]) for row in rows], dtype=np.float64)
    k_values = np.asarray([float(row["K_t"]) for row in rows], dtype=np.float64)
    coda = np.asarray([float(row["coda_calls"]) for row in rows], dtype=np.float64)
    eliminated = np.asarray([float(row["eliminated_coda_calls"]) for row in rows], dtype=np.float64)

    per_task: dict[str, Any] = {}
    task_means = []
    for task_id in frozen.PAPER_TASK_IDS:
        task_rows = [row for row in rows if int(row["task_id"]) == task_id]
        _require(bool(task_rows), f"task{task_id}: no replay rows")
        values = np.asarray([float(row["median_latency_ms"]) for row in task_rows], dtype=np.float64)
        task_means.append(float(values.mean()))
        per_task[str(task_id)] = {
            "prediction_count": len(task_rows),
            "mean_latency_ms": float(values.mean()),
            "median_latency_ms": float(np.median(values)),
            "p95_latency_ms": float(np.percentile(values, 95)),
            "mean_K": float(np.mean([float(row["K_t"]) for row in task_rows])),
            "mean_coda_calls": float(np.mean([float(row["coda_calls"]) for row in task_rows])),
        }

    episode_groups: dict[tuple[int, int], list[float]] = {}
    for row in rows:
        episode_groups.setdefault(
            (int(row["task_id"]), int(row["paired_trial_id"])), []
        ).append(float(row["median_latency_ms"]))
    _require(len(episode_groups) == 500, f"expected 500 source episodes, got {len(episode_groups)}")
    episode_means = np.asarray(
        [float(np.mean(values)) for values in episode_groups.values()], dtype=np.float64
    )

    return {
        "prediction_count": len(rows),
        "primary_prediction_weighted_mean_latency_ms": float(latency.mean()),
        "prediction_weighted_median_latency_ms": float(np.median(latency)),
        "prediction_weighted_p95_latency_ms": float(np.percentile(latency, 95)),
        "task_macro_mean_latency_ms": float(np.mean(task_means)),
        "episode_balanced_mean_latency_ms": float(episode_means.mean()),
        "mean_K": float(k_values.mean()),
        "mean_coda_calls": float(coda.mean()),
        "mean_eliminated_coda_calls": float(eliminated.mean()),
        "total_coda_calls": int(coda.sum()),
        "total_eliminated_coda_calls": int(eliminated.sum()),
        "per_task": per_task,
    }


def _paired_delta(
    medians: Sequence[Mapping[str, Any]],
    *,
    source_arm: str,
    reference_arm: str,
    variant_arm: str,
) -> dict[str, Any]:
    lookup: dict[tuple[int, int, int, str], float] = {}
    for row in medians:
        if row["source_arm"] != source_arm:
            continue
        key = (
            int(row["task_id"]),
            int(row["paired_trial_id"]),
            int(row["prediction_step"]),
            str(row["condition_id"]),
        )
        lookup[key] = float(row["median_latency_ms"])
    identities = sorted(
        {
            (task_id, trial_id, prediction_step)
            for task_id, trial_id, prediction_step, arm in lookup
            if arm == reference_arm
        }
    )
    deltas = []
    refs = []
    variants = []
    for task_id, trial_id, prediction_step in identities:
        ref = lookup[(task_id, trial_id, prediction_step, reference_arm)]
        variant = lookup[(task_id, trial_id, prediction_step, variant_arm)]
        refs.append(ref)
        variants.append(variant)
        deltas.append(variant - ref)
    ref_mean = float(np.mean(refs))
    variant_mean = float(np.mean(variants))
    return {
        "source_arm": source_arm,
        "reference_arm": reference_arm,
        "variant_arm": variant_arm,
        "paired_prediction_count": len(deltas),
        "mean_reference_latency_ms": ref_mean,
        "mean_variant_latency_ms": variant_mean,
        "mean_paired_delta_ms": float(np.mean(deltas)),
        "median_paired_delta_ms": float(np.median(deltas)),
        "variant_speedup_percent_from_aggregate_means": (
            100.0 * (ref_mean - variant_mean) / ref_mean
        ),
        "variant_faster_fraction": float(np.mean(np.asarray(deltas) < 0.0)),
    }


def _aggregate(medians: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for arm in ARMS:
        rows = [row for row in medians if row["condition_id"] == arm]
        _require(bool(rows), f"no medians for {arm}")
        arms[arm] = _arm_aggregate(rows)

    baseline_mean = float(arms["baseline"]["primary_prediction_weighted_mean_latency_ms"])
    for arm in ARMS:
        value = float(arms[arm]["primary_prediction_weighted_mean_latency_ms"])
        arms[arm]["relative_reduction_vs_baseline_percent"] = (
            100.0 * (baseline_mean - value) / baseline_mean
        )
        arms[arm]["comparison_to_baseline_is_same_workload_paired"] = arm in {"baseline", "ldce"}

    return {
        "primary_metric": "prediction-weighted mean of within-workload median Action-head latency",
        "arms": arms,
        "paired_effects": {
            "baseline_to_ldce": _paired_delta(
                medians,
                source_arm="baseline",
                reference_arm="baseline",
                variant_arm="ldce",
            ),
            "warm_start_to_combined": _paired_delta(
                medians,
                source_arm="warm_start",
                reference_arm="warm_start",
                variant_arm="combined",
            ),
        },
        "cross_distribution_note": (
            "Baseline/LDCE use baseline-source prediction distribution; "
            "Warm-start/Combined use warm-start-source prediction distribution. "
            "Absolute arm means represent each method family's observed source distribution; "
            "only Baseline-vs-LDCE and Warm-start-vs-Combined are same-workload paired."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--action-delta-artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--action-delta-sha256", default="")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--measurement-repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--warmup-rounds", type=int, default=DEFAULT_WARMUP_ROUNDS)
    parser.add_argument("--order-seed", type=int, default=DEFAULT_ORDER_SEED)
    parser.add_argument("--max-workloads", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    capture_root = args.capture_root.resolve()
    checkpoint = args.checkpoint.resolve()
    artifact_path = args.action_delta_artifact.resolve()
    output = args.output.resolve()
    measurement_path = output.parent / "measurements.jsonl"
    medians_path = output.parent / "workload_medians.jsonl"

    _require(args.measurement_repeats >= 3, "--measurement-repeats must be >= 3")
    _require(args.warmup_rounds >= 1, "--warmup-rounds must be >= 1")
    _require(checkpoint.exists(), f"checkpoint does not exist: {checkpoint}")
    _require(artifact_path.exists(), f"Action-Delta artifact does not exist: {artifact_path}")
    _require(capture_root.exists(), f"capture root does not exist: {capture_root}")

    descriptors = _load_capture_descriptors(capture_root)
    formal_descriptor_count = len(descriptors)
    if args.max_workloads is not None:
        _require(args.max_workloads >= 1, "--max-workloads must be >= 1")
        descriptors = descriptors[: int(args.max_workloads)]
    _require(bool(descriptors), "no captured workloads found")

    artifact_sha = frozen._artifact_sha256(
        artifact_path,
        args.action_delta_sha256 or None,
    )
    artifact_manifest, artifact_payload = load_action_delta_gate_artifact(
        artifact_path,
        expected_sha256=artifact_sha,
    )

    if args.dry_run:
        by_source = {
            source: sum(item["source_arm"] == source for item in descriptors)
            for source in SOURCE_TO_ARMS
        }
        print(
            json.dumps(
                {
                    "protocol": PROTOCOL_NAME,
                    "workloads": len(descriptors),
                    "formal_workloads": formal_descriptor_count,
                    "by_source": by_source,
                    "measurement_repeats": args.measurement_repeats,
                    "timed_calls": len(descriptors) * 2 * args.measurement_repeats,
                    "primary_metric": "prediction-weighted mean of workload medians",
                    "artifact_sha256": artifact_sha,
                },
                indent=2,
            )
        )
        return 0

    _require(torch.cuda.is_available(), "full-distribution Action-head replay requires CUDA")
    if output.exists() and not (args.overwrite or args.resume):
        raise FileExistsError(f"report already exists: {output}")
    if args.overwrite and not args.resume:
        for path in (output, measurement_path, medians_path):
            if path.exists():
                path.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)

    existing_measurements: list[dict[str, Any]] = []
    completed_blocks: set[tuple[str, int, int, int]] = set()
    if args.resume and measurement_path.is_file():
        existing_measurements = load_jsonl(measurement_path)
        completed_blocks = _validate_resume_measurements(
            existing_measurements,
            repeats=args.measurement_repeats,
        )
        print(f"[resume] completed workload blocks={len(completed_blocks)}", flush=True)

    device = torch.device("cuda:0")
    torch.manual_seed(args.order_seed)
    torch.cuda.manual_seed_all(args.order_seed)
    random.seed(args.order_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    action_head, proprio_projector, checkpoint_inputs = load_benchmark_modules(
        checkpoint, device
    )
    required_layers = validate_frozen_spatial_action_head(action_head)

    # Bind once per task. The same frozen Spatial predictor weights are used for
    # all tasks; task_id only chooses the existing validated materialization path.
    bound_task_id = None
    measured_new = 0
    with torch.inference_mode(), _deferred_validator_without_inner_coda_profiling():
        # Warm the GPU with one representative workload from each source family.
        if not existing_measurements:
            for warmup_index, source_arm in enumerate(SOURCE_TO_ARMS):
                representative = next(item for item in descriptors if item["source_arm"] == source_arm)
                task_id = int(representative["identity"]["task_id"])
                _bind_shared_predictor(
                    action_head,
                    artifact_payload,
                    task_id=task_id,
                    device=device,
                )
                _run_workload(
                    descriptor=representative,
                    block_index=warmup_index,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    device=device,
                    required_layers=required_layers,
                    repeats=args.warmup_rounds,
                    order_seed=args.order_seed,
                    measured=False,
                )
            bound_task_id = None

        for block_index, descriptor in enumerate(descriptors):
            key = _descriptor_key(descriptor)
            if key in completed_blocks:
                continue
            task_id = int(descriptor["identity"]["task_id"])
            if bound_task_id != task_id:
                _bind_shared_predictor(
                    action_head,
                    artifact_payload,
                    task_id=task_id,
                    device=device,
                )
                bound_task_id = task_id
            block_rows = _run_workload(
                descriptor=descriptor,
                block_index=block_index,
                action_head=action_head,
                proprio_projector=proprio_projector,
                device=device,
                required_layers=required_layers,
                repeats=args.measurement_repeats,
                order_seed=args.order_seed,
                measured=True,
            )
            # Append only after the complete balanced workload block finishes so
            # --resume never intentionally creates partial blocks.
            append_jsonl(measurement_path, block_rows)
            existing_measurements.extend(block_rows)
            measured_new += 1
            if measured_new % 100 == 0 or measured_new == len(descriptors) - len(completed_blocks):
                print(
                    f"Measured new workloads: {measured_new}/"
                    f"{len(descriptors) - len(completed_blocks)}",
                    flush=True,
                )

    torch.cuda.synchronize(device)
    complete_blocks = _validate_resume_measurements(
        existing_measurements,
        repeats=args.measurement_repeats,
    )
    expected_keys = {_descriptor_key(item) for item in descriptors}
    _require(complete_blocks == expected_keys, "final replay workload set does not match capture set")

    medians = _workload_medians(
        existing_measurements,
        repeats=args.measurement_repeats,
    )
    if medians_path.exists():
        medians_path.unlink()
    append_jsonl(medians_path, medians)
    summary = _aggregate(medians)

    device_properties = torch.cuda.get_device_properties(device)
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL_NAME,
        "formal_run": (
            len(descriptors) == formal_descriptor_count
            and args.max_workloads is None
            and args.measurement_repeats == DEFAULT_REPEATS
        ),
        "code_git_commit": _git_commit(),
        "inputs": {
            "capture_root": str(capture_root),
            "capture_summary_sha256": sha256_file(capture_root / "capture_summary.json"),
            "checkpoint": checkpoint_inputs,
            "action_delta_artifact": {
                "path": str(artifact_path),
                "sha256": artifact_sha,
                "model_type": artifact_manifest.get("model_type"),
                "weights_shared_across_spatial_tasks": True,
            },
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
        "protocol_details": {
            "source_to_arms": {key: list(value) for key, value in SOURCE_TO_ARMS.items()},
            "selected_vlm_layers": list(required_layers),
            "measurement_repeats": int(args.measurement_repeats),
            "warmup_rounds": int(args.warmup_rounds),
            "workload_count": len(descriptors),
            "timed_call_count": len(existing_measurements),
            "timer": (
                "CPU perf_counter_ns around ActionHeadRecurrent.predict_action "
                "with one outer CUDA synchronization before and after"
            ),
            "disk_io_inside_timer": False,
            "tensor_transfer_inside_timer": False,
            "selected_layer_expansion_inside_timer": False,
            "projector_validation_inside_timer": False,
            "VLM_inside_timer": False,
            "environment_inside_timer": False,
            "profile_coda_cost": False,
            "within_workload_estimator": "median of 5 repeats",
            "primary_aggregation": "prediction-weighted mean over workload medians",
            "source_condition_K_exact_replay_required": True,
            "old_prediction_step_1_microbenchmark_used": False,
        },
        "summary": summary,
        "measurement_file": str(measurement_path),
        "workload_medians_file": str(medians_path),
    }
    _write_json(output, report)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote report: {output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
