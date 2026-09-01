#!/usr/bin/env python3
"""Paired saved-workload ActionHeadRecurrent microbenchmark for the paper 4-arm study.

This runner follows the formal GPU replay protocol already used by the project:

* reuse frozen post-VLM action-head workloads captured during formal seed-7 calibration;
* move and validate tensors before timing;
* measure CPU wall-clock around ActionHeadRecurrent.predict_action with one CUDA
  synchronization immediately before and after the call;
* run every condition five times in a deterministic balanced complete-block order;
* take the median within workload/condition, then mean by episode/task, and use
  an equal-weight macro mean across the ten LIBERO-Spatial tasks.

Primary paired scope
--------------------
The formal calibration set contains one ACTUAL_WARM workload per task/episode
(prediction step 1), giving 100 primary workloads total. All four paper arms are
replayed on exactly these same post-VLM hidden states and proprio inputs:

    baseline    : cold initialization, no LDCE
    warm_start  : captured incoming midpoint cache, no LDCE
    ldce        : same cold initialization as baseline + frozen LDCE
    combined    : same captured midpoint cache as warm_start + frozen LDCE

Because an ACTUAL_WARM shard stores the accepted warm cache as its selected
initial state, it does not contain the counterfactual cold random state for that
same observation. For paired four-arm timing, this runner therefore creates one
deterministic production-distributed cold state per workload *before timing*.
During each cold-arm timed call it still executes the real production cold
initialization for its cost and then substitutes that frozen state, matching the
project's existing cold-replay technique. Baseline and LDCE therefore see the
same cold state on every repeat; Warm and Combined see the same captured warm
state. This makes the four latency conditions directly paired on the same
post-VLM workload while retaining the real cold-init cost.

This benchmark is for component latency only. Closed-loop success, K, and the
online get_action latency remain sourced from the completed 2,000-episode paper
rollout. No VLM or environment execution occurs here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import prismatic.models.action_heads as action_heads_module  # noqa: E402
import scripts.run_spatial_paper_4arm as frozen  # noqa: E402
from prismatic.models.action_delta_gate import load_action_delta_gate_artifact  # noqa: E402
from prismatic.models.action_head_workload import load_action_head_workload  # noqa: E402
from scripts.origin_aware_calibration_lib import validate_calibration_run  # noqa: E402
from scripts.origin_aware_gpu_microbenchmark_lib import (  # noqa: E402
    GPUMicrobenchmarkValidationError,
    balanced_condition_order,
    sha256_file,
)
from scripts.run_origin_aware_gpu_microbenchmark import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_INITIAL_STATE_MANIFEST,
    DEFAULT_RUN_ROOT,
    captured_cold_initial_state,
    load_benchmark_modules,
    load_workload_descriptors,
)


ARMS = ("baseline", "warm_start", "ldce", "combined")
DEFAULT_ARTIFACT = (
    REPO_ROOT
    / "benchmark_results/coda_anchor_feasibility/action_delta_gate_fold4"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "benchmark_results/paper_4arm_action_head_microbenchmark/seed7/report.json"
)
PROTOCOL_NAME = "paper-4arm-action-head-saved-workload-replay-v1"
DEFAULT_ORDER_SEED = 20260801
DEFAULT_REPEATS = 5
DEFAULT_WARMUP_ROUNDS = 2
DEFAULT_BOOTSTRAP_DRAWS = 20000


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GPUMicrobenchmarkValidationError(message)


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _identity_seed(identity: Mapping[str, Any], *, base_seed: int) -> int:
    payload = (
        f"paper4arm-cold-v1|{int(base_seed)}|"
        f"{int(identity['task_id'])}|{int(identity['episode_id'])}|"
        f"{int(identity['paired_trial_id'])}|{int(identity['prediction_step'])}|"
        f"{int(identity['initial_state_id'])}|{int(identity['episode_seed'])}"
    ).encode("ascii")
    digest = hashlib.sha256(payload).digest()
    # torch manual_seed accepts 64-bit values, but keep the value in the
    # conservative signed-63-bit range for portability.
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def _prepare_tensors(payload: Mapping[str, Any], device: torch.device):
    tensors = {
        name: (None if tensor is None else tensor.to(device=device, non_blocking=False))
        for name, tensor in payload["tensors"].items()
    }
    for name, tensor in tensors.items():
        if tensor is not None:
            _require(tensor.is_contiguous(), f"GPU workload tensor {name} is non-contiguous")
            _require(bool(torch.isfinite(tensor).all().item()), f"GPU workload tensor {name} is non-finite")
    return tensors


def _validate_projector(proprio_projector, tensors, device: torch.device) -> None:
    projected = proprio_projector(
        tensors["proprio_input"].reshape(1, -1).to(torch.bfloat16)
    ).unsqueeze(1)
    torch.cuda.synchronize(device)
    expected = tensors["proprio_features"]
    if not torch.equal(projected, expected):
        max_error = float(torch.max(torch.abs(projected.float() - expected.float())).item())
        raise GPUMicrobenchmarkValidationError(
            f"captured proprio feature mismatch; max absolute error={max_error}"
        )


def _make_counterfactual_cold_state(
    action_head,
    tensors: Mapping[str, torch.Tensor | None],
    identity: Mapping[str, Any],
    *,
    base_seed: int,
) -> torch.Tensor:
    """Create one frozen production-distributed cold state without perturbing benchmark RNG."""

    h = tensors["actions_hidden_states"]
    _require(h is not None, "actions_hidden_states is missing")
    batch_size = int(h.shape[0])
    device = h.device
    dtype = h.dtype

    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all()
    seed = _identity_seed(identity, base_seed=base_seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        selected = action_head.model.init_state(batch_size, device, dtype).detach().clone()
        torch.cuda.synchronize(device)
    finally:
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)

    _require(bool(torch.isfinite(selected).all().item()), "generated cold state is non-finite")
    return selected


@contextmanager
def _deferred_validator_without_inner_coda_profiling():
    """Bypass only the development-only profile_coda_cost=True requirement."""

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


def _condition_kwargs(arm: str, warm_state: torch.Tensor | None) -> dict[str, Any]:
    _require(arm in ARMS, f"unknown arm: {arm}")
    use_warm = arm in {"warm_start", "combined"}
    use_ldce = arm in {"ldce", "combined"}
    return {
        "phase": "Inference",
        "num_iter": None,
        "convergence_strategy": "adjacent_action_mse",
        "kl_thresh": 0.001,
        "cos_thresh": 0.999,
        "max_iter": 32,
        "warm_start_state": warm_state if use_warm else None,
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


def _bind_shared_predictor(action_head, artifact_payload, *, task_id: int, device) -> None:
    action_head.clear_scalar_task_policy()
    prepared = frozen._prepare_shared_spatial_predictor(
        artifact_payload,
        device=device,
        task_id=int(task_id),
    )
    action_head.configure_action_delta_gate(prepared, deferred_scorer_backend="eager")


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
        applied = bool(debug["action_delta_deferred_backfill_filter_applied"])
    else:
        coda_calls = int(K_t)
        eliminated = 0
        score_calls = 0
        applied = False
    return {
        "K_t": int(K_t),
        "coda_calls": coda_calls,
        "eliminated_coda_calls": eliminated,
        "ldce_score_calls": score_calls,
        "ldce_applied": applied,
        "actual_origin": "ACTUAL_WARM" if debug.get("warm_start_state_used") else "COLD",
        "cached_final_output_reused": bool(debug.get("cached_final_output_reused", True)),
        "numerical_retry_attempted": bool(debug.get("numerical_retry_attempted", False)),
    }


def _execute(
    *,
    action_head,
    proprio_projector,
    tensors,
    arm: str,
    cold_state: torch.Tensor,
):
    use_warm = arm in {"warm_start", "combined"}
    warm_state = tensors["incoming_warm_start_state"] if use_warm else None
    _require(
        (warm_state is not None) == use_warm,
        f"{arm}: ACTUAL_WARM workload/cache contract is inconsistent",
    )
    cold_origin = "ACTUAL_WARM" if use_warm else "COLD"
    with captured_cold_initial_state(action_head.model, cold_state, cold_origin):
        output, K_t, final_score = action_head.predict_action(
            tensors["actions_hidden_states"],
            proprio=tensors["proprio_input"],
            proprio_projector=proprio_projector,
            **_condition_kwargs(arm, warm_state),
        )
    K_t = int(K_t)
    return output, K_t, final_score, _schedule(action_head, arm, K_t)


def _timed_execute(
    *,
    action_head,
    proprio_projector,
    tensors,
    arm: str,
    cold_state: torch.Tensor,
    device: torch.device,
):
    torch.cuda.synchronize(device)
    start_ns = time.perf_counter_ns()
    output, K_t, final_score, schedule = _execute(
        action_head=action_head,
        proprio_projector=proprio_projector,
        tensors=tensors,
        arm=arm,
        cold_state=cold_state,
    )
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
    _require(math.isfinite(elapsed_ms) and elapsed_ms > 0, "latency must be finite and positive")
    _require(bool(torch.isfinite(output).all().item()), "action-head output is non-finite")
    _require(not schedule["numerical_retry_attempted"], "numerical retry occurred during replay")
    expected_origin = "ACTUAL_WARM" if arm in {"warm_start", "combined"} else "COLD"
    _require(schedule["actual_origin"] == expected_origin, f"{arm}: replay origin mismatch")
    if arm in {"ldce", "combined"}:
        _require(schedule["ldce_applied"], f"{arm}: LDCE was requested but not applied")
    return elapsed_ms, output, K_t, final_score, schedule


def _run_workload(
    *,
    descriptor: Mapping[str, Any],
    block_index: int,
    action_head,
    proprio_projector,
    artifact_payload,
    device: torch.device,
    repeats: int,
    order_seed: int,
    cold_seed: int,
    measured: bool,
):
    _require(descriptor["actual_origin"] == "ACTUAL_WARM", "primary replay requires ACTUAL_WARM workload")
    payload = load_action_head_workload(
        descriptor["path"],
        expected_sha256=descriptor["sha256"],
        expected_identity=descriptor["identity"],
        expected_origin="ACTUAL_WARM",
    )
    tensors = _prepare_tensors(payload, device)
    _validate_projector(proprio_projector, tensors, device)
    _bind_shared_predictor(
        action_head,
        artifact_payload,
        task_id=int(descriptor["identity"]["task_id"]),
        device=device,
    )
    cold_state = _make_counterfactual_cold_state(
        action_head,
        tensors,
        descriptor["identity"],
        base_seed=cold_seed,
    )

    measurements = []
    schedules_by_arm: dict[str, dict[str, Any]] = {}
    for repeat_index in range(repeats):
        order = balanced_condition_order(
            ARMS,
            block_index=block_index * repeats,
            repeat_index=repeat_index,
            seed=order_seed,
        )
        for order_position, arm in enumerate(order):
            if measured:
                elapsed_ms, output, K_t, final_score, schedule = _timed_execute(
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    tensors=tensors,
                    arm=arm,
                    cold_state=cold_state,
                    device=device,
                )
            else:
                output, K_t, final_score, schedule = _execute(
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    tensors=tensors,
                    arm=arm,
                    cold_state=cold_state,
                )
                torch.cuda.synchronize(device)
                elapsed_ms = None

            prior = schedules_by_arm.setdefault(arm, schedule)
            _require(prior == schedule, f"{arm}: replay schedule changed across repeats for {descriptor['identity']}")
            if measured:
                measurements.append(
                    {
                        **descriptor["identity"],
                        "captured_origin": descriptor["actual_origin"],
                        "replay_origin": schedule["actual_origin"],
                        "condition_id": arm,
                        "repeat_index": int(repeat_index),
                        "order_position": int(order_position),
                        "latency_ms": float(elapsed_ms),
                        "K_t": int(K_t),
                        "final_score": None if final_score is None else float(final_score),
                        "schedule": schedule,
                    }
                )

    del cold_state
    del tensors
    del payload
    return measurements


def _workload_medians(measurements: Sequence[Mapping[str, Any]], repeats: int):
    grouped: dict[tuple[int, int, int, str], list[Mapping[str, Any]]] = {}
    for row in measurements:
        key = (
            int(row["task_id"]),
            int(row["episode_id"]),
            int(row["prediction_step"]),
            str(row["condition_id"]),
        )
        grouped.setdefault(key, []).append(row)

    medians = []
    for key, rows in sorted(grouped.items()):
        _require(len(rows) == repeats, f"{key}: expected {repeats} repeats, got {len(rows)}")
        latency = float(np.median(np.asarray([float(row["latency_ms"]) for row in rows], dtype=np.float64)))
        schedule = rows[0]["schedule"]
        _require(all(row["schedule"] == schedule for row in rows), f"{key}: schedule differs across repeats")
        medians.append(
            {
                "task_id": key[0],
                "episode_id": key[1],
                "prediction_step": key[2],
                "condition_id": key[3],
                "median_latency_ms": latency,
                "K_t": int(schedule["K_t"]),
                "coda_calls": int(schedule["coda_calls"]),
                "eliminated_coda_calls": int(schedule["eliminated_coda_calls"]),
                "ldce_score_calls": int(schedule["ldce_score_calls"]),
            }
        )
    return medians


def _aggregate(medians: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for arm in ARMS:
        rows = [row for row in medians if row["condition_id"] == arm]
        _require(len(rows) == 100, f"{arm}: expected 100 workload medians, got {len(rows)}")
        task_means = []
        task_k = []
        task_coda = []
        task_eliminated = []
        per_task = {}
        for task_id in range(10):
            task_rows = [row for row in rows if int(row["task_id"]) == task_id]
            _require(len(task_rows) == 10, f"{arm}/task{task_id}: expected 10 workloads")
            latency_values = np.asarray([float(row["median_latency_ms"]) for row in task_rows], dtype=np.float64)
            k_values = np.asarray([float(row["K_t"]) for row in task_rows], dtype=np.float64)
            coda_values = np.asarray([float(row["coda_calls"]) for row in task_rows], dtype=np.float64)
            elim_values = np.asarray([float(row["eliminated_coda_calls"]) for row in task_rows], dtype=np.float64)
            per_task[str(task_id)] = {
                "mean_latency_ms": float(latency_values.mean()),
                "mean_K": float(k_values.mean()),
                "mean_coda_calls": float(coda_values.mean()),
                "mean_eliminated_coda_calls": float(elim_values.mean()),
            }
            task_means.append(float(latency_values.mean()))
            task_k.append(float(k_values.mean()))
            task_coda.append(float(coda_values.mean()))
            task_eliminated.append(float(elim_values.mean()))

        pooled = np.asarray([float(row["median_latency_ms"]) for row in rows], dtype=np.float64)
        result[arm] = {
            "primary_task_macro_mean_latency_ms": float(np.mean(task_means)),
            "task_macro_mean_K": float(np.mean(task_k)),
            "task_macro_mean_coda_calls": float(np.mean(task_coda)),
            "task_macro_mean_eliminated_coda_calls": float(np.mean(task_eliminated)),
            "pooled_workload_median_latency_ms": float(np.median(pooled)),
            "pooled_workload_p95_latency_ms": float(np.percentile(pooled, 95)),
            "per_task": per_task,
        }

    baseline = float(result["baseline"]["primary_task_macro_mean_latency_ms"])
    for arm in ARMS:
        value = float(result[arm]["primary_task_macro_mean_latency_ms"])
        result[arm]["speedup_vs_baseline_percent"] = 100.0 * (baseline - value) / baseline
    return result


def _bootstrap_speedups(
    medians: Sequence[Mapping[str, Any]], *, draws: int, seed: int
) -> dict[str, Any]:
    by_arm_task: dict[str, dict[int, np.ndarray]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for task_id in range(10):
            rows = sorted(
                (
                    row
                    for row in medians
                    if row["condition_id"] == arm and int(row["task_id"]) == task_id
                ),
                key=lambda row: (int(row["episode_id"]), int(row["prediction_step"])),
            )
            _require(len(rows) == 10, f"bootstrap {arm}/task{task_id}: expected 10 rows")
            by_arm_task[arm][task_id] = np.asarray(
                [float(row["median_latency_ms"]) for row in rows], dtype=np.float64
            )

    rng = np.random.default_rng(int(seed))
    samples = {arm: np.empty(draws, dtype=np.float64) for arm in ARMS if arm != "baseline"}
    for draw in range(draws):
        task_baseline = []
        task_other = {arm: [] for arm in ARMS if arm != "baseline"}
        for task_id in range(10):
            indices = rng.integers(0, 10, size=10)
            base_mean = float(by_arm_task["baseline"][task_id][indices].mean())
            task_baseline.append(base_mean)
            for arm in task_other:
                task_other[arm].append(float(by_arm_task[arm][task_id][indices].mean()))
        base_macro = float(np.mean(task_baseline))
        for arm, values in task_other.items():
            arm_macro = float(np.mean(values))
            samples[arm][draw] = 100.0 * (base_macro - arm_macro) / base_macro

    return {
        arm: {
            "draws": int(draws),
            "speedup_percent_p2_5": float(np.percentile(values, 2.5)),
            "speedup_percent_p50": float(np.percentile(values, 50)),
            "speedup_percent_p97_5": float(np.percentile(values, 97.5)),
        }
        for arm, values in samples.items()
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--initial-state-manifest", type=Path, default=DEFAULT_INITIAL_STATE_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--action-delta-artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--action-delta-sha256", default="")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-seed", type=int, default=7)
    parser.add_argument("--order-seed", type=int, default=DEFAULT_ORDER_SEED)
    parser.add_argument("--cold-seed", type=int, default=20260901)
    parser.add_argument("--measurement-repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--warmup-rounds", type=int, default=DEFAULT_WARMUP_ROUNDS)
    parser.add_argument("--bootstrap-draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS)
    parser.add_argument("--max-workloads", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.run_root = _resolve_repo_path(args.run_root)
    args.initial_state_manifest = _resolve_repo_path(args.initial_state_manifest)
    args.checkpoint = _resolve_repo_path(args.checkpoint)
    args.action_delta_artifact = _resolve_repo_path(args.action_delta_artifact)
    args.output = _resolve_repo_path(args.output)

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite report: {args.output}")
    if args.measurement_repeats < 3:
        raise ValueError("--measurement-repeats must be >= 3")
    if args.warmup_rounds < 1:
        raise ValueError("--warmup-rounds must be >= 1")
    if args.bootstrap_draws < 1000:
        raise ValueError("--bootstrap-draws must be >= 1000")

    artifact_sha = frozen._artifact_sha256(
        args.action_delta_artifact,
        args.action_delta_sha256 or None,
    )
    artifact_manifest, artifact_payload = load_action_delta_gate_artifact(
        args.action_delta_artifact,
        expected_sha256=artifact_sha,
    )

    calibration_validation = validate_calibration_run(
        str(args.run_root),
        str(args.initial_state_manifest),
        base_seed=args.base_seed,
    )
    _require(
        calibration_validation.get("complete_10_task_gate") is True,
        "benchmark requires the complete ten-task formal calibration run",
    )
    descriptors = [
        descriptor
        for descriptor in load_workload_descriptors(args.run_root)
        if descriptor["actual_origin"] == "ACTUAL_WARM"
    ]
    _require(len(descriptors) == 100, f"expected 100 ACTUAL_WARM workloads, got {len(descriptors)}")
    if args.max_workloads is not None:
        if args.max_workloads < 1:
            raise ValueError("--max-workloads must be >= 1")
        descriptors = descriptors[: int(args.max_workloads)]

    plan = {
        "schema_version": 1,
        "protocol": PROTOCOL_NAME,
        "source_commit": _git_commit(),
        "run_root": str(args.run_root.resolve()),
        "run_root_validation_complete_10_task_gate": True,
        "checkpoint": str(args.checkpoint.resolve()),
        "initial_state_manifest": str(args.initial_state_manifest.resolve()),
        "action_delta_artifact": str(args.action_delta_artifact.resolve()),
        "action_delta_artifact_sha256": artifact_sha,
        "action_delta_artifact_model_type": artifact_manifest.get("model_type"),
        "arms": list(ARMS),
        "captured_scope": "ACTUAL_WARM prediction-step-1 workloads",
        "workload_count": len(descriptors),
        "formal_workload_count": 100,
        "measurement_repeats": int(args.measurement_repeats),
        "warmup_rounds": int(args.warmup_rounds),
        "timed_calls": len(descriptors) * len(ARMS) * int(args.measurement_repeats),
        "timer": "CPU perf_counter_ns around ActionHeadRecurrent.predict_action with outer CUDA sync",
        "tensor_transfer_inside_timer": False,
        "VLM_inside_timer": False,
        "environment_inside_timer": False,
        "profile_coda_cost": False,
        "aggregation": "within-workload median -> within-task mean -> equal-task macro mean",
        "cold_counterfactual": (
            "one deterministic production-distributed cold state per ACTUAL_WARM workload; "
            "cold timed calls still execute real init_state for cost then substitute frozen state"
        ),
        "method_config": {
            "recurrence_strategy": "adjacent_action_mse",
            "recurrence_threshold": 0.001,
            "max_iter": 32,
            "warm_source": "midpoint",
            "warm_min_iter": 2,
            "cached_final_output": True,
            "ldce_threshold": 0.0015,
            "ldce_runtime_policy": "lazy_prefix_exact",
            "ldce_scorer_backend": "eager",
            "ldce_min_terminal_iter": 2,
        },
    }

    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("Action-head microbenchmark requires CUDA")

    device = torch.device("cuda:0")
    torch.manual_seed(int(args.order_seed))
    torch.cuda.manual_seed_all(int(args.order_seed))
    random.seed(int(args.order_seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    action_head, proprio_projector, checkpoint_inputs = load_benchmark_modules(
        args.checkpoint, device
    )
    plan["checkpoint_inputs"] = checkpoint_inputs
    plan["cuda"] = {
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }

    with torch.inference_mode(), _deferred_validator_without_inner_coda_profiling():
        representative = descriptors[0]
        print("Warm-up...", flush=True)
        _run_workload(
            descriptor=representative,
            block_index=0,
            action_head=action_head,
            proprio_projector=proprio_projector,
            artifact_payload=artifact_payload,
            device=device,
            repeats=int(args.warmup_rounds),
            order_seed=int(args.order_seed),
            cold_seed=int(args.cold_seed),
            measured=False,
        )

        measurements = []
        for block_index, descriptor in enumerate(descriptors):
            measurements.extend(
                _run_workload(
                    descriptor=descriptor,
                    block_index=block_index,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    artifact_payload=artifact_payload,
                    device=device,
                    repeats=int(args.measurement_repeats),
                    order_seed=int(args.order_seed),
                    cold_seed=int(args.cold_seed),
                    measured=True,
                )
            )
            if (block_index + 1) % 10 == 0 or block_index + 1 == len(descriptors):
                print(f"Measured workloads: {block_index + 1}/{len(descriptors)}", flush=True)

    medians = _workload_medians(measurements, int(args.measurement_repeats))
    formal_complete = len(descriptors) == 100
    summary = _aggregate(medians) if formal_complete else None
    bootstrap = (
        _bootstrap_speedups(
            medians,
            draws=int(args.bootstrap_draws),
            seed=int(args.order_seed),
        )
        if formal_complete
        else None
    )

    report = {
        **plan,
        "formal_complete": formal_complete,
        "primary_summary": summary,
        "paired_bootstrap_speedup_vs_baseline": bootstrap,
        "workload_medians": medians,
        "raw_measurements": measurements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote: {args.output}")
    if summary is not None:
        print("\nPrimary task-macro Action-head latency:")
        for arm in ARMS:
            row = summary[arm]
            print(
                f"  {arm:10s} {row['primary_task_macro_mean_latency_ms']:.4f} ms "
                f"({row['speedup_vs_baseline_percent']:+.2f}% vs baseline), "
                f"K={row['task_macro_mean_K']:.3f}, "
                f"Coda={row['task_macro_mean_coda_calls']:.3f}"
            )
    else:
        print("Subset run completed; no formal 100-workload aggregate was emitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
