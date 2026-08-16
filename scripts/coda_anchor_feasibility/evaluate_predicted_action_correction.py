"""Offline diagnostic for adding the frozen predicted delta to the anchor action."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from prismatic.models.action_delta_gate import (
    build_action_delta_gate_corrected_output,
    load_action_delta_gate_artifact,
    sha256_file,
)


EXPECTED_ARTIFACT_SHA256 = (
    "b4f9e938c72108f164b9d997a86eb4f5d9ea15b146754b9af83f9735e1ecfcf8"
)
EXPECTED_THRESHOLD = 0.000732466738008497
OUTER_FOLD = 4
HELD_OUT_TASK_IDS = (4, 5)
SAFE_ACTION_MSE = 0.001
OFFLINE_MIN_GATE_K = 4
RUNTIME_MIN_TERMINAL_ITER = OFFLINE_MIN_GATE_K + 1
PREFIX_STEPS = 5
EXPECTED_CADENCE_REPLAY = {
    "trajectory_count": 616,
    "score_call_count": 842,
    "activated": 205,
    "correct_safe": 203,
    "false_early": 2,
    "no_skip": 411,
}
MIN_RUNTIME_DTYPE_BENEFIT_RETENTION = 0.95
PERCENTILES = (50, 90, 95, 99)


def _distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
        }
    if not np.isfinite(values).all():
        raise RuntimeError("metric distribution contains non-finite values")
    percentiles = np.percentile(values, PERCENTILES)
    return {
        "mean": float(values.mean()),
        "median": float(percentiles[0]),
        "p90": float(percentiles[1]),
        "p95": float(percentiles[2]),
        "p99": float(percentiles[3]),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _percent(mask: np.ndarray) -> float | None:
    mask = np.asarray(mask, dtype=np.bool_)
    return float(mask.mean() * 100.0) if mask.size else None


def _ratio_values(correction: np.ndarray, reuse: np.ndarray) -> np.ndarray:
    correction = np.asarray(correction, dtype=np.float64)
    reuse = np.asarray(reuse, dtype=np.float64)
    ratios = np.empty_like(correction)
    positive = reuse > 0.0
    ratios[positive] = correction[positive] / reuse[positive]
    both_zero = (~positive) & (correction == 0.0)
    ratios[both_zero] = 1.0
    ratios[(~positive) & (~both_zero)] = np.inf
    return ratios


def _full_mse_comparison(reuse: np.ndarray, correction: np.ndarray) -> dict:
    ratios = _ratio_values(correction, reuse)
    finite = np.isfinite(ratios)
    return {
        "mean_reduction_ratio": (
            float(ratios.mean()) if finite.all() and ratios.size else None
        ),
        "median_reduction_ratio": (
            float(np.median(ratios)) if finite.all() and ratios.size else None
        ),
        "correction_better_percent": _percent(correction < reuse),
        "correction_reduces_mse_at_least_25_percent": _percent(ratios <= 0.75),
        "correction_reduces_mse_at_least_50_percent": _percent(ratios <= 0.50),
        "correction_reduces_mse_at_least_75_percent": _percent(ratios <= 0.25),
        "correction_reduces_mse_at_least_90_percent": _percent(ratios <= 0.10),
        "correction_worse_percent": _percent(correction > reuse),
        "correction_equal_percent": _percent(correction == reuse),
        "worst_correction_degradation_ratio": (
            float(ratios.max()) if finite.all() and ratios.size else None
        ),
        "infinite_ratio_count": int((~finite).sum()),
        "zero_reuse_mse_count": int((reuse == 0.0).sum()),
    }


def compute_error_metrics(error: torch.Tensor, prefix_steps: int) -> dict[str, np.ndarray]:
    if not torch.is_tensor(error) or error.ndim != 3:
        raise ValueError("action error must have shape [rows, steps, dimensions]")
    if error.shape[1] < prefix_steps:
        raise ValueError(
            f"action chunk has {error.shape[1]} steps, fewer than prefix {prefix_steps}"
        )
    error = error.detach().float().cpu()
    if not bool(torch.isfinite(error).all().item()):
        raise ValueError("action error contains non-finite values")
    squared = error.square()
    absolute = error.abs()
    prefix_squared = squared[:, :prefix_steps]
    prefix_absolute = absolute[:, :prefix_steps]
    return {
        "full_mse": squared.mean(dim=(1, 2)).numpy(),
        "prefix5_mse": prefix_squared.mean(dim=(1, 2)).numpy(),
        "full_max_abs": absolute.amax(dim=(1, 2)).numpy(),
        "prefix5_max_abs": prefix_absolute.amax(dim=(1, 2)).numpy(),
        "max_per_step_mse": squared.mean(dim=2).amax(dim=1).numpy(),
        "max_per_dim_mse": squared.mean(dim=1).amax(dim=1).numpy(),
    }


def summarize_population(
    name: str,
    row_indices: np.ndarray,
    reuse_metrics: dict[str, np.ndarray],
    correction_metrics: dict[str, np.ndarray],
    runtime_dtype_correction_metrics: dict[str, np.ndarray] | None = None,
) -> dict:
    row_indices = np.asarray(row_indices, dtype=np.int64)
    metric_summary = {}
    for metric_name in reuse_metrics:
        metric_summary[metric_name] = {
            "anchor_reuse": _distribution(reuse_metrics[metric_name][row_indices]),
            "predicted_correction": _distribution(
                correction_metrics[metric_name][row_indices]
            ),
        }
        if runtime_dtype_correction_metrics is not None:
            metric_summary[metric_name]["runtime_dtype_predicted_correction"] = (
                _distribution(
                    runtime_dtype_correction_metrics[metric_name][row_indices]
                )
            )
    reuse_full = reuse_metrics["full_mse"][row_indices]
    correction_full = correction_metrics["full_mse"][row_indices]
    result = {
        "name": name,
        "N": int(row_indices.size),
        "metrics": metric_summary,
        "full_mse_comparison": _full_mse_comparison(
            reuse_full, correction_full
        ),
    }
    if runtime_dtype_correction_metrics is not None:
        result["runtime_dtype_full_mse_comparison"] = _full_mse_comparison(
            reuse_full,
            runtime_dtype_correction_metrics["full_mse"][row_indices],
        )
    return result


def replay_first_hits(
    scores: np.ndarray,
    target_safe: np.ndarray,
    trajectory_ids: np.ndarray,
    ks: np.ndarray,
    threshold: float,
    min_gate_k: int,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    target_safe = np.asarray(target_safe, dtype=np.bool_)
    trajectory_ids = np.asarray(trajectory_ids)
    ks = np.asarray(ks)
    if not (
        len(scores) == len(target_safe) == len(trajectory_ids) == len(ks)
    ):
        raise ValueError("replay arrays must have equal lengths")

    members_by_trajectory = defaultdict(list)
    for local_index, trajectory_id in enumerate(trajectory_ids):
        members_by_trajectory[int(trajectory_id)].append(local_index)

    activated = []
    correct_safe = []
    false_early = []
    score_call_count = 0
    for members in members_by_trajectory.values():
        members.sort(key=lambda index: int(ks[index]))
        hit = None
        for local_index in members:
            if int(ks[local_index]) < min_gate_k:
                continue
            score_call_count += 1
            if float(scores[local_index]) <= threshold:
                hit = local_index
                break
        if hit is None:
            continue
        activated.append(hit)
        if bool(target_safe[hit]):
            correct_safe.append(hit)
        else:
            false_early.append(hit)

    trajectory_count = len(members_by_trajectory)
    replay = {
        "runtime_min_terminal_iteration": int(min_gate_k + 1),
        "offline_min_gate_k": int(min_gate_k),
        "trajectory_count": int(trajectory_count),
        "score_call_count": int(score_call_count),
        "activated": int(len(activated)),
        "correct_safe": int(len(correct_safe)),
        "false_early": int(len(false_early)),
        "no_skip": int(trajectory_count - len(activated)),
    }
    return (
        replay,
        np.asarray(activated, dtype=np.int64),
        np.asarray(correct_safe, dtype=np.int64),
        np.asarray(false_early, dtype=np.int64),
    )


def _validate_cache_contract(cache: dict, payload: dict) -> None:
    required = {
        "delta_states",
        "anchor_actions",
        "delta_actions",
        "task_ids",
        "folds",
        "ks",
        "target_mse",
        "target_safe",
        "trajectory_ids",
    }
    missing = sorted(required.difference(cache))
    if missing:
        raise RuntimeError(f"cache is missing required tensors: {missing}")
    tensors = {name: cache[name] for name in required}
    if not all(torch.is_tensor(value) for value in tensors.values()):
        raise RuntimeError("all required cache fields must be tensors")

    delta_states = cache["delta_states"]
    anchor_actions = cache["anchor_actions"]
    delta_actions = cache["delta_actions"]
    if delta_states.ndim != 3 or anchor_actions.ndim != 3 or delta_actions.ndim != 3:
        raise RuntimeError("cached state/action transitions must be rank 3")
    if tuple(anchor_actions.shape) != tuple(delta_actions.shape):
        raise RuntimeError("anchor_actions and delta_actions shapes differ")
    row_count = delta_states.shape[0]
    if anchor_actions.shape[0] != row_count:
        raise RuntimeError("state and action transition row counts differ")
    if delta_states.dtype != torch.bfloat16:
        raise RuntimeError("delta_states must preserve the BF16 cache contract")
    if delta_actions.dtype != torch.bfloat16 or anchor_actions.dtype != torch.bfloat16:
        raise RuntimeError("cached action tensors must preserve the BF16 contract")
    if delta_states.shape[1] != int(payload["action_chunk_len"]):
        raise RuntimeError("state chunk length differs from the artifact")
    if anchor_actions.shape[1] != int(payload["action_chunk_len"]):
        raise RuntimeError("action chunk length differs from the artifact")
    if delta_states.shape[2] != int(payload["hidden_dim"]):
        raise RuntimeError("state hidden dimension differs from the artifact")
    if anchor_actions.shape[2] != int(payload["action_dim"]):
        raise RuntimeError("action dimension differs from the artifact")
    if anchor_actions.shape[1] < PREFIX_STEPS:
        raise RuntimeError("action chunk is shorter than the required first-5 prefix")
    for name in required.difference(
        {"delta_states", "anchor_actions", "delta_actions"}
    ):
        if cache[name].ndim != 1 or cache[name].shape[0] != row_count:
            raise RuntimeError(f"cache field {name} has an invalid row shape")


def predict_frozen_delta(
    delta_states: torch.Tensor,
    payload: dict,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    tensors = {
        name: payload[name].detach().to(device=device, dtype=torch.float32)
        for name in (
            "x_mean",
            "x_std",
            "y_mean",
            "y_std",
            "linear_weight",
            "linear_bias",
        )
    }
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(delta_states), batch_size):
            # delta_states is already the cached BF16 runtime transition.
            delta_state = delta_states[start:start + batch_size].to(device)
            x = (delta_state.float() - tensors["x_mean"]) / tensors["x_std"]
            pred_norm = F.linear(
                x, tensors["linear_weight"], tensors["linear_bias"]
            )
            pred_delta = pred_norm * tensors["y_std"] + tensors["y_mean"]
            predictions.append(pred_delta.cpu())
    result = torch.cat(predictions, dim=0).float()
    if not bool(torch.isfinite(result).all().item()):
        raise RuntimeError("frozen predictor produced non-finite action deltas")
    return result


def _safe_scalar_ratio(numerator: float, denominator: float) -> float | None:
    if denominator > 0.0:
        return float(numerator / denominator)
    return 1.0 if numerator == 0.0 else None


def _false_early_table(
    local_indices: np.ndarray,
    global_indices: np.ndarray,
    cache: dict,
    scores: np.ndarray,
    reuse_metrics: dict[str, np.ndarray],
    correction_metrics: dict[str, np.ndarray],
) -> list[dict]:
    rows = []
    for local_index in local_indices:
        local_index = int(local_index)
        global_index = int(global_indices[local_index])
        reuse_full = float(reuse_metrics["full_mse"][local_index])
        correction_full = float(correction_metrics["full_mse"][local_index])
        reuse_prefix = float(reuse_metrics["prefix5_mse"][local_index])
        correction_prefix = float(correction_metrics["prefix5_mse"][local_index])
        k = int(cache["ks"][global_index])
        rows.append({
            "task": int(cache["task_ids"][global_index]),
            "trajectory_identifier": int(cache["trajectory_ids"][global_index]),
            "global_row_index": global_index,
            "offline_k": k,
            "runtime_terminal_iteration": k + 1,
            "predicted_gate_score": float(scores[local_index]),
            "exact_action_delta_mse": float(cache["target_mse"][global_index]),
            "anchor_reuse_mse": reuse_full,
            "predicted_correction_mse": correction_full,
            "correction_reuse_ratio": _safe_scalar_ratio(
                correction_full, reuse_full
            ),
            "anchor_reuse_prefix5_mse": reuse_prefix,
            "predicted_correction_prefix5_mse": correction_prefix,
            "prefix5_correction_reuse_ratio": _safe_scalar_ratio(
                correction_prefix, reuse_prefix
            ),
            "anchor_reuse_full_max_abs": float(
                reuse_metrics["full_max_abs"][local_index]
            ),
            "predicted_correction_full_max_abs": float(
                correction_metrics["full_max_abs"][local_index]
            ),
            "anchor_reuse_prefix5_max_abs": float(
                reuse_metrics["prefix5_max_abs"][local_index]
            ),
            "predicted_correction_prefix5_max_abs": float(
                correction_metrics["prefix5_max_abs"][local_index]
            ),
            "anchor_reuse_max_per_step_mse": float(
                reuse_metrics["max_per_step_mse"][local_index]
            ),
            "predicted_correction_max_per_step_mse": float(
                correction_metrics["max_per_step_mse"][local_index]
            ),
            "anchor_reuse_max_per_dim_mse": float(
                reuse_metrics["max_per_dim_mse"][local_index]
            ),
            "predicted_correction_max_per_dim_mse": float(
                correction_metrics["max_per_dim_mse"][local_index]
            ),
        })
    return rows


def _print_summary(results: dict) -> None:
    replay = results["terminal5_cadence_replay"]
    print(
        "Fold-4 terminal-5 replay: "
        f"trajectories={replay['trajectory_count']}, "
        f"score_calls={replay['score_call_count']}, "
        f"activated={replay['activated']}, "
        f"correct_safe={replay['correct_safe']}, "
        f"false_early={replay['false_early']}, "
        f"no_skip={replay['no_skip']}"
    )
    print()
    print(
        f"{'population':<36} {'N':>6} {'reuse mean':>13} "
        f"{'correction mean':>16} {'median ratio':>13} "
        f"{'better %':>10} {'worse %':>9}"
    )
    for population in results["populations"].values():
        reuse = population["metrics"]["full_mse"]["anchor_reuse"]["mean"]
        correction = population["metrics"]["full_mse"]["predicted_correction"]["mean"]
        comparison = population["full_mse_comparison"]
        print(
            f"{population['name']:<36} {population['N']:>6d} "
            f"{reuse:>13.8g} {correction:>16.8g} "
            f"{comparison['median_reduction_ratio']:>13.5f} "
            f"{comparison['correction_better_percent']:>10.2f} "
            f"{comparison['correction_worse_percent']:>9.2f}"
        )
    runtime_dtype = results["runtime_dtype_validation"]
    print(
        "\nRuntime-dtype validation on terminal-5 activations: "
        f"dtype={runtime_dtype['runtime_output_dtype']}, "
        "float32_mse="
        f"{runtime_dtype['float32_correction_full_mse_mean']:.9g}, "
        "runtime_dtype_mse="
        f"{runtime_dtype['runtime_dtype_correction_full_mse_mean']:.9g}, "
        "benefit_retention="
        f"{runtime_dtype['runtime_dtype_benefit_retention']:.6f}"
    )
    print()
    print("False-early terminal-5 activations:")
    for row in results["false_early_activations"]:
        print(
            f"  task={row['task']} traj={row['trajectory_identifier']} "
            f"k={row['offline_k']} terminal={row['runtime_terminal_iteration']} "
            f"score={row['predicted_gate_score']:.9g} "
            f"reuse={row['anchor_reuse_mse']:.9g} "
            f"correction={row['predicted_correction_mse']:.9g} "
            f"ratio={row['correction_reuse_ratio']:.5f}"
        )
    print(f"\nJSON written: {results['output_path']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(
            "benchmark_results/coda_anchor_feasibility/action_delta_cache.pt"
        ),
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path(
            "benchmark_results/coda_anchor_feasibility/"
            "action_delta_gate_fold4/action_delta_gate.pt"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmark_results/coda_anchor_feasibility/"
            "predicted_action_correction_results.json"
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if not args.cache.is_file() or not args.artifact.is_file():
        raise FileNotFoundError("the frozen cache and artifact are required")
    artifact_sha256 = sha256_file(args.artifact)
    if artifact_sha256 != EXPECTED_ARTIFACT_SHA256:
        raise RuntimeError(
            "fold-4 artifact hash mismatch: "
            f"expected={EXPECTED_ARTIFACT_SHA256}, actual={artifact_sha256}"
        )
    manifest, payload = load_action_delta_gate_artifact(
        args.artifact,
        expected_sha256=EXPECTED_ARTIFACT_SHA256,
    )
    threshold = float(payload["threshold"])
    if threshold != EXPECTED_THRESHOLD:
        raise RuntimeError(
            f"frozen threshold mismatch: {threshold} != {EXPECTED_THRESHOLD}"
        )
    if int(payload["outer_fold"]) != OUTER_FOLD:
        raise RuntimeError("artifact outer fold is not fold 4")
    if tuple(int(value) for value in payload["held_out_task_ids"]) != HELD_OUT_TASK_IDS:
        raise RuntimeError("artifact held-out tasks are not tasks 4 and 5")

    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    _validate_cache_contract(cache, payload)
    fold_mask = cache["folds"] == OUTER_FOLD
    global_indices = torch.where(fold_mask)[0].numpy()
    held_out_tasks = tuple(
        int(value) for value in torch.unique(cache["task_ids"][fold_mask]).tolist()
    )
    if held_out_tasks != HELD_OUT_TASK_IDS:
        raise RuntimeError(f"fold-4 cache tasks differ: {held_out_tasks}")

    delta_states = cache["delta_states"][fold_mask]
    anchor_actions_runtime_dtype = cache["anchor_actions"][fold_mask]
    anchor_actions = anchor_actions_runtime_dtype.float()
    target_delta = cache["delta_actions"][fold_mask].float()
    target_mse = cache["target_mse"][fold_mask].numpy()
    target_safe = cache["target_safe"][fold_mask].numpy()
    expected_target_safe = target_mse < SAFE_ACTION_MSE
    if not np.array_equal(target_safe, expected_target_safe):
        raise RuntimeError("target_safe differs from exact target_mse < 0.001")

    device = torch.device(args.device)
    pred_delta = predict_frozen_delta(
        delta_states, payload, device, args.batch_size
    )
    if tuple(pred_delta.shape) != tuple(target_delta.shape):
        raise RuntimeError(
            "predicted and target action delta shapes differ: "
            f"predicted={tuple(pred_delta.shape)}, target={tuple(target_delta.shape)}"
        )
    gate_scores = pred_delta.square().mean(dim=(1, 2)).numpy()
    reuse_error = target_delta
    correction_error = target_delta - pred_delta
    reuse_metrics = compute_error_metrics(reuse_error, PREFIX_STEPS)
    correction_metrics = compute_error_metrics(correction_error, PREFIX_STEPS)

    exact_terminal_action = anchor_actions + target_delta
    predicted_terminal_action = anchor_actions + pred_delta
    runtime_dtype_predicted_terminal_action = (
        build_action_delta_gate_corrected_output(
            anchor_actions_runtime_dtype,
            pred_delta,
        )
    )
    runtime_dtype_correction_error = (
        exact_terminal_action - runtime_dtype_predicted_terminal_action.float()
    )
    runtime_dtype_correction_metrics = compute_error_metrics(
        runtime_dtype_correction_error,
        PREFIX_STEPS,
    )
    reconstructed_reuse_error = exact_terminal_action - anchor_actions
    reconstructed_correction_error = exact_terminal_action - predicted_terminal_action
    target_delta_mse = target_delta.square().mean(dim=(1, 2)).numpy()
    target_mse_bf16_difference = np.abs(target_mse - target_delta_mse)
    target_safe_from_bf16_delta = target_delta_mse < SAFE_ACTION_MSE
    reuse_sanity_difference = np.abs(
        reuse_metrics["full_mse"] - target_delta_mse
    )
    reconstruction_reuse_difference = (
        reconstructed_reuse_error - reuse_error
    ).abs().max().item()
    reconstruction_correction_difference = (
        reconstructed_correction_error - correction_error
    ).abs().max().item()
    reconstruction_tolerance = 1e-6
    if float(reuse_sanity_difference.max()) > 1e-12:
        raise RuntimeError("anchor-reuse MSE sanity check failed")
    if max(
        reconstruction_reuse_difference,
        reconstruction_correction_difference,
    ) > reconstruction_tolerance:
        raise RuntimeError("terminal-action reconstruction sanity check failed")

    trajectory_ids = cache["trajectory_ids"][fold_mask].numpy()
    ks = cache["ks"][fold_mask].numpy()
    replay, activated, correct_safe, false_early = replay_first_hits(
        gate_scores,
        target_safe,
        trajectory_ids,
        ks,
        threshold,
        OFFLINE_MIN_GATE_K,
    )
    observed_replay = {
        key: replay[key] for key in EXPECTED_CADENCE_REPLAY
    }
    if observed_replay != EXPECTED_CADENCE_REPLAY:
        raise RuntimeError(
            "terminal-5 cadence replay mismatch: "
            f"expected={EXPECTED_CADENCE_REPLAY}, observed={observed_replay}"
        )

    activated_reuse_mean = float(reuse_metrics["full_mse"][activated].mean())
    activated_float_correction_mean = float(
        correction_metrics["full_mse"][activated].mean()
    )
    activated_runtime_correction_mean = float(
        runtime_dtype_correction_metrics["full_mse"][activated].mean()
    )
    float32_benefit = activated_reuse_mean - activated_float_correction_mean
    runtime_dtype_benefit = (
        activated_reuse_mean - activated_runtime_correction_mean
    )
    if float32_benefit <= 0.0:
        raise RuntimeError("float32 predicted correction has no activation benefit")
    runtime_dtype_benefit_retention = runtime_dtype_benefit / float32_benefit
    if (
        activated_runtime_correction_mean >= activated_reuse_mean
        or runtime_dtype_benefit_retention < MIN_RUNTIME_DTYPE_BENEFIT_RETENTION
    ):
        raise RuntimeError(
            "runtime-dtype conversion materially destroys correction benefit: "
            f"retention={runtime_dtype_benefit_retention}"
        )

    all_rows = np.arange(len(global_indices), dtype=np.int64)
    populations = {
        "all_fold4_rows": all_rows,
        "exact_next_action_mse_below_0_001": np.flatnonzero(target_safe),
        "frozen_gate_score_selected_rows": np.flatnonzero(
            gate_scores <= threshold
        ),
        "terminal5_sequential_activations": activated,
        "terminal5_correct_safe_activations": correct_safe,
        "terminal5_false_early_activations": false_early,
    }
    population_results = {
        key: summarize_population(
            key,
            indices,
            reuse_metrics,
            correction_metrics,
            runtime_dtype_correction_metrics,
        )
        for key, indices in populations.items()
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "schema_version": 1,
        "analysis": "frozen_fold4_predicted_action_correction",
        "diagnostic_only": True,
        "cache_path": str(args.cache),
        "cache_sha256": sha256_file(args.cache),
        "artifact_path": str(args.artifact),
        "artifact_sha256": artifact_sha256,
        "artifact_manifest": manifest,
        "threshold": threshold,
        "outer_fold": OUTER_FOLD,
        "held_out_task_ids": list(HELD_OUT_TASK_IDS),
        "device": str(device),
        "preprocessing_contract": {
            "delta_state_dtype": "bfloat16",
            "normalization": "(delta_state.float() - x_mean) / x_std",
            "predictor": "F.linear(x, linear_weight, linear_bias)",
            "denormalization": "pred_norm * y_std + y_mean",
        },
        "cache_contract": {
            "fold4_row_count": int(len(global_indices)),
            "action_chunk_shape": list(anchor_actions.shape[1:]),
            "state_delta_shape": list(delta_states.shape[1:]),
            "prefix_steps": PREFIX_STEPS,
            "exact_safety_label": "target_mse < 0.001",
            "error_source": "cached BF16 delta_actions converted to float32",
            "target_mse_source": "cache target_mse retained from source trace",
        },
        "terminal5_cadence_replay": replay,
        "expected_terminal5_cadence_replay": EXPECTED_CADENCE_REPLAY,
        "terminal5_cadence_replay_verified": True,
        "runtime_dtype_validation": {
            "runtime_output_dtype": str(
                runtime_dtype_predicted_terminal_action.dtype
            ),
            "population": "terminal5_sequential_activations",
            "N": int(len(activated)),
            "anchor_reuse_full_mse_mean": activated_reuse_mean,
            "float32_correction_full_mse_mean": (
                activated_float_correction_mean
            ),
            "runtime_dtype_correction_full_mse_mean": (
                activated_runtime_correction_mean
            ),
            "runtime_dtype_to_float32_correction_mse_ratio": (
                activated_runtime_correction_mean
                / activated_float_correction_mean
            ),
            "float32_benefit": float32_benefit,
            "runtime_dtype_benefit": runtime_dtype_benefit,
            "runtime_dtype_benefit_retention": (
                runtime_dtype_benefit_retention
            ),
            "minimum_required_benefit_retention": (
                MIN_RUNTIME_DTYPE_BENEFIT_RETENTION
            ),
            "material_benefit_preserved": True,
        },
        "sanity_checks": {
            "anchor_reuse_full_mse_equals_mean_target_delta_squared": True,
            "anchor_reuse_full_mse_max_abs_difference": float(
                reuse_sanity_difference.max()
            ),
            "cached_target_mse_vs_bf16_delta_mse_max_abs_difference": float(
                target_mse_bf16_difference.max()
            ),
            "safe_label_disagreement_if_recomputed_from_bf16_delta_count": int(
                np.count_nonzero(target_safe != target_safe_from_bf16_delta)
            ),
            "reconstructed_reuse_error_max_abs_difference": float(
                reconstruction_reuse_difference
            ),
            "reconstructed_correction_error_max_abs_difference": float(
                reconstruction_correction_difference
            ),
            "reconstruction_tolerance": reconstruction_tolerance,
        },
        "populations": population_results,
        "false_early_activations": _false_early_table(
            false_early,
            global_indices,
            cache,
            gate_scores,
            reuse_metrics,
            correction_metrics,
        ),
        "output_path": str(args.output),
    }
    args.output.write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _print_summary(results)


if __name__ == "__main__":
    main()
