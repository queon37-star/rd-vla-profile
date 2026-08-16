"""Offline causal diagnostics for Action-Delta Gate false-safe signals.

This script never changes runtime policy.  It uses the frozen fold-4 linear
predictor and only pre-terminal-Coda information to build candidate veto
features.  Cutoffs are evaluated with leave-one-task-out development replay;
task 4 is used only after feature definitions and cutoffs are frozen, and task
5 is intentionally not evaluated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from prismatic.models.action_delta_gate import (
    load_action_delta_gate_artifact,
    sha256_file,
)


EXPECTED_ARTIFACT_SHA256 = (
    "b4f9e938c72108f164b9d997a86eb4f5d9ea15b146754b9af83f9735e1ecfcf8"
)
EXPECTED_THRESHOLD = 0.000732466738008497
SAFE_ACTION_MSE = 0.001
OUTER_FOLD = 4
TRAIN_CAL_TASKS = (0, 1, 2, 3, 6, 7, 8, 9)
FORENSIC_TASK = 4
UNTOUCHED_TASK = 5
CURRENT_MIN_TERMINAL_ITER = 5
PREFIX_STEPS = 5
CUTOFF_SAFE_RETENTIONS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99)
STRICT_SCORE_FACTORS = (0.25, 0.50, 0.75, 0.90)

RUNTIME_FEATURE_NAMES = (
    "predicted_action_delta_mse",
    "threshold_margin",
    "normalized_margin",
    "predicted_prefix5_mse",
    "predicted_full_max_abs",
    "predicted_prefix5_max_abs",
    "predicted_max_per_step_mse",
    "predicted_per_step_mse_std",
    "predicted_per_step_mse_cv",
    "predicted_max_per_dim_mse",
    "predicted_per_dim_mse_std",
    "predicted_per_dim_mse_cv",
    "predicted_max_step_to_full_mse_ratio",
    "predicted_max_dim_to_full_mse_ratio",
    "latent_delta_full_rms",
    "latent_delta_token_rms_mean",
    "latent_delta_token_rms_max",
    "latent_delta_token_rms_std",
    "latent_delta_token_rms_cv",
    "latent_delta_max_abs",
    "normalized_x_l2",
    "normalized_x_rms",
    "normalized_x_max_abs",
    "normalized_x_token_norm_mean",
    "normalized_x_token_norm_max",
    "normalized_x_token_norm_std",
    "terminal_iteration",
    "previous_predicted_score",
    "score_ratio_current_to_previous",
    "score_difference_current_minus_previous",
    "relative_score_drop",
    "previous_latent_delta_rms",
    "latent_delta_rms_ratio_current_to_previous",
    "latent_delta_cosine_current_previous",
    "latent_delta_second_difference_rms",
)

EVALUATION_TARGET_NAMES = (
    "exact_adjacent_action_mse",
    "exact_safe",
    "false_safe",
    "residual_exact_minus_predicted",
)


def offline_anchor_k_to_runtime_terminal_iteration(offline_anchor_k: int) -> int:
    """Map cached S_k -> S_(k+1) row indexing to runtime terminal iteration."""

    if not isinstance(offline_anchor_k, (int, np.integer)) or isinstance(
        offline_anchor_k, (bool, np.bool_)
    ):
        raise TypeError("offline anchor k must be an integer")
    if int(offline_anchor_k) < 1:
        raise ValueError("offline anchor k must be >= 1")
    return int(offline_anchor_k) + 1


def _json_scalar(value):
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    value = float(value)
    return value if math.isfinite(value) else None


def _safe_divide_array(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    result = np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan)
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    positive = denominator != 0.0
    result[positive] = numerator[positive] / denominator[positive]
    both_zero = (~positive) & (numerator == 0.0)
    result[both_zero] = 0.0
    return result


def _distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "N": 0,
            **{
                key: None
                for key in ("mean", "median", "p10", "p90", "min", "max")
            },
        }
    p10, median, p90 = np.percentile(values, [10, 50, 90])
    return {
        "N": int(values.size),
        "mean": float(values.mean()),
        "median": float(median),
        "p10": float(p10),
        "p90": float(p90),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def binary_auroc(values: np.ndarray, positive: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    positive = np.asarray(positive, dtype=np.bool_)
    valid = np.isfinite(values)
    values = values[valid]
    positive = positive[valid]
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = _rankdata(values)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(values: np.ndarray, positive: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    positive = np.asarray(positive, dtype=np.bool_)
    valid = np.isfinite(values)
    values = values[valid]
    positive = positive[valid]
    n_pos = int(positive.sum())
    if n_pos == 0 or int((~positive).sum()) == 0:
        return None
    order = np.argsort(-values, kind="mergesort")
    labels = positive[order]
    precision = np.cumsum(labels) / np.arange(1, len(labels) + 1)
    return float(precision[labels].sum() / n_pos)


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 3:
        return None
    return _pearson(_rankdata(x[valid]), _rankdata(y[valid]))


def _cohen_d(false_values: np.ndarray, safe_values: np.ndarray) -> float | None:
    false_values = np.asarray(false_values, dtype=np.float64)
    safe_values = np.asarray(safe_values, dtype=np.float64)
    false_values = false_values[np.isfinite(false_values)]
    safe_values = safe_values[np.isfinite(safe_values)]
    if len(false_values) < 2 or len(safe_values) < 2:
        return None
    pooled_variance = (
        (len(false_values) - 1) * false_values.var(ddof=1)
        + (len(safe_values) - 1) * safe_values.var(ddof=1)
    ) / (len(false_values) + len(safe_values) - 2)
    if pooled_variance <= 0.0:
        return None
    return float((false_values.mean() - safe_values.mean()) / math.sqrt(pooled_variance))


def _validate_cache(cache: dict, payload: dict) -> None:
    required = {
        "delta_states",
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
        raise RuntimeError(f"cache is missing required fields: {missing}")
    row_count = int(cache["delta_states"].shape[0])
    for name in required:
        value = cache[name]
        if not torch.is_tensor(value) or value.shape[0] != row_count:
            raise RuntimeError(f"cache field {name} has an invalid row contract")
    if cache["delta_states"].ndim != 3 or cache["delta_states"].dtype != torch.bfloat16:
        raise RuntimeError("delta_states must be rank-3 BF16 runtime transitions")
    if cache["delta_actions"].ndim != 3:
        raise RuntimeError("delta_actions must be rank 3")
    if cache["delta_states"].shape[1:] != (
        int(payload["action_chunk_len"]),
        int(payload["hidden_dim"]),
    ):
        raise RuntimeError("cached latent shape differs from the artifact")
    if cache["delta_actions"].shape[1:] != (
        int(payload["action_chunk_len"]),
        int(payload["action_dim"]),
    ):
        raise RuntimeError("cached action shape differs from the artifact")
    exact_safe = cache["target_mse"].numpy() < SAFE_ACTION_MSE
    if not np.array_equal(exact_safe, cache["target_safe"].numpy()):
        raise RuntimeError("cache safe labels differ from exact MSE < 0.001")


def predict_frozen_delta(
    delta_states: torch.Tensor,
    payload: dict,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Apply the exact frozen runtime preprocessing and linear predictor."""

    tensors = {
        name: payload[name].detach().to(device=device, dtype=torch.float32)
        for name in ("x_mean", "x_std", "y_mean", "y_std", "linear_weight", "linear_bias")
    }
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(delta_states), batch_size):
            delta_state = delta_states[start : start + batch_size].to(device)
            x = (delta_state.float() - tensors["x_mean"]) / tensors["x_std"]
            pred_norm = F.linear(x, tensors["linear_weight"], tensors["linear_bias"])
            predictions.append((pred_norm * tensors["y_std"] + tensors["y_mean"]).cpu())
    result = torch.cat(predictions).float()
    if not bool(torch.isfinite(result).all().item()):
        raise RuntimeError("frozen predictor produced non-finite values")
    return result


def build_runtime_features(
    delta_states: torch.Tensor,
    predicted_delta: torch.Tensor,
    trajectory_ids: np.ndarray,
    ks: np.ndarray,
    gate_threshold: float,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    *,
    prefix_steps: int = PREFIX_STEPS,
    batch_size: int = 256,
) -> dict[str, np.ndarray]:
    """Build only information available before the candidate terminal Coda."""

    if delta_states.ndim != 3 or delta_states.dtype != torch.bfloat16:
        raise ValueError("delta_states must be BF16 [rows, tokens, hidden]")
    if predicted_delta.ndim != 3 or len(predicted_delta) != len(delta_states):
        raise ValueError("predicted_delta must be [rows, steps, dimensions]")
    if predicted_delta.shape[1] < prefix_steps:
        raise ValueError("predicted action chunk is shorter than the prefix")
    if len(trajectory_ids) != len(delta_states) or len(ks) != len(delta_states):
        raise ValueError("trajectory history arrays must match feature rows")
    if gate_threshold <= 0.0 or batch_size <= 0:
        raise ValueError("threshold and batch size must be positive")

    predicted_delta = predicted_delta.detach().float().cpu()
    pred_squared = predicted_delta.square()
    pred_step_mse = pred_squared.mean(dim=2)
    pred_dim_mse = pred_squared.mean(dim=1)
    predicted_score = pred_squared.mean(dim=(1, 2)).numpy().astype(np.float64)
    step_mean = pred_step_mse.mean(dim=1).numpy()
    dim_mean = pred_dim_mse.mean(dim=1).numpy()

    features: dict[str, np.ndarray] = {
        "predicted_action_delta_mse": predicted_score,
        "threshold_margin": gate_threshold - predicted_score,
        "normalized_margin": (gate_threshold - predicted_score) / gate_threshold,
        "predicted_prefix5_mse": pred_squared[:, :prefix_steps].mean(dim=(1, 2)).numpy(),
        "predicted_full_max_abs": predicted_delta.abs().amax(dim=(1, 2)).numpy(),
        "predicted_prefix5_max_abs": predicted_delta[:, :prefix_steps].abs().amax(dim=(1, 2)).numpy(),
        "predicted_max_per_step_mse": pred_step_mse.amax(dim=1).numpy(),
        "predicted_per_step_mse_std": pred_step_mse.std(dim=1, unbiased=False).numpy(),
        "predicted_max_per_dim_mse": pred_dim_mse.amax(dim=1).numpy(),
        "predicted_per_dim_mse_std": pred_dim_mse.std(dim=1, unbiased=False).numpy(),
    }
    features["predicted_per_step_mse_cv"] = _safe_divide_array(
        features["predicted_per_step_mse_std"], step_mean
    )
    features["predicted_per_dim_mse_cv"] = _safe_divide_array(
        features["predicted_per_dim_mse_std"], dim_mean
    )
    features["predicted_max_step_to_full_mse_ratio"] = _safe_divide_array(
        features["predicted_max_per_step_mse"], predicted_score
    )
    features["predicted_max_dim_to_full_mse_ratio"] = _safe_divide_array(
        features["predicted_max_per_dim_mse"], predicted_score
    )

    row_count = len(delta_states)
    latent_names = (
        "latent_delta_full_rms",
        "latent_delta_token_rms_mean",
        "latent_delta_token_rms_max",
        "latent_delta_token_rms_std",
        "latent_delta_token_rms_cv",
        "latent_delta_max_abs",
        "normalized_x_l2",
        "normalized_x_rms",
        "normalized_x_max_abs",
        "normalized_x_token_norm_mean",
        "normalized_x_token_norm_max",
        "normalized_x_token_norm_std",
    )
    for name in latent_names:
        features[name] = np.empty(row_count, dtype=np.float64)
    x_mean = x_mean.detach().float().cpu()
    x_std = x_std.detach().float().cpu()
    with torch.inference_mode():
        for start in range(0, row_count, batch_size):
            end = min(start + batch_size, row_count)
            delta = delta_states[start:end].float()
            token_rms = delta.square().mean(dim=2).sqrt()
            token_rms_mean = token_rms.mean(dim=1)
            token_rms_std = token_rms.std(dim=1, unbiased=False)
            x = (delta - x_mean) / x_std
            x_token_norm = torch.linalg.vector_norm(x, dim=2)
            batch_values = {
                "latent_delta_full_rms": delta.square().mean(dim=(1, 2)).sqrt(),
                "latent_delta_token_rms_mean": token_rms_mean,
                "latent_delta_token_rms_max": token_rms.amax(dim=1),
                "latent_delta_token_rms_std": token_rms_std,
                "latent_delta_token_rms_cv": torch.where(
                    token_rms_mean != 0,
                    token_rms_std / token_rms_mean,
                    torch.zeros_like(token_rms_mean),
                ),
                "latent_delta_max_abs": delta.abs().amax(dim=(1, 2)),
                "normalized_x_l2": torch.linalg.vector_norm(x.flatten(1), dim=1),
                "normalized_x_rms": x.square().mean(dim=(1, 2)).sqrt(),
                "normalized_x_max_abs": x.abs().amax(dim=(1, 2)),
                "normalized_x_token_norm_mean": x_token_norm.mean(dim=1),
                "normalized_x_token_norm_max": x_token_norm.amax(dim=1),
                "normalized_x_token_norm_std": x_token_norm.std(dim=1, unbiased=False),
            }
            for name, values in batch_values.items():
                features[name][start:end] = values.numpy()

    features["terminal_iteration"] = np.asarray(
        [
            offline_anchor_k_to_runtime_terminal_iteration(int(k))
            for k in ks
        ],
        dtype=np.float64,
    )
    history_names = (
        "previous_predicted_score",
        "score_ratio_current_to_previous",
        "score_difference_current_minus_previous",
        "relative_score_drop",
        "previous_latent_delta_rms",
        "latent_delta_rms_ratio_current_to_previous",
        "latent_delta_cosine_current_previous",
        "latent_delta_second_difference_rms",
    )
    for name in history_names:
        features[name] = np.full(row_count, np.nan, dtype=np.float64)

    previous_index = np.full(row_count, -1, dtype=np.int64)
    by_trajectory: dict[int, list[int]] = defaultdict(list)
    for index, trajectory_id in enumerate(np.asarray(trajectory_ids)):
        by_trajectory[int(trajectory_id)].append(index)
    for members in by_trajectory.values():
        members.sort(key=lambda index: int(ks[index]))
        for previous, current in zip(members, members[1:]):
            if int(ks[current]) == int(ks[previous]) + 1:
                previous_index[current] = previous

    valid_current = np.flatnonzero(previous_index >= 0)
    previous_scores = predicted_score[previous_index[valid_current]]
    current_scores = predicted_score[valid_current]
    features["previous_predicted_score"][valid_current] = previous_scores
    features["score_ratio_current_to_previous"][valid_current] = _safe_divide_array(
        current_scores, previous_scores
    )
    features["score_difference_current_minus_previous"][valid_current] = current_scores - previous_scores
    features["relative_score_drop"][valid_current] = _safe_divide_array(
        previous_scores - current_scores, previous_scores
    )
    previous_rms = features["latent_delta_full_rms"][previous_index[valid_current]]
    current_rms = features["latent_delta_full_rms"][valid_current]
    features["previous_latent_delta_rms"][valid_current] = previous_rms
    features["latent_delta_rms_ratio_current_to_previous"][valid_current] = _safe_divide_array(
        current_rms, previous_rms
    )

    with torch.inference_mode():
        for start in range(0, len(valid_current), batch_size):
            current_indices = valid_current[start : start + batch_size]
            previous_indices = previous_index[current_indices]
            current_delta = delta_states[current_indices].float().flatten(1)
            previous_delta = delta_states[previous_indices].float().flatten(1)
            current_norm = torch.linalg.vector_norm(current_delta, dim=1)
            previous_norm = torch.linalg.vector_norm(previous_delta, dim=1)
            denominator = current_norm * previous_norm
            cosine = torch.full_like(denominator, float("nan"))
            valid_norm = denominator != 0
            cosine[valid_norm] = (
                (current_delta[valid_norm] * previous_delta[valid_norm]).sum(dim=1)
                / denominator[valid_norm]
            )
            second_rms = (current_delta - previous_delta).square().mean(dim=1).sqrt()
            features["latent_delta_cosine_current_previous"][current_indices] = cosine.numpy()
            features["latent_delta_second_difference_rms"][current_indices] = second_rms.numpy()

    if set(features) != set(RUNTIME_FEATURE_NAMES):
        missing = sorted(set(RUNTIME_FEATURE_NAMES).difference(features))
        extra = sorted(set(features).difference(RUNTIME_FEATURE_NAMES))
        raise RuntimeError(f"runtime feature schema mismatch: missing={missing}, extra={extra}")
    for name, values in features.items():
        if len(values) != row_count:
            raise RuntimeError(f"feature {name} has an invalid row count")
    return {name: features[name] for name in RUNTIME_FEATURE_NAMES}


def build_evaluation_targets(exact_mse: np.ndarray, predicted_score: np.ndarray, threshold: float) -> dict:
    exact_mse = np.asarray(exact_mse, dtype=np.float64)
    predicted_score = np.asarray(predicted_score, dtype=np.float64)
    exact_safe = exact_mse < SAFE_ACTION_MSE
    predicted_trigger = predicted_score <= threshold
    return {
        "exact_adjacent_action_mse": exact_mse,
        "exact_safe": exact_safe,
        "false_safe": predicted_trigger & (~exact_safe),
        "residual_exact_minus_predicted": exact_mse - predicted_score,
    }


def select_analysis_indices(task_ids: np.ndarray) -> np.ndarray:
    """Select development plus task-4 forensic rows, never task 5."""

    task_ids = np.asarray(task_ids, dtype=np.int64)
    allowed_tasks = TRAIN_CAL_TASKS + (FORENSIC_TASK,)
    indices = np.flatnonzero(np.isin(task_ids, allowed_tasks))
    if np.any(task_ids[indices] == UNTOUCHED_TASK):
        raise RuntimeError("task 5 entered the analysis selection")
    return indices


def descriptive_feature_analysis(
    features: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    task_ids: np.ndarray,
    row_mask: np.ndarray,
) -> tuple[list[dict], dict[str, list[dict]]]:
    trigger_mask = row_mask & (
        features["predicted_action_delta_mse"] <= EXPECTED_THRESHOLD
    )

    def analyze(mask: np.ndarray) -> list[dict]:
        rows = []
        false_label = targets["false_safe"][mask]
        for name in RUNTIME_FEATURE_NAMES:
            values = features[name][mask]
            safe_values = values[~false_label]
            false_values = values[false_label]
            raw_auc = binary_auroc(values, false_label)
            if raw_auc is None:
                direction = None
                oriented_auc = None
                oriented_values = values
            elif raw_auc >= 0.5:
                direction = "higher_is_false_safe"
                oriented_auc = raw_auc
                oriented_values = values
            else:
                direction = "lower_is_false_safe"
                oriented_auc = 1.0 - raw_auc
                oriented_values = -values
            safe_finite = safe_values[np.isfinite(safe_values)]
            false_finite = false_values[np.isfinite(false_values)]
            false_percentiles = np.asarray([], dtype=np.float64)
            if len(safe_finite) and len(false_finite):
                false_percentiles = np.asarray(
                    [100.0 * np.mean(safe_finite <= value) for value in false_finite]
                )
                if direction == "lower_is_false_safe":
                    false_percentiles = 100.0 - false_percentiles
            rows.append(
                {
                    "feature": name,
                    "trigger_candidate_count": int(mask.sum()),
                    "exact_safe_count": int((~false_label).sum()),
                    "false_safe_count": int(false_label.sum()),
                    "missing_count": int((~np.isfinite(values)).sum()),
                    "exact_safe_distribution": _distribution(safe_values),
                    "false_safe_distribution": _distribution(false_values),
                    "standardized_effect_false_minus_safe": _cohen_d(false_values, safe_values),
                    "false_safe_high_auroc": raw_auc,
                    "risk_direction": direction,
                    "best_direction_auroc": oriented_auc,
                    "false_safe_auprc": average_precision(oriented_values, false_label),
                    "false_safe_percentile_among_safe": _distribution(false_percentiles),
                }
            )
        return rows

    overall = analyze(trigger_mask)
    by_task = {
        str(int(task)): analyze(trigger_mask & (task_ids == task))
        for task in sorted(np.unique(task_ids[row_mask]).tolist())
    }
    return overall, by_task


def underestimation_analysis(
    features: dict[str, np.ndarray], targets: dict[str, np.ndarray], row_mask: np.ndarray
) -> list[dict]:
    residual = targets["residual_exact_minus_predicted"]
    rows = []
    for name in RUNTIME_FEATURE_NAMES:
        valid = row_mask & np.isfinite(features[name]) & np.isfinite(residual)
        rows.append(
            {
                "feature": name,
                "N": int(valid.sum()),
                "pearson": _pearson(features[name][valid], residual[valid]),
                "spearman": _spearman(features[name][valid], residual[valid]),
            }
        )
    rows.sort(key=lambda row: abs(row["spearman"] or 0.0), reverse=True)
    return rows


def build_underestimation_hypothesis_summary(correlations: list[dict]) -> dict:
    by_name = {row["feature"]: row for row in correlations}
    groups = {
        "latent_ood_magnitude": (
            "latent_delta_full_rms",
            "latent_delta_max_abs",
            "normalized_x_l2",
            "normalized_x_rms",
            "normalized_x_max_abs",
        ),
        "sharp_score_drop": (
            "relative_score_drop",
            "score_difference_current_minus_previous",
            "score_ratio_current_to_previous",
        ),
        "concentrated_predicted_action_delta": (
            "predicted_per_step_mse_cv",
            "predicted_per_dim_mse_cv",
            "predicted_max_step_to_full_mse_ratio",
            "predicted_max_dim_to_full_mse_ratio",
        ),
        "early_terminal_iteration": ("terminal_iteration",),
    }
    result = {}
    for hypothesis, feature_names in groups.items():
        members = [by_name[name] for name in feature_names]
        strongest = max(members, key=lambda row: abs(row["spearman"] or 0.0))
        result[hypothesis] = {
            "features": members,
            "strongest_absolute_spearman_feature": strongest["feature"],
            "strongest_absolute_spearman": strongest["spearman"],
            "association_strength_interpretation": (
                "weak_or_negligible"
                if abs(strongest["spearman"] or 0.0) < 0.1
                else "modest"
                if abs(strongest["spearman"] or 0.0) < 0.3
                else "strong"
            ),
            "sign_is_feature_definition_dependent": True,
        }
    terminal_spearman = by_name["terminal_iteration"]["spearman"]
    result["early_terminal_iteration"]["earlier_iterations_associated_with_larger_residual"] = bool(
        terminal_spearman is not None and terminal_spearman < 0.0
    )
    return result


def fit_feature_cutoff(
    feature_values: np.ndarray,
    predicted_scores: np.ndarray,
    exact_safe: np.ndarray,
    terminal_iterations: np.ndarray,
    task_ids: np.ndarray,
    *,
    feature_name: str,
    calibration_tasks: Iterable[int],
    safe_retention: float,
    gate_threshold: float = EXPECTED_THRESHOLD,
    min_terminal_iteration: int = CURRENT_MIN_TERMINAL_ITER,
) -> dict:
    calibration_tasks = tuple(sorted(int(task) for task in calibration_tasks))
    if not calibration_tasks or not set(calibration_tasks).issubset(TRAIN_CAL_TASKS):
        raise ValueError("cutoff calibration tasks must be a non-empty subset of development tasks")
    if UNTOUCHED_TASK in calibration_tasks or FORENSIC_TASK in calibration_tasks:
        raise ValueError("task 4/5 cannot enter cutoff calibration")
    if not 0.0 < safe_retention <= 1.0:
        raise ValueError("safe retention must be in (0, 1]")
    candidate = (
        np.isin(task_ids, calibration_tasks)
        & (terminal_iterations >= min_terminal_iteration)
        & (predicted_scores <= gate_threshold)
        & np.isfinite(feature_values)
    )
    safe_values = np.asarray(feature_values[candidate & exact_safe], dtype=np.float64)
    false_values = np.asarray(feature_values[candidate & (~exact_safe)], dtype=np.float64)
    if safe_values.size == 0:
        raise RuntimeError(f"no safe calibration trigger rows for feature {feature_name}")
    raw_auc = binary_auroc(
        np.concatenate([safe_values, false_values]),
        np.concatenate(
            [np.zeros(len(safe_values), dtype=np.bool_), np.ones(len(false_values), dtype=np.bool_)]
        ),
    )
    if raw_auc is None:
        direction = "high_risk"
    else:
        direction = "high_risk" if raw_auc >= 0.5 else "low_risk"
    if direction == "high_risk":
        cutoff = float(np.quantile(safe_values, safe_retention))
        accepted_region = "value <= cutoff"
    else:
        cutoff = float(np.quantile(safe_values, 1.0 - safe_retention))
        accepted_region = "value >= cutoff"
    return {
        "feature": feature_name,
        "safe_retention_target": float(safe_retention),
        "calibration_tasks": list(calibration_tasks),
        "calibration_candidate_count": int(candidate.sum()),
        "calibration_safe_count": int(len(safe_values)),
        "calibration_false_safe_count": int(len(false_values)),
        "risk_direction": direction,
        "cutoff": cutoff,
        "accepted_region": accepted_region,
        "missing_policy": "accept",
    }


def cutoff_accept_mask(feature_values: np.ndarray, cutoff: dict) -> np.ndarray:
    values = np.asarray(feature_values, dtype=np.float64)
    accepted = np.ones(len(values), dtype=np.bool_)
    finite = np.isfinite(values)
    if cutoff["risk_direction"] == "high_risk":
        accepted[finite] = values[finite] <= float(cutoff["cutoff"])
    elif cutoff["risk_direction"] == "low_risk":
        accepted[finite] = values[finite] >= float(cutoff["cutoff"])
    else:
        raise ValueError("unknown cutoff risk direction")
    return accepted


def sequential_first_hit_replay(
    row_indices: np.ndarray,
    predicted_scores: np.ndarray,
    exact_safe: np.ndarray,
    trajectory_ids: np.ndarray,
    terminal_iterations: np.ndarray,
    *,
    gate_threshold: float = EXPECTED_THRESHOLD,
    min_terminal_iteration: int = CURRENT_MIN_TERMINAL_ITER,
    veto_accept_mask: np.ndarray | None = None,
) -> tuple[dict, np.ndarray]:
    row_indices = np.asarray(row_indices, dtype=np.int64)
    if veto_accept_mask is None:
        veto_accept_mask = np.ones(len(predicted_scores), dtype=np.bool_)
    by_trajectory: dict[int, list[int]] = defaultdict(list)
    for index in row_indices:
        by_trajectory[int(trajectory_ids[index])].append(int(index))
    activated = []
    score_calls = 0
    terminal_distribution = Counter()
    for members in by_trajectory.values():
        members.sort(key=lambda index: int(terminal_iterations[index]))
        for index in members:
            if int(terminal_iterations[index]) < min_terminal_iteration:
                continue
            score_calls += 1
            if predicted_scores[index] <= gate_threshold and veto_accept_mask[index]:
                activated.append(index)
                terminal_distribution[int(terminal_iterations[index])] += 1
                break
    activated = np.asarray(activated, dtype=np.int64)
    safe_count = int(exact_safe[activated].sum()) if len(activated) else 0
    false_count = int(len(activated) - safe_count)
    trajectory_count = len(by_trajectory)
    replay = {
        "trajectory_count": int(trajectory_count),
        "score_call_count": int(score_calls),
        "accepted_triggers": int(len(activated)),
        "exact_safe_accepted": safe_count,
        "false_safe_accepted": false_count,
        "false_safe_rate": float(false_count / len(activated)) if len(activated) else None,
        "precision": float(safe_count / len(activated)) if len(activated) else None,
        "capture": float(safe_count / trajectory_count) if trajectory_count else None,
        "no_skip_count": int(trajectory_count - len(activated)),
        "coda_calls_saved_proxy": int(len(activated)),
        "score_calls_per_saved_coda": float(score_calls / len(activated)) if len(activated) else None,
        "sequential_first_hit_terminal_distribution": {
            str(key): int(value) for key, value in sorted(terminal_distribution.items())
        },
    }
    return replay, activated


def _aggregate_replays(task_replays: dict[int, dict]) -> dict:
    summed_keys = (
        "trajectory_count",
        "score_call_count",
        "accepted_triggers",
        "exact_safe_accepted",
        "false_safe_accepted",
        "no_skip_count",
        "coda_calls_saved_proxy",
    )
    result = {key: int(sum(replay[key] for replay in task_replays.values())) for key in summed_keys}
    accepted = result["accepted_triggers"]
    trajectories = result["trajectory_count"]
    result.update(
        {
            "false_safe_rate": result["false_safe_accepted"] / accepted if accepted else None,
            "precision": result["exact_safe_accepted"] / accepted if accepted else None,
            "capture": result["exact_safe_accepted"] / trajectories if trajectories else None,
            "score_calls_per_saved_coda": result["score_call_count"] / accepted if accepted else None,
        }
    )
    terminal_distribution = Counter()
    for replay in task_replays.values():
        terminal_distribution.update(
            {int(key): value for key, value in replay["sequential_first_hit_terminal_distribution"].items()}
        )
    result["sequential_first_hit_terminal_distribution"] = {
        str(key): int(value) for key, value in sorted(terminal_distribution.items())
    }
    result["by_evaluation_task"] = {str(task): replay for task, replay in sorted(task_replays.items())}
    return result


def evaluate_inner_oof_feature_safeguard(
    feature_name: str,
    safe_retention: float,
    features: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    task_ids: np.ndarray,
    trajectory_ids: np.ndarray,
) -> dict:
    cutoffs = {}
    task_replays = {}
    predicted_scores = features["predicted_action_delta_mse"]
    terminal_iterations = features["terminal_iteration"]
    for evaluation_task in TRAIN_CAL_TASKS:
        calibration_tasks = tuple(task for task in TRAIN_CAL_TASKS if task != evaluation_task)
        cutoff = fit_feature_cutoff(
            features[feature_name],
            predicted_scores,
            targets["exact_safe"],
            terminal_iterations,
            task_ids,
            feature_name=feature_name,
            calibration_tasks=calibration_tasks,
            safe_retention=safe_retention,
        )
        if evaluation_task in cutoff["calibration_tasks"]:
            raise RuntimeError("inner evaluation task leaked into cutoff calibration")
        accept_mask = cutoff_accept_mask(features[feature_name], cutoff)
        replay, _ = sequential_first_hit_replay(
            np.flatnonzero(task_ids == evaluation_task),
            predicted_scores,
            targets["exact_safe"],
            trajectory_ids,
            terminal_iterations,
            veto_accept_mask=accept_mask,
        )
        cutoffs[str(evaluation_task)] = cutoff
        task_replays[evaluation_task] = replay
    result = _aggregate_replays(task_replays)
    result.update(
        {
            "kind": "single_feature_veto",
            "feature": feature_name,
            "safe_retention_target": float(safe_retention),
            "inner_task_cutoffs": cutoffs,
            "cutoff_selection": "leave-one-development-task-out",
        }
    )
    return result


def _annotate_vs_baseline(point: dict, baseline: dict) -> None:
    point["false_safe_removed_vs_current"] = baseline["false_safe_accepted"] - point["false_safe_accepted"]
    point["safe_skips_lost_vs_current"] = baseline["exact_safe_accepted"] - point["exact_safe_accepted"]
    point["safe_skip_retention_vs_current"] = (
        point["exact_safe_accepted"] / baseline["exact_safe_accepted"]
        if baseline["exact_safe_accepted"]
        else None
    )
    baseline_risk = baseline["false_safe_rate"]
    point_risk = point["false_safe_rate"]
    point["false_safe_rate_reduction_vs_current"] = (
        baseline_risk - point_risk
        if baseline_risk is not None and point_risk is not None
        else None
    )


def _pareto_frontier(points: list[dict]) -> list[dict]:
    frontier = []
    for candidate in points:
        dominated = False
        for other in points:
            if other is candidate:
                continue
            no_more_false = other["false_safe_accepted"] <= candidate["false_safe_accepted"]
            no_less_safe = other["exact_safe_accepted"] >= candidate["exact_safe_accepted"]
            strictly_better = (
                other["false_safe_accepted"] < candidate["false_safe_accepted"]
                or other["exact_safe_accepted"] > candidate["exact_safe_accepted"]
            )
            if no_more_false and no_less_safe and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return sorted(
        frontier,
        key=lambda point: (point["false_safe_accepted"], -point["exact_safe_accepted"]),
    )


def _simple_vs_complex(point: dict, min6: dict) -> dict:
    point_risk = point["false_safe_rate"] if point["false_safe_rate"] is not None else 1.0
    min6_risk = min6["false_safe_rate"] if min6["false_safe_rate"] is not None else 1.0
    similar_capture_floor = 0.95 * min6["exact_safe_accepted"]
    similar_risk_ceiling = min6_risk + max(0.001, 0.10 * min6_risk)
    lower_risk_similar_capture = point_risk < min6_risk and point["exact_safe_accepted"] >= similar_capture_floor
    higher_capture_similar_risk = (
        point["exact_safe_accepted"] > min6["exact_safe_accepted"]
        and point_risk <= similar_risk_ceiling
    )
    return {
        "reference": "min_terminal_6",
        "similar_capture_floor_exact_safe": float(similar_capture_floor),
        "similar_false_safe_rate_ceiling": float(similar_risk_ceiling),
        "lower_risk_at_similar_capture": bool(lower_risk_similar_capture),
        "higher_capture_at_similar_risk": bool(higher_capture_similar_risk),
        "interesting": bool(lower_risk_similar_capture or higher_capture_similar_risk),
    }


def _measure_veto_overhead(features: dict[str, np.ndarray], cutoff_specs: list[dict]) -> dict:
    if not cutoff_specs:
        return {"measured": False, "reason": "no feature safeguards selected"}
    samples = []
    repetitions = 100
    operations_per_repetition = sum(
        len(features[spec["feature"]]) for spec in cutoff_specs
    )
    checksum = 0
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        for spec in cutoff_specs:
            values = features[spec["feature"]]
            cutoff = float(spec["final_cutoff"]["cutoff"])
            high_risk = spec["final_cutoff"]["risk_direction"] == "high_risk"
            for value in values:
                checksum += int(
                    not math.isfinite(value)
                    or (value <= cutoff if high_risk else value >= cutoff)
                )
        samples.append(time.perf_counter_ns() - start)
    per_operation = np.asarray(samples, dtype=np.float64) / operations_per_repetition
    return {
        "measured": True,
        "device": "CPU/NumPy",
        "scope": (
            "scalar feature finite-check and one cutoff comparison; feature "
            "construction and frozen predictor excluded"
        ),
        "repetitions": repetitions,
        "p50_nanoseconds_per_row": float(np.percentile(per_operation, 50)),
        "p95_nanoseconds_per_row": float(np.percentile(per_operation, 95)),
        "gpu_scorer_overhead_measured": False,
        "feature_construction_overhead_measured": False,
        "checksum": int(checksum),
    }


def _task4_forensics(
    selected_specs: list[dict],
    features: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    task_ids: np.ndarray,
    trajectory_ids: np.ndarray,
    ks: np.ndarray,
) -> dict:
    task4_rows = np.flatnonzero(task_ids == FORENSIC_TASK)
    baseline, baseline_activated = sequential_first_hit_replay(
        task4_rows,
        features["predicted_action_delta_mse"],
        targets["exact_safe"],
        trajectory_ids,
        features["terminal_iteration"],
    )
    # These are offline calibration-distribution candidates only.  Episode and
    # prediction counters are not global identities across protocol partitions,
    # so they must not be labeled as closed-loop oracle-confirm events.
    offline_false_safe_candidates = task4_rows[
        (features["predicted_action_delta_mse"][task4_rows] <= EXPECTED_THRESHOLD)
        & (~targets["exact_safe"][task4_rows])
    ]
    safe_cases = baseline_activated[targets["exact_safe"][baseline_activated]]
    safeguards = []
    decisions_by_row: dict[int, list[dict]] = defaultdict(list)
    for spec in selected_specs:
        cutoff = spec["final_cutoff"]
        accept_mask = cutoff_accept_mask(features[spec["feature"]], cutoff)
        replay, activated = sequential_first_hit_replay(
            task4_rows,
            features["predicted_action_delta_mse"],
            targets["exact_safe"],
            trajectory_ids,
            features["terminal_iteration"],
            veto_accept_mask=accept_mask,
        )
        baseline_safe_vetoed = safe_cases[~accept_mask[safe_cases]]
        safeguards.append(
            {
                "feature": spec["feature"],
                "safe_retention_target": spec["safe_retention_target"],
                "cutoff": cutoff,
                "sequential_replay": replay,
                "baseline_exact_safe_first_hits_vetoed": int(len(baseline_safe_vetoed)),
                "task4_exact_safe_accepted_change_vs_baseline": (
                    replay["exact_safe_accepted"] - baseline["exact_safe_accepted"]
                ),
            }
        )
        for index in offline_false_safe_candidates:
            value = features[spec["feature"]][index]
            accepted = bool(accept_mask[index])
            decisions_by_row[int(index)].append(
                {
                    "feature": spec["feature"],
                    "safe_retention_target": spec["safe_retention_target"],
                    "feature_value": _json_scalar(value),
                    "accepted": accepted,
                    "decision": "accepted" if accepted else "vetoed",
                    "why": (
                        "history unavailable; missing-policy accepts"
                        if not np.isfinite(value)
                        else f"{value:.9g} satisfies {cutoff['accepted_region']} ({cutoff['cutoff']:.9g})"
                        if accepted
                        else f"{value:.9g} violates {cutoff['accepted_region']} ({cutoff['cutoff']:.9g})"
                    ),
                }
            )
    offline_false_safe_rows = []
    for index in offline_false_safe_candidates:
        offline_false_safe_rows.append(
            {
                "row_index_in_analysis": int(index),
                "task": FORENSIC_TASK,
                "trajectory_id": int(trajectory_ids[index]),
                "offline_k": int(ks[index]),
                "runtime_terminal_iteration": int(features["terminal_iteration"][index]),
                "eligible_under_min_terminal_5": bool(
                    features["terminal_iteration"][index]
                    >= CURRENT_MIN_TERMINAL_ITER
                ),
                "current_min_terminal_5_cadence_decision": (
                    "eligible"
                    if features["terminal_iteration"][index]
                    >= CURRENT_MIN_TERMINAL_ITER
                    else "not_scored_ineligible"
                ),
                "predicted_gate_score": float(features["predicted_action_delta_mse"][index]),
                "exact_adjacent_action_mse": float(targets["exact_adjacent_action_mse"][index]),
                "safeguard_decisions": decisions_by_row[int(index)],
            }
        )
    return {
        "task": FORENSIC_TASK,
        "evaluated_after_cutoff_freeze": True,
        "baseline_min_terminal_5": baseline,
        "offline_false_safe_candidate_count": int(
            len(offline_false_safe_candidates)
        ),
        "offline_false_safe_candidates_are_runtime_identity_matches": False,
        "offline_false_safe_candidates": offline_false_safe_rows,
        "baseline_exact_safe_first_hit_count": int(len(safe_cases)),
        "safeguards": safeguards,
    }


def _write_feature_csv(
    path: Path,
    original_indices: np.ndarray,
    task_ids: np.ndarray,
    trajectory_ids: np.ndarray,
    ks: np.ndarray,
    features: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
) -> None:
    fieldnames = [
        "original_cache_row_index",
        "data_role",
        "task_id",
        "trajectory_id",
        "offline_k",
        *RUNTIME_FEATURE_NAMES,
        *(f"evaluation_{name}" for name in EVALUATION_TARGET_NAMES),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(len(task_ids)):
            task = int(task_ids[index])
            row = {
                "original_cache_row_index": int(original_indices[index]),
                "data_role": "train_cal" if task in TRAIN_CAL_TASKS else "task4_posthoc_forensic",
                "task_id": task,
                "trajectory_id": int(trajectory_ids[index]),
                "offline_k": int(ks[index]),
            }
            row.update({name: _json_scalar(features[name][index]) for name in RUNTIME_FEATURE_NAMES})
            row.update(
                {
                    f"evaluation_{name}": _json_scalar(targets[name][index])
                    for name in EVALUATION_TARGET_NAMES
                }
            )
            writer.writerow(row)


def _write_table_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(results: dict) -> None:
    baseline = results["sequential_analysis"]["baseline_min_terminal_5"]
    min6 = results["sequential_analysis"]["baseline_min_terminal_6"]
    print(
        "Development sequential replay: "
        f"min-T5 safe={baseline['exact_safe_accepted']} false={baseline['false_safe_accepted']} "
        f"calls={baseline['score_call_count']}; "
        f"min-T6 safe={min6['exact_safe_accepted']} false={min6['false_safe_accepted']} "
        f"calls={min6['score_call_count']}"
    )
    print("\nTop descriptive false-safe features:")
    for row in results["ranked_feature_table"][:10]:
        print(
            f"  {row['feature']:<48} AUROC={row['best_direction_auroc']!s:<8} "
            f"AUPRC={row['false_safe_auprc']!s:<8} direction={row['risk_direction']}"
        )
    print("\nRisk-capture frontier:")
    for row in results["risk_capture_frontier"]:
        label = row.get("feature", row.get("name", row["kind"]))
        print(
            f"  {label:<48} safe={row['exact_safe_accepted']:>4} "
            f"false={row['false_safe_accepted']:>3} "
            f"risk={row['false_safe_rate']!s:<10} calls={row['score_call_count']}"
        )
    task4 = results["task4_posthoc_forensics"]
    print("\nOffline Task-4 false-safe candidates (not runtime-identity matched):")
    for case in task4["offline_false_safe_candidates"]:
        vetoed = sum(not decision["accepted"] for decision in case["safeguard_decisions"])
        print(
            f"  traj={case['trajectory_id']} k={case['offline_k']} "
            f"terminal={case['runtime_terminal_iteration']} score={case['predicted_gate_score']:.9g} "
            f"exact_mse={case['exact_adjacent_action_mse']:.9g} "
            f"vetoed_by={vetoed}/{len(case['safeguard_decisions'])}"
        )
    print(f"\nJSON: {results['outputs']['json']}")
    print(f"Features CSV: {results['outputs']['features_csv']}")
    print(f"Ranked CSV: {results['outputs']['ranked_features_csv']}")
    print(f"Frontier CSV: {results['outputs']['risk_capture_frontier_csv']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("benchmark_results/coda_anchor_feasibility/action_delta_cache.pt"),
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path(
            "benchmark_results/coda_anchor_feasibility/action_delta_gate_fold4/action_delta_gate.pt"
        ),
    )
    output_root = Path("benchmark_results/coda_anchor_feasibility/false_safe_signal_diagnostics")
    parser.add_argument("--output", type=Path, default=output_root / "results.json")
    parser.add_argument("--features-csv", type=Path, default=output_root / "transition_features.csv")
    parser.add_argument("--ranked-csv", type=Path, default=output_root / "ranked_features.csv")
    parser.add_argument("--frontier-csv", type=Path, default=output_root / "risk_capture_frontier.csv")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    artifact_sha = sha256_file(args.artifact)
    if artifact_sha != EXPECTED_ARTIFACT_SHA256:
        raise RuntimeError(f"frozen artifact hash mismatch: {artifact_sha}")
    manifest, payload = load_action_delta_gate_artifact(
        args.artifact, expected_sha256=EXPECTED_ARTIFACT_SHA256
    )
    threshold = float(payload["threshold"])
    if threshold != EXPECTED_THRESHOLD:
        raise RuntimeError(f"frozen threshold mismatch: {threshold}")
    if int(payload["outer_fold"]) != OUTER_FOLD:
        raise RuntimeError("artifact is not the fold-4 predictor")

    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    _validate_cache(cache, payload)
    source_task_ids = cache["task_ids"].numpy().astype(np.int64)
    analysis_indices = select_analysis_indices(source_task_ids)
    delta_states = cache["delta_states"][analysis_indices]
    task_ids = source_task_ids[analysis_indices]
    trajectory_ids = cache["trajectory_ids"].numpy()[analysis_indices].astype(np.int64)
    ks = cache["ks"].numpy()[analysis_indices].astype(np.int64)
    exact_mse = cache["target_mse"].numpy()[analysis_indices].astype(np.float64)

    predicted_delta = predict_frozen_delta(
        delta_states, payload, torch.device(args.device), args.batch_size
    )
    features = build_runtime_features(
        delta_states,
        predicted_delta,
        trajectory_ids,
        ks,
        threshold,
        payload["x_mean"],
        payload["x_std"],
        batch_size=args.batch_size,
    )
    targets = build_evaluation_targets(
        exact_mse, features["predicted_action_delta_mse"], threshold
    )
    if set(RUNTIME_FEATURE_NAMES) & set(EVALUATION_TARGET_NAMES):
        raise RuntimeError("exact-action evaluation targets leaked into runtime feature names")

    development_mask = np.isin(task_ids, TRAIN_CAL_TASKS)
    descriptive, descriptive_by_task = descriptive_feature_analysis(
        features, targets, task_ids, development_mask
    )
    underestimation = underestimation_analysis(features, targets, development_mask)
    underestimation_hypotheses = build_underestimation_hypothesis_summary(
        underestimation
    )
    ranked_features = sorted(
        descriptive,
        key=lambda row: (
            -(row["best_direction_auroc"] or 0.0),
            -(row["false_safe_auprc"] or 0.0),
        ),
    )

    development_rows = np.flatnonzero(development_mask)
    baseline5, _ = sequential_first_hit_replay(
        development_rows,
        features["predicted_action_delta_mse"],
        targets["exact_safe"],
        trajectory_ids,
        features["terminal_iteration"],
    )
    baseline5.update({"kind": "baseline", "name": "frozen_gate_min_terminal_5"})
    baseline6, _ = sequential_first_hit_replay(
        development_rows,
        features["predicted_action_delta_mse"],
        targets["exact_safe"],
        trajectory_ids,
        features["terminal_iteration"],
        min_terminal_iteration=6,
    )
    baseline6.update({"kind": "baseline", "name": "frozen_gate_min_terminal_6"})

    scalar_points = []
    for factor in STRICT_SCORE_FACTORS:
        replay, _ = sequential_first_hit_replay(
            development_rows,
            features["predicted_action_delta_mse"],
            targets["exact_safe"],
            trajectory_ids,
            features["terminal_iteration"],
            gate_threshold=threshold * factor,
        )
        replay.update(
            {
                "kind": "diagnostic_stricter_scalar_threshold",
                "name": f"score_threshold_x_{factor:g}",
                "threshold_factor": factor,
                "threshold": threshold * factor,
                "not_recalibrated": True,
            }
        )
        scalar_points.append(replay)

    feature_points = []
    for feature_name in RUNTIME_FEATURE_NAMES:
        for retention in CUTOFF_SAFE_RETENTIONS:
            feature_points.append(
                evaluate_inner_oof_feature_safeguard(
                    feature_name,
                    retention,
                    features,
                    targets,
                    task_ids,
                    trajectory_ids,
                )
            )
    all_points = [baseline5, baseline6, *scalar_points, *feature_points]
    for point in all_points:
        _annotate_vs_baseline(point, baseline5)
    for point in feature_points:
        point["simple_vs_min_terminal_6"] = _simple_vs_complex(point, baseline6)
    frontier = _pareto_frontier(all_points)

    frontier_feature_points = [point for point in frontier if point["kind"] == "single_feature_veto"]
    if not frontier_feature_points:
        frontier_feature_points = sorted(
            feature_points,
            key=lambda point: (point["false_safe_accepted"], -point["exact_safe_accepted"]),
        )[:5]
    selected_specs = []
    for point in frontier_feature_points:
        final_cutoff = fit_feature_cutoff(
            features[point["feature"]],
            features["predicted_action_delta_mse"],
            targets["exact_safe"],
            features["terminal_iteration"],
            task_ids,
            feature_name=point["feature"],
            calibration_tasks=TRAIN_CAL_TASKS,
            safe_retention=point["safe_retention_target"],
        )
        selected_specs.append(
            {
                "feature": point["feature"],
                "safe_retention_target": point["safe_retention_target"],
                "development_oof_result": point,
                "final_cutoff": final_cutoff,
            }
        )

    task4_forensics = _task4_forensics(
        selected_specs, features, targets, task_ids, trajectory_ids, ks
    )
    veto_overhead = _measure_veto_overhead(features, selected_specs)

    ranked_rows = []
    best_point_by_feature = {}
    for point in feature_points:
        existing = best_point_by_feature.get(point["feature"])
        key = (point["false_safe_accepted"], -point["exact_safe_accepted"])
        if existing is None or key < (
            existing["false_safe_accepted"],
            -existing["exact_safe_accepted"],
        ):
            best_point_by_feature[point["feature"]] = point
    for rank, row in enumerate(ranked_features, start=1):
        best = best_point_by_feature[row["feature"]]
        ranked_rows.append(
            {
                "rank": rank,
                "feature": row["feature"],
                "risk_direction": row["risk_direction"],
                "best_direction_auroc": row["best_direction_auroc"],
                "false_safe_auprc": row["false_safe_auprc"],
                "standardized_effect_false_minus_safe": row["standardized_effect_false_minus_safe"],
                "best_oof_safe_retention_target": best["safe_retention_target"],
                "best_oof_exact_safe_accepted": best["exact_safe_accepted"],
                "best_oof_false_safe_accepted": best["false_safe_accepted"],
                "best_oof_false_safe_rate": best["false_safe_rate"],
                "interesting_vs_min_terminal_6": best["simple_vs_min_terminal_6"]["interesting"],
            }
        )

    _write_feature_csv(
        args.features_csv,
        analysis_indices,
        task_ids,
        trajectory_ids,
        ks,
        features,
        targets,
    )
    _write_table_csv(args.ranked_csv, ranked_rows, list(ranked_rows[0]))
    frontier_rows = []
    for point in frontier:
        frontier_rows.append(
            {
                "kind": point["kind"],
                "name": point.get("name"),
                "feature": point.get("feature"),
                "safe_retention_target": point.get("safe_retention_target"),
                "score_call_count": point["score_call_count"],
                "accepted_triggers": point["accepted_triggers"],
                "exact_safe_accepted": point["exact_safe_accepted"],
                "false_safe_accepted": point["false_safe_accepted"],
                "false_safe_rate": point["false_safe_rate"],
                "precision": point["precision"],
                "capture": point["capture"],
                "safe_skip_retention_vs_current": point["safe_skip_retention_vs_current"],
                "false_safe_removed_vs_current": point["false_safe_removed_vs_current"],
                "safe_skips_lost_vs_current": point["safe_skips_lost_vs_current"],
            }
        )
    _write_table_csv(args.frontier_csv, frontier_rows, list(frontier_rows[0]))

    results = {
        "schema_version": 1,
        "analysis": "offline_action_delta_gate_false_safe_signals",
        "diagnostic_only": True,
        "runtime_code_modified": False,
        "libero_run": False,
        "artifact": {
            "path": str(args.artifact),
            "sha256": artifact_sha,
            "manifest": manifest,
            "threshold": threshold,
            "outer_fold": OUTER_FOLD,
        },
        "cache": {"path": str(args.cache), "sha256": sha256_file(args.cache)},
        "data_policy": {
            "train_cal_tasks": list(TRAIN_CAL_TASKS),
            "inner_protocol": "leave-one-task-out",
            "posthoc_forensic_task": FORENSIC_TASK,
            "untouched_task": UNTOUCHED_TASK,
            "task5_rows_loaded_into_feature_analysis": 0,
            "analysis_row_count": int(len(analysis_indices)),
            "train_cal_row_count": int(development_mask.sum()),
            "task4_row_count": int((task_ids == FORENSIC_TASK).sum()),
        },
        "causal_contract": {
            "row_k_predicts_runtime_terminal_iteration": "k + 1",
            "latent_delta": "cached BF16 (S_(k+1) - S_k)",
            "predictor_input": "(delta_state.float() - x_mean) / x_std",
            "predictor": "F.linear(x, linear_weight, linear_bias)",
            "prediction_denormalization": "pred_norm * y_std + y_mean",
            "history_requires_same_trajectory_immediate_k_minus_1": True,
            "missing_history_representation": "NaN in memory, empty field in CSV, null in JSON summaries",
            "exact_action_targets_used_as_runtime_features": False,
        },
        "runtime_feature_names": list(RUNTIME_FEATURE_NAMES),
        "evaluation_target_names": list(EVALUATION_TARGET_NAMES),
        "descriptive_separation": {
            "population": "development-task frozen predicted-trigger candidate rows",
            "overall": descriptive,
            "by_task": descriptive_by_task,
        },
        "underestimation_analysis": {
            "target": "exact_adjacent_action_mse - predicted_action_delta_mse",
            "correlations": underestimation,
            "hypothesis_summary": underestimation_hypotheses,
        },
        "sequential_analysis": {
            "protocol": "inner leave-one-task-out cutoff fit; exact sequential first hit",
            "baseline_min_terminal_5": baseline5,
            "baseline_min_terminal_6": baseline6,
            "stricter_scalar_thresholds_diagnostic_only": scalar_points,
            "single_feature_veto_points": feature_points,
            "cutoff_safe_retention_grid": list(CUTOFF_SAFE_RETENTIONS),
            "veto_overhead": veto_overhead,
        },
        "simple_vs_complex": {
            "reference": baseline6,
            "interesting_feature_points": [
                point
                for point in feature_points
                if point["simple_vs_min_terminal_6"]["interesting"]
            ],
        },
        "risk_capture_frontier": frontier,
        "ranked_feature_table": ranked_rows,
        "frozen_task4_safeguards": selected_specs,
        "task4_posthoc_forensics": task4_forensics,
        "limitations": {
            "development_predicted_trigger_false_safe_row_count": int(
                sum(
                    row["false_safe_count"]
                    for row in descriptive[:1]
                )
            ),
            "development_min_terminal_5_sequential_false_safe_count": int(
                baseline5["false_safe_accepted"]
            ),
            "rare_event_warning": (
                "Feature ranking and cutoff frontiers are exploratory because "
                "the development data contain very few false-safe events."
            ),
            "task4_known_case_cadence_note": (
                "The two cached task-4 false-safe candidates are calibration-"
                "distribution rows at mapped runtime terminals 2 and 4. They "
                "are not the closed-loop oracle-confirm events."
            ),
            "task5_evaluated": False,
        },
        "outputs": {
            "json": str(args.output),
            "features_csv": str(args.features_csv),
            "ranked_features_csv": str(args.ranked_csv),
            "risk_capture_frontier_csv": str(args.frontier_csv),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _print_summary(results)


if __name__ == "__main__":
    main()
