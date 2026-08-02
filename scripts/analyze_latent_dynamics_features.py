#!/usr/bin/env python3
"""Leakage-safe offline analysis of LIBERO latent-dynamics traces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.check_latent_dynamics_trace import (  # noqa: E402
    HISTORY_DEPENDENT_FIELDS,
    LATENT_DYNAMICS_FIELDS,
    load_jsonl,
    validate_records,
)
from scripts.origin_aware_replay_lib import parse_fold_manifest  # noqa: E402


SCHEMA_VERSION = 1
EXPECTED_IDENTITY_SHA256 = (
    "11e9625e136e2c1c08255a020b10a4b6645f8136a9c49d6bbf383f30d987b268"
)
EXPECTED_PREDICTION_COUNT = 2398
EXPECTED_TRANSITION_COUNT = 74338
DEFAULT_INPUT_ROOT = (
    REPO_ROOT / "benchmark_results/latent_dynamics_features/calibration_b17fe9d"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_ROOT / "analysis"
DEFAULT_FOLD_MANIFEST = (
    REPO_ROOT
    / "experiments/robot/libero/manifests/libero_spatial_task_oof_5fold_v1.json"
)

EXISTING_METRICS = (
    "raw_mse",
    "relative_mse",
    "relative_l2",
    "cosine_distance",
)
UPDATE_DYNAMICS = (
    "contraction_ratio",
    "update_turning_cosine",
    "acceleration_rms",
    "acceleration_ratio",
    "state_norm_ratio",
)
TOKEN_DYNAMICS = (
    "token_update_p50",
    "token_update_p90",
    "token_update_p95",
    "token_update_max",
    "token_update_cv",
    "token_update_energy_entropy",
    "token_update_top10_fraction",
)
WARM_ANCHOR = (
    "warm_anchor_relative_l2",
    "warm_anchor_cosine_distance",
)
INDIVIDUAL_FEATURES = (
    "iteration_index",
    *EXISTING_METRICS,
    *LATENT_DYNAMICS_FIELDS,
)
FEATURE_GROUPS = {
    "iteration_index": ("iteration_index",),
    "raw_mse": ("raw_mse",),
    "existing_metrics": EXISTING_METRICS,
    "update_dynamics": UPDATE_DYNAMICS,
    "token_dynamics": TOKEN_DYNAMICS,
    "warm_anchor": WARM_ANCHOR,
    "combined": (*EXISTING_METRICS, *UPDATE_DYNAMICS, *TOKEN_DYNAMICS, *WARM_ANCHOR),
}
LEAKAGE_EXCLUSIONS = frozenset(
    {
        "K_t",
        "baseline_k",
        "baseline_stopping_iteration",
        "adjacent_action_mse",
        "action_mse_below_0_001",
        "stop_reason",
        "canonical_stop_reason",
        "success",
        "activation_due",
        "activation_target",
        "relative_iteration",
    }
)
OUTPUT_FILENAMES = (
    "dataset_summary.json",
    "prediction_summary.csv",
    "feature_distribution_summary.csv",
    "univariate_oof_results.csv",
    "feature_group_oof_results.csv",
    "trajectory_relative_to_activation.csv",
    "analysis_report.json",
)


class LatentDynamicsAnalysisError(ValueError):
    """Raised when an input or analysis invariant is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LatentDynamicsAnalysisError(message)


def _json_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    _require(math.isfinite(result), "required feature value is non-finite")
    return result


def _mean(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _std(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return float(np.std(finite)) if finite else None


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile)) if values else None


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_trace_paths(input_root: Path) -> list[Path]:
    paths = [input_root / f"task{task_id}/steps.jsonl" for task_id in range(10)]
    missing = [str(path) for path in paths if not path.is_file()]
    _require(not missing, "missing trace files: " + ", ".join(missing))
    return paths


def validate_input_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_identity_sha256: str,
    expected_prediction_count: int,
    expected_transition_count: int,
) -> dict[str, Any]:
    contract = validate_records(
        records, expected_identity_sha256=expected_identity_sha256
    )
    _require(
        contract["prediction_count"] == expected_prediction_count,
        "prediction count mismatch: "
        f"expected={expected_prediction_count}, actual={contract['prediction_count']}",
    )
    _require(
        contract["transition_count"] == expected_transition_count,
        "transition count mismatch: "
        f"expected={expected_transition_count}, actual={contract['transition_count']}",
    )
    return contract


def difficulty_for_k(baseline_k: int) -> str:
    if baseline_k <= 4:
        return "easy"
    if baseline_k <= 7:
        return "medium"
    return "hard"


def activation_target_for_k(baseline_k: int) -> int:
    return max(2, int(baseline_k) - 1)


def build_analysis_dataset(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    for record in records:
        key = (
            int(record["task_id"]),
            int(record["episode_id"]),
            int(record["prediction_step"]),
        )
        _require(key not in seen, f"duplicate prediction identity: {key}")
        seen.add(key)
        baseline_k = int(record["K_t"])
        activation_target = activation_target_for_k(baseline_k)
        origin = str(record["actual_origin"])
        _require(origin in {"ACTUAL_WARM", "COLD"}, f"invalid origin: {origin}")
        prediction = {
            "task_id": key[0],
            "episode_id": key[1],
            "prediction_id": key[2],
            "actual_origin": origin,
            "baseline_k": baseline_k,
            "activation_target": activation_target,
            "difficulty": difficulty_for_k(baseline_k),
            "full_transition_count": len(record["latent_metric_trace"]),
            "primary_transition_count": sum(
                int(item["iteration_index"]) <= baseline_k
                for item in record["latent_metric_trace"]
            ),
        }
        predictions.append(prediction)
        for item in record["latent_metric_trace"]:
            k = int(item["iteration_index"])
            features = {
                feature: (
                    float(k)
                    if feature == "iteration_index"
                    else _json_float(item.get(feature))
                )
                for feature in INDIVIDUAL_FEATURES
            }
            for feature in HISTORY_DEPENDENT_FIELDS:
                _require(
                    (features[feature] is None) == (k == 2),
                    f"{key} k={k}: {feature} null contract mismatch",
                )
            transitions.append(
                {
                    "task_id": key[0],
                    "episode_id": key[1],
                    "prediction_id": key[2],
                    "actual_origin": origin,
                    "baseline_k": baseline_k,
                    "activation_target": activation_target,
                    "iteration_index": k,
                    "relative_iteration": k - activation_target,
                    "activation_due": bool(k >= activation_target),
                    "primary_window": bool(k <= baseline_k),
                    "difficulty": prediction["difficulty"],
                    "features": features,
                }
            )
    predictions.sort(key=lambda item: (item["task_id"], item["episode_id"], item["prediction_id"]))
    transitions.sort(
        key=lambda item: (
            item["task_id"],
            item["episode_id"],
            item["prediction_id"],
            item["iteration_index"],
        )
    )
    return predictions, transitions


def primary_rows(
    transitions: Sequence[Mapping[str, Any]], *, origin: str = "ACTUAL_WARM"
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in transitions
        if row["actual_origin"] == origin and bool(row["primary_window"])
    ]


def secondary_rows(
    transitions: Sequence[Mapping[str, Any]], *, origin: str
) -> list[Mapping[str, Any]]:
    return [row for row in transitions if row["actual_origin"] == origin]


def build_dataset_summary(
    predictions: Sequence[Mapping[str, Any]],
    transitions: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    by_task = Counter(str(item["task_id"]) for item in predictions)
    by_episode = Counter(
        f"{item['task_id']}:{item['episode_id']}" for item in predictions
    )
    by_origin = Counter(str(item["actual_origin"]) for item in predictions)
    by_terminal_k = Counter(str(item["baseline_k"]) for item in predictions)
    transition_origin = Counter(str(item["actual_origin"]) for item in transitions)
    primary = [item for item in transitions if item["primary_window"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": dict(contract),
        "prediction_count": len(predictions),
        "transition_count_full_depth_shadow_inclusive": len(transitions),
        "transition_count_primary_production_window": len(primary),
        "duplicate_prediction_identity_count": 0,
        "counts_by_task": dict(sorted(by_task.items(), key=lambda item: int(item[0]))),
        "counts_by_task_episode": dict(
            sorted(
                by_episode.items(),
                key=lambda item: tuple(int(value) for value in item[0].split(":")),
            )
        ),
        "prediction_counts_by_actual_origin": dict(sorted(by_origin.items())),
        "transition_counts_by_actual_origin_full_depth": dict(
            sorted(transition_origin.items())
        ),
        "prediction_counts_by_terminal_k": dict(
            sorted(by_terminal_k.items(), key=lambda item: int(item[0]))
        ),
        "primary_population": "ACTUAL_WARM",
        "secondary_population": "COLD descriptive only",
        "primary_window": "iteration_index <= K_t",
        "secondary_window": "full k=2..32 shadow-inclusive trace",
        "history_null_policy": "history-dependent fields are null exactly at k=2",
    }


def load_fold_assignment(path: Path, task_ids: Iterable[int]) -> tuple[dict[str, Any], dict[str, int]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assignment = parse_fold_manifest(manifest, {str(task) for task in task_ids})
    _require(len(set(assignment.values())) == 5, "expected exactly five task folds")
    return manifest, assignment


def build_task_splits(
    rows: Sequence[Mapping[str, Any]], assignment: Mapping[str, int]
) -> list[dict[str, Any]]:
    splits = []
    for fold_id in sorted(set(assignment.values())):
        train_tasks = sorted(
            (task for task, fold in assignment.items() if fold != fold_id), key=int
        )
        validation_tasks = sorted(
            (task for task, fold in assignment.items() if fold == fold_id), key=int
        )
        _require(set(train_tasks).isdisjoint(validation_tasks), f"fold {fold_id}: task leakage")
        train_rows = [row for row in rows if str(row["task_id"]) in train_tasks]
        validation_rows = [
            row for row in rows if str(row["task_id"]) in validation_tasks
        ]
        _require(train_rows and validation_rows, f"fold {fold_id}: empty split")
        splits.append(
            {
                "fold_id": fold_id,
                "training_task_ids": train_tasks,
                "validation_task_ids": validation_tasks,
                "train_rows": train_rows,
                "validation_rows": validation_rows,
            }
        )
    return splits


def _feature_matrix_raw(
    rows: Sequence[Mapping[str, Any]], feature_names: Sequence[str]
) -> np.ndarray:
    _require(not (set(feature_names) & LEAKAGE_EXCLUSIONS), "label/leakage field requested as input")
    _require(set(feature_names) <= set(INDIVIDUAL_FEATURES), "unknown input feature")
    return np.asarray(
        [
            [
                np.nan if row["features"][name] is None else float(row["features"][name])
                for name in feature_names
            ]
            for row in rows
        ],
        dtype=np.float64,
    )


def fit_preprocessor(
    rows: Sequence[Mapping[str, Any]], feature_names: Sequence[str]
) -> dict[str, Any]:
    raw = _feature_matrix_raw(rows, feature_names)
    imputation = []
    expanded_names = []
    columns = []
    for index, name in enumerate(feature_names):
        column = raw[:, index]
        available = np.isfinite(column)
        _require(bool(available.any()), f"training feature has no available values: {name}")
        median = float(np.median(column[available]))
        imputation.append(median)
        columns.append(np.where(available, column, median))
        expanded_names.append(name)
        if name in HISTORY_DEPENDENT_FIELDS:
            columns.append(available.astype(np.float64))
            expanded_names.append(f"{name}__available")
    matrix = np.column_stack(columns)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return {
        "feature_names": list(feature_names),
        "expanded_feature_names": expanded_names,
        "imputation_medians": imputation,
        "scaling_mean": mean.tolist(),
        "scaling_scale": scale.tolist(),
        "fit_task_ids": sorted({str(row["task_id"]) for row in rows}, key=int),
    }


def transform_features(
    rows: Sequence[Mapping[str, Any]], preprocessor: Mapping[str, Any]
) -> np.ndarray:
    names = tuple(preprocessor["feature_names"])
    raw = _feature_matrix_raw(rows, names)
    columns = []
    for index, name in enumerate(names):
        column = raw[:, index]
        available = np.isfinite(column)
        median = float(preprocessor["imputation_medians"][index])
        columns.append(np.where(available, column, median))
        if name in HISTORY_DEPENDENT_FIELDS:
            columns.append(available.astype(np.float64))
    matrix = np.column_stack(columns)
    mean = np.asarray(preprocessor["scaling_mean"], dtype=np.float64)
    scale = np.asarray(preprocessor["scaling_scale"], dtype=np.float64)
    transformed = (matrix - mean) / scale
    _require(np.isfinite(transformed).all(), "non-finite transformed feature matrix")
    return transformed


def labels_for_rows(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([bool(row["activation_due"]) for row in rows], dtype=bool)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def binary_auroc(labels: Sequence[bool], scores: Sequence[float]) -> float | None:
    y = np.asarray(labels, dtype=bool)
    score = np.asarray(scores, dtype=np.float64)
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = _rankdata(score)
    return float(
        (ranks[y].sum() - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def binary_auprc(labels: Sequence[bool], scores: Sequence[float]) -> float | None:
    y = np.asarray(labels, dtype=bool)
    score = np.asarray(scores, dtype=np.float64)
    positives = int(y.sum())
    if positives == 0:
        return None
    order = np.argsort(-score, kind="mergesort")
    sorted_score = score[order]
    sorted_y = y[order]
    cumulative_positive = np.cumsum(sorted_y)
    area = 0.0
    previous_recall = 0.0
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_score[end] == sorted_score[start]:
            end += 1
        true_positive = int(cumulative_positive[end - 1])
        recall = true_positive / positives
        precision = true_positive / end
        area += (recall - previous_recall) * precision
        previous_recall = recall
        start = end
    return float(area)


def classification_metrics(
    labels: Sequence[bool], scores: Sequence[float], predicted: Sequence[bool]
) -> dict[str, Any]:
    y = np.asarray(labels, dtype=bool)
    score = np.asarray(scores, dtype=np.float64)
    pred = np.asarray(predicted, dtype=bool)
    _require(len(y) == len(score) == len(pred), "classification length mismatch")
    tp = int(np.sum(pred & y))
    tn = int(np.sum(~pred & ~y))
    fp = int(np.sum(pred & ~y))
    fn = int(np.sum(~pred & y))
    positives = tp + fn
    negatives = tn + fp
    recall = tp / positives if positives else None
    specificity = tn / negatives if negatives else None
    return {
        "transition_count": len(y),
        "positive_count": positives,
        "negative_count": negatives,
        "positive_prevalence": positives / len(y) if len(y) else None,
        "random_auprc_baseline": positives / len(y) if len(y) else None,
        "auroc": binary_auroc(y, score),
        "auprc": binary_auprc(y, score),
        "balanced_accuracy": (
            (recall + specificity) / 2.0
            if recall is not None and specificity is not None
            else None
        ),
        "recall_activation_due": recall,
        "false_activation_rate_before_target": fp / negatives if negatives else None,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def evaluate_rows(
    rows: Sequence[Mapping[str, Any]], scores: np.ndarray, predicted: np.ndarray
) -> dict[str, Any]:
    labels = labels_for_rows(rows)
    micro = classification_metrics(labels, scores, predicted)
    per_task = {}
    for task_id in sorted({int(row["task_id"]) for row in rows}):
        indices = np.asarray(
            [index for index, row in enumerate(rows) if int(row["task_id"]) == task_id],
            dtype=np.int64,
        )
        per_task[str(task_id)] = classification_metrics(
            labels[indices], scores[indices], predicted[indices]
        )
    metric_names = (
        "auroc",
        "auprc",
        "balanced_accuracy",
        "recall_activation_due",
        "false_activation_rate_before_target",
        "positive_prevalence",
    )
    task_macro = {
        name: _mean(item[name] for item in per_task.values()) for name in metric_names
    }
    task_macro["random_auprc_baseline"] = task_macro["positive_prevalence"]
    task_macro["task_count"] = len(per_task)
    return {"micro": micro, "task_macro": task_macro, "per_task": per_task}


def select_threshold(labels: Sequence[bool], scores: Sequence[float]) -> float:
    y = np.asarray(labels, dtype=bool)
    score = np.asarray(scores, dtype=np.float64)
    _require(bool(y.any()) and bool((~y).any()), "threshold training requires both classes")
    order = np.argsort(-score, kind="mergesort")
    sorted_score = score[order]
    sorted_y = y[order]
    total_positive = int(y.sum())
    total_negative = len(y) - total_positive
    candidates = [(float(np.nextafter(np.max(score), math.inf)), 0, 0)]
    tp = 0
    fp = 0
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_score[end] == sorted_score[start]:
            end += 1
        tp += int(sorted_y[start:end].sum())
        fp += int((~sorted_y[start:end]).sum())
        candidates.append((float(sorted_score[start]), tp, fp))
        start = end
    best_threshold = candidates[0][0]
    best_key = None
    for threshold, candidate_tp, candidate_fp in candidates:
        recall = candidate_tp / total_positive
        specificity = 1.0 - candidate_fp / total_negative
        balanced = (recall + specificity) / 2.0
        false_rate = candidate_fp / total_negative
        key = (balanced, recall, -false_rate, threshold)
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = threshold
    return float(best_threshold)


def select_univariate_direction(labels: Sequence[bool], scores: Sequence[float]) -> int:
    auc = binary_auroc(labels, scores)
    if auc is None or auc >= 0.5:
        return 1
    return -1


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def fit_logistic(x: np.ndarray, labels: Sequence[bool]) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.float64)
    design = np.column_stack([x, np.ones(len(x), dtype=np.float64)])
    beta = np.zeros(design.shape[1], dtype=np.float64)
    positives = int(y.sum())
    negatives = len(y) - positives
    _require(positives > 0 and negatives > 0, "logistic training requires both classes")
    sample_weight = np.where(
        y == 1.0,
        len(y) / (2.0 * positives),
        len(y) / (2.0 * negatives),
    )
    regularization = 1e-3
    regularizer = np.full(design.shape[1], regularization, dtype=np.float64)
    regularizer[-1] = 0.0
    iterations = 0
    for iterations in range(1, 61):
        probability = _sigmoid(design @ beta)
        curvature = sample_weight * probability * (1.0 - probability)
        gradient = design.T @ (sample_weight * (probability - y)) / sample_weight.sum()
        gradient += regularizer * beta
        hessian = (design.T @ (design * curvature[:, None])) / sample_weight.sum()
        hessian += np.diag(regularizer + 1e-9)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        beta -= step
        if float(np.max(np.abs(step))) < 1e-8:
            break
    return {
        "weights": beta[:-1].tolist(),
        "bias": float(beta[-1]),
        "iterations": iterations,
        "class_weighted": True,
        "l2_regularization": regularization,
    }


def predict_logistic(model: Mapping[str, Any], x: np.ndarray) -> np.ndarray:
    weights = np.asarray(model["weights"], dtype=np.float64)
    return _sigmoid(x @ weights + float(model["bias"]))


def _balanced_weights(labels: np.ndarray) -> np.ndarray:
    positives = int(labels.sum())
    negatives = len(labels) - positives
    _require(positives > 0 and negatives > 0, "tree training requires both classes")
    return np.where(
        labels,
        len(labels) / (2.0 * positives),
        len(labels) / (2.0 * negatives),
    )


def _weighted_gini(labels: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    if total <= 0:
        return 0.0
    positive = float(weights[labels].sum()) / total
    return 2.0 * positive * (1.0 - positive)


def fit_shallow_tree(
    x: np.ndarray,
    labels: Sequence[bool],
    *,
    max_depth: int = 3,
    min_leaf: int = 20,
) -> dict[str, Any]:
    y = np.asarray(labels, dtype=bool)
    weights = _balanced_weights(y)
    importances = np.zeros(x.shape[1], dtype=np.float64)

    def build(indices: np.ndarray, depth: int) -> dict[str, Any]:
        node_y = y[indices]
        node_weights = weights[indices]
        probability = float(node_weights[node_y].sum() / node_weights.sum())
        node = {
            "leaf": True,
            "probability": probability,
            "sample_count": int(len(indices)),
        }
        parent_impurity = _weighted_gini(node_y, node_weights)
        if depth >= max_depth or len(indices) < 2 * min_leaf or parent_impurity <= 0:
            return node
        best = None
        for feature_index in range(x.shape[1]):
            values = x[indices, feature_index]
            order = np.argsort(values, kind="mergesort")
            ordered_indices = indices[order]
            ordered_values = values[order]
            ordered_y = y[ordered_indices]
            ordered_weights = weights[ordered_indices]
            cumulative_weight = np.cumsum(ordered_weights)
            cumulative_positive = np.cumsum(ordered_weights * ordered_y)
            total_weight = float(cumulative_weight[-1])
            total_positive = float(cumulative_positive[-1])
            for split in range(min_leaf, len(indices) - min_leaf + 1):
                if ordered_values[split - 1] == ordered_values[split]:
                    continue
                left_weight = float(cumulative_weight[split - 1])
                left_positive = float(cumulative_positive[split - 1])
                right_weight = total_weight - left_weight
                right_positive = total_positive - left_positive
                left_rate = left_positive / left_weight
                right_rate = right_positive / right_weight
                impurity = (
                    left_weight * 2.0 * left_rate * (1.0 - left_rate)
                    + right_weight * 2.0 * right_rate * (1.0 - right_rate)
                ) / total_weight
                gain = parent_impurity - impurity
                threshold = float(
                    ordered_values[split - 1]
                    + (ordered_values[split] - ordered_values[split - 1]) / 2.0
                )
                candidate = (gain, -feature_index, -threshold, split, ordered_indices)
                if best is None or candidate[:3] > best[:3]:
                    best = candidate
        if best is None or best[0] <= 1e-12:
            return node
        gain, negative_feature, negative_threshold, split, ordered_indices = best
        feature_index = -negative_feature
        threshold = -negative_threshold
        importances[feature_index] += gain * float(node_weights.sum())
        return {
            "leaf": False,
            "probability": probability,
            "sample_count": int(len(indices)),
            "feature_index": int(feature_index),
            "threshold": float(threshold),
            "left": build(ordered_indices[:split], depth + 1),
            "right": build(ordered_indices[split:], depth + 1),
        }

    root = build(np.arange(len(x), dtype=np.int64), 0)
    total_importance = float(importances.sum())
    if total_importance > 0:
        importances /= total_importance
    return {
        "max_depth": max_depth,
        "min_leaf": min_leaf,
        "root": root,
        "feature_importances": importances.tolist(),
    }


def predict_tree(model: Mapping[str, Any], x: np.ndarray) -> np.ndarray:
    scores = np.empty(len(x), dtype=np.float64)
    for row_index, values in enumerate(x):
        node = model["root"]
        while not node["leaf"]:
            node = (
                node["left"]
                if values[int(node["feature_index"])] <= float(node["threshold"])
                else node["right"]
            )
        scores[row_index] = float(node["probability"])
    return scores


def _parameter_stability(
    parameters: Sequence[Sequence[float]], names: Sequence[str]
) -> list[dict[str, Any]]:
    matrix = np.asarray(parameters, dtype=np.float64)
    rows = []
    for index, name in enumerate(names):
        values = matrix[:, index]
        nonzero = values[np.abs(values) > 1e-12]
        if len(nonzero):
            positive = float(np.mean(nonzero > 0))
            sign_agreement = max(positive, 1.0 - positive)
        else:
            sign_agreement = 1.0
        rows.append(
            {
                "feature": name,
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
                "sign_agreement": sign_agreement,
            }
        )
    return rows


def _flatten_metric_row(
    *,
    scope: str,
    model: str,
    feature_or_group: str,
    metrics: Mapping[str, Any],
    fold_id: int | str = "all_oof",
    details: Any = None,
) -> dict[str, Any]:
    return {
        "scope": scope,
        "model": model,
        "feature_or_group": feature_or_group,
        "fold_id": fold_id,
        "transition_count": metrics.get("transition_count"),
        "positive_prevalence": metrics.get("positive_prevalence"),
        "random_auprc_baseline": metrics.get("random_auprc_baseline"),
        "auroc": metrics.get("auroc"),
        "auprc": metrics.get("auprc"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "recall_activation_due": metrics.get("recall_activation_due"),
        "false_activation_rate_before_target": metrics.get(
            "false_activation_rate_before_target"
        ),
        "details_json": json.dumps(details, sort_keys=True, separators=(",", ":")) if details is not None else "",
    }


def evaluate_univariate_oof(
    rows: Sequence[Mapping[str, Any]], assignment: Mapping[str, int]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    csv_rows = []
    report = {}
    splits = build_task_splits(rows, assignment)
    for feature in INDIVIDUAL_FEATURES:
        fold_reports = []
        all_validation_rows = []
        all_scores = []
        all_predictions = []
        for split in splits:
            train_rows = split["train_rows"]
            validation_rows = split["validation_rows"]
            preprocessor = fit_preprocessor(train_rows, (feature,))
            train_x = transform_features(train_rows, preprocessor)[:, 0]
            validation_x = transform_features(validation_rows, preprocessor)[:, 0]
            train_y = labels_for_rows(train_rows)
            direction = select_univariate_direction(train_y, train_x)
            train_scores = direction * train_x
            threshold = select_threshold(train_y, train_scores)
            validation_scores = direction * validation_x
            validation_predictions = validation_scores >= threshold
            evaluation = evaluate_rows(
                validation_rows, validation_scores, validation_predictions
            )
            fold_report = {
                "fold_id": split["fold_id"],
                "training_task_ids": split["training_task_ids"],
                "validation_task_ids": split["validation_task_ids"],
                "direction": direction,
                "threshold": threshold,
                "preprocessor": preprocessor,
                "metrics": evaluation,
            }
            fold_reports.append(fold_report)
            csv_rows.append(
                _flatten_metric_row(
                    scope="fold_micro",
                    model="univariate",
                    feature_or_group=feature,
                    fold_id=split["fold_id"],
                    metrics=evaluation["micro"],
                    details={
                        "direction": direction,
                        "threshold": threshold,
                        "training_task_ids": split["training_task_ids"],
                        "validation_task_ids": split["validation_task_ids"],
                    },
                )
            )
            all_validation_rows.extend(validation_rows)
            all_scores.extend(validation_scores.tolist())
            all_predictions.extend(validation_predictions.tolist())
        oof = evaluate_rows(
            all_validation_rows,
            np.asarray(all_scores, dtype=np.float64),
            np.asarray(all_predictions, dtype=bool),
        )
        directions = [item["direction"] for item in fold_reports]
        thresholds = [item["threshold"] for item in fold_reports]
        stability = {
            "direction_by_fold": directions,
            "direction_agreement": max(
                directions.count(1), directions.count(-1)
            )
            / len(directions),
            "threshold_mean": float(np.mean(thresholds)),
            "threshold_std": float(np.std(thresholds)),
            "threshold_min": float(np.min(thresholds)),
            "threshold_max": float(np.max(thresholds)),
        }
        csv_rows.extend(
            [
                _flatten_metric_row(
                    scope="oof_micro",
                    model="univariate",
                    feature_or_group=feature,
                    metrics=oof["micro"],
                    details=stability,
                ),
                _flatten_metric_row(
                    scope="oof_task_macro",
                    model="univariate",
                    feature_or_group=feature,
                    metrics=oof["task_macro"],
                    details=stability,
                ),
            ]
        )
        report[feature] = {
            "folds": fold_reports,
            "oof_metrics": oof,
            "stability": stability,
        }
    return csv_rows, report


def evaluate_feature_groups_oof(
    rows: Sequence[Mapping[str, Any]], assignment: Mapping[str, int]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    csv_rows = []
    report = {}
    splits = build_task_splits(rows, assignment)
    for group_name, features in FEATURE_GROUPS.items():
        report[group_name] = {}
        for model_name in ("logistic_regression", "shallow_decision_tree"):
            fold_reports = []
            all_validation_rows = []
            all_scores = []
            all_predictions = []
            parameters = []
            expanded_names = None
            for split in splits:
                train_rows = split["train_rows"]
                validation_rows = split["validation_rows"]
                preprocessor = fit_preprocessor(train_rows, features)
                train_x = transform_features(train_rows, preprocessor)
                validation_x = transform_features(validation_rows, preprocessor)
                train_y = labels_for_rows(train_rows)
                expanded_names = preprocessor["expanded_feature_names"]
                if model_name == "logistic_regression":
                    model = fit_logistic(train_x, train_y)
                    train_scores = predict_logistic(model, train_x)
                    validation_scores = predict_logistic(model, validation_x)
                    parameters.append(model["weights"])
                    serialized_model = model
                else:
                    model = fit_shallow_tree(train_x, train_y)
                    train_scores = predict_tree(model, train_x)
                    validation_scores = predict_tree(model, validation_x)
                    parameters.append(model["feature_importances"])
                    serialized_model = model
                threshold = select_threshold(train_y, train_scores)
                validation_predictions = validation_scores >= threshold
                evaluation = evaluate_rows(
                    validation_rows, validation_scores, validation_predictions
                )
                fold_report = {
                    "fold_id": split["fold_id"],
                    "training_task_ids": split["training_task_ids"],
                    "validation_task_ids": split["validation_task_ids"],
                    "threshold": threshold,
                    "preprocessor": preprocessor,
                    "model": serialized_model,
                    "metrics": evaluation,
                }
                fold_reports.append(fold_report)
                csv_rows.append(
                    _flatten_metric_row(
                        scope="fold_micro",
                        model=model_name,
                        feature_or_group=group_name,
                        fold_id=split["fold_id"],
                        metrics=evaluation["micro"],
                        details={
                            "threshold": threshold,
                            "training_task_ids": split["training_task_ids"],
                            "validation_task_ids": split["validation_task_ids"],
                        },
                    )
                )
                all_validation_rows.extend(validation_rows)
                all_scores.extend(validation_scores.tolist())
                all_predictions.extend(validation_predictions.tolist())
            oof = evaluate_rows(
                all_validation_rows,
                np.asarray(all_scores, dtype=np.float64),
                np.asarray(all_predictions, dtype=bool),
            )
            stability_kind = (
                "coefficient" if model_name == "logistic_regression" else "feature_importance"
            )
            stability = {
                "kind": stability_kind,
                "features": _parameter_stability(parameters, expanded_names or []),
            }
            csv_rows.extend(
                [
                    _flatten_metric_row(
                        scope="oof_micro",
                        model=model_name,
                        feature_or_group=group_name,
                        metrics=oof["micro"],
                        details=stability,
                    ),
                    _flatten_metric_row(
                        scope="oof_task_macro",
                        model=model_name,
                        feature_or_group=group_name,
                        metrics=oof["task_macro"],
                        details=stability,
                    ),
                ]
            )
            report[group_name][model_name] = {
                "folds": fold_reports,
                "oof_metrics": oof,
                "stability": stability,
            }
    return csv_rows, report


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 2:
        return None
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    x_rank = _rankdata(x_values)
    y_rank = _rankdata(y_values)
    if float(x_rank.std()) == 0.0 or float(y_rank.std()) == 0.0:
        return None
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def summarize_feature_distributions(
    transitions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    populations = {
        "primary_actual_warm_production_window": primary_rows(
            transitions, origin="ACTUAL_WARM"
        ),
        "cold_descriptive_production_window": primary_rows(
            transitions, origin="COLD"
        ),
        "secondary_actual_warm_full_depth_shadow_inclusive": secondary_rows(
            transitions, origin="ACTUAL_WARM"
        ),
        "secondary_cold_full_depth_shadow_inclusive": secondary_rows(
            transitions, origin="COLD"
        ),
    }
    summaries = []
    for population, rows in populations.items():
        for feature in INDIVIDUAL_FEATURES:
            available_rows = [
                row for row in rows if row["features"][feature] is not None
            ]
            values = [float(row["features"][feature]) for row in available_rows]
            remaining = [
                float(row["activation_target"] - row["iteration_index"])
                for row in available_rows
            ]
            task_correlations = []
            for task_id in sorted({int(row["task_id"]) for row in available_rows}):
                task_rows = [row for row in available_rows if int(row["task_id"]) == task_id]
                correlation = spearman_correlation(
                    [float(row["features"][feature]) for row in task_rows],
                    [
                        float(row["activation_target"] - row["iteration_index"])
                        for row in task_rows
                    ],
                )
                if correlation is not None:
                    task_correlations.append(correlation)
            summaries.append(
                {
                    "population": population,
                    "shadow_inclusive": "shadow_inclusive" in population,
                    "feature": feature,
                    "sample_count": len(rows),
                    "available_count": len(values),
                    "missing_count": len(rows) - len(values),
                    "mean": float(np.mean(values)) if values else None,
                    "std": float(np.std(values)) if values else None,
                    "p25": _percentile(values, 0.25),
                    "median": _percentile(values, 0.50),
                    "p75": _percentile(values, 0.75),
                    "spearman_remaining_micro": spearman_correlation(values, remaining),
                    "spearman_remaining_task_macro": _mean(task_correlations),
                    "spearman_task_count": len(task_correlations),
                }
            )
    return summaries


def summarize_trajectories(
    transitions: Sequence[Mapping[str, Any]], *, min_samples: int = 5
) -> list[dict[str, Any]]:
    rows = [row for row in transitions if row["primary_window"]]
    summaries = []
    for origin in ("ACTUAL_WARM", "COLD"):
        origin_rows = [row for row in rows if row["actual_origin"] == origin]
        for difficulty in ("all", "easy", "medium", "hard"):
            difficulty_rows = [
                row
                for row in origin_rows
                if difficulty == "all" or row["difficulty"] == difficulty
            ]
            for relative_iteration in range(-5, 2):
                aligned_rows = [
                    row
                    for row in difficulty_rows
                    if int(row["relative_iteration"]) == relative_iteration
                ]
                for feature in INDIVIDUAL_FEATURES:
                    values = [
                        float(row["features"][feature])
                        for row in aligned_rows
                        if row["features"][feature] is not None
                    ]
                    if len(values) < min_samples:
                        continue
                    summaries.append(
                        {
                            "population": (
                                "primary_actual_warm"
                                if origin == "ACTUAL_WARM"
                                else "secondary_cold_descriptive"
                            ),
                            "window": "production_window_only_not_shadow_inclusive",
                            "actual_origin": origin,
                            "difficulty": difficulty,
                            "relative_iteration": relative_iteration,
                            "feature": feature,
                            "sample_count": len(values),
                            "p25": _percentile(values, 0.25),
                            "median": _percentile(values, 0.50),
                            "p75": _percentile(values, 0.75),
                        }
                    )
    return summaries


def build_prediction_summary(
    predictions: Sequence[Mapping[str, Any]], assignment: Mapping[str, int]
) -> list[dict[str, Any]]:
    return [
        {
            **prediction,
            "fold_id": assignment[str(prediction["task_id"])],
        }
        for prediction in predictions
    ]


def leakage_audit(
    rows: Sequence[Mapping[str, Any]], assignment: Mapping[str, int]
) -> dict[str, Any]:
    folds = []
    for split in build_task_splits(rows, assignment):
        train_predictions = {
            (row["task_id"], row["episode_id"], row["prediction_id"])
            for row in split["train_rows"]
        }
        validation_predictions = {
            (row["task_id"], row["episode_id"], row["prediction_id"])
            for row in split["validation_rows"]
        }
        overlap = train_predictions & validation_predictions
        _require(not overlap, f"fold {split['fold_id']}: prediction leakage")
        folds.append(
            {
                "fold_id": split["fold_id"],
                "training_task_ids": split["training_task_ids"],
                "validation_task_ids": split["validation_task_ids"],
                "training_prediction_count": len(train_predictions),
                "validation_prediction_count": len(validation_predictions),
                "prediction_overlap_count": 0,
                "task_overlap_count": 0,
            }
        )
    return {
        "passed": True,
        "feature_inputs": list(INDIVIDUAL_FEATURES),
        "excluded_fields": sorted(LEAKAGE_EXCLUSIONS),
        "post_terminal_rows_in_primary_count": sum(
            int(row["iteration_index"] > row["baseline_k"]) for row in rows
        ),
        "folds": folds,
    }


def _metric_rankings(group_report: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for group, models in group_report.items():
        for model, result in models.items():
            metrics = result["oof_metrics"]["micro"]
            rows.append(
                {
                    "group": group,
                    "model": model,
                    "auroc": metrics["auroc"],
                    "auprc": metrics["auprc"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                }
            )
    rows.sort(
        key=lambda item: (
            -(item["auprc"] if item["auprc"] is not None else -math.inf),
            -(item["auroc"] if item["auroc"] is not None else -math.inf),
            item["group"],
            item["model"],
        )
    )
    baselines = {
        group: next(item for item in rows if item["group"] == group and item["model"] == "logistic_regression")
        for group in ("iteration_index", "raw_mse")
    }
    best = rows[0]
    return {
        "ranking_by_micro_auprc_then_auroc": rows,
        "best_diagnostic": best,
        "best_delta_vs_iteration_index_logistic": {
            metric: best[metric] - baselines["iteration_index"][metric]
            for metric in ("auroc", "auprc", "balanced_accuracy")
        },
        "best_delta_vs_raw_mse_logistic": {
            metric: best[metric] - baselines["raw_mse"][metric]
            for metric in ("auroc", "auprc", "balanced_accuracy")
        },
        "deployment_threshold_selected": False,
        "runtime_scheduler_implemented": False,
    }


def analyze_records(
    records: Sequence[Mapping[str, Any]],
    *,
    fold_manifest: Mapping[str, Any],
    assignment: Mapping[str, int],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    predictions, transitions = build_analysis_dataset(records)
    dataset_summary = build_dataset_summary(predictions, transitions, contract)
    primary = primary_rows(transitions, origin="ACTUAL_WARM")
    _require(primary, "ACTUAL_WARM primary population is empty")
    audit = leakage_audit(primary, assignment)
    _require(audit["post_terminal_rows_in_primary_count"] == 0, "primary dataset contains shadow rows")
    univariate_csv, univariate_report = evaluate_univariate_oof(primary, assignment)
    group_csv, group_report = evaluate_feature_groups_oof(primary, assignment)
    distributions = summarize_feature_distributions(transitions)
    trajectories = summarize_trajectories(transitions)
    return {
        "dataset_summary": dataset_summary,
        "prediction_summary": build_prediction_summary(predictions, assignment),
        "feature_distributions": distributions,
        "univariate_csv": univariate_csv,
        "feature_group_csv": group_csv,
        "trajectories": trajectories,
        "analysis_report": {
            "schema_version": SCHEMA_VERSION,
            "status": "offline_diagnostic_only",
            "primary_population": "ACTUAL_WARM",
            "primary_decision_window": "iteration_index <= K_t",
            "secondary_population": "COLD descriptive only",
            "secondary_diagnostic_window": "full k=2..32 shadow-inclusive",
            "activation_target_definition": "max(2, K_t - 1)",
            "label_definition": "iteration_index >= activation_target",
            "feature_groups": {name: list(values) for name, values in FEATURE_GROUPS.items()},
            "individual_features": list(INDIVIDUAL_FEATURES),
            "leakage_audit": audit,
            "fold_manifest": dict(fold_manifest),
            "dataset_summary": dataset_summary,
            "univariate_oof": univariate_report,
            "feature_group_oof": group_report,
            "comparative_summary": _metric_rankings(group_report),
            "statistical_scope": (
                "Transitions are not treated as independent confidence units; "
                "task-macro metrics and per-task Spearman summaries are reported."
            ),
            "runtime_defaults_modified": False,
            "deployment_threshold_selected": False,
        },
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_outputs(
    output_dir: Path,
    analysis: Mapping[str, Any],
    *,
    inputs: Mapping[str, Any],
    overwrite: bool,
) -> None:
    existing = [output_dir / name for name in OUTPUT_FILENAMES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError("refusing to overwrite analysis outputs: " + ", ".join(map(str, existing)))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_summary.json").write_text(
        canonical_json(analysis["dataset_summary"]), encoding="utf-8"
    )
    prediction_fields = (
        "task_id",
        "episode_id",
        "prediction_id",
        "actual_origin",
        "baseline_k",
        "activation_target",
        "difficulty",
        "full_transition_count",
        "primary_transition_count",
        "fold_id",
    )
    _write_csv(
        output_dir / "prediction_summary.csv",
        analysis["prediction_summary"],
        prediction_fields,
    )
    distribution_fields = (
        "population",
        "shadow_inclusive",
        "feature",
        "sample_count",
        "available_count",
        "missing_count",
        "mean",
        "std",
        "p25",
        "median",
        "p75",
        "spearman_remaining_micro",
        "spearman_remaining_task_macro",
        "spearman_task_count",
    )
    _write_csv(
        output_dir / "feature_distribution_summary.csv",
        analysis["feature_distributions"],
        distribution_fields,
    )
    metric_fields = (
        "scope",
        "model",
        "feature_or_group",
        "fold_id",
        "transition_count",
        "positive_prevalence",
        "random_auprc_baseline",
        "auroc",
        "auprc",
        "balanced_accuracy",
        "recall_activation_due",
        "false_activation_rate_before_target",
        "details_json",
    )
    _write_csv(
        output_dir / "univariate_oof_results.csv",
        analysis["univariate_csv"],
        metric_fields,
    )
    _write_csv(
        output_dir / "feature_group_oof_results.csv",
        analysis["feature_group_csv"],
        metric_fields,
    )
    trajectory_fields = (
        "population",
        "window",
        "actual_origin",
        "difficulty",
        "relative_iteration",
        "feature",
        "sample_count",
        "p25",
        "median",
        "p75",
    )
    _write_csv(
        output_dir / "trajectory_relative_to_activation.csv",
        analysis["trajectories"],
        trajectory_fields,
    )
    report = dict(analysis["analysis_report"])
    report["inputs"] = dict(inputs)
    report["outputs"] = {name: name for name in OUTPUT_FILENAMES}
    (output_dir / "analysis_report.json").write_text(
        canonical_json(report), encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--trace", action="append", type=Path)
    parser.add_argument("--fold-manifest", type=Path, default=DEFAULT_FOLD_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    trace_paths = args.trace or default_trace_paths(args.input_root)
    records = load_jsonl(trace_paths)
    contract = validate_input_records(
        records,
        expected_identity_sha256=EXPECTED_IDENTITY_SHA256,
        expected_prediction_count=EXPECTED_PREDICTION_COUNT,
        expected_transition_count=EXPECTED_TRANSITION_COUNT,
    )
    fold_manifest, assignment = load_fold_assignment(
        args.fold_manifest, {int(record["task_id"]) for record in records}
    )
    print(
        "Validated dataset: "
        f"{contract['prediction_count']} predictions, "
        f"{contract['transition_count']} transitions, identity matched"
    )
    analysis = analyze_records(
        records,
        fold_manifest=fold_manifest,
        assignment=assignment,
        contract=contract,
    )
    inputs = {
        "trace_files": [str(path.resolve()) for path in trace_paths],
        "trace_sha256": {str(path.resolve()): sha256_file(path) for path in trace_paths},
        "fold_manifest": str(args.fold_manifest.resolve()),
        "fold_manifest_sha256": sha256_file(args.fold_manifest),
        "workload_identity_sha256": contract["workload_identity_sha256"],
    }
    write_outputs(args.output_dir, analysis, inputs=inputs, overwrite=args.overwrite)
    comparison = analysis["analysis_report"]["comparative_summary"]
    best = comparison["best_diagnostic"]
    print(
        "Best offline diagnostic by OOF micro AUPRC: "
        f"{best['group']}/{best['model']} "
        f"(AUPRC={best['auprc']:.6f}, AUROC={best['auroc']:.6f})"
    )
    print(f"Wrote analysis outputs to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
