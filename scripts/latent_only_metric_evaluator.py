"""Leakage-safe offline evaluation for scalar latent convergence traces."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

from prismatic.models.latent_metrics import LATENT_METRIC_NAMES
from scripts.origin_aware_replay_lib import parse_fold_manifest, percentile


ORIGINS = ("ACTUAL_WARM", "COLD")
CAPTURE_TARGET = 0.995
LOGGER = logging.getLogger(__name__)


class LatentTraceValidationError(ValueError):
    """Raised when scalar trace inputs violate the evaluation contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LatentTraceValidationError(message)


def load_jsonl_records(paths: Sequence[Path]) -> list[Dict[str, Any]]:
    records = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            _require(isinstance(value, dict), f"expected object at {path}:{line_number}")
            records.append(value)
    return records


def parse_trace_predictions(records: Iterable[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    predictions = []
    seen = set()
    for record in records:
        trace = record.get("latent_metric_trace")
        if not trace:
            continue
        key = (
            str(record.get("task_id")),
            int(record.get("episode_id")),
            int(record.get("prediction_step", record.get("action_prediction_index"))),
        )
        _require(key not in seen, f"duplicate latent trace prediction: {key}")
        seen.add(key)
        origin = record.get("actual_origin", trace[0].get("actual_origin"))
        _require(origin in ORIGINS, f"{key}: unsupported origin {origin}")
        baseline_k = int(
            record.get("K_t", trace[0].get("baseline_stopping_iteration"))
        )
        max_iter = int(record.get("max_recurrent_iteration", max(item["iteration_index"] for item in trace)))
        transitions = []
        for item in trace:
            values = {name: float(item[name]) for name in LATENT_METRIC_NAMES}
            action_mse = float(item["adjacent_action_mse"])
            _require(
                all(math.isfinite(value) for value in (*values.values(), action_mse)),
                f"{key}: non-finite scalar trace",
            )
            transitions.append(
                {
                    "k": int(item["iteration_index"]),
                    **values,
                    "action_mse": action_mse,
                    "label": bool(item.get("action_mse_below_0_001", action_mse < 0.001)),
                }
            )
        transitions.sort(key=lambda item: item["k"])
        expected = list(range(2, max_iter + 1))
        _require(
            [item["k"] for item in transitions] == expected,
            f"{key}: trace must contain every eligible iteration 2..{max_iter}",
        )
        predictions.append(
            {
                "key": key,
                "task_id": key[0],
                "episode_id": key[1],
                "prediction_id": key[2],
                "actual_origin": origin,
                "baseline_k": baseline_k,
                "max_iter": max_iter,
                "transitions": transitions,
            }
        )
    _require(bool(predictions), "no latent_metric_trace predictions found")
    return predictions


def load_fold_assignment(path: Path, task_ids: Iterable[str]):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return manifest, parse_fold_manifest(manifest, task_ids)


def binary_auroc(labels: Sequence[bool], scores: Sequence[float]) -> float | None:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum = float(ranks[labels].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def binary_auprc(labels: Sequence[bool], scores: Sequence[float]) -> float | None:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    if positives == 0:
        return None
    thresholds = np.unique(scores)[::-1]
    previous_recall = 0.0
    area = 0.0
    for threshold in thresholds:
        selected = scores >= threshold
        true_positive = int((selected & labels).sum())
        precision = true_positive / int(selected.sum())
        recall = true_positive / positives
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return float(area)


def _classification(predictions, metric_name: str) -> Dict[str, Any]:
    labels = []
    scores = []
    by_task: Dict[str, tuple[list, list]] = {}
    for prediction in predictions:
        task_labels, task_scores = by_task.setdefault(prediction["task_id"], ([], []))
        for transition in prediction["transitions"]:
            label = bool(transition["label"])
            score = -float(transition[metric_name])
            labels.append(label)
            scores.append(score)
            task_labels.append(label)
            task_scores.append(score)
    task_metrics = {
        task_id: {
            "auroc": binary_auroc(task_labels, task_scores),
            "auprc": binary_auprc(task_labels, task_scores),
            "transition_count": len(task_labels),
        }
        for task_id, (task_labels, task_scores) in sorted(
            by_task.items(), key=lambda item: int(item[0])
        )
    }
    return {
        "auroc": binary_auroc(labels, scores),
        "auprc": binary_auprc(labels, scores),
        "transition_count": len(labels),
        "task_macro_auroc": _mean(
            [item["auroc"] for item in task_metrics.values() if item["auroc"] is not None]
        ),
        "task_macro_auprc": _mean(
            [item["auprc"] for item in task_metrics.values() if item["auprc"] is not None]
        ),
        "task_metrics": task_metrics,
    }


def replay_predictions(
    predictions: Sequence[Mapping[str, Any]],
    metric_name: str,
    threshold: float,
    *,
    min_iter: int,
) -> list[Dict[str, Any]]:
    replays = []
    for prediction in predictions:
        eligible = [item for item in prediction["transitions"] if item["k"] >= min_iter]
        selected = next(
            (item for item in eligible if float(item[metric_name]) <= threshold), None
        )
        terminal_k = int(selected["k"]) if selected else int(prediction["max_iter"])
        stopped = selected is not None
        true_stop = bool(selected and selected["label"])
        reference = next(
            (int(item["k"]) for item in eligible if item["label"]), None
        )
        replays.append(
            {
                "key": prediction["key"],
                "task_id": prediction["task_id"],
                "actual_origin": prediction["actual_origin"],
                "baseline_k": int(prediction["baseline_k"]),
                "max_iter": int(prediction["max_iter"]),
                "terminal_k": terminal_k,
                "stopped": stopped,
                "true_stop": true_stop,
                "false_convergence": bool(stopped and not true_stop),
                "captured_convergence": (
                    None if reference is None else bool(stopped and true_stop)
                ),
                "delta_k": terminal_k - int(prediction["baseline_k"]),
                "early_stop": terminal_k < int(prediction["max_iter"]),
                "max_iteration": terminal_k == int(prediction["max_iter"]),
            }
        )
    return replays


def _mean(values) -> float | None:
    return float(np.mean(values)) if values else None


def _aggregate_flat(replays: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    _require(bool(replays), "cannot aggregate empty replay set")
    capture = [item for item in replays if item["captured_convergence"] is not None]
    return {
        "prediction_count": len(replays),
        "false_convergence_count": int(sum(item["false_convergence"] for item in replays)),
        "convergence_capture_eligible_count": len(capture),
        "convergence_capture": _mean([item["captured_convergence"] for item in capture]),
        "recall": _mean([item["captured_convergence"] for item in capture]),
        "mean_delta_K": _mean([item["delta_k"] for item in replays]),
        "p95_delta_K": percentile([item["delta_k"] for item in replays], 0.95),
        "early_stop_rate": _mean([item["early_stop"] for item in replays]),
        "max_iteration_rate": _mean([item["max_iteration"] for item in replays]),
    }


def aggregate_replays(replays: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    result = _aggregate_flat(replays)
    by_task: Dict[str, list] = {}
    for item in replays:
        by_task.setdefault(item["task_id"], []).append(item)
    task_metrics = {
        task_id: _aggregate_flat(items)
        for task_id, items in sorted(by_task.items(), key=lambda item: int(item[0]))
    }
    for field in (
        "false_convergence_count",
        "convergence_capture",
        "recall",
        "mean_delta_K",
        "p95_delta_K",
        "early_stop_rate",
        "max_iteration_rate",
    ):
        result[f"task_macro_{field}"] = _mean(
            [item[field] for item in task_metrics.values() if item[field] is not None]
        )
    result["task_metrics"] = task_metrics
    return result


def select_training_threshold(
    predictions: Sequence[Mapping[str, Any]],
    metric_name: str,
    *,
    min_iter: int,
    capture_target: float = CAPTURE_TARGET,
) -> Dict[str, Any]:
    _require(bool(predictions), "threshold selection requires training predictions")
    event_values = []
    event_prediction_indices = []
    event_iterations = []
    event_labels = []
    convergence_eligible = np.zeros(len(predictions), dtype=bool)
    max_iterations = np.empty(len(predictions), dtype=np.int64)
    for prediction_index, prediction in enumerate(predictions):
        max_iterations[prediction_index] = int(prediction["max_iter"])
        for item in prediction["transitions"]:
            if item["k"] < min_iter:
                continue
            event_values.append(float(item[metric_name]))
            event_prediction_indices.append(prediction_index)
            event_iterations.append(int(item["k"]))
            event_labels.append(bool(item["label"]))
            convergence_eligible[prediction_index] |= bool(item["label"])

    values = np.asarray(event_values, dtype=np.float64)
    candidates = np.concatenate(
        ([np.nextafter(values.min(), -math.inf)], np.unique(values))
    )

    # Sweep the exact candidate set in ascending order. A prediction changes only
    # when the newly eligible value exposes an earlier iteration than its current
    # selection, so the selection statistics can be updated in constant time.
    order = np.argsort(values, kind="mergesort")
    event_prediction_indices = np.asarray(event_prediction_indices, dtype=np.int64)[order]
    event_iterations = np.asarray(event_iterations, dtype=np.int64)[order]
    event_labels = np.asarray(event_labels, dtype=bool)[order]
    sorted_values = values[order]
    selected_iterations = np.full(len(predictions), -1, dtype=np.int64)
    selected_labels = np.zeros(len(predictions), dtype=bool)
    false_convergence_count = 0
    captured_convergence_count = 0
    early_stop_count = 0
    capture_eligible_count = int(convergence_eligible.sum())
    prediction_count = len(predictions)

    best_feasible = None
    best_infeasible = None

    def consider(threshold: float) -> None:
        nonlocal best_feasible, best_infeasible
        convergence_capture = (
            None
            if capture_eligible_count == 0
            else captured_convergence_count / capture_eligible_count
        )
        early_stop_rate = early_stop_count / prediction_count
        summary = (
            float(threshold),
            false_convergence_count,
            convergence_capture,
            early_stop_rate,
        )
        if convergence_capture is not None and convergence_capture >= capture_target:
            key = (false_convergence_count, -early_stop_rate, float(threshold))
            if best_feasible is None or key < best_feasible[0]:
                best_feasible = (key, summary)
        fallback_key = (
            -(convergence_capture or 0.0),
            false_convergence_count,
            -early_stop_rate,
            float(threshold),
        )
        if best_infeasible is None or fallback_key < best_infeasible[0]:
            best_infeasible = (fallback_key, summary)

    consider(float(candidates[0]))
    event_start = 0
    for threshold in candidates[1:]:
        event_end = event_start + 1
        while event_end < len(sorted_values) and sorted_values[event_end] == threshold:
            event_end += 1

        pending_updates = {}
        for event_index in range(event_start, event_end):
            prediction_index = int(event_prediction_indices[event_index])
            iteration = int(event_iterations[event_index])
            current = pending_updates.get(prediction_index)
            if current is None or iteration < current[0]:
                pending_updates[prediction_index] = (
                    iteration,
                    bool(event_labels[event_index]),
                )

        for prediction_index, (iteration, label) in pending_updates.items():
            previous_iteration = int(selected_iterations[prediction_index])
            if previous_iteration >= 0 and previous_iteration <= iteration:
                continue
            if previous_iteration >= 0:
                previous_label = bool(selected_labels[prediction_index])
                false_convergence_count -= int(not previous_label)
                captured_convergence_count -= int(previous_label)
                early_stop_count -= int(
                    previous_iteration < max_iterations[prediction_index]
                )
            selected_iterations[prediction_index] = iteration
            selected_labels[prediction_index] = label
            false_convergence_count += int(not label)
            captured_convergence_count += int(label)
            early_stop_count += int(iteration < max_iterations[prediction_index])

        consider(float(threshold))
        event_start = event_end

    if best_feasible is not None:
        threshold = best_feasible[1][0]
        status = "capture_feasible"
    else:
        threshold = best_infeasible[1][0]
        status = "capture_infeasible_fail_closed"
    metrics = aggregate_replays(
        replay_predictions(predictions, metric_name, threshold, min_iter=min_iter)
    )
    return {
        "threshold": threshold,
        "threshold_hex": float(threshold).hex(),
        "selection_status": status,
        "capture_target": capture_target,
        "candidate_count": len(candidates),
        "training_metrics": metrics,
    }


def evaluate_oof(
    predictions: Sequence[Mapping[str, Any]],
    fold_assignment: Mapping[str, int],
    *,
    min_iter: int = 2,
    capture_target: float = CAPTURE_TARGET,
) -> Dict[str, Any]:
    _require(min_iter >= 2, "min_iter must be >= 2")
    task_ids = {prediction["task_id"] for prediction in predictions}
    _require(task_ids == set(fold_assignment), "trace/fold task IDs differ")
    fold_ids = sorted(set(fold_assignment.values()))
    metrics_output = {}
    leakage_folds = []
    for metric_name in LATENT_METRIC_NAMES:
        origins_output = {}
        for origin in ORIGINS:
            origin_predictions = [
                item for item in predictions if item["actual_origin"] == origin
            ]
            _require(bool(origin_predictions), f"missing {origin} predictions")
            oof_replays = []
            folds = []
            for fold_id in fold_ids:
                train = [
                    item
                    for item in origin_predictions
                    if fold_assignment[item["task_id"]] != fold_id
                ]
                validation = [
                    item
                    for item in origin_predictions
                    if fold_assignment[item["task_id"]] == fold_id
                ]
                _require(bool(train) and bool(validation), f"fold {fold_id} {origin}: empty split")
                LOGGER.info(
                    "evaluating metric=%s origin=%s fold=%s train=%d validation=%d",
                    metric_name,
                    origin,
                    fold_id,
                    len(train),
                    len(validation),
                )
                selection = select_training_threshold(
                    train,
                    metric_name,
                    min_iter=min_iter,
                    capture_target=capture_target,
                )
                held_out = replay_predictions(
                    validation,
                    metric_name,
                    selection["threshold"],
                    min_iter=min_iter,
                )
                oof_replays.extend(held_out)
                train_tasks = sorted({item["task_id"] for item in train}, key=int)
                validation_tasks = sorted(
                    {item["task_id"] for item in validation}, key=int
                )
                _require(
                    set(train_tasks).isdisjoint(validation_tasks),
                    f"fold {fold_id}: task leakage",
                )
                folds.append(
                    {
                        "fold_id": fold_id,
                        "training_task_ids": train_tasks,
                        "validation_task_ids": validation_tasks,
                        "training_prediction_count": len(train),
                        "validation_prediction_count": len(validation),
                        "threshold_selection": selection,
                        "held_out_metrics": aggregate_replays(held_out),
                    }
                )
                if metric_name == LATENT_METRIC_NAMES[0] and origin == ORIGINS[0]:
                    leakage_folds.append(
                        {
                            "fold_id": fold_id,
                            "training_task_ids": train_tasks,
                            "validation_task_ids": validation_tasks,
                            "task_overlap_count": 0,
                        }
                    )
            origins_output[origin] = {
                "classification": _classification(origin_predictions, metric_name),
                "selected_thresholds_per_fold": [
                    {
                        "fold_id": item["fold_id"],
                        "threshold": item["threshold_selection"]["threshold"],
                    }
                    for item in folds
                ],
                "oof_stopping": aggregate_replays(oof_replays),
                "folds": folds,
            }
        metrics_output[metric_name] = origins_output

    def nominal_rank(metric_name: str):
        origin_results = [
            metrics_output[metric_name][origin]["oof_stopping"] for origin in ORIGINS
        ]
        return (
            sum(item["false_convergence_count"] for item in origin_results),
            -sum(item["convergence_capture"] or 0.0 for item in origin_results),
            sum(abs(item["mean_delta_K"] or 0.0) for item in origin_results),
            sum(item["max_iteration_rate"] or 0.0 for item in origin_results),
            LATENT_METRIC_NAMES.index(metric_name),
        )

    selected = min(LATENT_METRIC_NAMES, key=nominal_rank)
    return {
        "schema_version": 1,
        "label_definition": "adjacent_action_mse < 0.001",
        "metric_direction": "lower_is_more_converged",
        "normalization": "none (scalar thresholds are fit directly on training folds)",
        "min_iter": min_iter,
        "capture_target": capture_target,
        "prediction_count": len(predictions),
        "task_ids": sorted(task_ids, key=int),
        "leakage_audit": {"passed": True, "folds": leakage_folds},
        "metrics": metrics_output,
        "nominal_best_metric": selected,
        "runtime_defaults_modified": False,
        "selection_order": [
            "minimize combined OOF false-convergence count",
            "maximize combined OOF convergence capture",
            "minimize combined absolute mean delta_K",
            "minimize combined max-iteration rate",
        ],
    }
