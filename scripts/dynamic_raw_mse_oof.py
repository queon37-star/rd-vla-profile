"""Task-level OOF evaluation for iteration-conditioned raw-MSE thresholds."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from scripts.latent_only_metric_evaluator import (
    CAPTURE_TARGET,
    LatentTraceValidationError,
    aggregate_replays,
    replay_predictions,
    select_training_threshold,
)


ORIGIN = "ACTUAL_WARM"
ELIGIBLE_ITERATIONS = tuple(range(2, 33))
DEFAULT_ALPHAS = (0.0, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01)
DEFAULT_MIN_NEGATIVE_SAMPLES = 20
FAIL_CLOSED_THRESHOLD = float(np.nextafter(0.0, -math.inf))
LOGGER = logging.getLogger(__name__)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LatentTraceValidationError(message)


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _validate_alphas(alphas: Sequence[float]) -> tuple[float, ...]:
    normalized = tuple(sorted({float(alpha) for alpha in alphas}))
    _require(bool(normalized), "at least one false-positive budget is required")
    _require(
        all(math.isfinite(alpha) and 0.0 <= alpha <= 1.0 for alpha in normalized),
        "false-positive budgets must be finite values in [0, 1]",
    )
    return normalized


def _transitions_by_iteration(
    predictions: Sequence[Mapping[str, Any]],
) -> Dict[int, list[Mapping[str, Any]]]:
    by_iteration = {iteration: [] for iteration in ELIGIBLE_ITERATIONS}
    for prediction in predictions:
        for transition in prediction["transitions"]:
            iteration = int(transition["k"])
            if iteration not in by_iteration:
                continue
            raw_mse = float(transition["raw_mse"])
            _require(
                math.isfinite(raw_mse) and raw_mse >= 0.0,
                f"{prediction['key']}: raw_mse at k={iteration} must be finite and non-negative",
            )
            by_iteration[iteration].append(transition)
    return by_iteration


def _largest_empirical_fpr_threshold(
    samples: Sequence[Mapping[str, Any]], alpha: float
) -> Dict[str, Any]:
    values = np.asarray([float(item["raw_mse"]) for item in samples], dtype=np.float64)
    labels = np.asarray([bool(item["label"]) for item in samples], dtype=bool)
    negative_values = np.sort(values[~labels])
    _require(bool(len(negative_values)), "threshold fitting requires negative samples")

    candidates = np.unique(values)
    false_positive_counts = np.searchsorted(
        negative_values, candidates, side="right"
    )
    false_positive_rates = false_positive_counts / len(negative_values)
    unsafe = np.flatnonzero(false_positive_rates > alpha)
    if len(unsafe):
        first_unsafe = int(unsafe[0])
        threshold = float(np.nextafter(candidates[first_unsafe], -math.inf))
        false_positive_count = int(
            np.searchsorted(negative_values, threshold, side="right")
        )
        false_positive_rate = false_positive_count / len(negative_values)
    else:
        threshold = float(candidates[-1])
        false_positive_count = len(negative_values)
        false_positive_rate = 1.0

    selected = values <= threshold
    return {
        "threshold": threshold,
        "threshold_hex": threshold.hex(),
        "candidate_count": len(candidates) + 1,
        "empirical_false_positive_count": false_positive_count,
        "empirical_false_positive_rate": false_positive_rate,
        "selected_sample_count": int(selected.sum()),
        "selected_positive_count": int((selected & labels).sum()),
    }


def fit_threshold_curve(
    predictions: Sequence[Mapping[str, Any]],
    alpha: float,
    *,
    min_negative_samples: int = DEFAULT_MIN_NEGATIVE_SAMPLES,
) -> list[Dict[str, Any]]:
    """Fit one threshold per k using only the supplied training predictions."""

    _require(bool(predictions), "threshold-curve fitting requires predictions")
    _require(
        isinstance(min_negative_samples, int)
        and not isinstance(min_negative_samples, bool)
        and min_negative_samples >= 1,
        "min_negative_samples must be an integer >= 1",
    )
    alpha = _validate_alphas((alpha,))[0]
    by_iteration = _transitions_by_iteration(predictions)
    curve = []
    for iteration in ELIGIBLE_ITERATIONS:
        local = by_iteration[iteration]
        local_negative_count = sum(not bool(item["label"]) for item in local)
        local_positive_count = len(local) - local_negative_count
        calibration_iterations = [iteration]
        calibration = local
        source = "per_iteration"
        if local_negative_count < min_negative_samples:
            calibration_iterations = [
                neighbor
                for neighbor in (iteration - 1, iteration, iteration + 1)
                if neighbor in by_iteration
            ]
            calibration = [
                item
                for neighbor in calibration_iterations
                for item in by_iteration[neighbor]
            ]
            source = "pooled_neighborhood"

        calibration_negative_count = sum(
            not bool(item["label"]) for item in calibration
        )
        calibration_positive_count = len(calibration) - calibration_negative_count
        if calibration_negative_count < min_negative_samples:
            source = "fail_closed_insufficient_negatives"
            selection = {
                "threshold": FAIL_CLOSED_THRESHOLD,
                "threshold_hex": FAIL_CLOSED_THRESHOLD.hex(),
                "candidate_count": len(
                    {float(item["raw_mse"]) for item in calibration}
                ),
                "empirical_false_positive_count": 0,
                "empirical_false_positive_rate": (
                    0.0 if calibration_negative_count else None
                ),
                "selected_sample_count": 0,
                "selected_positive_count": 0,
            }
        else:
            selection = _largest_empirical_fpr_threshold(calibration, alpha)

        curve.append(
            {
                "iteration": iteration,
                "alpha": alpha,
                "threshold": selection["threshold"],
                "threshold_hex": selection["threshold_hex"],
                "calibration_source": source,
                "calibration_iterations": calibration_iterations,
                "minimum_negative_samples": min_negative_samples,
                "local_sample_count": len(local),
                "local_negative_count": local_negative_count,
                "local_positive_count": local_positive_count,
                "calibration_sample_count": len(calibration),
                "calibration_negative_count": calibration_negative_count,
                "calibration_positive_count": calibration_positive_count,
                "candidate_count": selection["candidate_count"],
                "empirical_false_positive_count": selection[
                    "empirical_false_positive_count"
                ],
                "empirical_false_positive_rate": selection[
                    "empirical_false_positive_rate"
                ],
                "selected_sample_count": selection["selected_sample_count"],
                "selected_positive_count": selection["selected_positive_count"],
            }
        )
    return curve


def replay_dynamic_thresholds(
    predictions: Sequence[Mapping[str, Any]],
    threshold_curve: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    """Replay the exact first k whose raw MSE crosses its k-specific threshold."""

    thresholds = {
        int(item["iteration"]): float(item["threshold"])
        for item in threshold_curve
    }
    _require(
        tuple(sorted(thresholds)) == ELIGIBLE_ITERATIONS,
        "threshold curve must cover every iteration k=2..32",
    )
    replays = []
    for prediction in predictions:
        selected = next(
            (
                item
                for item in prediction["transitions"]
                if int(item["k"]) in thresholds
                and float(item["raw_mse"]) <= thresholds[int(item["k"])]
            ),
            None,
        )
        terminal_k = int(selected["k"]) if selected else int(prediction["max_iter"])
        stopped = selected is not None
        true_stop = bool(selected and selected["label"])
        reference = next(
            (
                int(item["k"])
                for item in prediction["transitions"]
                if int(item["k"]) in thresholds and bool(item["label"])
            ),
            None,
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


def aggregate_oof_replays(
    replays: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Reuse fixed replay aggregation and add prediction/task false-positive rates."""

    result = aggregate_replays(replays)
    result["false_convergence_rate"] = (
        result["false_convergence_count"] / result["prediction_count"]
    )
    task_rates = []
    for metrics in result["task_metrics"].values():
        metrics["false_convergence_rate"] = (
            metrics["false_convergence_count"] / metrics["prediction_count"]
        )
        task_rates.append(metrics["false_convergence_rate"])
    result["task_macro_false_convergence_rate"] = _mean(task_rates)
    return result


COMPARISON_FIELDS = (
    "false_convergence_count",
    "false_convergence_rate",
    "convergence_capture",
    "mean_delta_K",
    "p95_delta_K",
    "early_stop_rate",
    "max_iteration_rate",
    "task_macro_false_convergence_count",
    "task_macro_false_convergence_rate",
    "task_macro_convergence_capture",
    "task_macro_mean_delta_K",
    "task_macro_p95_delta_K",
    "task_macro_early_stop_rate",
    "task_macro_max_iteration_rate",
)


def compare_metrics(
    dynamic: Mapping[str, Any], fixed: Mapping[str, Any]
) -> Dict[str, Any]:
    differences = {}
    for field in COMPARISON_FIELDS:
        dynamic_value = dynamic.get(field)
        fixed_value = fixed.get(field)
        differences[field] = (
            None
            if dynamic_value is None or fixed_value is None
            else dynamic_value - fixed_value
        )
    return {
        "reference": "existing_fixed_raw_mse_oof",
        "dynamic_minus_fixed": differences,
    }


def _pareto_frontier(schedules: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    def objectives(schedule: Mapping[str, Any]) -> tuple[float, ...]:
        metrics = schedule["oof_stopping"]
        capture = metrics["convergence_capture"]
        return (
            float(metrics["false_convergence_count"]),
            -(float(capture) if capture is not None else -math.inf),
            float(metrics["mean_delta_K"]),
            float(metrics["p95_delta_K"]),
            float(metrics["max_iteration_rate"]),
        )

    frontier = []
    for candidate in schedules:
        candidate_values = objectives(candidate)
        dominated = False
        for other in schedules:
            if other is candidate:
                continue
            other_values = objectives(other)
            if all(a <= b for a, b in zip(other_values, candidate_values)) and any(
                a < b for a, b in zip(other_values, candidate_values)
            ):
                dominated = True
                break
        if not dominated:
            metrics = candidate["oof_stopping"]
            frontier.append(
                {
                    "alpha": candidate["alpha"],
                    "false_convergence_count": metrics[
                        "false_convergence_count"
                    ],
                    "false_convergence_rate": metrics["false_convergence_rate"],
                    "convergence_capture": metrics["convergence_capture"],
                    "mean_delta_K": metrics["mean_delta_K"],
                    "p95_delta_K": metrics["p95_delta_K"],
                    "max_iteration_rate": metrics["max_iteration_rate"],
                }
            )
    return frontier


def evaluate_dynamic_raw_mse_oof(
    predictions: Sequence[Mapping[str, Any]],
    fold_assignment: Mapping[str, int],
    *,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    min_negative_samples: int = DEFAULT_MIN_NEGATIVE_SAMPLES,
) -> Dict[str, Any]:
    """Evaluate dynamic thresholds on ACTUAL_WARM held-out tasks only."""

    alphas = _validate_alphas(alphas)
    origin_predictions = [
        prediction
        for prediction in predictions
        if prediction["actual_origin"] == ORIGIN
    ]
    _require(bool(origin_predictions), f"missing {ORIGIN} predictions")
    task_ids = {str(prediction["task_id"]) for prediction in origin_predictions}
    _require(task_ids == set(fold_assignment), "trace/fold task IDs differ")
    fold_ids = sorted(set(fold_assignment.values()))
    _require(len(fold_ids) == 5, "dynamic evaluator requires the existing 5-fold split")

    fixed_replays = []
    fixed_folds = []
    schedules = {
        alpha: {"alpha": alpha, "alpha_hex": alpha.hex(), "replays": [], "folds": []}
        for alpha in alphas
    }
    threshold_curve_data = []
    leakage_folds = []

    for fold_id in fold_ids:
        train = [
            item
            for item in origin_predictions
            if fold_assignment[str(item["task_id"])] != fold_id
        ]
        validation = [
            item
            for item in origin_predictions
            if fold_assignment[str(item["task_id"])] == fold_id
        ]
        _require(bool(train) and bool(validation), f"fold {fold_id}: empty split")
        train_tasks = sorted({str(item["task_id"]) for item in train}, key=int)
        validation_tasks = sorted(
            {str(item["task_id"]) for item in validation}, key=int
        )
        train_keys = {tuple(item["key"]) for item in train}
        validation_keys = {tuple(item["key"]) for item in validation}
        _require(
            set(train_tasks).isdisjoint(validation_tasks),
            f"fold {fold_id}: task leakage",
        )
        _require(train_keys.isdisjoint(validation_keys), f"fold {fold_id}: prediction leakage")
        leakage_folds.append(
            {
                "fold_id": fold_id,
                "training_task_ids": train_tasks,
                "validation_task_ids": validation_tasks,
                "training_prediction_count": len(train),
                "validation_prediction_count": len(validation),
                "task_overlap_count": 0,
                "prediction_overlap_count": 0,
            }
        )

        LOGGER.info(
            "evaluating fixed raw_mse reference origin=%s fold=%s train=%d validation=%d",
            ORIGIN,
            fold_id,
            len(train),
            len(validation),
        )
        fixed_selection = select_training_threshold(
            train,
            "raw_mse",
            min_iter=2,
            capture_target=CAPTURE_TARGET,
        )
        fixed_held_out = replay_predictions(
            validation,
            "raw_mse",
            fixed_selection["threshold"],
            min_iter=2,
        )
        fixed_replays.extend(fixed_held_out)
        fixed_folds.append(
            {
                "fold_id": fold_id,
                "training_task_ids": train_tasks,
                "validation_task_ids": validation_tasks,
                "threshold_selection": fixed_selection,
                "held_out_metrics": aggregate_oof_replays(fixed_held_out),
            }
        )

        for alpha in alphas:
            LOGGER.info(
                "evaluating dynamic raw_mse origin=%s fold=%s alpha=%.17g",
                ORIGIN,
                fold_id,
                alpha,
            )
            curve = fit_threshold_curve(
                train,
                alpha,
                min_negative_samples=min_negative_samples,
            )
            held_out = replay_dynamic_thresholds(validation, curve)
            schedules[alpha]["replays"].extend(held_out)
            sample_counts = [
                {
                    key: row[key]
                    for key in (
                        "iteration",
                        "calibration_source",
                        "calibration_iterations",
                        "local_sample_count",
                        "local_negative_count",
                        "local_positive_count",
                        "calibration_sample_count",
                        "calibration_negative_count",
                        "calibration_positive_count",
                    )
                }
                for row in curve
            ]
            schedules[alpha]["folds"].append(
                {
                    "fold_id": fold_id,
                    "training_task_ids": train_tasks,
                    "validation_task_ids": validation_tasks,
                    "training_prediction_count": len(train),
                    "validation_prediction_count": len(validation),
                    "thresholds_by_iteration": curve,
                    "sample_counts_by_iteration": sample_counts,
                    "held_out_metrics": aggregate_oof_replays(held_out),
                }
            )
            threshold_curve_data.extend(
                {"fold_id": fold_id, **row} for row in curve
            )

    fixed_metrics = aggregate_oof_replays(fixed_replays)
    dynamic_output = []
    for alpha in alphas:
        schedule = schedules[alpha]
        metrics = aggregate_oof_replays(schedule.pop("replays"))
        schedule["oof_stopping"] = metrics
        schedule["comparison_to_fixed_raw_mse"] = compare_metrics(
            metrics, fixed_metrics
        )
        dynamic_output.append(schedule)

    return {
        "schema_version": 1,
        "status": "offline_diagnostic_only",
        "origin": ORIGIN,
        "origins_evaluated": [ORIGIN],
        "cold_evaluation_status": "not_evaluated_initial_scope",
        "label_definition": "adjacent_action_mse < 0.001",
        "stopping_rule": "first k where raw_mse(k) <= threshold[k]",
        "eligible_iterations": list(ELIGIBLE_ITERATIONS),
        "false_positive_budgets": list(alphas),
        "minimum_negative_samples": min_negative_samples,
        "sparse_iteration_fallback": (
            "pool k-1, k, and k+1; if still insufficient, use a threshold "
            "below every non-negative raw MSE"
        ),
        "threshold_candidate_definition": (
            "the largest representable float below the first raw_mse value whose "
            "inclusion would exceed alpha; otherwise the maximum observed raw_mse"
        ),
        "prediction_count": len(origin_predictions),
        "task_ids": sorted(task_ids, key=int),
        "leakage_audit": {"passed": True, "folds": leakage_folds},
        "fixed_raw_mse_reference": {
            "capture_target": CAPTURE_TARGET,
            "oof_stopping": fixed_metrics,
            "folds": fixed_folds,
        },
        "dynamic_schedules": dynamic_output,
        "pareto_frontier": _pareto_frontier(dynamic_output),
        "threshold_curve_data": threshold_curve_data,
        "runtime_defaults_modified": False,
        "runtime_inference_modified": False,
        "calibration_traces_modified": False,
        "fold_assignments_modified": False,
    }


def format_pareto_table(report: Mapping[str, Any]) -> str:
    """Format the requested concise dynamic-schedule Pareto table."""

    headers = (
        "alpha",
        "false convergence",
        "capture",
        "mean delta_K",
        "p95 delta_K",
        "max-iteration rate",
    )
    rows = [headers]
    for item in report["pareto_frontier"]:
        capture = item["convergence_capture"]
        rows.append(
            (
                f"{item['alpha']:.4g}",
                f"{item['false_convergence_count']} ({item['false_convergence_rate']:.4%})",
                "n/a" if capture is None else f"{capture:.4%}",
                f"{item['mean_delta_K']:.4f}",
                f"{item['p95_delta_K']:.4f}",
                f"{item['max_iteration_rate']:.4%}",
            )
        )
    widths = [max(len(row[index]) for row in rows) for index in range(len(headers))]
    rendered = []
    for row_index, row in enumerate(rows):
        rendered.append(
            "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        )
        if row_index == 0:
            rendered.append("  ".join("-" * width for width in widths))
    return "\n".join(rendered)
