"""ACTUAL_WARM task-level OOF evaluation for raw-MSE Coda activation."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from scripts.latent_only_metric_evaluator import LatentTraceValidationError
from scripts.origin_aware_replay_lib import percentile


ORIGIN = "ACTUAL_WARM"
ELIGIBLE_ITERATIONS = tuple(range(2, 33))
TRIGGER_SEARCH_ITERATIONS = tuple(range(2, 32))
FORCED_TRIGGER_ITERATION = 31
DEFAULT_BETAS = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10)
DEFAULT_MIN_ACTIVATION_DUE_SAMPLES = 20
FAIL_CLOSED_THRESHOLD = float(np.nextafter(0.0, -math.inf))
LOGGER = logging.getLogger(__name__)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LatentTraceValidationError(message)


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _validate_betas(betas: Sequence[float]) -> tuple[float, ...]:
    normalized = tuple(sorted({float(beta) for beta in betas}))
    _require(bool(normalized), "at least one miss budget is required")
    _require(
        all(math.isfinite(beta) and 0.0 <= beta < 1.0 for beta in normalized),
        "miss budgets must be finite values in [0, 1)",
    )
    return normalized


def activation_target(prediction: Mapping[str, Any]) -> int:
    return max(2, int(prediction["baseline_k"]) - 1)


def _transitions_by_iteration(
    predictions: Sequence[Mapping[str, Any]],
) -> Dict[int, list[tuple[Mapping[str, Any], Mapping[str, Any]]]]:
    by_iteration = {iteration: [] for iteration in ELIGIBLE_ITERATIONS}
    for prediction in predictions:
        _require(
            prediction["actual_origin"] == ORIGIN,
            f"{prediction['key']}: activation fitting supports {ORIGIN} only",
        )
        _require(
            int(prediction["max_iter"]) == 32,
            f"{prediction['key']}: activation replay requires max_iter=32",
        )
        seen_iterations = set()
        for transition in prediction["transitions"]:
            iteration = int(transition["k"])
            if iteration not in by_iteration:
                continue
            raw_mse = float(transition["raw_mse"])
            _require(
                math.isfinite(raw_mse) and raw_mse >= 0.0,
                f"{prediction['key']}: raw_mse at k={iteration} must be finite and non-negative",
            )
            _require(
                iteration not in seen_iterations,
                f"{prediction['key']}: duplicate transition at k={iteration}",
            )
            seen_iterations.add(iteration)
            by_iteration[iteration].append((prediction, transition))
        _require(
            tuple(sorted(seen_iterations)) == ELIGIBLE_ITERATIONS,
            f"{prediction['key']}: transitions must cover k=2..32",
        )
    return by_iteration


def _smallest_capture_threshold(values: Sequence[float], beta: float) -> Dict[str, Any]:
    _require(bool(values), "threshold fitting requires activation-due samples")
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    required_count = int(math.ceil((1.0 - beta) * len(ordered)))
    threshold = float(ordered[required_count - 1])
    captured_count = int(np.searchsorted(ordered, threshold, side="right"))
    return {
        "threshold": threshold,
        "threshold_hex": threshold.hex(),
        "required_capture_count": required_count,
        "captured_sample_count": captured_count,
        "empirical_capture": captured_count / len(ordered),
        "unique_candidate_count": len(np.unique(ordered)),
    }


def fit_dynamic_threshold_curve(
    predictions: Sequence[Mapping[str, Any]],
    beta: float,
    *,
    min_activation_due_samples: int = DEFAULT_MIN_ACTIVATION_DUE_SAMPLES,
) -> list[Dict[str, Any]]:
    """Fit one activation threshold per k from training tasks only."""

    _require(bool(predictions), "dynamic threshold fitting requires predictions")
    _require(
        isinstance(min_activation_due_samples, int)
        and not isinstance(min_activation_due_samples, bool)
        and min_activation_due_samples >= 1,
        "min_activation_due_samples must be an integer >= 1",
    )
    beta = _validate_betas((beta,))[0]
    by_iteration = _transitions_by_iteration(predictions)
    curve = []
    for iteration in ELIGIBLE_ITERATIONS:
        direct_observations = by_iteration[iteration]
        direct_due = [
            transition
            for prediction, transition in direct_observations
            if activation_target(prediction) <= iteration
        ]
        calibration_iterations = [iteration]
        calibration_due = direct_due
        source = "direct"
        if len(direct_due) < min_activation_due_samples:
            calibration_iterations = [
                neighbor
                for neighbor in (iteration - 1, iteration, iteration + 1)
                if neighbor in by_iteration
            ]
            calibration_due = [
                transition
                for neighbor in calibration_iterations
                for prediction, transition in by_iteration[neighbor]
                if activation_target(prediction) <= neighbor
            ]
            source = "pooled_neighborhood"

        if len(calibration_due) < min_activation_due_samples:
            source = "fail_closed_insufficient_activation_due_samples"
            selection = {
                "threshold": FAIL_CLOSED_THRESHOLD,
                "threshold_hex": FAIL_CLOSED_THRESHOLD.hex(),
                "required_capture_count": 0,
                "captured_sample_count": 0,
                "empirical_capture": None,
                "unique_candidate_count": len(
                    {float(item["raw_mse"]) for item in calibration_due}
                ),
            }
        else:
            selection = _smallest_capture_threshold(
                [float(item["raw_mse"]) for item in calibration_due], beta
            )

        curve.append(
            {
                "iteration": iteration,
                "beta": beta,
                "threshold": selection["threshold"],
                "threshold_hex": selection["threshold_hex"],
                "calibration_source": source,
                "calibration_iterations": calibration_iterations,
                "minimum_activation_due_samples": min_activation_due_samples,
                "direct_observation_count": len(direct_observations),
                "direct_activation_due_count": len(direct_due),
                "direct_activation_not_due_count": (
                    len(direct_observations) - len(direct_due)
                ),
                "calibration_activation_due_count": len(calibration_due),
                "required_capture_count": selection["required_capture_count"],
                "captured_sample_count": selection["captured_sample_count"],
                "empirical_capture": selection["empirical_capture"],
                "unique_candidate_count": selection["unique_candidate_count"],
            }
        )
    return curve


def fit_fixed_activation_threshold(
    predictions: Sequence[Mapping[str, Any]], beta: float
) -> Dict[str, Any]:
    """Fit the smallest fixed threshold meeting the training on-time target."""

    _require(bool(predictions), "fixed threshold fitting requires predictions")
    beta = _validate_betas((beta,))[0]
    _transitions_by_iteration(predictions)
    due_minima = []
    for prediction in predictions:
        target = activation_target(prediction)
        values = [
            float(transition["raw_mse"])
            for transition in prediction["transitions"]
            if int(transition["k"]) <= target
        ]
        _require(bool(values), f"{prediction['key']}: no raw-MSE values by activation target")
        due_minima.append(min(values))
    selection = _smallest_capture_threshold(due_minima, beta)
    return {
        "beta": beta,
        **selection,
        "training_prediction_count": len(predictions),
        "selection_rule": (
            "smallest fixed threshold with training on-time activation rate >= 1-beta"
        ),
    }


def fixed_threshold_curve(selection: Mapping[str, Any]) -> list[Dict[str, Any]]:
    return [
        {
            "iteration": iteration,
            "beta": float(selection["beta"]),
            "threshold": float(selection["threshold"]),
            "threshold_hex": selection["threshold_hex"],
            "calibration_source": "fixed_training_replay",
            "calibration_iterations": list(ELIGIBLE_ITERATIONS),
            "direct_observation_count": int(selection["training_prediction_count"]),
            "direct_activation_due_count": int(selection["training_prediction_count"]),
            "direct_activation_not_due_count": 0,
            "calibration_activation_due_count": int(
                selection["training_prediction_count"]
            ),
            "required_capture_count": int(selection["required_capture_count"]),
            "captured_sample_count": int(selection["captured_sample_count"]),
            "empirical_capture": float(selection["empirical_capture"]),
            "unique_candidate_count": int(selection["unique_candidate_count"]),
        }
        for iteration in ELIGIBLE_ITERATIONS
    ]


def replay_activation_policy(
    predictions: Sequence[Mapping[str, Any]],
    threshold_curve: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    """Replay Coda execution and action-MSE stopping from recorded transitions."""

    thresholds = {
        int(item["iteration"]): float(item["threshold"])
        for item in threshold_curve
    }
    _require(
        tuple(sorted(thresholds)) == ELIGIBLE_ITERATIONS,
        "activation threshold table must cover k=2..32",
    )
    replays = []
    for prediction in predictions:
        _transitions_by_iteration((prediction,))
        transitions = {
            int(item["k"]): item for item in prediction["transitions"]
        }
        crossing = next(
            (
                iteration
                for iteration in TRIGGER_SEARCH_ITERATIONS
                if float(transitions[iteration]["raw_mse"]) <= thresholds[iteration]
            ),
            None,
        )
        forced_trigger = crossing is None
        trigger_k = FORCED_TRIGGER_ITERATION if forced_trigger else int(crossing)
        first_check_k = 2 if trigger_k == 2 else trigger_k + 1
        stopping_k = next(
            (
                iteration
                for iteration in range(first_check_k, int(prediction["max_iter"]) + 1)
                if bool(transitions[iteration]["label"])
            ),
            None,
        )
        terminal_k = (
            int(stopping_k)
            if stopping_k is not None
            else int(prediction["max_iter"])
        )
        executed_coda_iterations = [1, *range(trigger_k, terminal_k + 1)]
        executed_action_mse_checks = list(range(first_check_k, terminal_k + 1))
        scheduled_coda_calls = len(executed_coda_iterations)
        expected_calls = 1 + (terminal_k - trigger_k + 1)
        _require(
            scheduled_coda_calls == expected_calls,
            f"{prediction['key']}: inconsistent Coda call count",
        )
        target = activation_target(prediction)
        baseline_k = int(prediction["baseline_k"])
        replays.append(
            {
                "key": prediction["key"],
                "task_id": str(prediction["task_id"]),
                "actual_origin": prediction["actual_origin"],
                "baseline_k": baseline_k,
                "max_iter": int(prediction["max_iter"]),
                "activation_target": target,
                "trigger_k": trigger_k,
                "forced_trigger": forced_trigger,
                "first_action_mse_check_k": first_check_k,
                "terminal_k": terminal_k,
                "stop_reason": (
                    "action_mse" if stopping_k is not None else "max_iter"
                ),
                "delta_k": terminal_k - baseline_k,
                "exact_k_preserved": terminal_k == baseline_k,
                "delta_k_gt_0": terminal_k > baseline_k,
                "trigger_delay": max(0, trigger_k - target),
                "early_trigger_distance": max(0, target - trigger_k),
                "max_iteration": terminal_k == int(prediction["max_iter"]),
                "baseline_coda_calls": baseline_k,
                "scheduled_coda_calls": scheduled_coda_calls,
                "executed_coda_iterations": executed_coda_iterations,
                "executed_action_mse_checks": executed_action_mse_checks,
            }
        )
    return replays


def replay_coda_every_iteration(
    predictions: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    replays = []
    for prediction in predictions:
        baseline_k = int(prediction["baseline_k"])
        target = activation_target(prediction)
        baseline_transition = next(
            item
            for item in prediction["transitions"]
            if int(item["k"]) == baseline_k
        )
        replays.append(
            {
                "key": prediction["key"],
                "task_id": str(prediction["task_id"]),
                "actual_origin": prediction["actual_origin"],
                "baseline_k": baseline_k,
                "max_iter": int(prediction["max_iter"]),
                "activation_target": target,
                "trigger_k": 1,
                "forced_trigger": False,
                "first_action_mse_check_k": 2,
                "terminal_k": baseline_k,
                "stop_reason": (
                    "action_mse" if baseline_transition["label"] else "max_iter"
                ),
                "delta_k": 0,
                "exact_k_preserved": True,
                "delta_k_gt_0": False,
                "trigger_delay": 0,
                "early_trigger_distance": target - 1,
                "max_iteration": baseline_k == int(prediction["max_iter"]),
                "baseline_coda_calls": baseline_k,
                "scheduled_coda_calls": baseline_k,
                "executed_coda_iterations": list(range(1, baseline_k + 1)),
                "executed_action_mse_checks": list(range(2, baseline_k + 1)),
            }
        )
    return replays


TASK_MACRO_FIELDS = (
    "mean_delta_K",
    "p95_delta_K",
    "exact_K_preservation_rate",
    "delta_K_gt_0_rate",
    "mean_trigger_delay",
    "p95_trigger_delay",
    "mean_early_trigger_distance",
    "forced_trigger_rate",
    "max_iteration_rate",
    "mean_coda_calls_per_prediction",
    "coda_call_reduction",
)


def _aggregate_flat(replays: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    _require(bool(replays), "cannot aggregate an empty activation replay set")
    baseline_calls = int(sum(item["baseline_coda_calls"] for item in replays))
    scheduled_calls = int(sum(item["scheduled_coda_calls"] for item in replays))
    return {
        "prediction_count": len(replays),
        "mean_delta_K": _mean([item["delta_k"] for item in replays]),
        "p95_delta_K": percentile([item["delta_k"] for item in replays], 0.95),
        "exact_K_preservation_rate": _mean(
            [item["exact_k_preserved"] for item in replays]
        ),
        "delta_K_gt_0_rate": _mean([item["delta_k_gt_0"] for item in replays]),
        "mean_trigger_delay": _mean([item["trigger_delay"] for item in replays]),
        "p95_trigger_delay": percentile(
            [item["trigger_delay"] for item in replays], 0.95
        ),
        "mean_early_trigger_distance": _mean(
            [item["early_trigger_distance"] for item in replays]
        ),
        "forced_trigger_rate": _mean([item["forced_trigger"] for item in replays]),
        "max_iteration_rate": _mean([item["max_iteration"] for item in replays]),
        "baseline_total_coda_calls": baseline_calls,
        "scheduled_total_coda_calls": scheduled_calls,
        "mean_coda_calls_per_prediction": scheduled_calls / len(replays),
        "coda_call_reduction": (baseline_calls - scheduled_calls) / baseline_calls,
        "coda_call_reduction_percentage": (
            100.0 * (baseline_calls - scheduled_calls) / baseline_calls
        ),
    }


def aggregate_activation_replays(
    replays: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    result = _aggregate_flat(replays)
    by_task: Dict[str, list[Mapping[str, Any]]] = {}
    for replay in replays:
        by_task.setdefault(str(replay["task_id"]), []).append(replay)
    task_metrics = {
        task_id: _aggregate_flat(items)
        for task_id, items in sorted(by_task.items(), key=lambda item: int(item[0]))
    }
    for field in TASK_MACRO_FIELDS:
        result[f"task_macro_{field}"] = _mean(
            [metrics[field] for metrics in task_metrics.values()]
        )
    result["task_metrics"] = task_metrics
    return result


COMPARISON_FIELDS = (
    *TASK_MACRO_FIELDS,
    *(f"task_macro_{field}" for field in TASK_MACRO_FIELDS),
)


def compare_activation_metrics(
    scheduled: Mapping[str, Any], reference: Mapping[str, Any], reference_name: str
) -> Dict[str, Any]:
    return {
        "reference": reference_name,
        "scheduled_minus_reference": {
            field: scheduled[field] - reference[field]
            for field in COMPARISON_FIELDS
        },
    }


def _pareto_frontier(schedules: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    def objectives(schedule: Mapping[str, Any]) -> tuple[float, ...]:
        metrics = schedule["oof_metrics"]
        return (
            -float(metrics["coda_call_reduction"]),
            float(metrics["mean_delta_K"]),
            float(metrics["p95_delta_K"]),
            -float(metrics["exact_K_preservation_rate"]),
            float(metrics["forced_trigger_rate"]),
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
            metrics = candidate["oof_metrics"]
            frontier.append(
                {
                    "beta": candidate["beta"],
                    "coda_call_reduction": metrics["coda_call_reduction"],
                    "mean_delta_K": metrics["mean_delta_K"],
                    "p95_delta_K": metrics["p95_delta_K"],
                    "exact_K_preservation_rate": metrics[
                        "exact_K_preservation_rate"
                    ],
                    "forced_trigger_rate": metrics["forced_trigger_rate"],
                    "max_iteration_rate": metrics["max_iteration_rate"],
                }
            )
    return frontier


def _select_schedule(schedules: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    mean_qualified = [
        schedule
        for schedule in schedules
        if schedule["oof_metrics"]["mean_delta_K"] <= 0.1
    ]
    p95_qualified = [
        schedule
        for schedule in schedules
        if schedule["oof_metrics"]["p95_delta_K"] <= 1.0
    ]
    joint = [schedule for schedule in mean_qualified if schedule in p95_qualified]
    if joint:
        winner = max(
            joint,
            key=lambda schedule: (
                schedule["oof_metrics"]["coda_call_reduction"],
                -schedule["oof_metrics"]["mean_delta_K"],
                -schedule["oof_metrics"]["p95_delta_K"],
                schedule["oof_metrics"]["exact_K_preservation_rate"],
                -schedule["beta"],
            ),
        )
        status = "qualifying_schedule_selected"
        selected = {
            "beta": winner["beta"],
            "coda_call_reduction": winner["oof_metrics"]["coda_call_reduction"],
            "mean_delta_K": winner["oof_metrics"]["mean_delta_K"],
            "p95_delta_K": winner["oof_metrics"]["p95_delta_K"],
        }
    else:
        status = "no_schedule_satisfies_both_constraints"
        selected = None
    return {
        "mean_delta_K_constraint": 0.1,
        "p95_delta_K_constraint": 1.0,
        "mean_delta_K_qualifying_betas": [item["beta"] for item in mean_qualified],
        "p95_delta_K_qualifying_betas": [item["beta"] for item in p95_qualified],
        "jointly_qualifying_betas": [item["beta"] for item in joint],
        "status": status,
        "selected_schedule": selected,
        "selection_rule": "largest Coda-call reduction among jointly qualifying schedules",
    }


def evaluate_coda_activation_oof(
    predictions: Sequence[Mapping[str, Any]],
    fold_assignment: Mapping[str, int],
    *,
    betas: Sequence[float] = DEFAULT_BETAS,
    min_activation_due_samples: int = DEFAULT_MIN_ACTIVATION_DUE_SAMPLES,
) -> Dict[str, Any]:
    """Evaluate dynamic and fixed activation policies on held-out ACTUAL_WARM tasks."""

    betas = _validate_betas(betas)
    origin_predictions = [
        prediction
        for prediction in predictions
        if prediction["actual_origin"] == ORIGIN
    ]
    _require(bool(origin_predictions), f"missing {ORIGIN} predictions")
    _transitions_by_iteration(origin_predictions)
    task_ids = {str(prediction["task_id"]) for prediction in origin_predictions}
    _require(task_ids == set(fold_assignment), "trace/fold task IDs differ")
    fold_ids = sorted(set(fold_assignment.values()))
    _require(len(fold_ids) == 5, "activation evaluator requires the existing 5-fold split")

    baseline_replays = []
    dynamic = {
        beta: {"beta": beta, "beta_hex": beta.hex(), "replays": [], "folds": []}
        for beta in betas
    }
    fixed = {
        beta: {"beta": beta, "beta_hex": beta.hex(), "replays": [], "folds": []}
        for beta in betas
    }
    leakage_folds = []
    threshold_curve_data = []

    for fold_id in fold_ids:
        train = [
            prediction
            for prediction in origin_predictions
            if fold_assignment[str(prediction["task_id"])] != fold_id
        ]
        validation = [
            prediction
            for prediction in origin_predictions
            if fold_assignment[str(prediction["task_id"])] == fold_id
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
        baseline_replays.extend(replay_coda_every_iteration(validation))

        for beta in betas:
            LOGGER.info(
                "evaluating Coda activation origin=%s fold=%s beta=%.17g",
                ORIGIN,
                fold_id,
                beta,
            )
            dynamic_curve = fit_dynamic_threshold_curve(
                train,
                beta,
                min_activation_due_samples=min_activation_due_samples,
            )
            dynamic_held_out = replay_activation_policy(validation, dynamic_curve)
            dynamic[beta]["replays"].extend(dynamic_held_out)
            dynamic[beta]["folds"].append(
                {
                    "fold_id": fold_id,
                    "training_task_ids": train_tasks,
                    "validation_task_ids": validation_tasks,
                    "training_prediction_count": len(train),
                    "validation_prediction_count": len(validation),
                    "thresholds_by_iteration": dynamic_curve,
                    "held_out_metrics": aggregate_activation_replays(
                        dynamic_held_out
                    ),
                }
            )
            threshold_curve_data.extend(
                {"fold_id": fold_id, **row} for row in dynamic_curve
            )

            fixed_selection = fit_fixed_activation_threshold(train, beta)
            fixed_curve = fixed_threshold_curve(fixed_selection)
            fixed_held_out = replay_activation_policy(validation, fixed_curve)
            fixed[beta]["replays"].extend(fixed_held_out)
            fixed[beta]["folds"].append(
                {
                    "fold_id": fold_id,
                    "training_task_ids": train_tasks,
                    "validation_task_ids": validation_tasks,
                    "training_prediction_count": len(train),
                    "validation_prediction_count": len(validation),
                    "threshold_selection": fixed_selection,
                    "held_out_metrics": aggregate_activation_replays(fixed_held_out),
                }
            )

    baseline_metrics = aggregate_activation_replays(baseline_replays)
    fixed_output = []
    dynamic_output = []
    for beta in betas:
        fixed_schedule = fixed[beta]
        fixed_metrics = aggregate_activation_replays(fixed_schedule.pop("replays"))
        fixed_schedule["oof_metrics"] = fixed_metrics
        fixed_schedule["comparison_to_coda_every_iteration"] = (
            compare_activation_metrics(
                fixed_metrics, baseline_metrics, "coda_every_iteration"
            )
        )
        fixed_output.append(fixed_schedule)

        dynamic_schedule = dynamic[beta]
        dynamic_metrics = aggregate_activation_replays(
            dynamic_schedule.pop("replays")
        )
        dynamic_schedule["oof_metrics"] = dynamic_metrics
        dynamic_schedule["comparison_to_coda_every_iteration"] = (
            compare_activation_metrics(
                dynamic_metrics, baseline_metrics, "coda_every_iteration"
            )
        )
        dynamic_schedule["comparison_to_fixed_raw_mse_activation"] = (
            compare_activation_metrics(
                dynamic_metrics, fixed_metrics, "fixed_raw_mse_activation_same_beta"
            )
        )
        dynamic_output.append(dynamic_schedule)

    selection = _select_schedule(dynamic_output)
    return {
        "schema_version": 1,
        "status": "offline_diagnostic_only",
        "origin": ORIGIN,
        "label_definition": "adjacent_action_mse < 0.001",
        "activation_target_definition": "max(2, baseline_k - 1)",
        "stopping_definition": (
            "first executed adjacent Coda pair with adjacent_action_mse < 0.001; "
            "otherwise max_iter"
        ),
        "eligible_iterations": list(ELIGIBLE_ITERATIONS),
        "forced_trigger_iteration": FORCED_TRIGGER_ITERATION,
        "miss_budgets": list(betas),
        "minimum_activation_due_samples": min_activation_due_samples,
        "sparse_iteration_fallback": (
            "pool activation-due samples from k-1, k, and k+1; if still "
            "insufficient, use a threshold below every non-negative raw MSE"
        ),
        "prediction_count": len(origin_predictions),
        "task_ids": sorted(task_ids, key=int),
        "leakage_audit": {"passed": True, "folds": leakage_folds},
        "coda_every_iteration_reference": baseline_metrics,
        "fixed_raw_mse_activation_schedules": fixed_output,
        "dynamic_activation_schedules": dynamic_output,
        "pareto_frontier": _pareto_frontier(dynamic_output),
        "selection": selection,
        "threshold_curve_data": threshold_curve_data,
        "runtime_inference_modified": False,
        "runtime_defaults_modified": False,
        "calibration_traces_modified": False,
        "fold_assignments_modified": False,
    }


def format_pareto_table(report: Mapping[str, Any]) -> str:
    selection = report["selection"]
    winner = selection["selected_schedule"]
    winner_beta = None if winner is None else winner["beta"]
    mean_qualified = set(selection["mean_delta_K_qualifying_betas"])
    p95_qualified = set(selection["p95_delta_K_qualifying_betas"])
    headers = (
        "beta",
        "Coda reduction",
        "mean delta_K",
        "p95 delta_K",
        "exact-K",
        "forced-trigger",
        "max-iteration",
    )
    rows = [headers]
    for item in report["pareto_frontier"]:
        flags = ""
        if item["beta"] in mean_qualified:
            flags += "M"
        if item["beta"] in p95_qualified:
            flags += "P"
        if item["beta"] == winner_beta:
            flags += "*"
        beta = f"{item['beta']:.4g}" + (f" [{flags}]" if flags else "")
        rows.append(
            (
                beta,
                f"{item['coda_call_reduction']:.4%}",
                f"{item['mean_delta_K']:.4f}",
                f"{item['p95_delta_K']:.4f}",
                f"{item['exact_K_preservation_rate']:.4%}",
                f"{item['forced_trigger_rate']:.4%}",
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
    rendered.append("M: mean delta_K <= 0.1; P: p95 delta_K <= 1; *: selected")
    return "\n".join(rendered)
