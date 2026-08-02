import copy
import math

import numpy as np
import pytest

from scripts.latent_only_metric_evaluator import (
    aggregate_replays,
    evaluate_oof,
    parse_trace_predictions,
    replay_predictions,
    select_training_threshold,
)


def _step(task_id, origin, prediction_id=0):
    metrics = {
        2: (0.5, False),
        3: (0.1, True),
        4: (0.05, True),
    }
    trace = []
    for k, (value, label) in metrics.items():
        trace.append(
            {
                "iteration_index": k,
                "phase": "production" if k <= 3 else "shadow_tail",
                "actual_origin": origin,
                "raw_mse": value,
                "relative_mse": value * 2,
                "cosine_distance": value * 0.5,
                "relative_l2": value * 3,
                "adjacent_action_mse": 0.0005 if label else 0.01,
                "action_mse_below_0_001": label,
                "baseline_stopping_iteration": 3,
                "task_id": task_id,
                "episode_id": 0,
                "prediction_id": prediction_id,
            }
        )
    return {
        "task_id": task_id,
        "episode_id": 0,
        "prediction_step": prediction_id,
        "actual_origin": origin,
        "K_t": 3,
        "max_recurrent_iteration": 4,
        "latent_metric_trace": trace,
    }


def _records():
    return [
        _step(task_id, origin, prediction_id=index)
        for task_id in range(10)
        for index, origin in enumerate(("COLD", "ACTUAL_WARM"))
    ]


def _assignment():
    return {str(task_id): min(task_id, 9 - task_id) for task_id in range(10)}


def _prediction(task_id, prediction_id, origin, values, labels, *, baseline_k=None):
    transitions = [
        {
            "k": k,
            "raw_mse": float(value),
            "relative_mse": float(value),
            "cosine_distance": float(value),
            "relative_l2": float(value),
            "action_mse": 0.0005 if label else 0.01,
            "label": bool(label),
        }
        for k, (value, label) in enumerate(zip(values, labels), start=2)
    ]
    max_iter = len(transitions) + 1
    return {
        "key": (str(task_id), 0, prediction_id),
        "task_id": str(task_id),
        "episode_id": 0,
        "prediction_id": prediction_id,
        "actual_origin": origin,
        "baseline_k": max_iter if baseline_k is None else baseline_k,
        "max_iter": max_iter,
        "transitions": transitions,
    }


def _brute_force_threshold(
    predictions, metric_name="raw_mse", *, min_iter=2, capture_target=0.995
):
    values = np.asarray(
        [
            item[metric_name]
            for prediction in predictions
            for item in prediction["transitions"]
            if item["k"] >= min_iter
        ],
        dtype=np.float64,
    )
    candidates = np.concatenate(
        ([np.nextafter(values.min(), -math.inf)], np.unique(values))
    )
    evaluated = [
        (
            float(threshold),
            aggregate_replays(
                replay_predictions(
                    predictions, metric_name, float(threshold), min_iter=min_iter
                )
            ),
        )
        for threshold in candidates
    ]
    feasible = [
        item
        for item in evaluated
        if item[1]["convergence_capture"] is not None
        and item[1]["convergence_capture"] >= capture_target
    ]
    if feasible:
        threshold, metrics = min(
            feasible,
            key=lambda item: (
                item[1]["false_convergence_count"],
                -item[1]["early_stop_rate"],
                item[0],
            ),
        )
        status = "capture_feasible"
    else:
        threshold, metrics = min(
            evaluated,
            key=lambda item: (
                -(item[1]["convergence_capture"] or 0.0),
                item[1]["false_convergence_count"],
                -item[1]["early_stop_rate"],
                item[0],
            ),
        )
        status = "capture_infeasible_fail_closed"
    return {
        "threshold": threshold,
        "threshold_hex": float(threshold).hex(),
        "selection_status": status,
        "capture_target": capture_target,
        "candidate_count": len(candidates),
        "training_metrics": metrics,
    }


def _assert_matches_brute_force(predictions, *, min_iter=2, capture_target=0.995):
    optimized = select_training_threshold(
        predictions,
        "raw_mse",
        min_iter=min_iter,
        capture_target=capture_target,
    )
    reference = _brute_force_threshold(
        predictions,
        min_iter=min_iter,
        capture_target=capture_target,
    )
    assert optimized == reference


def test_oof_evaluator_reports_required_metrics_by_origin_and_task():
    predictions = parse_trace_predictions(_records())
    result = evaluate_oof(predictions, _assignment())

    assert result["leakage_audit"]["passed"] is True
    assert result["runtime_defaults_modified"] is False
    assert result["nominal_best_metric"] in result["metrics"]
    for metric in ("raw_mse", "relative_mse", "cosine_distance", "relative_l2"):
        for origin in ("ACTUAL_WARM", "COLD"):
            output = result["metrics"][metric][origin]
            assert output["classification"]["auroc"] == pytest.approx(1.0)
            assert output["classification"]["auprc"] == pytest.approx(1.0)
            assert len(output["selected_thresholds_per_fold"]) == 5
            stopping = output["oof_stopping"]
            assert stopping["false_convergence_count"] == 0
            assert stopping["convergence_capture"] == pytest.approx(1.0)
            assert stopping["mean_delta_K"] == pytest.approx(0.0)
            assert stopping["p95_delta_K"] == pytest.approx(0.0)
            assert stopping["early_stop_rate"] == pytest.approx(1.0)
            assert stopping["max_iteration_rate"] == pytest.approx(0.0)
            assert len(stopping["task_metrics"]) == 10


def test_held_out_values_do_not_change_their_fold_threshold():
    original_records = _records()
    original = evaluate_oof(parse_trace_predictions(original_records), _assignment())
    changed_records = copy.deepcopy(original_records)
    for record in changed_records:
        if record["task_id"] in (0, 9):
            for item in record["latent_metric_trace"]:
                item["raw_mse"] *= 1000.0
    changed = evaluate_oof(parse_trace_predictions(changed_records), _assignment())

    for origin in ("ACTUAL_WARM", "COLD"):
        original_fold = original["metrics"]["raw_mse"][origin]["folds"][0]
        changed_fold = changed["metrics"]["raw_mse"][origin]["folds"][0]
        assert original_fold["validation_task_ids"] == ["0", "9"]
        assert original_fold["threshold_selection"]["threshold"] == changed_fold[
            "threshold_selection"
        ]["threshold"]


@pytest.mark.parametrize("origin", ["COLD", "ACTUAL_WARM"])
def test_optimized_threshold_matches_brute_force_on_randomized_traces(origin):
    for seed in range(10):
        rng = np.random.default_rng(seed)
        predictions = []
        for prediction_id in range(8):
            transition_count = int(rng.integers(2, 7))
            values = rng.choice(
                np.asarray([0.01, 0.05, 0.1, 0.2, 0.5]),
                size=transition_count,
                replace=True,
            )
            labels = rng.random(transition_count) < 0.35
            predictions.append(
                _prediction(
                    prediction_id % 4,
                    prediction_id,
                    origin,
                    values,
                    labels,
                    baseline_k=int(rng.integers(2, transition_count + 2)),
                )
            )
        _assert_matches_brute_force(
            predictions,
            min_iter=2 + seed % 2,
            capture_target=(0.5, 0.995, 1.1)[seed % 3],
        )


def test_optimized_threshold_matches_brute_force_with_tied_metric_values():
    predictions = [
        _prediction(0, 0, "COLD", [0.2, 0.1, 0.1, 0.3], [False, True, False, True]),
        _prediction(1, 1, "COLD", [0.1, 0.1, 0.2, 0.2], [False, True, True, False]),
        _prediction(2, 2, "COLD", [0.3, 0.2, 0.1, 0.1], [True, False, True, False]),
    ]
    _assert_matches_brute_force(predictions, capture_target=0.5)


def test_optimized_threshold_matches_brute_force_without_converged_iterations():
    predictions = [
        _prediction(0, 0, "ACTUAL_WARM", [0.3, 0.2, 0.1], [False] * 3),
        _prediction(1, 1, "ACTUAL_WARM", [0.1, 0.2, 0.3], [False] * 3),
    ]
    _assert_matches_brute_force(predictions)
    selection = select_training_threshold(predictions, "raw_mse", min_iter=2)
    assert selection["selection_status"] == "capture_infeasible_fail_closed"
    assert selection["training_metrics"]["convergence_capture"] is None


def test_optimized_threshold_preserves_feasible_and_infeasible_selection():
    predictions = [
        _prediction(0, 0, "COLD", [0.1, 0.2], [True, True]),
        _prediction(1, 1, "COLD", [0.05, 0.1], [False, True]),
    ]
    for target, expected_status in (
        (0.5, "capture_feasible"),
        (0.75, "capture_infeasible_fail_closed"),
    ):
        _assert_matches_brute_force(predictions, capture_target=target)
        assert select_training_threshold(
            predictions,
            "raw_mse",
            min_iter=2,
            capture_target=target,
        )["selection_status"] == expected_status


def test_oof_evaluator_logs_progress_for_every_metric_origin_and_fold(caplog):
    caplog.set_level("INFO", logger="scripts.latent_only_metric_evaluator")
    evaluate_oof(parse_trace_predictions(_records()), _assignment())

    progress = [
        record.message
        for record in caplog.records
        if record.message.startswith("evaluating metric=")
    ]
    assert len(progress) == 4 * 2 * 5
    assert all("origin=" in message and "fold=" in message for message in progress)
