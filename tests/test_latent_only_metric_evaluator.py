import copy

import pytest

from scripts.latent_only_metric_evaluator import evaluate_oof, parse_trace_predictions


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
