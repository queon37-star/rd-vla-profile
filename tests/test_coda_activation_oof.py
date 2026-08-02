import copy
import json

from scripts.coda_activation_oof import (
    DEFAULT_BETAS,
    ELIGIBLE_ITERATIONS,
    FAIL_CLOSED_THRESHOLD,
    activation_target,
    evaluate_coda_activation_oof,
    fit_dynamic_threshold_curve,
    format_pareto_table,
    replay_activation_policy,
    replay_coda_every_iteration,
)


def _prediction(
    task_id,
    prediction_id,
    *,
    baseline_k,
    raw_overrides=None,
    true_iterations=None,
):
    raw_overrides = raw_overrides or {}
    true_iterations = {baseline_k} if true_iterations is None else set(true_iterations)
    transitions = [
        {
            "k": iteration,
            "raw_mse": float(raw_overrides.get(iteration, 1.0)),
            "action_mse": 0.0005 if iteration in true_iterations else 0.01,
            "label": iteration in true_iterations,
        }
        for iteration in ELIGIBLE_ITERATIONS
    ]
    return {
        "key": (str(task_id), 0, prediction_id),
        "task_id": str(task_id),
        "episode_id": 0,
        "prediction_id": prediction_id,
        "actual_origin": "ACTUAL_WARM",
        "baseline_k": baseline_k,
        "max_iter": 32,
        "transitions": transitions,
    }


def _curve(threshold=FAIL_CLOSED_THRESHOLD):
    return [
        {"iteration": iteration, "threshold": float(threshold)}
        for iteration in ELIGIBLE_ITERATIONS
    ]


def _set_threshold(curve, iteration, threshold):
    curve[iteration - 2]["threshold"] = float(threshold)


def _assignment():
    return {str(task_id): min(task_id, 9 - task_id) for task_id in range(10)}


def _oof_predictions():
    predictions = []
    for task_id in range(10):
        baseline_k = 3 + task_id % 5
        target = max(2, baseline_k - 1)
        raw = {
            iteration: (
                0.5 + 0.001 * task_id
                if iteration < target
                else 0.2 / (iteration - target + 1) + 0.001 * task_id
            )
            for iteration in ELIGIBLE_ITERATIONS
        }
        predictions.append(
            _prediction(
                task_id,
                task_id,
                baseline_k=baseline_k,
                raw_overrides=raw,
                true_iterations={baseline_k, min(32, baseline_k + 2)},
            )
        )
    return predictions


def test_mandatory_k1_call_and_baseline_k2_special_case():
    prediction = _prediction(0, 0, baseline_k=2, raw_overrides={2: 0.05})
    curve = _curve()
    _set_threshold(curve, 2, 0.05)

    replay = replay_activation_policy([prediction], curve)[0]
    baseline = replay_coda_every_iteration([prediction])[0]

    assert activation_target(prediction) == 2
    assert replay["trigger_k"] == 2
    assert replay["first_action_mse_check_k"] == 2
    assert replay["terminal_k"] == 2
    assert replay["executed_coda_iterations"] == [1, 2]
    assert replay["scheduled_coda_calls"] == 2
    assert baseline["executed_coda_iterations"] == [1, 2]
    assert baseline["scheduled_coda_calls"] == prediction["baseline_k"]


def test_trigger_before_at_and_after_activation_target():
    curve = _curve(0.1)
    predictions = [
        _prediction(0, 0, baseline_k=6, raw_overrides={3: 0.05}),
        _prediction(1, 1, baseline_k=6, raw_overrides={5: 0.05}),
        _prediction(
            2,
            2,
            baseline_k=6,
            raw_overrides={7: 0.05},
            true_iterations={6, 8},
        ),
    ]

    before, at, after = replay_activation_policy(predictions, curve)

    assert (before["trigger_k"], before["early_trigger_distance"]) == (3, 2)
    assert before["terminal_k"] == 6
    assert (at["trigger_k"], at["trigger_delay"]) == (5, 0)
    assert at["terminal_k"] == 6
    assert (after["trigger_k"], after["trigger_delay"]) == (7, 2)
    assert after["first_action_mse_check_k"] == 8
    assert after["terminal_k"] == 8
    for replay in (before, at, after):
        assert replay["executed_coda_iterations"][0] == 1
        assert replay["scheduled_coda_calls"] == (
            1 + replay["terminal_k"] - replay["trigger_k"] + 1
        )


def test_non_monotonic_action_labels_are_replayed_instead_of_approximated():
    prediction = _prediction(
        0,
        0,
        baseline_k=4,
        raw_overrides={4: 0.05},
        true_iterations={4, 6},
    )
    curve = _curve()
    _set_threshold(curve, 4, 0.05)

    replay = replay_activation_policy([prediction], curve)[0]

    assert replay["trigger_k"] == 4
    assert replay["first_action_mse_check_k"] == 5
    assert replay["executed_action_mse_checks"] == [5, 6]
    assert replay["terminal_k"] == 6
    assert replay["terminal_k"] != max(prediction["baseline_k"], 5)


def test_forced_k31_activation_and_explicit_coda_call_count():
    prediction = _prediction(
        0,
        0,
        baseline_k=5,
        true_iterations={5, 32},
    )

    replay = replay_activation_policy([prediction], _curve())[0]

    assert replay["forced_trigger"] is True
    assert replay["trigger_k"] == 31
    assert replay["first_action_mse_check_k"] == 32
    assert replay["terminal_k"] == 32
    assert replay["executed_coda_iterations"] == [1, 31, 32]
    assert replay["scheduled_coda_calls"] == 3


def test_tied_raw_mse_and_sparse_neighborhood_fallback_are_deterministic():
    tied = [
        _prediction(0, 0, baseline_k=3, raw_overrides={2: 0.1}),
        _prediction(1, 1, baseline_k=3, raw_overrides={2: 0.1}),
        _prediction(2, 2, baseline_k=3, raw_overrides={2: 0.2}),
        _prediction(3, 3, baseline_k=3, raw_overrides={2: 0.05}),
    ]
    tied_curve = fit_dynamic_threshold_curve(
        tied, 0.5, min_activation_due_samples=1
    )
    assert tied_curve[0]["threshold"] == 0.1
    assert tied_curve[0]["required_capture_count"] == 2
    assert tied_curve[0]["captured_sample_count"] == 3

    sparse = [
        _prediction(0, 0, baseline_k=3, raw_overrides={2: 0.2, 3: 0.1}),
        _prediction(1, 1, baseline_k=5, raw_overrides={2: 0.4, 3: 0.3}),
    ]
    first = fit_dynamic_threshold_curve(
        sparse, 0.0, min_activation_due_samples=2
    )
    second = fit_dynamic_threshold_curve(
        sparse, 0.0, min_activation_due_samples=2
    )
    by_iteration = {row["iteration"]: row for row in first}
    assert first == second
    assert by_iteration[2]["calibration_source"] == "pooled_neighborhood"
    assert by_iteration[2]["calibration_iterations"] == [2, 3]
    assert by_iteration[2]["direct_activation_due_count"] == 1
    assert by_iteration[2]["calibration_activation_due_count"] == 2
    fail_closed = fit_dynamic_threshold_curve(
        sparse, 0.0, min_activation_due_samples=100
    )
    assert fail_closed[0]["calibration_source"] == (
        "fail_closed_insufficient_activation_due_samples"
    )
    assert fail_closed[0]["threshold"] == FAIL_CLOSED_THRESHOLD


def test_oof_output_has_no_leakage_complete_tables_references_and_determinism():
    predictions = _oof_predictions()
    kwargs = {"betas": (0.0, 0.1), "min_activation_due_samples": 1}

    first = evaluate_coda_activation_oof(predictions, _assignment(), **kwargs)
    second = evaluate_coda_activation_oof(predictions, _assignment(), **kwargs)

    assert json.dumps(first, sort_keys=True, allow_nan=False) == json.dumps(
        second, sort_keys=True, allow_nan=False
    )
    assert first["leakage_audit"]["passed"] is True
    assert len(first["leakage_audit"]["folds"]) == 5
    for fold in first["leakage_audit"]["folds"]:
        assert fold["task_overlap_count"] == 0
        assert fold["prediction_overlap_count"] == 0
        assert set(fold["training_task_ids"]).isdisjoint(fold["validation_task_ids"])
    assert len(first["dynamic_activation_schedules"]) == 2
    assert len(first["fixed_raw_mse_activation_schedules"]) == 2
    for schedule in first["dynamic_activation_schedules"]:
        assert schedule["comparison_to_coda_every_iteration"]["reference"] == (
            "coda_every_iteration"
        )
        assert schedule["comparison_to_fixed_raw_mse_activation"]["reference"] == (
            "fixed_raw_mse_activation_same_beta"
        )
        for fold in schedule["folds"]:
            assert [
                row["iteration"] for row in fold["thresholds_by_iteration"]
            ] == list(ELIGIBLE_ITERATIONS)
    assert "Coda reduction" in format_pareto_table(first)
    assert first["runtime_inference_modified"] is False
    assert first["runtime_defaults_modified"] is False


def test_held_out_raw_mse_changes_do_not_change_fold_training_thresholds():
    original_predictions = _oof_predictions()
    changed_predictions = copy.deepcopy(original_predictions)
    for prediction in changed_predictions:
        if prediction["task_id"] in ("0", "9"):
            for transition in prediction["transitions"]:
                transition["raw_mse"] *= 1000.0

    kwargs = {"betas": DEFAULT_BETAS, "min_activation_due_samples": 1}
    original = evaluate_coda_activation_oof(
        original_predictions, _assignment(), **kwargs
    )
    changed = evaluate_coda_activation_oof(
        changed_predictions, _assignment(), **kwargs
    )

    for original_schedule, changed_schedule in zip(
        original["dynamic_activation_schedules"],
        changed["dynamic_activation_schedules"],
    ):
        original_fold = original_schedule["folds"][0]
        changed_fold = changed_schedule["folds"][0]
        assert original_fold["validation_task_ids"] == ["0", "9"]
        assert original_fold["thresholds_by_iteration"] == changed_fold[
            "thresholds_by_iteration"
        ]
    for original_schedule, changed_schedule in zip(
        original["fixed_raw_mse_activation_schedules"],
        changed["fixed_raw_mse_activation_schedules"],
    ):
        assert original_schedule["folds"][0]["threshold_selection"] == (
            changed_schedule["folds"][0]["threshold_selection"]
        )
