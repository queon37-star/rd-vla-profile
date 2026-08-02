import copy

from scripts.dynamic_raw_mse_oof import (
    DEFAULT_ALPHAS,
    ELIGIBLE_ITERATIONS,
    FAIL_CLOSED_THRESHOLD,
    evaluate_dynamic_raw_mse_oof,
    fit_threshold_curve,
    format_pareto_table,
    replay_dynamic_thresholds,
)


def _prediction(task_id, prediction_id, values, labels, *, baseline_k=3):
    transitions = [
        {
            "k": iteration,
            "raw_mse": float(value),
            "action_mse": 0.0005 if label else 0.01,
            "label": bool(label),
        }
        for iteration, (value, label) in enumerate(zip(values, labels), start=2)
    ]
    return {
        "key": (str(task_id), 0, prediction_id),
        "task_id": str(task_id),
        "episode_id": 0,
        "prediction_id": prediction_id,
        "actual_origin": "ACTUAL_WARM",
        "baseline_k": baseline_k,
        "max_iter": len(transitions) + 1,
        "transitions": transitions,
    }


def _predictions():
    return [
        _prediction(
            task_id,
            task_id,
            [0.30 + 0.01 * (task_id % 3), 0.10 + 0.005 * (task_id % 2), 0.05],
            [False, True, True],
        )
        for task_id in range(10)
    ]


def _assignment():
    return {str(task_id): min(task_id, 9 - task_id) for task_id in range(10)}


def _fail_closed_curve():
    return [
        {"iteration": iteration, "threshold": FAIL_CLOSED_THRESHOLD}
        for iteration in ELIGIBLE_ITERATIONS
    ]


def test_dynamic_replay_uses_exact_first_crossing_semantics():
    curve = _fail_closed_curve()
    curve[0]["threshold"] = 0.15
    curve[1]["threshold"] = 0.10
    prediction = _prediction(0, 0, [0.20, 0.05, 0.01], [False, True, True])

    replay = replay_dynamic_thresholds([prediction], curve)[0]

    assert replay["terminal_k"] == 3
    assert replay["stopped"] is True
    assert replay["true_stop"] is True
    assert replay["false_convergence"] is False
    assert replay["delta_k"] == 0


def test_sparse_iteration_uses_neighbor_pool_then_fails_closed():
    predictions = [
        _prediction(0, 0, [0.4, 0.1, 0.3], [False, True, False]),
        _prediction(1, 1, [0.5, 0.2, 0.35], [False, True, True]),
    ]

    curve = fit_threshold_curve(
        predictions, 0.0, min_negative_samples=2
    )
    by_iteration = {row["iteration"]: row for row in curve}

    assert by_iteration[3]["calibration_source"] == "pooled_neighborhood"
    assert by_iteration[3]["calibration_iterations"] == [2, 3, 4]
    assert by_iteration[3]["local_negative_count"] == 0
    assert by_iteration[3]["calibration_negative_count"] == 3
    assert by_iteration[10]["calibration_source"] == "fail_closed_insufficient_negatives"
    assert by_iteration[10]["threshold"] == FAIL_CLOSED_THRESHOLD


def test_tied_raw_mse_values_are_never_split_by_threshold_selection():
    predictions = [
        _prediction(0, 0, [0.1], [False], baseline_k=2),
        _prediction(1, 1, [0.1], [True], baseline_k=2),
        _prediction(2, 2, [0.2], [False], baseline_k=2),
        _prediction(3, 3, [0.05], [True], baseline_k=2),
    ]

    zero_budget = fit_threshold_curve(
        predictions, 0.0, min_negative_samples=1
    )[0]
    half_budget = fit_threshold_curve(
        predictions, 0.5, min_negative_samples=1
    )[0]

    assert 0.05 < zero_budget["threshold"] < 0.1
    assert zero_budget["empirical_false_positive_count"] == 0
    assert 0.1 < half_budget["threshold"] < 0.2
    assert half_budget["empirical_false_positive_count"] == 1
    assert half_budget["selected_sample_count"] == 3
    assert half_budget["selected_positive_count"] == 2


def test_oof_report_has_no_leakage_complete_tables_comparisons_and_is_deterministic():
    predictions = _predictions()
    kwargs = {
        "alphas": (0.0, 0.25),
        "min_negative_samples": 1,
    }

    first = evaluate_dynamic_raw_mse_oof(predictions, _assignment(), **kwargs)
    second = evaluate_dynamic_raw_mse_oof(predictions, _assignment(), **kwargs)

    assert first == second
    assert first["leakage_audit"]["passed"] is True
    assert len(first["leakage_audit"]["folds"]) == 5
    for fold in first["leakage_audit"]["folds"]:
        assert fold["task_overlap_count"] == 0
        assert fold["prediction_overlap_count"] == 0
        assert set(fold["training_task_ids"]).isdisjoint(fold["validation_task_ids"])
    assert first["runtime_inference_modified"] is False
    assert first["runtime_defaults_modified"] is False
    assert len(first["dynamic_schedules"]) == 2
    for schedule in first["dynamic_schedules"]:
        assert schedule["comparison_to_fixed_raw_mse"]["reference"] == (
            "existing_fixed_raw_mse_oof"
        )
        for fold in schedule["folds"]:
            assert [
                row["iteration"] for row in fold["thresholds_by_iteration"]
            ] == list(ELIGIBLE_ITERATIONS)
            assert [
                row["iteration"] for row in fold["sample_counts_by_iteration"]
            ] == list(ELIGIBLE_ITERATIONS)
    assert "false convergence" in format_pareto_table(first)


def test_held_out_changes_do_not_alter_fold_training_thresholds():
    original_predictions = _predictions()
    changed_predictions = copy.deepcopy(original_predictions)
    for prediction in changed_predictions:
        if prediction["task_id"] in ("0", "9"):
            for transition in prediction["transitions"]:
                transition["raw_mse"] *= 1000.0

    kwargs = {"alphas": DEFAULT_ALPHAS, "min_negative_samples": 1}
    original = evaluate_dynamic_raw_mse_oof(
        original_predictions, _assignment(), **kwargs
    )
    changed = evaluate_dynamic_raw_mse_oof(
        changed_predictions, _assignment(), **kwargs
    )

    original_fixed = original["fixed_raw_mse_reference"]["folds"][0]
    changed_fixed = changed["fixed_raw_mse_reference"]["folds"][0]
    assert original_fixed["validation_task_ids"] == ["0", "9"]
    assert original_fixed["threshold_selection"] == changed_fixed["threshold_selection"]
    for original_schedule, changed_schedule in zip(
        original["dynamic_schedules"], changed["dynamic_schedules"]
    ):
        original_curve = original_schedule["folds"][0]["thresholds_by_iteration"]
        changed_curve = changed_schedule["folds"][0]["thresholds_by_iteration"]
        assert original_curve == changed_curve
