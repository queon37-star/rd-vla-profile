import copy
import csv
import hashlib
import json

import numpy as np
import pytest

from prismatic.models.latent_dynamics import (
    HISTORY_DEPENDENT_FIELDS,
    LATENT_DYNAMICS_FIELDS,
    WARM_ANCHOR_FIELDS,
)
from scripts.analyze_latent_dynamics_features import (
    FEATURE_GROUPS,
    INDIVIDUAL_FEATURES,
    LatentDynamicsAnalysisError,
    _feature_matrix_raw,
    activation_target_for_k,
    analyze_records,
    build_analysis_dataset,
    build_task_splits,
    canonical_json,
    evaluate_feature_groups_oof,
    fit_logistic,
    fit_preprocessor,
    primary_rows,
    select_univariate_direction,
    secondary_rows,
    summarize_feature_distributions,
    summarize_trajectories,
    transform_features,
    validate_input_records,
    write_outputs,
)
from scripts.check_latent_dynamics_trace import validate_records


def _trace_item(task_id, episode_id, prediction_id, origin, baseline_k, iteration):
    scale = 0.01 * (task_id + 1) + 0.001 * prediction_id
    dynamics = {
        field: scale + 0.02 * iteration + 0.0001 * index
        for index, field in enumerate(LATENT_DYNAMICS_FIELDS)
    }
    if iteration == 2:
        for field in HISTORY_DEPENDENT_FIELDS:
            dynamics[field] = None
    if origin == "COLD":
        for field in WARM_ANCHOR_FIELDS:
            dynamics[field] = None
    dynamics["token_update_energy_entropy"] = min(1.0, 0.02 * iteration)
    dynamics["token_update_top10_fraction"] = min(1.0, 0.1 + 0.01 * iteration)
    return {
        "iteration_index": iteration,
        "phase": "production" if iteration <= baseline_k else "shadow_tail",
        "actual_origin": origin,
        "raw_mse": 1.0 / (iteration + task_id + 1),
        "relative_mse": 1.0 / (iteration + task_id + 2),
        "relative_l2": 1.0 / (iteration + task_id + 3),
        "cosine_distance": 1.0 / (iteration + task_id + 4),
        "adjacent_action_mse": 0.0005 if iteration >= baseline_k - 1 else 0.01,
        "action_mse_below_0_001": iteration >= baseline_k - 1,
        "baseline_stopping_iteration": baseline_k,
        "task_id": task_id,
        "episode_id": episode_id,
        "prediction_id": prediction_id,
        **dynamics,
    }


def _record(task_id, prediction_id, *, origin="ACTUAL_WARM", baseline_k=6):
    episode_id = prediction_id // 2
    return {
        "task_id": task_id,
        "episode_id": episode_id,
        "prediction_step": prediction_id,
        "action_prediction_index": prediction_id,
        "paired_trial_id": episode_id,
        "initial_state_id": task_id,
        "episode_seed": 100 + task_id,
        "actual_origin": origin,
        "K_t": baseline_k,
        "max_recurrent_iteration": 32,
        "latent_metric_trace_enabled": True,
        "latent_dynamics_trace_enabled": True,
        "latent_dynamics_warm_anchor_available": origin == "ACTUAL_WARM",
        "shadow_full_depth_enabled": True,
        "shadow_trace_complete": True,
        "shadow_error": None,
        "latent_metric_trace": [
            _trace_item(
                task_id,
                episode_id,
                prediction_id,
                origin,
                baseline_k,
                iteration,
            )
            for iteration in range(2, 33)
        ],
    }


def _records():
    records = []
    for task_id in range(10):
        records.append(_record(task_id, 0, origin="COLD", baseline_k=4 + task_id % 3))
        records.append(
            _record(
                task_id,
                1,
                origin="ACTUAL_WARM",
                baseline_k=4 + task_id % 5,
            )
        )
    return records


def _assignment():
    return {
        "0": 0,
        "9": 0,
        "1": 1,
        "8": 1,
        "2": 2,
        "7": 2,
        "3": 3,
        "6": 3,
        "4": 4,
        "5": 4,
    }


def _manifest():
    assignment = _assignment()
    return {
        "schema_version": 1,
        "folds": [
            {
                "fold_id": fold_id,
                "validation_task_ids": [
                    int(task) for task, assigned in assignment.items() if assigned == fold_id
                ],
            }
            for fold_id in range(5)
        ],
    }


def _dataset():
    return build_analysis_dataset(_records())


def test_activation_target_primary_window_and_relative_alignment():
    assert activation_target_for_k(2) == 2
    assert activation_target_for_k(3) == 2
    assert activation_target_for_k(8) == 7

    _, transitions = _dataset()
    warm_primary = primary_rows(transitions)
    assert all(row["iteration_index"] <= row["baseline_k"] for row in warm_primary)
    assert all(
        row["activation_due"]
        == (row["iteration_index"] >= row["activation_target"])
        for row in warm_primary
    )
    assert all(
        row["relative_iteration"]
        == row["iteration_index"] - row["activation_target"]
        for row in warm_primary
    )
    assert max(row["iteration_index"] for row in secondary_rows(transitions, origin="ACTUAL_WARM")) == 32
    assert any(
        row["iteration_index"] > row["baseline_k"]
        for row in secondary_rows(transitions, origin="ACTUAL_WARM")
    )


def test_feature_matrices_exclude_labels_and_leakage_fields():
    _, transitions = _dataset()
    rows = primary_rows(transitions)
    for features in FEATURE_GROUPS.values():
        matrix = _feature_matrix_raw(rows, features)
        assert matrix.shape == (len(rows), len(features))
    with pytest.raises(LatentDynamicsAnalysisError, match="leakage"):
        _feature_matrix_raw(rows, ("adjacent_action_mse",))
    assert "activation_due" not in INDIVIDUAL_FEATURES
    assert "baseline_k" not in INDIVIDUAL_FEATURES


def test_task_folds_are_disjoint_and_cover_validation_once():
    _, transitions = _dataset()
    splits = build_task_splits(primary_rows(transitions), _assignment())
    validation_tasks = []
    for split in splits:
        assert set(split["training_task_ids"]).isdisjoint(split["validation_task_ids"])
        validation_tasks.extend(split["validation_task_ids"])
    assert sorted(validation_tasks, key=int) == [str(value) for value in range(10)]


def test_train_only_preprocessing_and_heldout_perturbation_invariance():
    _, transitions = _dataset()
    rows = primary_rows(transitions)
    split = build_task_splits(rows, _assignment())[0]
    features = FEATURE_GROUPS["combined"]
    preprocessor = fit_preprocessor(split["train_rows"], features)
    train_x = transform_features(split["train_rows"], preprocessor)
    train_y = np.asarray([row["activation_due"] for row in split["train_rows"]])
    model = fit_logistic(train_x, train_y)

    perturbed_validation = copy.deepcopy(split["validation_rows"])
    for row in perturbed_validation:
        row["activation_due"] = not row["activation_due"]
        for feature in features:
            if row["features"][feature] is not None:
                row["features"][feature] *= 1e6
    assert fit_preprocessor(split["train_rows"], features) == preprocessor
    assert fit_logistic(train_x, train_y) == model
    assert set(preprocessor["fit_task_ids"]) == set(split["training_task_ids"])


def test_k2_null_imputation_has_explicit_train_fitted_availability_indicators():
    _, transitions = _dataset()
    rows = primary_rows(transitions)
    features = ("contraction_ratio", "raw_mse")
    preprocessor = fit_preprocessor(rows, features)
    assert preprocessor["expanded_feature_names"] == [
        "contraction_ratio",
        "contraction_ratio__available",
        "raw_mse",
    ]
    transformed = transform_features(rows, preprocessor)
    k2_index = next(index for index, row in enumerate(rows) if row["iteration_index"] == 2)
    k3_index = next(index for index, row in enumerate(rows) if row["iteration_index"] == 3)
    assert transformed[k2_index, 1] < transformed[k3_index, 1]
    nonnull = [
        row["features"]["contraction_ratio"]
        for row in rows
        if row["features"]["contraction_ratio"] is not None
    ]
    assert preprocessor["imputation_medians"][0] == pytest.approx(np.median(nonnull))


def test_univariate_direction_uses_training_labels_only():
    training_scores = np.asarray([0.0, 1.0, 2.0, 3.0])
    training_labels = np.asarray([True, True, False, False])
    direction = select_univariate_direction(training_labels, training_scores)
    assert direction == -1
    heldout_scores = np.asarray([-100.0, 100.0])
    heldout_labels = np.asarray([False, True])
    heldout_scores *= -1
    heldout_labels = ~heldout_labels
    assert select_univariate_direction(training_labels, training_scores) == direction


def test_distribution_and_trajectory_windows_are_labeled_and_aligned():
    _, transitions = _dataset()
    distributions = summarize_feature_distributions(transitions)
    raw_rows = [row for row in distributions if row["feature"] == "raw_mse"]
    primary = next(
        row for row in raw_rows if row["population"] == "primary_actual_warm_production_window"
    )
    secondary = next(
        row
        for row in raw_rows
        if row["population"] == "secondary_actual_warm_full_depth_shadow_inclusive"
    )
    assert primary["sample_count"] < secondary["sample_count"]
    assert primary["shadow_inclusive"] is False
    assert secondary["shadow_inclusive"] is True

    trajectory = summarize_trajectories(transitions, min_samples=1)
    aligned = next(
        row
        for row in trajectory
        if row["actual_origin"] == "ACTUAL_WARM"
        and row["difficulty"] == "all"
        and row["relative_iteration"] == 0
        and row["feature"] == "iteration_index"
    )
    expected = [
        row["iteration_index"]
        for row in primary_rows(transitions)
        if row["relative_iteration"] == 0
    ]
    assert aligned["median"] == pytest.approx(np.median(expected))
    assert aligned["window"] == "production_window_only_not_shadow_inclusive"


def test_exact_dataset_identity_and_counts_are_required():
    records = _records()
    contract = validate_records(records)
    validated = validate_input_records(
        records,
        expected_identity_sha256=contract["workload_identity_sha256"],
        expected_prediction_count=20,
        expected_transition_count=620,
    )
    assert validated == contract
    with pytest.raises(Exception, match="SHA-256 mismatch"):
        validate_input_records(
            records,
            expected_identity_sha256="0" * 64,
            expected_prediction_count=20,
            expected_transition_count=620,
        )
    with pytest.raises(LatentDynamicsAnalysisError, match="prediction count mismatch"):
        validate_input_records(
            records,
            expected_identity_sha256=contract["workload_identity_sha256"],
            expected_prediction_count=21,
            expected_transition_count=620,
        )


def test_oof_training_parameters_ignore_heldout_values_and_labels():
    _, transitions = _dataset()
    rows = primary_rows(transitions)
    _, original = evaluate_feature_groups_oof(rows, _assignment())
    changed = copy.deepcopy(rows)
    for row in changed:
        if str(row["task_id"]) in {"0", "9"}:
            row["activation_due"] = not row["activation_due"]
            row["features"]["raw_mse"] *= 1e5
    _, perturbed = evaluate_feature_groups_oof(changed, _assignment())
    original_fold = original["raw_mse"]["logistic_regression"]["folds"][0]
    perturbed_fold = perturbed["raw_mse"]["logistic_regression"]["folds"][0]
    assert original_fold["preprocessor"] == perturbed_fold["preprocessor"]
    assert original_fold["model"] == perturbed_fold["model"]
    assert original_fold["threshold"] == perturbed_fold["threshold"]


def test_analysis_and_written_outputs_are_deterministic(tmp_path):
    records = _records()
    contract = validate_records(records)
    kwargs = {
        "fold_manifest": _manifest(),
        "assignment": _assignment(),
        "contract": contract,
    }
    first = analyze_records(records, **kwargs)
    second = analyze_records(copy.deepcopy(records), **kwargs)
    assert canonical_json(first) == canonical_json(second)

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    inputs = {"fixture_sha256": hashlib.sha256(b"fixture").hexdigest()}
    write_outputs(first_dir, first, inputs=inputs, overwrite=False)
    write_outputs(second_dir, second, inputs=inputs, overwrite=False)
    expected_files = {
        "dataset_summary.json",
        "prediction_summary.csv",
        "feature_distribution_summary.csv",
        "univariate_oof_results.csv",
        "feature_group_oof_results.csv",
        "trajectory_relative_to_activation.csv",
        "analysis_report.json",
    }
    assert {path.name for path in first_dir.iterdir()} == expected_files
    for name in expected_files:
        assert (first_dir / name).read_bytes() == (second_dir / name).read_bytes()
    with (first_dir / "prediction_summary.csv").open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 20
    report = json.loads((first_dir / "analysis_report.json").read_text())
    assert report["runtime_defaults_modified"] is False
    assert report["deployment_threshold_selected"] is False
