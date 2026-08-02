import copy
import json

import pytest

from prismatic.models.latent_dynamics import HISTORY_DEPENDENT_FIELDS
from scripts.adaptive_coda_gate_oof import (
    LEAKAGE_EXCLUSIONS,
    METRIC_FIELDS,
    MODEL_FEATURES,
    activation_target,
    exact_threshold_sweep,
    exact_threshold_sweep_bruteforce,
    evaluate_nested_oof,
    fit_gate_model,
    fit_nested_outer_candidate,
    full_scoring_rows,
    inner_leave_one_task_out_splits,
    primary_training_rows,
    qualify_and_select_threshold,
    recorded_action_mse_by_iteration,
    replay_scored_predictions,
    replay_trigger,
    split_predictions_by_outer_fold,
    write_evaluation_outputs,
)
from scripts.analyze_latent_dynamics_features import canonical_json
from scripts.coda_activation_oof import fit_fixed_activation_threshold


ALL_MODEL_FEATURES = sorted({feature for features in MODEL_FEATURES.values() for feature in features})


def _prediction(
    task_id,
    prediction_id,
    *,
    baseline_k=6,
    true_iterations=None,
    feature_offset=0.0,
):
    target = activation_target(baseline_k)
    true_iterations = {baseline_k} if true_iterations is None else set(true_iterations)
    transitions = []
    for iteration in range(2, 33):
        features = {}
        for feature_index, feature in enumerate(ALL_MODEL_FEATURES):
            if feature == "iteration_index":
                value = float(iteration)
            elif feature == "raw_mse":
                value = 1.0 / (iteration + task_id + 1) + feature_offset
            else:
                value = (
                    0.01 * (feature_index + 1)
                    + 0.02 * iteration
                    + 0.001 * task_id
                    + feature_offset
                )
            if iteration == 2 and feature in HISTORY_DEPENDENT_FIELDS:
                value = None
            features[feature] = value
        action_mse = 0.0005 if iteration in true_iterations else 0.01
        transitions.append(
            {
                "task_id": str(task_id),
                "episode_id": 0,
                "prediction_id": prediction_id,
                "actual_origin": "ACTUAL_WARM",
                "baseline_k": baseline_k,
                "activation_target": target,
                "iteration_index": iteration,
                "k": iteration,
                "relative_iteration": iteration - target,
                "activation_due": iteration >= target,
                "primary_window": iteration <= baseline_k,
                "difficulty": "easy" if baseline_k <= 4 else "medium" if baseline_k <= 7 else "hard",
                "features": features,
                "raw_mse": features["raw_mse"],
                "action_mse": action_mse,
                "label": iteration in true_iterations,
            }
        )
    return {
        "key": (str(task_id), 0, prediction_id),
        "task_id": str(task_id),
        "episode_id": 0,
        "prediction_id": prediction_id,
        "actual_origin": "ACTUAL_WARM",
        "baseline_k": baseline_k,
        "activation_target": target,
        "difficulty": "easy" if baseline_k <= 4 else "medium" if baseline_k <= 7 else "hard",
        "max_iter": 32,
        "transitions": transitions,
    }


def _predictions():
    return [
        _prediction(
            task_id,
            task_id,
            baseline_k=4 + task_id % 5,
            true_iterations={4 + task_id % 5, min(32, 6 + task_id % 5)},
        )
        for task_id in range(10)
    ]


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


def _scored(predictions, score_by_k):
    return [
        {
            "prediction": prediction,
            "scores_by_k": {
                iteration: float(score_by_k(prediction, iteration))
                for iteration in range(2, 32)
            },
        }
        for prediction in predictions
    ]


def test_activation_target_and_model_fitting_window_are_exact():
    assert activation_target(2) == 2
    assert activation_target(3) == 2
    assert activation_target(8) == 7
    predictions = _predictions()
    rows = primary_training_rows(predictions)
    assert all(row["k"] <= row["baseline_k"] for row in rows)
    assert len(rows) == sum(prediction["baseline_k"] - 1 for prediction in predictions)


def test_full_scoring_includes_post_terminal_but_training_never_does():
    predictions = _predictions()
    training = primary_training_rows(predictions)
    scoring = full_scoring_rows(predictions)
    assert all(row["k"] <= row["baseline_k"] for row in training)
    assert any(row["k"] > row["baseline_k"] for row in scoring)
    assert {row["k"] for row in scoring} == set(range(2, 32))
    fitted = fit_gate_model(predictions, "combined")
    assert fitted["training_transition_count"] == len(training)
    assert fitted["maximum_training_iteration_by_prediction"] == "baseline_k"


def test_outer_and_inner_task_splits_are_disjoint():
    folds = split_predictions_by_outer_fold(_predictions(), _assignment())
    assert len(folds) == 5
    held_out_tasks = []
    for fold in folds:
        assert set(fold["training_task_ids"]).isdisjoint(fold["held_out_task_ids"])
        held_out_tasks.extend(fold["held_out_task_ids"])
        inner = inner_leave_one_task_out_splits(fold["training_predictions"])
        assert len(inner) == 8
        for split in inner:
            assert split["omitted_task_id"] not in split["training_task_ids"]
            assert {item["task_id"] for item in split["held_out_predictions"]} == {
                split["omitted_task_id"]
            }
    assert sorted(held_out_tasks, key=int) == [str(value) for value in range(10)]


def test_train_only_preprocessing_scaling_and_k2_availability_indicator():
    predictions = _predictions()
    fitted = fit_gate_model(predictions[2:], "update_dynamics")
    preprocessor = fitted["preprocessor"]
    assert set(preprocessor["fit_task_ids"]) == {str(value) for value in range(2, 10)}
    assert "contraction_ratio__available" in preprocessor["expanded_feature_names"]
    assert len(preprocessor["scaling_mean"]) == len(preprocessor["expanded_feature_names"])
    assert len(preprocessor["scaling_scale"]) == len(preprocessor["expanded_feature_names"])


def test_outer_heldout_perturbation_does_not_change_fitted_parameters_or_q():
    original = _predictions()
    changed = copy.deepcopy(original)
    for prediction in changed:
        if prediction["task_id"] in {"0", "9"}:
            for row in prediction["transitions"]:
                row["activation_due"] = not row["activation_due"]
                for feature in row["features"]:
                    if row["features"][feature] is not None:
                        row["features"][feature] *= 10000.0
    original_train = split_predictions_by_outer_fold(original, _assignment())[0][
        "training_predictions"
    ]
    changed_train = split_predictions_by_outer_fold(changed, _assignment())[0][
        "training_predictions"
    ]
    first = fit_nested_outer_candidate(original_train, "iteration_raw_mse")
    second = fit_nested_outer_candidate(changed_train, "iteration_raw_mse")
    assert canonical_json(first) == canonical_json(second)


def test_t2_first_check_and_exact_coda_calls():
    prediction = _prediction(0, 0, baseline_k=4, true_iterations={4})
    replay = replay_trigger(prediction, 2, forced_trigger=False)
    assert replay["first_action_mse_check_k"] == 2
    assert replay["terminal_k"] == 4
    assert replay["executed_coda_iterations"] == [1, 2, 3, 4]
    assert replay["scheduled_coda_calls"] == 4
    assert replay["scheduled_coda_calls"] == 1 + replay["terminal_k"] - 2 + 1


def test_recorded_production_action_metric_is_authoritative_before_shadow_tail():
    record = {
        "K_t": 4,
        "iteration_mse": [0.1, 0.01, 0.0009],
        "latent_metric_trace": [
            {
                "iteration_index": iteration,
                "adjacent_action_mse": 0.0011 if iteration == 4 else 0.02,
            }
            for iteration in range(2, 33)
        ],
    }
    values = recorded_action_mse_by_iteration(record)
    assert values[4] == {
        "action_mse": 0.0009,
        "label": True,
        "source": "production_iteration_mse",
    }
    assert values[5] == {
        "action_mse": 0.02,
        "label": False,
        "source": "shadow_fp32_adjacent_action_mse",
    }


def test_trigger_after_k2_checks_at_t_plus_one_and_replays_nonmonotonic_labels():
    prediction = _prediction(0, 0, baseline_k=4, true_iterations={4, 6})
    replay = replay_trigger(prediction, 4, forced_trigger=False)
    assert replay["first_action_mse_check_k"] == 5
    assert replay["executed_action_mse_checks"] == [5, 6]
    assert replay["terminal_k"] == 6
    assert replay["terminal_k"] != max(prediction["baseline_k"], 5)


def test_forced_trigger_at_k31_and_call_count():
    prediction = _prediction(0, 0, baseline_k=5, true_iterations={5, 32})
    replay = replay_scored_predictions(
        _scored([prediction], lambda _prediction, _iteration: 0.0), 1.0
    )[0]
    assert replay["forced_trigger"] is True
    assert replay["trigger_k"] == 31
    assert replay["first_action_mse_check_k"] == 32
    assert replay["terminal_k"] == 32
    assert replay["executed_coda_iterations"] == [1, 31, 32]
    assert replay["scheduled_coda_calls"] == 3


def test_safe_all_trigger_candidate_preserves_k_and_calls():
    predictions = _predictions()
    scored = _scored(predictions, lambda prediction, iteration: 0.1 * iteration + int(prediction["task_id"]))
    sweep = exact_threshold_sweep(scored)
    safe = sweep[-1]
    assert safe["candidate_type"] == "all_trigger_at_k2_boundary"
    assert safe["metrics"]["mean_delta_K"] == 0.0
    assert safe["metrics"]["p95_delta_K"] == 0.0
    assert safe["metrics"]["exact_K_preservation_rate"] == 1.0
    assert safe["metrics"]["coda_call_reduction"] == 0.0
    assert safe["metrics"]["forced_trigger_rate"] == 0.0


def test_optimized_threshold_sweep_matches_bruteforce_exactly():
    predictions = _predictions()[:4]
    scored = _scored(
        predictions,
        lambda prediction, iteration: (
            ((iteration * 7 + int(prediction["task_id"]) * 3) % 11) / 10.0
        ),
    )
    optimized = exact_threshold_sweep(scored)
    brute = exact_threshold_sweep_bruteforce(scored)
    optimized_by_signature = {item["signature"]: item for item in optimized}
    brute_by_signature = {item["signature"]: item for item in brute}
    assert set(optimized_by_signature) == set(brute_by_signature)
    for signature, item in optimized_by_signature.items():
        reference = brute_by_signature[signature]
        assert item["threshold"] == reference["threshold"]
        assert item["threshold_hex"] == reference["threshold_hex"]
        assert item["metrics"] == reference["metrics"]


def _qualifying_metrics(reduction=0.1, mean=0.0, p95=0.0, exact=1.0):
    metrics = {
        "prediction_count": 2,
        "baseline_total_coda_calls": 10,
        "scheduled_total_coda_calls": int(10 * (1.0 - reduction)),
        "coda_call_reduction": reduction,
        "mean_delta_K": mean,
        "median_delta_K": mean,
        "p95_delta_K": p95,
        "max_delta_K": int(p95),
        "exact_K_preservation_rate": exact,
        "delta_K_gt_0_rate": 1.0 - exact,
        "mean_trigger_delay": 0.0,
        "median_trigger_delay": 0.0,
        "p95_trigger_delay": 0.0,
        "mean_early_trigger_distance": 0.0,
        "forced_trigger_rate": 0.0,
        "max_iteration_rate": 0.0,
        "mean_coda_calls_per_prediction": 4.5,
    }
    assert set(METRIC_FIELDS) <= set(metrics)
    return metrics


def test_threshold_tie_breaking_prefers_lower_q_deterministically():
    metrics = _qualifying_metrics()
    sweep = [
        {
            "threshold": threshold,
            "threshold_hex": threshold.hex(),
            "candidate_type": "score_event",
            "metrics": dict(metrics),
            "signature": ((2, False), (3, False)),
        }
        for threshold in (0.6, 0.4)
    ]
    baseline = _qualifying_metrics(reduction=0.0)
    first = qualify_and_select_threshold(copy.deepcopy(sweep), baseline)
    second = qualify_and_select_threshold(copy.deepcopy(sweep), baseline)
    assert first == second
    assert first["selected_threshold"] == 0.4
    assert first["selected_threshold_hex"] == (0.4).hex()


def test_fixed_raw_mse_reference_is_fit_from_outer_training_only():
    predictions = _predictions()
    fold = split_predictions_by_outer_fold(predictions, _assignment())[0]
    first = fit_fixed_activation_threshold(fold["training_predictions"], 0.05)
    changed = copy.deepcopy(fold["held_out_predictions"])
    for prediction in changed:
        for transition in prediction["transitions"]:
            transition["raw_mse"] *= 100000.0
    second = fit_fixed_activation_threshold(fold["training_predictions"], 0.05)
    assert first == second
    assert first["beta"] == 0.05


def test_no_leakage_fields_are_model_inputs():
    for features in MODEL_FEATURES.values():
        assert set(features).isdisjoint(LEAKAGE_EXCLUSIONS)
        assert "adjacent_action_mse" not in features
        assert "action_mse_below_0_001" not in features
        assert "baseline_k" not in features


def test_nested_evaluation_and_outputs_are_byte_deterministic(tmp_path):
    predictions = _predictions()
    first = evaluate_nested_oof(
        predictions,
        _assignment(),
        prior_classification_context={"scope": "fixture classification only"},
    )
    second = evaluate_nested_oof(
        copy.deepcopy(predictions),
        _assignment(),
        prior_classification_context={"scope": "fixture classification only"},
    )
    assert canonical_json(first["metric_report"]) == canonical_json(second["metric_report"])
    assert canonical_json(first["model_summary"]) == canonical_json(second["model_summary"])
    assert first["threshold_sweeps"] == second["threshold_sweeps"]
    assert first["all_replays"] == second["all_replays"]

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    inputs = {"fixture": True}
    write_evaluation_outputs(first_dir, first, inputs=inputs)
    write_evaluation_outputs(second_dir, second, inputs=inputs)
    expected = {
        "metric_report.json",
        "model_summary.json",
        "threshold_sweeps.csv",
        "oof_prediction_replays.csv",
        "task_metrics.csv",
        "difficulty_metrics.csv",
        "output_hashes.json",
    }
    assert {path.name for path in first_dir.iterdir()} == expected
    for name in expected:
        assert (first_dir / name).read_bytes() == (second_dir / name).read_bytes()
    hashes = json.loads((first_dir / "output_hashes.json").read_text())
    assert set(hashes["files"]) == expected - {"output_hashes.json"}
    report = json.loads((first_dir / "metric_report.json").read_text())
    assert report["runtime_inference_modified"] is False
    assert report["global_or_deployment_threshold_selected"] is False
    assert report["classification_scheduler_distinction"]
