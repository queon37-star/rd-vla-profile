import inspect

import numpy as np
import pytest
import torch

from scripts.coda_anchor_feasibility.explore_false_safe_signals import (
    EVALUATION_TARGET_NAMES,
    RUNTIME_FEATURE_NAMES,
    TRAIN_CAL_TASKS,
    build_evaluation_targets,
    build_runtime_features,
    cutoff_accept_mask,
    evaluate_inner_oof_feature_safeguard,
    fit_feature_cutoff,
    offline_anchor_k_to_runtime_terminal_iteration,
    select_analysis_indices,
    sequential_first_hit_replay,
)


@pytest.mark.parametrize(
    ("offline_anchor_k", "runtime_terminal_iteration"),
    [(1, 2), (3, 4), (4, 5), (5, 6)],
)
def test_offline_anchor_k_maps_to_next_runtime_terminal_iteration(
    offline_anchor_k,
    runtime_terminal_iteration,
):
    assert (
        offline_anchor_k_to_runtime_terminal_iteration(offline_anchor_k)
        == runtime_terminal_iteration
    )


def test_offline_anchor_k_mapping_rejects_ambiguous_or_invalid_values():
    with pytest.raises(TypeError, match="integer"):
        offline_anchor_k_to_runtime_terminal_iteration(True)
    with pytest.raises(ValueError, match=">= 1"):
        offline_anchor_k_to_runtime_terminal_iteration(0)


def _causal_features():
    delta_states = torch.stack(
        [
            torch.full((5, 2), value, dtype=torch.bfloat16)
            for value in (1.0, 4.0, 3.0, 8.0)
        ]
    )
    predicted_delta = torch.stack(
        [
            torch.full((5, 2), value, dtype=torch.float32)
            for value in (0.1, 0.4, 0.2, 0.8)
        ]
    )
    features = build_runtime_features(
        delta_states,
        predicted_delta,
        trajectory_ids=np.array([10, 20, 10, 20]),
        ks=np.array([1, 1, 2, 3]),
        gate_threshold=1.0,
        x_mean=torch.zeros(2),
        x_std=torch.ones(2),
        prefix_steps=5,
        batch_size=2,
    )
    return features


def test_causal_history_uses_only_same_trajectory_immediate_previous_row():
    features = _causal_features()

    assert np.isnan(features["previous_predicted_score"][0])
    assert np.isnan(features["previous_predicted_score"][1])
    assert features["previous_predicted_score"][2] == pytest.approx(0.01)
    assert features["score_ratio_current_to_previous"][2] == pytest.approx(4.0)
    assert features["score_difference_current_minus_previous"][2] == pytest.approx(0.03)
    assert features["relative_score_drop"][2] == pytest.approx(-3.0)
    assert features["previous_latent_delta_rms"][2] == pytest.approx(1.0)
    assert features[
        "latent_delta_rms_ratio_current_to_previous"
    ][2] == pytest.approx(3.0)
    assert features["latent_delta_cosine_current_previous"][2] == pytest.approx(1.0)
    assert features["latent_delta_second_difference_rms"][2] == pytest.approx(2.0)
    # Trajectory 20 has k=1 followed by k=3.  No k=2 history is invented.
    assert np.isnan(features["previous_predicted_score"][3])
    assert np.isnan(features["latent_delta_second_difference_rms"][3])


def test_runtime_features_cannot_receive_or_expose_exact_action_targets():
    parameters = set(inspect.signature(build_runtime_features).parameters)
    assert "exact_mse" not in parameters
    assert "exact_safe" not in parameters
    assert set(RUNTIME_FEATURE_NAMES).isdisjoint(EVALUATION_TARGET_NAMES)

    features = _causal_features()
    targets = build_evaluation_targets(
        np.array([0.0005, 0.002, 0.001, 0.0]),
        features["predicted_action_delta_mse"],
        threshold=1.0,
    )
    assert set(features) == set(RUNTIME_FEATURE_NAMES)
    assert set(targets) == set(EVALUATION_TARGET_NAMES)


def test_sequential_replay_scores_vetoed_hits_and_stops_at_first_accepted_hit():
    scores = np.array([0.01, 0.04, 0.0, 0.10, 0.01])
    exact_safe = np.array([False, True, True, True, True])
    trajectory_ids = np.array([10, 10, 10, 20, 20])
    terminal_iterations = np.array([5, 6, 7, 5, 6])
    veto_accept = np.array([False, True, True, True, True])

    replay, activated = sequential_first_hit_replay(
        np.arange(5),
        scores,
        exact_safe,
        trajectory_ids,
        terminal_iterations,
        gate_threshold=0.05,
        veto_accept_mask=veto_accept,
    )

    np.testing.assert_array_equal(activated, np.array([1, 4]))
    assert replay["trajectory_count"] == 2
    assert replay["score_call_count"] == 4
    assert replay["accepted_triggers"] == 2
    assert replay["exact_safe_accepted"] == 2
    assert replay["false_safe_accepted"] == 0
    assert replay["no_skip_count"] == 0
    assert replay["sequential_first_hit_terminal_distribution"] == {"6": 2}


def test_task_selection_keeps_task4_forensic_and_excludes_task5():
    task_ids = np.arange(10)
    selected = select_analysis_indices(task_ids)
    selected_tasks = set(task_ids[selected].tolist())

    assert selected_tasks == set(TRAIN_CAL_TASKS) | {4}
    assert 5 not in selected_tasks


def test_cutoff_uses_only_declared_calibration_tasks():
    values = np.array([1.0, 2.0, 10.0, 20.0, 1000.0, -1000.0])
    scores = np.zeros(6)
    exact_safe = np.array([True, False, True, False, True, False])
    terminal = np.full(6, 5)
    task_ids = np.array([0, 0, 1, 1, 2, 2])

    cutoff = fit_feature_cutoff(
        values,
        scores,
        exact_safe,
        terminal,
        task_ids,
        feature_name="synthetic",
        calibration_tasks=(0, 1),
        safe_retention=0.5,
        gate_threshold=1.0,
    )
    changed_evaluation_values = values.copy()
    changed_evaluation_values[task_ids == 2] *= 1_000_000
    repeated = fit_feature_cutoff(
        changed_evaluation_values,
        scores,
        exact_safe,
        terminal,
        task_ids,
        feature_name="synthetic",
        calibration_tasks=(0, 1),
        safe_retention=0.5,
        gate_threshold=1.0,
    )

    assert cutoff["calibration_tasks"] == [0, 1]
    assert repeated["cutoff"] == cutoff["cutoff"]
    assert repeated["risk_direction"] == cutoff["risk_direction"]
    with pytest.raises(ValueError, match="development tasks"):
        fit_feature_cutoff(
            values,
            scores,
            exact_safe,
            terminal,
            task_ids,
            feature_name="synthetic",
            calibration_tasks=(0, 4),
            safe_retention=0.5,
            gate_threshold=1.0,
        )


def test_inner_oof_cutoffs_exclude_each_evaluation_task():
    task_ids = np.repeat(np.array(TRAIN_CAL_TASKS), 2)
    trajectory_ids = np.arange(len(task_ids))
    features = {
        "synthetic": np.tile(np.array([0.0, 1.0]), len(TRAIN_CAL_TASKS)),
        "predicted_action_delta_mse": np.zeros(len(task_ids)),
        "terminal_iteration": np.full(len(task_ids), 5.0),
    }
    targets = {
        "exact_safe": np.tile(np.array([True, False]), len(TRAIN_CAL_TASKS)),
    }

    result = evaluate_inner_oof_feature_safeguard(
        "synthetic",
        0.9,
        features,
        targets,
        task_ids,
        trajectory_ids,
    )

    assert set(result["inner_task_cutoffs"]) == {
        str(task) for task in TRAIN_CAL_TASKS
    }
    for evaluation_task, cutoff in result["inner_task_cutoffs"].items():
        assert int(evaluation_task) not in cutoff["calibration_tasks"]
        assert set(cutoff["calibration_tasks"]) == set(TRAIN_CAL_TASKS) - {
            int(evaluation_task)
        }


def test_missing_history_is_explicit_and_cutoff_policy_accepts_it():
    features = _causal_features()
    cutoff = {
        "risk_direction": "high_risk",
        "cutoff": 0.02,
        "missing_policy": "accept",
    }
    accepted = cutoff_accept_mask(features["previous_predicted_score"], cutoff)

    assert accepted[0]
    assert accepted[1]
    assert accepted[2]
    assert accepted[3]
