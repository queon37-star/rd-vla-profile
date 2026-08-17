from scripts.coda_anchor_feasibility.analyze_deployment_matched_shadow_first_hit import (
    replay_predictions,
)


THRESHOLD = 0.1


def _row(terminal, score, exact_mse):
    exact_safe = exact_mse < 0.001
    return {
        "anchor_iteration": terminal - 1,
        "terminal_iteration": terminal,
        "gate_score": score,
        "gate_threshold": THRESHOLD,
        "predicted_trigger": score <= THRESHOLD,
        "exact_adjacent_action_mse": exact_mse,
        "recurrence_mse_threshold": 0.001,
        "exact_safe": exact_safe,
        "false_safe": score <= THRESHOLD and not exact_safe,
        "residual": exact_mse - score,
    }


def _prediction(prediction_index, transitions):
    identity = {
        "trajectory_id": "trajectory-1",
        "task_id": 0,
        "episode_id": 2,
        "initial_state_id": 3,
        "paired_trial_id": 4,
        "episode_seed": 5,
        "action_prediction_index": prediction_index,
        "environment_timestep": 10 + prediction_index,
    }
    return {
        "prediction_id": f"prediction-{prediction_index}",
        "task_name": "synthetic task",
        "identity": identity,
        "transitions": transitions,
    }


def test_replay_sorts_transitions_and_stops_at_first_threshold_hit():
    prediction = _prediction(
        0,
        [
            _row(7, 0.01, 0.002),
            _row(5, 0.20, 0.003),
            _row(6, 0.05, 0.0005),
        ],
    )

    replay, events = replay_predictions(
        [prediction], threshold=THRESHOLD, min_terminal_iter=5
    )

    assert replay["global"]["first_hit_activations"] == 1
    assert replay["global"]["first_hit_safe_activations"] == 1
    assert replay["global"]["first_hit_false_safe_activations"] == 0
    assert replay["global"]["nominal_coda_calls_saved"] == 1
    assert replay["global"]["first_hit_terminal_iteration_distribution"] == {"6": 1}
    assert len(events) == 1
    event = events[0]
    assert event["terminal_iteration"] == 6
    assert event["gate_score"] == 0.05
    assert event["previous_eligible_terminal_iteration"] == 5
    assert event["previous_eligible_gate_score"] == 0.20
    assert event["next_eligible_terminal_iteration"] == 7
    assert event["next_eligible_gate_score"] == 0.01


def test_replay_counts_above_threshold_and_ineligible_predictions_as_no_trigger():
    predictions = [
        _prediction(0, [_row(6, 0.20, 0.0005), _row(5, 0.30, 0.002)]),
        _prediction(1, []),
    ]

    replay, events = replay_predictions(
        predictions, threshold=THRESHOLD, min_terminal_iter=5
    )

    assert events == []
    assert replay["global"] == {
        "trajectory_count": 1,
        "total_predictions": 2,
        "predictions_eligible_for_gate_evaluation": 1,
        "ineligible_predictions": 1,
        "first_hit_activations": 0,
        "first_hit_safe_activations": 0,
        "first_hit_false_safe_activations": 0,
        "eligible_no_trigger_predictions": 1,
        "no_trigger_predictions": 2,
        "activation_rate": 0.0,
        "activation_rate_among_eligible_predictions": 0.0,
        "false_safe_rate_among_first_hit_activations": None,
        "nominal_coda_calls_saved": 0,
        "false_safe_trajectory_count": 0,
        "first_hit_terminal_iteration_distribution": {},
    }
