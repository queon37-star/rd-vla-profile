from scripts.coda_anchor_feasibility.analyze_action_delta_nonconvergence_filter import (
    causal_replay,
    row_metrics,
)


def _row(terminal, score, exact_mse, two_step=None):
    return {
        "task_id": 0,
        "prediction_id": "prediction",
        "trajectory_id": "trajectory",
        "terminal_iteration": terminal,
        "gate_score": score,
        "exact_adjacent_action_mse": exact_mse,
        "exact_safe": exact_mse < 0.001,
        "two_step_exact_mse_after_previous_skip": two_step,
    }


def _prediction(rows):
    return {
        "task_id": 0,
        "prediction_id": "prediction",
        "trajectory_id": "trajectory",
        "baseline_coda_calls": 7,
        "baseline_stop_reason": "adjacent_action_mse",
        "rows": rows,
    }


def test_high_side_row_metrics_count_false_nonconvergence_explicitly():
    rows = [
        _row(5, 0.8, 0.01),
        _row(6, 0.7, 0.0005, 0.02),
        _row(7, 0.2, 0.02, 0.03),
    ]

    observed = row_metrics(rows, threshold=0.5)

    assert observed["proposed_skips"] == 2
    assert observed["true_nonconvergence_skips"] == 1
    assert observed["exact_safe_rows_incorrectly_skipped"] == 1
    assert observed["precision"] == 0.5
    assert observed["recall_exact_nonconverged"] == 0.5


def test_causal_replay_forces_exact_after_skip_and_recomputes_history():
    rows = [
        _row(5, 0.8, 0.01),
        # Native adjacent MSE is safe, but a6 vs last-executed a4 is unsafe.
        _row(6, 0.9, 0.0005, 0.02),
        _row(7, 0.7, 0.0005, 0.03),
    ]

    observed = causal_replay([_prediction(rows)], threshold=0.5)["global"]

    assert observed["scorer_call_count"] == 2
    assert observed["nominal_skipped_coda_calls"] == 2
    assert observed["true_nonconvergence_skips"] == 1
    assert observed["exact_safe_coda_calls_incorrectly_skipped"] == 1
    assert observed["forced_exact_coda_calls"] == 1
    assert observed["adjacent_history_difference_events"] == 1
    assert observed["altered_history_stop_decision_changes"] == 1
    assert observed["censored_after_skipped_terminal_exact_safe"] == 1
    assert observed["censored_predictions"] == 1


def test_exact_coda_alone_stops_when_filter_does_not_skip():
    rows = [
        _row(5, 0.8, 0.01),
        _row(6, 0.7, 0.0005, 0.02),
        _row(7, 0.9, 0.02, 0.03),
    ]

    observed = causal_replay([_prediction(rows)], threshold=1.0)["global"]

    assert observed["scorer_call_count"] == 2
    assert observed["nominal_skipped_coda_calls"] == 0
    assert observed["forced_exact_coda_calls"] == 0
    assert observed["censored_predictions"] == 0
