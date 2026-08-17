from scripts.coda_anchor_feasibility.analyze_action_delta_deferred_backfill import (
    replay_prediction,
)


def prediction(rows, *, baseline_k, adaptive_stop=True):
    return {
        "task_id": 0,
        "prediction_id": "prediction-0",
        "trajectory_id": "trajectory-0",
        "baseline_k": baseline_k,
        "baseline_coda_calls": baseline_k,
        "baseline_stop_reason": "kl_divergence" if adaptive_stop else "max_iter",
        "baseline_adaptive_stop": adaptive_stop,
        "rows": rows,
    }


def row(terminal, score, mse):
    return {
        "terminal_iteration": terminal,
        "gate_score": score,
        "exact_adjacent_action_mse": mse,
        "exact_safe": mse < 0.001,
    }


def test_single_high_region_followed_by_low_is_fully_backfilled():
    replay = replay_prediction(
        prediction(
            [row(5, 0.002, 0.004), row(6, 0.0008, 0.0005)],
            baseline_k=6,
        )
    )

    assert replay["baseline_k_agrees"] is True
    assert replay["policy_k"] == 6
    assert replay["baseline_coda_calls"] == 6
    assert replay["actual_coda_calls"] == 6
    assert replay["deferred_coda_calls"] == 1
    assert replay["deferred_calls_later_backfilled"] == 1
    assert replay["truly_eliminated_coda_calls"] == 0
    assert replay["runs"][0]["eliminated_coda_calls"] == 0


def test_prediction_without_eligible_shadow_rows_is_unchanged():
    replay = replay_prediction(prediction([], baseline_k=6))

    assert replay["baseline_k_agrees"] is True
    assert replay["actual_coda_calls"] == 6
    assert replay["scorer_calls"] == 0
    assert replay["truly_eliminated_coda_calls"] == 0


def test_long_high_region_backfills_only_last_adjacent_action():
    replay = replay_prediction(
        prediction(
            [
                row(5, 0.003, 0.006),
                row(6, 0.0025, 0.004),
                row(7, 0.002, 0.002),
                row(8, 0.0008, 0.0005),
            ],
            baseline_k=8,
        )
    )

    assert replay["baseline_k_agrees"] is True
    assert replay["scorer_calls"] == 4
    assert replay["deferred_coda_calls"] == 3
    assert replay["deferred_calls_later_backfilled"] == 1
    assert replay["truly_eliminated_coda_calls"] == 2
    assert replay["actual_coda_calls"] == 6
    assert replay["runs"][0]["length"] == 3


def test_terminal_high_region_has_explicit_return_action_accounting_variants():
    value = prediction(
        [row(5, 0.003, 0.006), row(6, 0.002, 0.004)],
        baseline_k=6,
        adaptive_stop=False,
    )

    stopping_only = replay_prediction(value)
    exact_terminal = replay_prediction(value, require_exact_terminal_output=True)

    assert stopping_only["baseline_k_agrees"] is True
    assert stopping_only["actual_coda_calls"] == 4
    assert stopping_only["truly_eliminated_coda_calls"] == 2
    assert exact_terminal["actual_coda_calls"] == 5
    assert exact_terminal["deferred_calls_later_backfilled"] == 1
    assert exact_terminal["truly_eliminated_coda_calls"] == 1


def test_high_score_exact_safe_row_invalidates_assumption_and_k_agreement():
    replay = replay_prediction(
        prediction([row(5, 0.002, 0.0005)], baseline_k=5)
    )

    assert replay["high_score_exact_safe_assumption_violations"] == 1
    assert replay["baseline_k_agrees"] is False
    assert replay["policy_k"] is None
