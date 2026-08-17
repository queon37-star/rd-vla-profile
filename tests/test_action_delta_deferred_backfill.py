import pytest

from scripts.coda_anchor_feasibility.analyze_action_delta_deferred_backfill import (
    DeferredBackfillAnalysisError,
    replay_prediction,
    slice_prediction,
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
        "source_min_terminal_iteration": (
            int(rows[0]["terminal_iteration"]) if rows else 2
        ),
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


def test_terminal_high_region_executes_exact_terminal_action():
    value = prediction(
        [row(5, 0.003, 0.006), row(6, 0.002, 0.004)],
        baseline_k=6,
        adaptive_stop=False,
    )

    replay = replay_prediction(value)

    assert replay["baseline_k_agrees"] is True
    assert replay["actual_coda_calls"] == 5
    assert replay["backfilled_coda_calls"] == 0
    assert replay["terminal_exact_fallback_coda_calls"] == 1
    assert replay["truly_eliminated_coda_calls"] == 1

    with pytest.raises(DeferredBackfillAnalysisError, match="unresolved"):
        replay_prediction(value, require_exact_terminal_output=False)


def test_high_score_exact_safe_row_invalidates_assumption_and_k_agreement():
    replay = replay_prediction(
        prediction([row(5, 0.002, 0.0005)], baseline_k=5)
    )

    assert replay["high_score_exact_safe_assumption_violations"] == 1
    assert replay["violation_rows"] == [
        {
            "terminal_iteration": 5,
            "gate_score": 0.002,
            "exact_adjacent_action_mse": 0.0005,
        }
    ]
    assert replay["baseline_k_agrees"] is False
    assert replay["policy_k"] is None


def test_replay_minimum_slices_earlier_terminal_rows():
    source = prediction(
        [
            row(2, 0.0004, 0.004),
            row(3, 0.0004, 0.004),
            row(4, 0.0004, 0.004),
            row(5, 0.0004, 0.0005),
        ],
        baseline_k=5,
    )

    assert [item["terminal_iteration"] for item in slice_prediction(source, 3)["rows"]] == [3, 4, 5]
    assert [item["terminal_iteration"] for item in slice_prediction(source, 4)["rows"]] == [4, 5]


def test_min_five_slice_reproduces_equivalent_min_five_source_policy():
    rows = [
        row(2, 0.002, 0.004),
        row(3, 0.002, 0.004),
        row(4, 0.002, 0.004),
        row(5, 0.002, 0.004),
        row(6, 0.0005, 0.0005),
    ]
    full_source = prediction(rows, baseline_k=6)
    old_source = prediction(rows[3:], baseline_k=6)
    old_source["source_min_terminal_iteration"] = 5

    from_full = replay_prediction(full_source, min_terminal_iter=5)
    from_old = replay_prediction(old_source, min_terminal_iter=5)

    for name in (
        "scorer_calls",
        "deferred_coda_calls",
        "backfilled_coda_calls",
        "truly_eliminated_coda_calls",
        "baseline_k_agrees",
        "policy_k",
    ):
        assert from_full[name] == from_old[name]


@pytest.mark.parametrize(("length", "expected_eliminated"), [(1, 0), (2, 1), (3, 2)])
def test_length_l_high_run_followed_by_low_eliminates_l_minus_one(
    length, expected_eliminated
):
    rows = [row(2 + offset, 0.002, 0.004) for offset in range(length)]
    rows.append(row(2 + length, 0.0005, 0.0005))

    replay = replay_prediction(
        prediction(rows, baseline_k=2 + length)
    )

    assert replay["truly_eliminated_coda_calls"] == expected_eliminated
    assert replay["runs"][0]["length"] == length
