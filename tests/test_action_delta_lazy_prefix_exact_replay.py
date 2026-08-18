import pytest

from scripts.coda_anchor_feasibility.analyze_action_delta_lazy_prefix_exact import (
    LazyPrefixReplayError,
    analyze_records,
    replay_prediction,
)


def record(*, k, scores, exact_trace, applied=True, frozen_coda_calls=None):
    return {
        "task_id": 0,
        "episode_id": 1,
        "action_prediction_index": 2,
        "actual_origin": "ACTUAL_WARM" if applied else "COLD",
        "recurrent_iteration_count": k,
        "action_delta_deferred_backfill_filter_applied": applied,
        "action_delta_deferred_backfill_filter_score_call_count": len(scores),
        "action_delta_deferred_backfill_filter_score_trace": [
            {
                "anchor_terminal_iteration": terminal - 1,
                "terminal_iteration": terminal,
                "score": score,
            }
            for terminal, score in enumerate(scores, start=2)
        ],
        "action_delta_deferred_backfill_filter_exact_stop_mse_trace": exact_trace,
        "action_delta_deferred_backfill_filter_total_exact_coda_call_count": (
            k if frozen_coda_calls is None else frozen_coda_calls
        ),
    }


def exact(terminal, *, stopped):
    return {
        "anchor_terminal_iteration": terminal - 1,
        "terminal_iteration": terminal,
        "exact_adjacent_mse": 0.0005 if stopped else 0.004,
        "stopped": stopped,
    }


def test_first_low_at_t2_starts_exact_only_and_saves_no_coda():
    replay = replay_prediction(
        record(k=5, scores=[0.0005, 0.003, 0.003, 0.003], exact_trace=[exact(2, stopped=False)])
    )

    assert replay["first_nonhigh_terminal_iteration"] == 2
    assert replay["predicted_scorer_calls"] == 1
    assert replay["predicted_exact_coda_calls"] == 5
    assert replay["predicted_eliminated_coda_calls"] == 0
    assert replay["first_coda_avoided"] is False
    assert replay["exact_only_would_begin"] is True


def test_high_prefix_then_stopping_low_avoids_first_and_prefix_coda():
    replay = replay_prediction(
        record(k=4, scores=[0.002, 0.002, 0.0005], exact_trace=[exact(4, stopped=True)])
    )

    assert replay["first_nonhigh_terminal_iteration"] == 4
    assert replay["predicted_scorer_calls"] == 3
    assert replay["predicted_exact_coda_calls"] == 2
    assert replay["predicted_eliminated_coda_calls"] == 2
    assert replay["first_coda_avoided"] is True
    assert replay["exact_only_would_begin"] is False


def test_all_high_through_max_keeps_only_exact_terminal_coda():
    replay = replay_prediction(
        record(k=5, scores=[0.002, 0.002, 0.002, 0.002], exact_trace=[], frozen_coda_calls=3)
    )

    assert replay["first_nonhigh_terminal_iteration"] is None
    assert replay["predicted_scorer_calls"] == 4
    assert replay["predicted_exact_coda_calls"] == 1
    assert replay["predicted_eliminated_coda_calls"] == 4


def test_cold_prediction_remains_exact_and_unscored():
    replay = replay_prediction(record(k=3, scores=[], exact_trace=[], applied=False))

    assert replay["filter_applied"] is False
    assert replay["predicted_scorer_calls"] == 0
    assert replay["predicted_exact_coda_calls"] == 3


def test_analyzer_aggregates_and_asserts_accounting():
    results = analyze_records(
        [
            record(k=4, scores=[0.002, 0.002, 0.0005], exact_trace=[exact(4, stopped=True)]),
            record(k=3, scores=[], exact_trace=[], applied=False),
        ]
    )
    summary = results["aggregate"]
    assert summary["prediction_count"] == 2
    assert summary["lazy_prefix_expected_coda_calls"] == 5
    assert summary["warm_reference_coda_calls"] == 7
    assert summary["expected_coda_reduction_vs_warm"] == 2

    with pytest.raises(LazyPrefixReplayError, match="contiguous"):
        broken = record(k=4, scores=[0.002], exact_trace=[])
        broken["action_delta_deferred_backfill_filter_score_trace"][0][
            "terminal_iteration"
        ] = 3
        replay_prediction(broken)
