from __future__ import annotations

import math

from scripts.preconvergence_trigger_lib import (
    project_latency,
    replay_confirm_next,
    select_training_threshold,
    trigger_category,
)
from test_preconvergence_dataset import make_sequence


def scores(max_iter: int, positive_k: int | None) -> dict[int, float]:
    return {
        k: (0.9 if positive_k is not None and k == positive_k else 0.1)
        for k in range(3, max_iter + 1)
    }


def test_confirm_next_ideal_trigger_state_machine() -> None:
    sequence = make_sequence(0, k_action=5)
    replay = replay_confirm_next(sequence, scores(sequence.max_iter, 4), 0.5)
    assert replay.trigger_category == "ideal"
    assert replay.trigger_offset == 0
    assert replay.coda_iterations == (1, 2, 4, 5)
    assert replay.terminal_k == 5
    assert replay.delta_k == 0


def test_early_ideal_late_and_missed_categories() -> None:
    assert trigger_category(3, 5) == ("early", -1)
    assert trigger_category(4, 5) == ("ideal", 0)
    assert trigger_category(6, 5) == ("late", 2)
    assert trigger_category(None, 5) == ("missed", None)


def test_non_monotonic_action_labels_are_replayed_after_late_trigger() -> None:
    sequence = make_sequence(0, k_action=5)
    mse = list(sequence.action_mse)
    mse[6] = 0.01
    mse[7] = 0.02
    mse[8] = 0.0004
    sequence = sequence.__class__(
        identity=sequence.identity,
        actual_origin=sequence.actual_origin,
        states=sequence.states,
        actions=sequence.actions,
        action_mse=tuple(mse),
        action_mse_phase=sequence.action_mse_phase,
        baseline_k=sequence.baseline_k,
        max_iter=sequence.max_iter,
    )
    replay = replay_confirm_next(sequence, scores(sequence.max_iter, 6), 0.5)
    assert replay.coda_iterations == (1, 2, 6, 7, 8)
    assert replay.terminal_k == 8
    assert replay.delta_k == 3


def test_coda_call_accounting_for_missed_trigger() -> None:
    sequence = make_sequence(0, k_action=5)
    replay = replay_confirm_next(sequence, scores(sequence.max_iter, None), 0.5)
    assert replay.trigger_category == "missed"
    assert replay.coda_iterations == (1, 2, sequence.max_iter)
    assert replay.coda_call_count == 3
    assert replay.baseline_coda_call_count == 5
    assert replay.saved_coda_calls == 2
    assert replay.gate_evaluation_count == sequence.max_iter - 2


def test_projected_latency_includes_all_cost_terms() -> None:
    sequence = make_sequence(0, k_action=5)
    replay = replay_confirm_next(sequence, scores(sequence.max_iter, 4), 0.5)
    projection = project_latency(
        [replay],
        coda_latency_ms=2.0,
        recurrent_iteration_latency_ms=3.0,
        gate_latency_ms=0.25,
    )
    assert projection["gross_coda_saving_ms"] == 2.0
    assert projection["additional_recurrent_cost_ms"] == 0.0
    assert projection["gate_overhead_ms"] == 0.5
    assert projection["projected_net_saving_ms"] == 1.5


def test_training_threshold_selection_is_tied_and_deterministic() -> None:
    left = make_sequence(0, k_action=5)
    right = make_sequence(1, k_action=5)
    tied_scores = {k: (0.8 if k == 4 else 0.2) for k in range(3, 9)}
    first = select_training_threshold([(left, tied_scores), (right, tied_scores)])
    second = select_training_threshold([(left, tied_scores), (right, tied_scores)])
    assert first == second
    assert first["selection_status"] == "no_late_or_missed_feasible"
    assert math.isfinite(first["selected_threshold"])
