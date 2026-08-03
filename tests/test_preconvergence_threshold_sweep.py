from __future__ import annotations

import math
import random
from dataclasses import replace

import numpy as np
import pytest

from scripts.preconvergence_trigger_lib import (
    PreconvergenceValidationError,
    RawPreconvergenceSequence,
    TrainingConfig,
    _threshold_candidates,
    fit_oof_bundle,
    replay_confirm_next,
    select_training_threshold,
)
from test_preconvergence_dataset import make_sequence


def reference_select_training_threshold(scored_sequences):
    """Test-only exhaustive reference preserving the pre-optimization code."""

    candidates = _threshold_candidates(scored_sequences)
    evaluated = []
    for threshold in candidates:
        replays = [
            replay_confirm_next(sequence, scores, threshold)
            for sequence, scores in scored_sequences
        ]
        late_missed = sum(
            replay.trigger_category in {"late", "missed"} for replay in replays
        )
        offsets = [
            replay.trigger_offset
            for replay in replays
            if replay.trigger_offset is not None
        ]
        evaluated.append(
            {
                "threshold": threshold,
                "late_or_missed_count": late_missed,
                "scheduled_coda_calls": sum(r.coda_call_count for r in replays),
                "mean_absolute_trigger_offset": (
                    float(np.mean(np.abs(offsets))) if offsets else math.inf
                ),
                "mean_trigger_lead": (
                    float(np.mean([-offset for offset in offsets]))
                    if offsets
                    else None
                ),
            }
        )
    feasible = [item for item in evaluated if item["late_or_missed_count"] == 0]
    selected = min(
        feasible if feasible else evaluated,
        key=lambda item: (
            item["late_or_missed_count"],
            item["scheduled_coda_calls"],
            item["mean_absolute_trigger_offset"],
            -item["threshold"],
        ),
    )
    return {
        "selected_threshold": selected["threshold"],
        "selected_threshold_hex": float(selected["threshold"]).hex(),
        "selection_status": (
            "no_late_or_missed_feasible"
            if feasible
            else "no_safe_threshold_fail_closed"
        ),
        "candidate_count": len(candidates),
        "selection_order": [
            "require zero late or missed train triggers when feasible",
            "minimize exact CONFIRM_NEXT scheduled Coda calls",
            "minimize mean absolute trigger offset",
            "maximize threshold",
        ],
        "train_metrics": selected,
    }


def assert_selection_equivalent(optimized, reference):
    for field in (
        "selected_threshold",
        "selected_threshold_hex",
        "selection_status",
        "candidate_count",
        "selection_order",
    ):
        assert optimized[field] == reference[field]
    for field in (
        "threshold",
        "late_or_missed_count",
        "scheduled_coda_calls",
    ):
        assert optimized["train_metrics"][field] == reference["train_metrics"][field]
    for field in ("mean_absolute_trigger_offset", "mean_trigger_lead"):
        actual = optimized["train_metrics"][field]
        expected = reference["train_metrics"][field]
        if expected is None or math.isinf(expected):
            assert actual == expected
        else:
            assert actual == pytest.approx(expected, rel=1e-15, abs=1e-15)


def sequence_with_action_mse(
    prediction_id: int,
    *,
    max_iter: int,
    k_action: int,
    tail_values: dict[int, float] | None = None,
):
    sequence = make_sequence(
        prediction_id,
        task_id=prediction_id % 10,
        k_action=k_action,
        max_iter=max_iter,
    )
    mse = list(sequence.action_mse)
    for k, value in (tail_values or {}).items():
        if k > k_action:
            mse[k] = value
    result = replace(sequence, action_mse=tuple(mse))
    result.validate()
    assert result.k_action == k_action
    return result


def test_event_sweep_matches_reference_for_all_predeclared_edge_cases():
    non_monotonic = sequence_with_action_mse(
        0,
        max_iter=8,
        k_action=5,
        tail_values={6: 0.02, 7: 0.03, 8: 0.0004},
    )
    forced_prefix = sequence_with_action_mse(1, max_iter=8, k_action=2)
    max_boundary = sequence_with_action_mse(2, max_iter=8, k_action=8)
    scored = [
        (
            non_monotonic,
            {3: 0.2, 4: 0.5, 5: 0.5, 6: 0.8, 7: 0.8, 8: 0.1},
        ),
        (
            forced_prefix,
            {3: 0.2, 4: 0.2, 5: 0.5, 6: 0.5, 7: 0.8, 8: 0.8},
        ),
        (
            max_boundary,
            {3: 0.2, 4: 0.5, 5: 0.5, 6: 0.8, 7: 0.8, 8: 0.8},
        ),
    ]
    candidates = _threshold_candidates(scored)
    categories = {
        replay_confirm_next(non_monotonic, scored[0][1], threshold).trigger_category
        for threshold in candidates
    }
    assert categories == {"early", "ideal", "late", "missed"}
    assert replay_confirm_next(non_monotonic, scored[0][1], 0.8).terminal_k == 8

    assert_selection_equivalent(
        select_training_threshold(scored),
        reference_select_training_threshold(scored),
    )


@pytest.mark.parametrize("seed", range(40))
def test_event_sweep_matches_reference_on_randomized_duplicate_scores(seed):
    rng = random.Random(seed)
    scored = []
    score_pool = (-0.5, -0.1, -0.1, 0.0, 0.2, 0.2, 0.7, 0.9)
    for prediction_id in range(rng.randint(2, 9)):
        max_iter = rng.randint(3, 11)
        k_action = rng.randint(2, max_iter)
        tail = {
            k: rng.choice((0.02, 0.01, 0.0009, 0.0004))
            for k in range(k_action + 1, max_iter + 1)
        }
        sequence = sequence_with_action_mse(
            prediction_id,
            max_iter=max_iter,
            k_action=k_action,
            tail_values=tail,
        )
        scores = {
            k: rng.choice(score_pool) for k in range(3, max_iter + 1)
        }
        scored.append((sequence, scores))

    assert_selection_equivalent(
        select_training_threshold(scored),
        reference_select_training_threshold(scored),
    )


def test_selector_validates_each_sequence_once_before_candidate_sweep(monkeypatch):
    sequences = [
        sequence_with_action_mse(index, max_iter=8, k_action=4 + index)
        for index in range(3)
    ]
    scored = [
        (sequence, {k: float((k + index) % 3) for k in range(3, 9)})
        for index, sequence in enumerate(sequences)
    ]
    original_validate = RawPreconvergenceSequence.validate
    calls = {id(sequence): 0 for sequence in sequences}

    def counted_validate(self):
        calls[id(self)] += 1
        return original_validate(self)

    monkeypatch.setattr(RawPreconvergenceSequence, "validate", counted_validate)
    select_training_threshold(scored)
    assert calls == {id(sequence): 1 for sequence in sequences}


def test_selector_fail_closed_score_coverage_and_finiteness():
    sequence = sequence_with_action_mse(0, max_iter=6, k_action=5)
    with pytest.raises(PreconvergenceValidationError, match="missing gate score at k=4"):
        select_training_threshold(
            [(sequence, {3: 0.1, 5: 0.2, 6: 0.3})]
        )
    with pytest.raises(PreconvergenceValidationError, match="non-finite gate score"):
        select_training_threshold(
            [(sequence, {3: 0.1, 4: float("nan"), 5: 0.2, 6: 0.3})]
        )


def test_fit_oof_bundle_prints_flushed_stage_progress(capsys):
    sequences = [
        make_sequence(task_id, task_id=task_id, k_action=5 + task_id % 2)
        for task_id in range(4)
    ]
    assignment = {"0": 0, "1": 0, "2": 1, "3": 1}
    bundle = fit_oof_bundle(
        sequences,
        assignment,
        ranks=(4,),
        variants=("no_auxiliary",),
        config=TrainingConfig(seed=7, steps=1),
    )
    output = capsys.readouterr().out
    assert "model=rank4_no_auxiliary stage=model start" in output
    assert "fold=0 stage=train start" in output
    assert "fold=0 stage=train end elapsed_seconds=" in output
    assert "fold=0 stage=threshold start" in output
    assert "fold=0 stage=threshold end candidates=" in output
    assert "model=rank4_no_auxiliary stage=model end elapsed_seconds=" in output
    assert bundle["global_model_fitted"] is False
    assert bundle["global_threshold_fitted"] is False
