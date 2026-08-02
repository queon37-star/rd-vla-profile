from __future__ import annotations

import torch
from dataclasses import replace

import scripts.preconvergence_trigger_lib as trigger_lib
from scripts.preconvergence_trigger_lib import (
    LowRankPreconvergenceTrigger,
    TrainingConfig,
    evaluate_oof_bundle,
    fit_oof_bundle,
    tensor_scorer_item_call_count,
    train_trigger,
)
from test_preconvergence_dataset import make_sequence


def test_low_rank_model_parameter_and_flop_accounting() -> None:
    no_aux = LowRankPreconvergenceTrigger(3, 4)
    auxiliary = LowRankPreconvergenceTrigger(3, 4, auxiliary_action_dim=4)
    assert no_aux.parameter_count() == (6 * 4 + 4) + (4 + 1)
    assert auxiliary.parameter_count() == no_aux.parameter_count() + (4 * 4 + 4)
    assert no_aux.inference_flops() == auxiliary.inference_flops()


def test_training_is_deterministic_for_seed_7() -> None:
    sequences = [make_sequence(0, k_action=5), make_sequence(1, k_action=7)]
    config = TrainingConfig(seed=7, steps=12, learning_rate=0.01)
    left = train_trigger(sequences, rank=4, use_auxiliary=True, config=config)
    right = train_trigger(sequences, rank=4, use_auxiliary=True, config=config)
    assert left.normalizer.mean.equal(right.normalizer.mean)
    assert left.normalizer.scale.equal(right.normalizer.scale)
    for name, value in left.model.state_dict().items():
        assert torch.equal(value, right.model.state_dict()[name])


def test_auxiliary_action_is_not_an_inference_input() -> None:
    sequence = make_sequence(0, k_action=6)
    changed_actions = sequence.actions.clone().mul_(1000.0)
    before = sequence.states.clone()
    model = LowRankPreconvergenceTrigger(3, 4, auxiliary_action_dim=4)
    feature = torch.cat(
        (
            (sequence.states[3] - sequence.states[2]).mean(dim=0),
            (sequence.states[2] - sequence.states[1]).mean(dim=0),
        )
    ).unsqueeze(0)
    score_before = model.score_tensor(feature)
    score_after = model.score_tensor(feature)
    assert torch.equal(score_before, score_after)
    assert not torch.equal(sequence.actions, changed_actions)
    assert torch.equal(sequence.states, before)


def test_tensor_scorer_has_no_item_or_host_transfer() -> None:
    assert tensor_scorer_item_call_count() == 0
    model = LowRankPreconvergenceTrigger(3, 4)
    score = model.score_tensor(torch.zeros(2, 6))
    assert torch.is_tensor(score)
    assert score.shape == (2,)


def test_outer_held_out_changes_do_not_alter_fold_training_or_threshold(
    monkeypatch,
) -> None:
    sequences = [
        make_sequence(task_id, task_id=task_id, k_action=5 + task_id % 2)
        for task_id in range(10)
    ]
    assignment = {
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
    config = TrainingConfig(seed=7, steps=2)
    original = fit_oof_bundle(
        sequences,
        assignment,
        ranks=(4,),
        variants=("no_auxiliary",),
        config=config,
    )
    changed = [
        replace(sequence, states=sequence.states + 999.0)
        if sequence.identity.task_id in {0, 9}
        else sequence
        for sequence in sequences
    ]
    modified = fit_oof_bundle(
        changed,
        assignment,
        ranks=(4,),
        variants=("no_auxiliary",),
        config=config,
    )
    left_fold = original["models"]["rank4_no_auxiliary"]["folds"][0]
    right_fold = modified["models"]["rank4_no_auxiliary"]["folds"][0]
    assert left_fold["threshold_selection"] == right_fold["threshold_selection"]
    for name, value in left_fold["fitted_trigger"]["state_dict"].items():
        assert torch.equal(value, right_fold["fitted_trigger"]["state_dict"][name])

    def forbidden_refit(*args, **kwargs):
        raise AssertionError("evaluation attempted to refit a model")

    monkeypatch.setattr(trigger_lib, "train_trigger", forbidden_refit)
    report = evaluate_oof_bundle(
        sequences,
        assignment,
        original,
        latency={
            "coda_latency_ms": 2.0,
            "recurrent_iteration_latency_ms": 3.0,
            "gate_latency_ms": 0.1,
        },
    )
    result = report["models"]["rank4_no_auxiliary"]
    assert len(result["prediction_replays"]) == 10
    assert result["primary_actual_warm"]["history_coverage"] == {
        "model_applicable_prediction_count": 10,
        "history_unavailable_prediction_count": 0,
        "history_unavailable_definition": "K_action - 1 < 3",
    }
    assert report["online_integration_implemented"] is False
