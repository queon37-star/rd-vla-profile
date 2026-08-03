from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import torch

from scripts.boundary_latent_oof_lib import (
    MODEL_CONFIGS,
    aggregate_row_metrics,
    build_boundary_dataset_payload,
    exact_average_precision,
    evaluate_boundary_oof,
    fit_boundary_oof_bundle,
    fit_model,
    fit_normalizer,
    gate_timing,
    iteration_matched_metrics,
    leakage_audit,
    select_gate_threshold,
)
from scripts.action_latent_audit_lib import exact_rank_auc
from test_boundary_latent_dataset import record_with_k


def scalar_row(value: float, label: int, prediction_id: int) -> dict:
    features = torch.zeros(7)
    features[1] = value
    return {
        "task_id": 0,
        "episode_id": 0,
        "prediction_id": prediction_id,
        "target_reference": "K_first_authoritative",
        "k": 3,
        "K_reference": 4,
        "label": label,
        "boundary_offset": 0 if label else -1,
        "scalar_features": features,
        "current_mean_pooled_delta": torch.zeros(2),
        "previous_mean_pooled_delta": torch.zeros(2),
        "weight": 1.0,
    }


def test_task_fold_leakage_audit_has_zero_overlap() -> None:
    payload, _ = build_boundary_dataset_payload(
        [record_with_k(0, 0, 5), record_with_k(1, 1, 5)]
    )
    audit = leakage_audit(payload, {0: 0, 1: 1})
    assert audit["passed"] is True
    assert all(fold["task_overlap_count"] == 0 for fold in audit["folds"])
    assert all(fold["prediction_overlap_count"] == 0 for fold in audit["folds"])


def test_normalizer_uses_only_supplied_training_rows() -> None:
    training = torch.tensor([[1.0], [3.0]])
    held_out = torch.tensor([[1000.0]])
    assert fit_normalizer(training)["mean"].item() == 2.0
    assert fit_normalizer(torch.cat((training, held_out)))["mean"].item() != 2.0


def test_learned_scalar_direction_can_reverse() -> None:
    ascending = [scalar_row(0.0, 0, 0), scalar_row(1.0, 0, 1), scalar_row(2.0, 1, 2), scalar_row(3.0, 1, 3)]
    descending = [scalar_row(row["scalar_features"][1].item(), 1 - row["label"], row["prediction_id"]) for row in ascending]
    positive = fit_model(ascending, "delta_rms", seed=7, steps=200)
    negative = fit_model(descending, "delta_rms", seed=7, steps=200)
    assert positive["state_dict"]["linear.weight"].item() > 0
    assert negative["state_dict"]["linear.weight"].item() < 0


def test_exact_auc_and_average_precision_with_ties() -> None:
    labels = [1, 0, 1, 0]
    scores = [0.9, 0.8, 0.8, 0.1]
    assert exact_rank_auc(labels, scores) == pytest.approx(0.875)
    assert exact_average_precision(labels, scores) == pytest.approx(5 / 6)


def test_iteration_matched_auc_and_iteration_only_null_contract() -> None:
    rows = [
        {"task_id": task, "k": k, "label": label, "score": score}
        for k in (3, 4)
        for task, label, score in ((0, 0, 0.1), (1, 1, 0.9))
    ]
    matched = iteration_matched_metrics(rows, "delta_rms")
    assert matched["valid_k_count"] == 2
    assert matched["unweighted_valid_k_macro_roc_auc"] == 1.0
    iteration_only = iteration_matched_metrics(rows, "iteration_only")
    assert iteration_only["valid_k_count"] == 0
    assert iteration_only["unweighted_valid_k_macro_roc_auc"] is None


def exhaustive_selector(predictions: list[dict]) -> dict:
    scores = sorted({float(score) for prediction in predictions for score in prediction["scores_by_k"].values()})
    candidates = scores + [float(np.nextafter(scores[-1], math.inf))]
    best = None
    best_key = None
    for threshold in candidates:
        timings = [
            gate_timing(prediction["scores_by_k"], threshold, prediction["K_reference"], prediction["max_iter"])
            for prediction in predictions
        ]
        counts = {category: sum(item["category"] == category for item in timings) for category in ("early", "ideal", "late", "missed")}
        offsets = [item["trigger_offset"] for item in timings if item["trigger_offset"] is not None]
        metrics = {
            "late_plus_missed_count": counts["late"] + counts["missed"],
            "missed_count": counts["missed"],
            "total_late_offset": sum(value for value in offsets if value > 0),
            "early_count": counts["early"],
            "total_absolute_early_offset": sum(-value for value in offsets if value < 0),
            "ideal_count": counts["ideal"],
        }
        key = (metrics["late_plus_missed_count"], metrics["missed_count"], metrics["total_late_offset"], metrics["early_count"], metrics["total_absolute_early_offset"], -metrics["ideal_count"], -threshold)
        if best_key is None or key < best_key:
            best_key = key
            best = {"threshold": threshold, **metrics}
    return best


@pytest.mark.parametrize("seed", range(8))
def test_event_sweep_exactly_matches_exhaustive_randomized(seed: int) -> None:
    rng = np.random.default_rng(seed)
    predictions = []
    for prediction_id in range(8):
        max_iter = 7
        scores = {k: float(rng.choice([0.1, 0.2, 0.5, 0.8])) for k in range(3, max_iter + 1)}
        predictions.append({"prediction_id": prediction_id, "K_reference": int(rng.integers(4, 8)), "max_iter": max_iter, "scores_by_k": scores})
    optimized = select_gate_threshold(predictions)
    reference = exhaustive_selector(predictions)
    assert optimized["selected_threshold"] == reference["threshold"]
    assert optimized["selected_threshold_hex"] == reference["threshold"].hex()
    assert optimized["train_metrics"] == reference


def test_gate_timing_all_categories_and_max_iter_is_missed() -> None:
    scores = {3: 0.1, 4: 0.9, 5: 0.1, 6: 0.1}
    assert gate_timing(scores, 0.05, 6, 6)["category"] == "early"
    assert gate_timing(scores, 0.8, 5, 6)["category"] == "ideal"
    assert gate_timing(scores, 0.8, 4, 6)["category"] == "late"
    assert gate_timing(scores, 0.95, 5, 6)["category"] == "missed"
    max_only = {3: 0.1, 4: 0.1, 5: 0.1, 6: 0.9}
    assert gate_timing(max_only, 0.8, 6, 6)["category"] == "missed"


def test_threshold_tie_break_is_deterministic_and_prefers_higher_threshold() -> None:
    predictions = [{"K_reference": 4, "max_iter": 5, "scores_by_k": {3: 0.8, 4: 0.2, 5: 0.2}}]
    first = select_gate_threshold(predictions)
    second = select_gate_threshold(copy.deepcopy(predictions))
    assert first == second
    assert first["selected_threshold_hex"] == (0.8).hex()


def test_training_threshold_ignores_held_out_task_changes() -> None:
    payload, _ = build_boundary_dataset_payload(
        [record_with_k(0, 0, 5), record_with_k(1, 1, 5)]
    )
    payload["boundary_dataset_sha256"] = "synthetic"
    kwargs = dict(
        assignment={0: 0, 1: 1},
        source_bundle_sha256="source",
        boundary_dataset_sha256="synthetic",
        fold_manifest_sha256="fold",
        git_commit="commit",
        steps=2,
    )
    first = fit_boundary_oof_bundle(payload, **kwargs)
    changed = copy.deepcopy(payload)
    for trajectory in changed["scoring_trajectories"]:
        if trajectory["task_id"] == 0:
            trajectory["scalar_features"] += 1000
            trajectory["mean_pooled_features"] += 1000
    second = fit_boundary_oof_bundle(changed, **kwargs)
    fold0_first = next(fold for fold in first["folds"] if fold["fold_id"] == 0)
    fold0_second = next(fold for fold in second["folds"] if fold["fold_id"] == 0)
    assert {
        name: model["threshold_selection"] for name, model in fold0_first["models"].items()
    } == {
        name: model["threshold_selection"] for name, model in fold0_second["models"].items()
    }
    assert set(MODEL_CONFIGS) == {
        "iteration_only", "delta_rms", "relative_delta_rms", "delta_ratio", "scalar_combo", "mean_pooled_low_rank4"
    }


def test_end_to_end_oof_report_keeps_targets_and_folds_separate() -> None:
    payload, _ = build_boundary_dataset_payload(
        [record_with_k(0, 0, 5), record_with_k(1, 1, 5)]
    )
    payload["boundary_dataset_sha256"] = "synthetic"
    payload["source_trajectory_bundle_sha256"] = "source"
    assignment = {0: 0, 1: 1}
    bundle = fit_boundary_oof_bundle(
        payload,
        assignment,
        source_bundle_sha256="source",
        boundary_dataset_sha256="synthetic",
        fold_manifest_sha256="fold",
        git_commit="commit",
        steps=1,
    )
    report, predictions, task_rows = evaluate_boundary_oof(
        payload, bundle, assignment
    )
    assert len(report["models"]) == 2 * len(MODEL_CONFIGS)
    assert len(predictions) == 2 * len(MODEL_CONFIGS) * 2
    assert all(row["outer_fold"] == assignment[row["task_id"]] for row in predictions)
    assert {row["target_reference"] for row in task_rows} == set(payload["target_references"])
    assert report["cold_population"]["status"] == "excluded from OOF population"
