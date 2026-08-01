import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from scripts.learned_convergence_probe_lib import (
    FEATURE_NAMES,
    LearnedProbeValidationError,
    aggregate_scheduler_metrics,
    fit_normalizer,
    fit_tiny_mlp,
    leakage_audit,
    prediction_to_dataset_record,
    replay_scored_records,
    select_train_threshold,
)
from scripts.origin_aware_replay_lib import parse_shadow_prediction


def _shadow_record(*, warm=True):
    return {
        "task_id": 0,
        "episode_id": 0,
        "prediction_step": 0,
        "K_t": 3,
        "max_recurrent_iteration": 4,
        "action_mse_threshold": 0.001,
        "effective_min_iter": 2,
        "latent_precheck_min_iter": 2,
        "warm_start_used": warm,
        "numerical_retry_attempted": False,
        "shadow_full_depth_enabled": True,
        "shadow_trace_complete": True,
        "shadow_error": None,
        "iteration_mse": [0.002, 0.0009],
        "action_delta_list": [0.2, 0.1],
        "shadow_production_snapshot": {
            "K_t": 3,
            "terminal_iteration": 3,
            "cached_final_output_reused": True,
        },
        "shadow_trace": [
            {
                "k": k,
                "phase": "production" if k <= 3 else "shadow_tail",
                "state_finite": True,
                "output_finite": True,
                "latent_mse": [1.0, 0.5, 0.1, 0.05][k - 1],
                "latent_l2": [10.0, 7.0, 3.0, 2.0][k - 1],
                "action_mse": [None, 0.0008, 0.0007, 0.0006][k - 1],
                "action_l2": [None, 0.19, 0.09, 0.08][k - 1],
            }
            for k in range(1, 5)
        ],
    }


def _dataset_record(task_id, episode_id, prediction_index, labels, scores=None, origin="ACTUAL_WARM"):
    transitions = []
    for offset, label in enumerate(labels, start=2):
        transitions.append(
            {
                "k": offset,
                "phase": "production",
                "action_mse": 0.0005 if label else 0.01,
                "label": int(label),
                "features": [float(offset)] * len(FEATURE_NAMES),
            }
        )
    record = {
        "key": [str(task_id), episode_id, prediction_index],
        "task_id": str(task_id),
        "episode_id": episode_id,
        "prediction_index": prediction_index,
        "actual_origin": origin,
        "baseline_k": next((index + 2 for index, label in enumerate(labels) if label), len(labels) + 1),
        "baseline_decode_calls": next((index + 2 for index, label in enumerate(labels) if label), len(labels) + 1),
        "max_iter": len(labels) + 1,
        "transitions": transitions,
    }
    if scores is not None:
        record["scores"] = scores
    return record


def test_dataset_uses_native_production_mse_and_shadow_tail_diagnostic():
    prediction = parse_shadow_prediction(_shadow_record())
    record = prediction_to_dataset_record(prediction)

    assert [item["action_mse"] for item in record["transitions"]] == [
        0.002,
        0.0009,
        0.0006,
    ]
    assert [item["label"] for item in record["transitions"]] == [0, 1, 1]
    assert record["transitions"][0]["phase"] == "production"
    assert record["transitions"][-1]["phase"] == "shadow_tail"
    assert all(
        len(item["features"]) == len(FEATURE_NAMES)
        and all(math.isfinite(value) for value in item["features"])
        for item in record["transitions"]
    )
    assert record["transitions"][0]["features"][FEATURE_NAMES.index("prev2_available")] == 0.0


def test_leakage_audit_uses_whole_tasks_without_prediction_or_episode_overlap():
    records = [
        _dataset_record(0, 0, 0, [0, 1]),
        _dataset_record(0, 0, 1, [0, 1]),
        _dataset_record(1, 0, 0, [0, 1]),
    ]
    audit = leakage_audit(records, {"0": 0, "1": 1})
    assert audit["passed"] is True
    assert all(item["prediction_overlap_count"] == 0 for item in audit["folds"])
    assert all(item["episode_overlap_count"] == 0 for item in audit["folds"])

    assert audit["folds"][0]["validation_task_ids"] == ["0"]
    assert audit["folds"][1]["validation_task_ids"] == ["1"]


def test_normalizer_is_fit_only_from_supplied_training_rows():
    training = np.asarray([[0.0] * len(FEATURE_NAMES), [2.0] * len(FEATURE_NAMES)])
    held_out_outlier = np.asarray([[1000.0] * len(FEATURE_NAMES)])
    fitted = fit_normalizer(training)
    assert fitted["mean"] == [1.0] * len(FEATURE_NAMES)
    assert fitted["scale"] == [1.0] * len(FEATURE_NAMES)
    assert fit_normalizer(np.vstack([training, held_out_outlier]))["mean"][0] != 1.0


def test_train_threshold_selection_meets_capture_then_minimizes_false_stops():
    records = [
        _dataset_record(0, 0, 0, [0, 1, 1], [0.1, 0.9, 0.95]),
        _dataset_record(0, 1, 0, [0, 1, 1], [0.2, 0.85, 0.9]),
    ]
    selection = select_train_threshold(records)
    replay = replay_scored_records(records, selection["threshold"])
    metrics = aggregate_scheduler_metrics(replay)
    assert selection["selection_status"] == "capture_feasible"
    assert metrics["false_convergence_count"] == 0
    assert metrics["convergence_capture"] == 1.0


def test_scheduler_replay_omits_current_decode_and_counts_false_convergence():
    true_record = _dataset_record(0, 0, 0, [0, 1, 1], [0.1, 0.9, 0.9])
    false_record = _dataset_record(1, 0, 0, [0, 1, 1], [0.8, 0.9, 0.9])
    replay = replay_scored_records([true_record, false_record], threshold=0.5)
    assert replay[0]["terminal_k"] == 3
    assert replay[0]["decode_calls"] == 2
    assert replay[0]["false_convergence"] is False
    assert replay[1]["terminal_k"] == 2
    assert replay[1]["decode_calls"] == 1
    assert replay[1]["false_convergence"] is True


def test_tiny_mlp_respects_size_limit_and_is_deterministic():
    x = np.linspace(-1.0, 1.0, 40 * len(FEATURE_NAMES)).reshape(40, len(FEATURE_NAMES))
    y = (x[:, 0] > 0).astype(np.float64)
    first = fit_tiny_mlp(x, y, seed=7, width=16, steps=3)
    second = fit_tiny_mlp(x, y, seed=7, width=16, steps=3)
    assert first == second
    assert first["hidden_width"] <= 16
    assert first["affine_layer_count"] == 2
    assert first["parameter_count"] == len(FEATURE_NAMES) * 16 + 16 + 16 + 1


def test_frozen_model_artifact_and_fold_normalization_are_unchanged():
    repo_root = Path(__file__).resolve().parents[1]
    artifact_path = (
        repo_root
        / "experiments/robot/libero/manifests/learned_convergence_probe_seed7_model_v1.json"
    )
    payload = artifact_path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == (
        "2cd710fde8d088955dad85f55563cfb24f7d7bdfbfa55623968d0d999173f0c5"
    )

    artifact = json.loads(payload)
    models = artifact["oof_models_thresholds_and_normalization"]
    assert set(models) == {
        "latent_mse_threshold",
        "logistic_regression",
        "class_weighted_logistic_regression",
        "tiny_mlp",
    }
    for model_name, model in models.items():
        assert len(model["folds"]) == 5
        for fold in model["folds"]:
            normalization = fold["normalization"]
            if model_name == "latent_mse_threshold":
                assert normalization is None
                continue
            assert normalization is not None
            assert len(normalization["mean"]) == len(FEATURE_NAMES)
            assert len(normalization["scale"]) == len(FEATURE_NAMES)
