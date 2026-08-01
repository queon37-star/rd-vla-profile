import copy
import json
from pathlib import Path

import pytest

from scripts.formal_origin_aware_oof_lib import (
    FormalOOFValidationError,
    run_formal_cost_sensitivity_oof,
    validate_cost_sensitivity_manifest,
)
from scripts.origin_aware_replay_lib import parse_shadow_predictions


def _shadow_record(*, task_id, latent_mse):
    action_mse = [None, 1.0, 1.0, 0.2, 0.05, 0.01]
    baseline_k = 5
    max_iter = 6
    return {
        "task_id": task_id,
        "episode_id": 0,
        "prediction_step": 0,
        "K_t": baseline_k,
        "max_recurrent_iteration": max_iter,
        "action_mse_threshold": 0.1,
        "effective_min_iter": 2,
        "latent_precheck_min_iter": 3,
        "warm_start_used": True,
        "numerical_retry_attempted": False,
        "shadow_full_depth_enabled": True,
        "shadow_trace_complete": True,
        "shadow_error": None,
        "shadow_production_snapshot": {
            "K_t": baseline_k,
            "terminal_iteration": baseline_k,
            "stop_reason": "adjacent_action_mse",
            "midpoint_source_iteration": 2,
            "cached_final_output_reused": True,
        },
        "shadow_trace": [
            {
                "k": k,
                "phase": "production" if k <= baseline_k else "shadow_tail",
                "state_finite": True,
                "output_finite": True,
                "latent_mse": latent_mse[k - 1],
                "latent_l2": latent_mse[k - 1] ** 0.5,
                "action_mse": action_mse[k - 1],
                "action_l2": (
                    None if action_mse[k - 1] is None else action_mse[k - 1] ** 0.5
                ),
            }
            for k in range(1, max_iter + 1)
        ],
    }


def _cost_manifest():
    return {
        "schema_version": 1,
        "scope": "baseline-conditioned offline pruning only",
        "scenarios": [
            {
                "id": "primary",
                "description": "primary",
                "primary": True,
                "candidate_favorable": True,
                "cost_model": {
                    "recurrent_ms": 1.0,
                    "decode_ms": 1.0,
                    "latent_gate_ms": 0.0,
                    "action_compare_ms": 0.0,
                    "finite_check_ms": 0.0,
                },
            },
            {
                "id": "overhead",
                "description": "overhead",
                "primary": False,
                "candidate_favorable": False,
                "cost_model": {
                    "recurrent_ms": 1.0,
                    "decode_ms": 1.0,
                    "latent_gate_ms": 0.1,
                    "action_compare_ms": 0.1,
                    "finite_check_ms": 0.1,
                },
            },
        ],
        "promotion_gate": {"min_action_head_improvement": 0.05},
    }


def _predictions():
    records = []
    for task_id in (0, 1):
        records.append(
            _shadow_record(
                task_id=task_id,
                latent_mse=[1.0, 1.0, 0.1, 0.1, 0.1, 0.1],
            )
        )
    return parse_shadow_predictions(records)


def _folds():
    return {
        "folds": [
            {"fold_id": 0, "validation_task_ids": [0]},
            {"fold_id": 1, "validation_task_ids": [1]},
        ]
    }


def test_cost_manifest_requires_one_primary_and_candidate_favorable_scenario():
    manifest = _cost_manifest()
    validate_cost_sensitivity_manifest(manifest)

    broken = copy.deepcopy(manifest)
    broken["scenarios"][1]["primary"] = True
    with pytest.raises(FormalOOFValidationError, match="exactly one primary"):
        validate_cost_sensitivity_manifest(broken)

    broken = copy.deepcopy(manifest)
    broken["scenarios"][0]["candidate_favorable"] = False
    with pytest.raises(FormalOOFValidationError, match="candidate-favorable"):
        validate_cost_sensitivity_manifest(broken)


def test_formal_oof_shortlist_is_primary_ranked_and_requires_microbenchmark(monkeypatch):
    import scripts.formal_origin_aware_oof_lib as module

    monkeypatch.setattr(module, "FIXED_WARM_THRESHOLDS", (0.1,))
    monkeypatch.setattr(module, "WARM_QUANTILES", ())
    monkeypatch.setattr(module, "MAX_SKIP_VALUES", (1,))
    monkeypatch.setattr(module, "CONFIRMATION_MODES", ("next_iter",))
    report = run_formal_cost_sensitivity_oof(
        _predictions(), _folds(), _cost_manifest(), top_n=1
    )

    assert report["family_grid"]["size"] == 1
    assert report["passing_family_count"] == 1
    assert report["microbenchmark_shortlist_count"] == 1
    assert report["microbenchmark_shortlist"][0]["status"] == (
        "gpu_schedule_microbenchmark_required"
    )
    assert report["online_screening_allowed"] is False
    assert len(report["scenario_reports"]) == 2
    assert "full_data_refit_metrics_diagnostic" in (
        report["microbenchmark_shortlist"][0]
    )


def test_formal_oof_rejects_insufficient_distinct_configs(monkeypatch):
    import scripts.formal_origin_aware_oof_lib as module

    monkeypatch.setattr(module, "FIXED_WARM_THRESHOLDS", (0.1,))
    monkeypatch.setattr(module, "WARM_QUANTILES", ())
    monkeypatch.setattr(module, "MAX_SKIP_VALUES", (1,))
    monkeypatch.setattr(module, "CONFIRMATION_MODES", ("next_iter",))
    with pytest.raises(FormalOOFValidationError, match="requires 2 distinct"):
        run_formal_cost_sensitivity_oof(
            _predictions(), _folds(), _cost_manifest(), top_n=2
        )


def test_committed_shortlist_is_distinct_and_not_online_promoted():
    repo_root = Path(__file__).resolve().parents[1]
    cost_path = (
        repo_root
        / "experiments/robot/libero/manifests/origin_aware_oof_cost_sensitivity_v1.json"
    )
    shortlist_path = (
        repo_root
        / "experiments/robot/libero/manifests/origin_aware_oof_seed7_shortlist_v1.json"
    )
    cost_manifest = json.loads(cost_path.read_text(encoding="utf-8"))
    shortlist = json.loads(shortlist_path.read_text(encoding="utf-8"))
    validate_cost_sensitivity_manifest(cost_manifest)

    assert shortlist["status"] == "gpu_schedule_microbenchmark_required"
    assert shortlist["online_screening_allowed"] is False
    assert shortlist["linear_model_5pct_gate_met"] is False
    assert shortlist["tested_candidate_favorable_best_improvement"] < 0.05
    assert shortlist["source"]["formal_report"].endswith(
        "/20260801_seed7_e93dbb2/report.json"
    )

    candidates = shortlist["candidates"]
    expected_configs = [
        (0.075, 3, "next_iter"),
        (0.075, 2, "next_iter"),
        (0.08, 3, "next_iter"),
        (0.08, 2, "next_iter"),
        (0.075, 1, "next_iter"),
        (0.08, 1, "next_iter"),
    ]
    actual_configs = [
        (
            candidate["warm_threshold"],
            candidate["max_skip_iters"],
            candidate["confirmation_mode"],
        )
        for candidate in candidates
    ]
    assert [candidate["rank"] for candidate in candidates] == list(range(1, 7))
    assert actual_configs == expected_configs
    assert len(set(actual_configs)) == len(actual_configs)
    for candidate in candidates:
        assert float.fromhex(candidate["warm_threshold_hex"]) == (
            candidate["warm_threshold"]
        )
        assert candidate["cold_threshold"] == 0.2
