import json
from pathlib import Path

import pytest
import torch

from experiments.robot.libero.evaluation_protocol import (
    derive_paired_episode_seed,
    load_protocol_manifest,
)
from prismatic.models.action_head_workload import (
    ACTION_HEAD_WORKLOAD_SCHEMA_VERSION,
    ACTION_HEAD_WORKLOAD_TENSORS,
    build_action_head_workload,
    save_action_head_workload,
)
from scripts.origin_aware_calibration_lib import (
    CalibrationValidationError,
    validate_calibration_run,
)
from scripts.origin_aware_replay_lib import parse_fold_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "experiments/robot/libero/manifests/libero_spatial_official_50_v1.json"
FOLD_MANIFEST_PATH = REPO_ROOT / "experiments/robot/libero/manifests/libero_spatial_task_oof_5fold_v1.json"


def _trace():
    return [
        {
            "k": 1,
            "phase": "production",
            "state_finite": True,
            "output_finite": True,
            "latent_mse": 1.0,
            "latent_l2": 1.0,
            "action_mse": None,
            "action_l2": None,
        },
        {
            "k": 2,
            "phase": "production",
            "state_finite": True,
            "output_finite": True,
            "latent_mse": 0.1,
            "latent_l2": 0.2,
            "action_mse": 0.0005,
            "action_l2": 0.1,
        },
        {
            "k": 3,
            "phase": "shadow_tail",
            "state_finite": True,
            "output_finite": True,
            "latent_mse": 0.05,
            "latent_l2": 0.1,
            "action_mse": 0.0001,
            "action_l2": 0.05,
        },
    ]


def _workload(origin):
    selected = torch.zeros(1, 2, 4)
    incoming = selected.clone() if origin == "ACTUAL_WARM" else None
    return build_action_head_workload(
        actions_hidden_states=torch.zeros(1, 1, 513, 4),
        proprio_input=torch.zeros(1, 4),
        proprio_features=torch.zeros(1, 1, 4),
        incoming_warm_start_state=incoming,
        selected_initial_state=selected,
        actual_origin=origin,
    )


def _write_task(run_root, task_id, manifest, manifest_sha256):
    task_dir = run_root / f"task{task_id}"
    workload_dir = task_dir / "workloads"
    workload_dir.mkdir(parents=True)
    task_manifest = manifest["tasks"][str(task_id)]
    episode_stats = []
    records = []
    for paired_trial_id, initial_state_id in enumerate(task_manifest["partitions"]["calibration"]):
        episode_seed = derive_paired_episode_seed(
            base_seed=7,
            phase="calibration",
            task_suite_name="libero_spatial",
            task_id=task_id,
            initial_state_id=initial_state_id,
            paired_trial_id=paired_trial_id,
        )
        episode_stats.append(
            {
                "episode": paired_trial_id,
                "evaluation_protocol_phase": "calibration",
                "paired_trial_id": paired_trial_id,
                "initial_state_id": initial_state_id,
                "episode_seed": episode_seed,
                "initial_state_manifest_sha256": manifest_sha256,
                "smoke_excluded_from_fitting": False,
                "success": paired_trial_id != 9,
                "num_predictions": 3,
            }
        )
        for prediction_step in range(3):
            origin = "COLD" if prediction_step == 0 else "ACTUAL_WARM"
            identity = {
                "task_id": task_id,
                "episode_id": paired_trial_id,
                "paired_trial_id": paired_trial_id,
                "prediction_step": prediction_step,
                "initial_state_id": initial_state_id,
                "episode_seed": episode_seed,
            }
            record = {
                "task_id": task_id,
                "episode_id": paired_trial_id,
                "prediction_step": prediction_step,
                "K_t": 2,
                "max_recurrent_iteration": 3,
                "action_mse_threshold": 0.001,
                "effective_min_iter": 2,
                "latent_precheck_min_iter": 2,
                "evaluation_protocol_phase": "calibration",
                "initial_state_partition": "calibration",
                "paired_trial_id": paired_trial_id,
                "initial_state_id": initial_state_id,
                "initial_states_sha256": task_manifest["initial_states_sha256"],
                "initial_states_file": task_manifest["initial_states_file"],
                "initial_states_file_sha256": task_manifest["initial_states_file_sha256"],
                "initial_state_manifest_sha256": manifest_sha256,
                "paired_rng": True,
                "episode_seed": episode_seed,
                "episode_seed_source": "paired_protocol",
                "environment_seed_applied": True,
                "smoke_excluded_from_fitting": False,
                "use_latent_precheck": False,
                "latent_precheck_mode": "off",
                "latent_precheck_trace_level_requested": "off",
                "latent_precheck_trace_level_applied": "off",
                "latent_precheck_trace_collected": False,
                "latent_mse_list": [],
                "latent_l2_list": [],
                "latent_precheck_decisions": [],
                "nonfinite_policy": "legacy",
                "use_cached_final_output": True,
                "warm_start_enabled": True,
                "warm_start_source": "midpoint",
                "warm_start_used": origin == "ACTUAL_WARM",
                "execution_path": None,
                "numerical_retry_attempted": False,
                "numerical_retry_count": 0,
                "shadow_full_depth_enabled": True,
                "shadow_trace_complete": True,
                "shadow_error": None,
                "shadow_trace": _trace(),
                "shadow_production_snapshot": {
                    "K_t": 2,
                    "terminal_iteration": 2,
                    "stop_reason": "adjacent_action_mse",
                    "midpoint_source_iteration": 1,
                    "cached_final_output_reused": True,
                },
            }
            if prediction_step < 2:
                filename = f"task{task_id}_trial{paired_trial_id}_pred{prediction_step}.pt"
                path = workload_dir / filename
                digest = save_action_head_workload(path, _workload(origin), identity=identity)
                record.update(
                    {
                        "action_head_workload_requested": True,
                        "action_head_workload_captured": True,
                        "action_head_workload_file": f"workloads/{filename}",
                        "action_head_workload_sha256": digest,
                        "action_head_workload_schema_version": ACTION_HEAD_WORKLOAD_SCHEMA_VERSION,
                        "action_head_workload_tensor_fields": list(ACTION_HEAD_WORKLOAD_TENSORS),
                        "action_head_workload_capture_in_action_latency": True,
                    }
                )
            else:
                record.update(
                    {
                        "action_head_workload_requested": False,
                        "action_head_workload_captured": False,
                        "action_head_workload_file": None,
                        "action_head_workload_sha256": None,
                        "action_head_workload_schema_version": None,
                        "action_head_workload_tensor_fields": [],
                        "action_head_workload_capture_in_action_latency": False,
                    }
                )
            records.append(record)

    result = {
        "evaluation_protocol": {
            "phase": "calibration",
            "manifest_sha256": manifest_sha256,
            "task_suite_name": "libero_spatial",
            "num_trials_per_task": 10,
            "paired_rng": True,
            "action_head_workload_schema_version": ACTION_HEAD_WORKLOAD_SCHEMA_VERSION,
            "calibration_workload_predictions_per_episode": 2,
        },
        "tasks": {f"task_{task_id}": episode_stats},
        "total_episodes": 10,
        "total_successes": 9,
    }
    (task_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (task_dir / "steps.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


@pytest.fixture
def calibration_task(tmp_path):
    manifest, manifest_sha256 = load_protocol_manifest(
        str(MANIFEST_PATH), require_source_file_hashes=True
    )
    _write_task(tmp_path, 0, manifest, manifest_sha256)
    return tmp_path


def test_single_task_calibration_validates_scalar_and_tensor_artifacts(calibration_task):
    report = validate_calibration_run(
        str(calibration_task), str(MANIFEST_PATH), task_ids=[0]
    )
    assert report["valid"] is True
    assert report["complete_10_task_gate"] is False
    assert report["totals"] == {
        "tasks": 1,
        "episodes": 10,
        "successes": 9,
        "predictions": 30,
        "actual_warm_predictions": 20,
        "workload_shards": 20,
        "cold_workloads": 10,
        "actual_warm_workloads": 10,
    }


def test_calibration_rejects_corrupt_workload(calibration_task):
    path = calibration_task / "task0/workloads/task0_trial0_pred0.pt"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(CalibrationValidationError, match="SHA-256 mismatch"):
        validate_calibration_run(str(calibration_task), str(MANIFEST_PATH), task_ids=[0])


def test_calibration_rejects_workload_path_escape(calibration_task):
    step_path = calibration_task / "task0/steps.jsonl"
    records = [json.loads(line) for line in step_path.read_text(encoding="utf-8").splitlines()]
    records[0]["action_head_workload_file"] = "../outside.pt"
    step_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    with pytest.raises(CalibrationValidationError, match="escapes task workload directory"):
        validate_calibration_run(str(calibration_task), str(MANIFEST_PATH), task_ids=[0])


def test_oof_manifest_is_frozen_outcome_independent_and_exhaustive():
    fold_manifest = json.loads(FOLD_MANIFEST_PATH.read_text(encoding="utf-8"))
    assignment = parse_fold_manifest(fold_manifest, range(10))
    assert fold_manifest["assignment_algorithm"] == "task-id-symmetric-pairs-v1"
    assert fold_manifest["source_initial_state_manifest_sha256"] == (
        "0e3c6609b719d6b0a05f79efd769dff67141b52d00b42d9e0bea904ecf493144"
    )
    assert len(assignment) == 10
    assert sorted(assignment.values()).count(0) == 2
    assert all(list(assignment.values()).count(fold_id) == 2 for fold_id in range(5))
