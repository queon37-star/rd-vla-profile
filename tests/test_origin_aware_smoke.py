import json
from pathlib import Path

import pytest

from experiments.robot.libero.evaluation_protocol import (
    derive_paired_episode_seed,
    load_protocol_manifest,
)
from scripts.origin_aware_smoke_lib import SmokeValidationError, validate_smoke_run


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "experiments/robot/libero/manifests/libero_spatial_official_50_v1.json"


def _shadow_trace():
    return [
        {"k": 1, "phase": "production", "state_finite": True, "output_finite": True},
        {"k": 2, "phase": "production", "state_finite": True, "output_finite": True},
        {"k": 3, "phase": "shadow_tail", "state_finite": True, "output_finite": True},
    ]


def _step_record(task_id, paired_trial_id, prediction_step, manifest, manifest_sha256):
    task_manifest = manifest["tasks"][str(task_id)]
    initial_state_id = task_manifest["partitions"]["calibration"][paired_trial_id]
    episode_seed = derive_paired_episode_seed(
        base_seed=7,
        phase="smoke",
        task_suite_name="libero_spatial",
        task_id=task_id,
        initial_state_id=initial_state_id,
        paired_trial_id=paired_trial_id,
    )
    return {
        "task_id": task_id,
        "episode_id": paired_trial_id,
        "prediction_step": prediction_step,
        "K_t": 2,
        "max_recurrent_iteration": 3,
        "evaluation_protocol_phase": "smoke",
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
        "smoke_excluded_from_fitting": True,
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
        "warm_start_used": prediction_step > 0,
        "execution_path": "single_attempt",
        "numerical_retry_attempted": False,
        "numerical_retry_count": 0,
        "shadow_full_depth_enabled": True,
        "shadow_trace_complete": True,
        "shadow_error": None,
        "shadow_trace": _shadow_trace(),
        "shadow_production_snapshot": {
            "K_t": 2,
            "terminal_iteration": 2,
            "cached_final_output_reused": True,
        },
    }


def _write_task(run_root, task_id, manifest, manifest_sha256):
    task_dir = run_root / f"task{task_id}"
    task_dir.mkdir(parents=True)
    task_manifest = manifest["tasks"][str(task_id)]
    episode_stats = []
    step_records = []
    for paired_trial_id in range(3):
        initial_state_id = task_manifest["partitions"]["calibration"][paired_trial_id]
        episode_seed = derive_paired_episode_seed(
            base_seed=7,
            phase="smoke",
            task_suite_name="libero_spatial",
            task_id=task_id,
            initial_state_id=initial_state_id,
            paired_trial_id=paired_trial_id,
        )
        episode_stats.append(
            {
                "episode": paired_trial_id,
                "evaluation_protocol_phase": "smoke",
                "paired_trial_id": paired_trial_id,
                "initial_state_id": initial_state_id,
                "episode_seed": episode_seed,
                "initial_state_manifest_sha256": manifest_sha256,
                "smoke_excluded_from_fitting": True,
                "success": paired_trial_id != 2,
                "num_predictions": 2,
            }
        )
        step_records.extend(
            _step_record(task_id, paired_trial_id, prediction_step, manifest, manifest_sha256)
            for prediction_step in range(2)
        )

    result = {
        "evaluation_protocol": {
            "phase": "smoke",
            "manifest_sha256": manifest_sha256,
            "task_suite_name": "libero_spatial",
            "num_trials_per_task": 3,
            "paired_rng": True,
        },
        "tasks": {f"task_{task_id}": episode_stats},
        "total_episodes": 3,
        "total_successes": 2,
    }
    (task_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (task_dir / "steps.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in step_records),
        encoding="utf-8",
    )


@pytest.fixture
def valid_smoke_run(tmp_path):
    manifest, manifest_sha256 = load_protocol_manifest(
        str(MANIFEST_PATH), require_source_file_hashes=True
    )
    for task_id in range(10):
        _write_task(tmp_path, task_id, manifest, manifest_sha256)
    return tmp_path, manifest, manifest_sha256


def test_complete_smoke_gate_validates_all_ten_tasks(valid_smoke_run):
    run_root, _, manifest_sha256 = valid_smoke_run

    report = validate_smoke_run(str(run_root), str(MANIFEST_PATH))

    assert report["valid"] is True
    assert report["complete_10_task_gate"] is True
    assert report["manifest_sha256"] == manifest_sha256
    assert report["totals"] == {
        "tasks": 10,
        "episodes": 30,
        "successes": 20,
        "predictions": 60,
        "actual_warm_predictions": 30,
    }


def test_single_task_validation_is_explicitly_not_complete_gate(valid_smoke_run):
    run_root, _, _ = valid_smoke_run

    report = validate_smoke_run(str(run_root), str(MANIFEST_PATH), task_ids=[4])

    assert report["valid"] is True
    assert report["complete_10_task_gate"] is False
    assert report["task_ids"] == [4]


def test_smoke_rejects_partition_state_mismatch(valid_smoke_run):
    run_root, _, _ = valid_smoke_run
    result_path = run_root / "task2/result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    episode_stats = next(iter(result["tasks"].values()))
    episode_stats[0]["initial_state_id"] = 49
    result_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(SmokeValidationError, match="initial_state_id.*mismatch"):
        validate_smoke_run(str(run_root), str(MANIFEST_PATH), task_ids=[2])


def test_smoke_rejects_nonfinite_shadow_output(valid_smoke_run):
    run_root, _, _ = valid_smoke_run
    step_path = run_root / "task7/steps.jsonl"
    records = [json.loads(line) for line in step_path.read_text(encoding="utf-8").splitlines()]
    records[0]["shadow_trace"][1]["output_finite"] = False
    step_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    with pytest.raises(SmokeValidationError, match="non-finite shadow output"):
        validate_smoke_run(str(run_root), str(MANIFEST_PATH), task_ids=[7])


def test_smoke_rejects_missing_prediction_records(valid_smoke_run):
    run_root, _, _ = valid_smoke_run
    step_path = run_root / "task0/steps.jsonl"
    records = [json.loads(line) for line in step_path.read_text(encoding="utf-8").splitlines()]
    records = [record for record in records if record["episode_id"] != 1]
    step_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    with pytest.raises(SmokeValidationError, match="no action predictions"):
        validate_smoke_run(str(run_root), str(MANIFEST_PATH), task_ids=[0])


def test_smoke_rejects_retry_metadata(valid_smoke_run):
    run_root, _, _ = valid_smoke_run
    step_path = run_root / "task3/steps.jsonl"
    records = [json.loads(line) for line in step_path.read_text(encoding="utf-8").splitlines()]
    records[0]["numerical_retry_attempted"] = True
    records[0]["numerical_retry_count"] = 1
    step_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    with pytest.raises(SmokeValidationError, match="numerical_retry_attempted must be false"):
        validate_smoke_run(str(run_root), str(MANIFEST_PATH), task_ids=[3])


def test_smoke_rejects_latent_trace_in_clean_off_mode(valid_smoke_run):
    run_root, _, _ = valid_smoke_run
    step_path = run_root / "task5/steps.jsonl"
    records = [json.loads(line) for line in step_path.read_text(encoding="utf-8").splitlines()]
    records[0]["latent_mse_list"] = [0.1]
    step_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    with pytest.raises(SmokeValidationError, match="clean off mode populated latent_mse_list"):
        validate_smoke_run(str(run_root), str(MANIFEST_PATH), task_ids=[5])


def test_smoke_rejects_uncached_shadow_final_output(valid_smoke_run):
    run_root, _, _ = valid_smoke_run
    step_path = run_root / "task8/steps.jsonl"
    records = [json.loads(line) for line in step_path.read_text(encoding="utf-8").splitlines()]
    records[0]["shadow_production_snapshot"]["cached_final_output_reused"] = False
    step_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    with pytest.raises(SmokeValidationError, match="cached_final_output_reused must be true"):
        validate_smoke_run(str(run_root), str(MANIFEST_PATH), task_ids=[8])
