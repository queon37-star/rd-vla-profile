"""Fail-closed validation for formal origin-aware calibration artifacts."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from experiments.robot.libero.evaluation_protocol import (
    derive_paired_episode_seed,
    load_protocol_manifest,
)
from prismatic.models.action_head_workload import (
    ACTION_HEAD_WORKLOAD_SCHEMA_VERSION,
    ACTION_HEAD_WORKLOAD_TENSORS,
    ActionHeadWorkloadError,
    load_action_head_workload,
)
from scripts.origin_aware_replay_lib import (
    ShadowTraceValidationError,
    parse_shadow_prediction,
)


CALIBRATION_TASK_IDS = tuple(range(10))
CALIBRATION_EPISODES_PER_TASK = 10
CALIBRATION_WORKLOADS_PER_EPISODE = 2


class CalibrationValidationError(ValueError):
    """Raised when formal calibration artifacts violate the frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CalibrationValidationError(message)


def _load_json(path: Path) -> Dict[str, Any]:
    _require(path.is_file(), f"missing JSON result: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CalibrationValidationError(f"invalid JSON result {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON result root must be an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[Dict[str, Any]]:
    _require(path.is_file(), f"missing JSONL step log: {path}")
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CalibrationValidationError(
                f"invalid JSONL at {path}:{line_number}: {exc}"
            ) from exc
        _require(isinstance(value, dict), f"JSONL record must be an object at {path}:{line_number}")
        records.append(value)
    _require(records, f"step log is empty: {path}")
    return records


def _resolve_workload_path(task_dir: Path, recorded_path: str) -> Path:
    path = Path(recorded_path)
    if not path.is_absolute():
        path = task_dir / path
    resolved = path.resolve()
    expected_root = (task_dir / "workloads").resolve()
    try:
        resolved.relative_to(expected_root)
    except ValueError as exc:
        raise CalibrationValidationError(
            f"workload path escapes task workload directory: {recorded_path}"
        ) from exc
    return resolved


def _validate_workload(
    task_dir: Path,
    record: Mapping[str, Any],
    *,
    context: str,
    expected_identity: Mapping[str, Any],
    expected_origin: str,
) -> Path:
    _require(record.get("action_head_workload_requested") is True, f"{context}: workload not requested")
    _require(record.get("action_head_workload_captured") is True, f"{context}: workload not captured")
    _require(
        record.get("action_head_workload_capture_in_action_latency") is True,
        f"{context}: workload latency inclusion flag mismatch",
    )
    _require(
        record.get("action_head_workload_schema_version") == ACTION_HEAD_WORKLOAD_SCHEMA_VERSION,
        f"{context}: workload schema mismatch",
    )
    _require(
        record.get("action_head_workload_tensor_fields") == list(ACTION_HEAD_WORKLOAD_TENSORS),
        f"{context}: workload tensor field list mismatch",
    )
    recorded_path = record.get("action_head_workload_file")
    digest = record.get("action_head_workload_sha256")
    _require(isinstance(recorded_path, str) and recorded_path, f"{context}: workload file is missing")
    _require(isinstance(digest, str) and len(digest) == 64, f"{context}: workload SHA-256 is invalid")
    workload_path = _resolve_workload_path(task_dir, recorded_path)
    try:
        load_action_head_workload(
            workload_path,
            expected_sha256=digest,
            expected_identity=expected_identity,
            expected_origin=expected_origin,
        )
    except ActionHeadWorkloadError as exc:
        raise CalibrationValidationError(f"{context}: {exc}") from exc
    return workload_path


def _validate_unsampled_workload(record: Mapping[str, Any], context: str) -> None:
    expected = {
        "action_head_workload_requested": False,
        "action_head_workload_captured": False,
        "action_head_workload_file": None,
        "action_head_workload_sha256": None,
        "action_head_workload_schema_version": None,
        "action_head_workload_tensor_fields": [],
        "action_head_workload_capture_in_action_latency": False,
    }
    for field, value in expected.items():
        _require(record.get(field) == value, f"{context}: unsampled workload field {field!r} mismatch")


def _validate_task(
    run_root: Path,
    task_id: int,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    *,
    base_seed: int,
) -> Dict[str, Any]:
    task_dir = run_root / f"task{task_id}"
    result = _load_json(task_dir / "result.json")
    records = _load_jsonl(task_dir / "steps.jsonl")
    task_manifest = manifest["tasks"][str(task_id)]

    protocol = result.get("evaluation_protocol")
    _require(isinstance(protocol, Mapping), f"task {task_id}: result is missing evaluation_protocol")
    expected_protocol = {
        "phase": "calibration",
        "manifest_sha256": manifest_sha256,
        "task_suite_name": "libero_spatial",
        "num_trials_per_task": CALIBRATION_EPISODES_PER_TASK,
        "paired_rng": True,
        "action_head_workload_schema_version": ACTION_HEAD_WORKLOAD_SCHEMA_VERSION,
        "calibration_workload_predictions_per_episode": CALIBRATION_WORKLOADS_PER_EPISODE,
    }
    for field, expected in expected_protocol.items():
        _require(protocol.get(field) == expected, f"task {task_id}: protocol field {field!r} mismatch")
    _require(
        result.get("total_episodes") == CALIBRATION_EPISODES_PER_TASK,
        f"task {task_id}: total_episodes != 10",
    )

    result_tasks = result.get("tasks")
    _require(isinstance(result_tasks, Mapping) and len(result_tasks) == 1, f"task {task_id}: expected one task result")
    episode_stats = next(iter(result_tasks.values()))
    _require(isinstance(episode_stats, list), f"task {task_id}: episode stats must be a list")
    _require(
        len(episode_stats) == CALIBRATION_EPISODES_PER_TASK,
        f"task {task_id}: expected ten episode stats",
    )

    records_by_episode: Dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        episode_id = record.get("episode_id")
        _require(
            isinstance(episode_id, int) and 0 <= episode_id < CALIBRATION_EPISODES_PER_TASK,
            f"task {task_id}: invalid step episode_id {episode_id!r}",
        )
        records_by_episode[episode_id].append(record)

    expected_state_ids = task_manifest["partitions"]["calibration"]
    successes = 0
    total_predictions = 0
    actual_warm_predictions = 0
    workload_paths = set()
    cold_workloads = 0
    warm_workloads = 0
    for paired_trial_id, episode_stat in enumerate(episode_stats):
        context = f"task {task_id} trial {paired_trial_id}"
        _require(isinstance(episode_stat, Mapping), f"{context}: episode stat must be an object")
        initial_state_id = expected_state_ids[paired_trial_id]
        expected_seed = derive_paired_episode_seed(
            base_seed=base_seed,
            phase="calibration",
            task_suite_name="libero_spatial",
            task_id=task_id,
            initial_state_id=initial_state_id,
            paired_trial_id=paired_trial_id,
        )
        expected_stat_fields = {
            "episode": paired_trial_id,
            "evaluation_protocol_phase": "calibration",
            "paired_trial_id": paired_trial_id,
            "initial_state_id": initial_state_id,
            "episode_seed": expected_seed,
            "initial_state_manifest_sha256": manifest_sha256,
            "smoke_excluded_from_fitting": False,
        }
        for field, expected in expected_stat_fields.items():
            _require(episode_stat.get(field) == expected, f"{context}: episode field {field!r} mismatch")
        if episode_stat.get("success") is True:
            successes += 1

        episode_records = sorted(records_by_episode.get(paired_trial_id, []), key=lambda item: item["prediction_step"])
        _require(episode_records, f"{context}: no action predictions were recorded")
        _require(
            episode_stat.get("num_predictions") == len(episode_records),
            f"{context}: result/step prediction counts differ",
        )
        _require(
            [record.get("prediction_step") for record in episode_records] == list(range(len(episode_records))),
            f"{context}: prediction_step must be contiguous from zero",
        )

        for prediction_step, record in enumerate(episode_records):
            prediction_context = f"{context} prediction {prediction_step}"
            expected_step_fields = {
                "task_id": task_id,
                "episode_id": paired_trial_id,
                "evaluation_protocol_phase": "calibration",
                "initial_state_partition": "calibration",
                "paired_trial_id": paired_trial_id,
                "initial_state_id": initial_state_id,
                "initial_states_sha256": task_manifest["initial_states_sha256"],
                "initial_states_file": task_manifest["initial_states_file"],
                "initial_states_file_sha256": task_manifest["initial_states_file_sha256"],
                "initial_state_manifest_sha256": manifest_sha256,
                "paired_rng": True,
                "episode_seed": expected_seed,
                "episode_seed_source": "paired_protocol",
                "environment_seed_applied": True,
                "smoke_excluded_from_fitting": False,
                "use_latent_precheck": False,
                "latent_precheck_mode": "off",
                "latent_precheck_trace_level_requested": "off",
                "latent_precheck_trace_level_applied": "off",
                "latent_precheck_trace_collected": False,
                "nonfinite_policy": "legacy",
                "use_cached_final_output": True,
                "warm_start_enabled": True,
                "warm_start_source": "midpoint",
            }
            for field, expected in expected_step_fields.items():
                _require(record.get(field) == expected, f"{prediction_context}: field {field!r} mismatch")
            _require(record.get("execution_path") != "numerical_abort", f"{prediction_context}: numerical abort")
            _require(
                record.get("numerical_retry_attempted") is False,
                f"{prediction_context}: numerical_retry_attempted must be false",
            )
            _require(record.get("numerical_retry_count") == 0, f"{prediction_context}: unexpected retry count")
            for trace_field in ("latent_mse_list", "latent_l2_list", "latent_precheck_decisions"):
                _require(
                    record.get(trace_field) in (None, []),
                    f"{prediction_context}: clean off mode populated {trace_field}",
                )
            snapshot = record.get("shadow_production_snapshot")
            _require(
                isinstance(snapshot, Mapping) and snapshot.get("cached_final_output_reused") is True,
                f"{prediction_context}: cached final output was not reused",
            )
            try:
                parsed = parse_shadow_prediction(record)
            except ShadowTraceValidationError as exc:
                raise CalibrationValidationError(f"{prediction_context}: {exc}") from exc
            expected_origin = "COLD" if prediction_step == 0 else "ACTUAL_WARM"
            _require(parsed.actual_origin == expected_origin, f"{prediction_context}: actual origin mismatch")
            if expected_origin == "ACTUAL_WARM":
                actual_warm_predictions += 1

            capture_expected = prediction_step < CALIBRATION_WORKLOADS_PER_EPISODE
            if capture_expected:
                identity = {
                    "task_id": task_id,
                    "episode_id": paired_trial_id,
                    "paired_trial_id": paired_trial_id,
                    "prediction_step": prediction_step,
                    "initial_state_id": initial_state_id,
                    "episode_seed": expected_seed,
                }
                workload_path = _validate_workload(
                    task_dir,
                    record,
                    context=prediction_context,
                    expected_identity=identity,
                    expected_origin=expected_origin,
                )
                _require(workload_path not in workload_paths, f"{prediction_context}: duplicate workload path")
                workload_paths.add(workload_path)
                if expected_origin == "COLD":
                    cold_workloads += 1
                else:
                    warm_workloads += 1
            else:
                _validate_unsampled_workload(record, prediction_context)
            total_predictions += 1

    _require(result.get("total_successes") == successes, f"task {task_id}: total_successes mismatch")
    expected_workload_count = CALIBRATION_EPISODES_PER_TASK * CALIBRATION_WORKLOADS_PER_EPISODE
    _require(len(workload_paths) == expected_workload_count, f"task {task_id}: workload count mismatch")
    disk_workloads = {path.resolve() for path in (task_dir / "workloads").glob("*.pt")}
    _require(disk_workloads == workload_paths, f"task {task_id}: unreferenced or missing workload shards")
    return {
        "task_id": task_id,
        "episodes": CALIBRATION_EPISODES_PER_TASK,
        "successes": successes,
        "predictions": total_predictions,
        "actual_warm_predictions": actual_warm_predictions,
        "workload_shards": len(workload_paths),
        "cold_workloads": cold_workloads,
        "actual_warm_workloads": warm_workloads,
        "initial_state_ids": list(expected_state_ids),
    }


def validate_calibration_run(
    run_root: str,
    manifest_path: str,
    *,
    base_seed: int = 7,
    task_ids: Iterable[int] = CALIBRATION_TASK_IDS,
) -> Dict[str, Any]:
    """Validate selected formal calibration artifacts."""

    normalized_task_ids = tuple(int(task_id) for task_id in task_ids)
    _require(normalized_task_ids, "at least one task ID is required")
    _require(len(normalized_task_ids) == len(set(normalized_task_ids)), "task IDs must be unique")
    _require(all(task_id in CALIBRATION_TASK_IDS for task_id in normalized_task_ids), "task IDs must be in 0..9")
    manifest, manifest_sha256 = load_protocol_manifest(
        manifest_path, require_source_file_hashes=True
    )
    root = Path(run_root)
    _require(root.is_dir(), f"calibration run root does not exist: {root}")
    task_reports = [
        _validate_task(root, task_id, manifest, manifest_sha256, base_seed=base_seed)
        for task_id in normalized_task_ids
    ]
    return {
        "schema_version": 1,
        "valid": True,
        "phase": "calibration",
        "manifest_path": str(Path(manifest_path)),
        "manifest_sha256": manifest_sha256,
        "base_seed": int(base_seed),
        "task_ids": list(normalized_task_ids),
        "complete_10_task_gate": normalized_task_ids == CALIBRATION_TASK_IDS,
        "totals": {
            "tasks": len(task_reports),
            "episodes": sum(report["episodes"] for report in task_reports),
            "successes": sum(report["successes"] for report in task_reports),
            "predictions": sum(report["predictions"] for report in task_reports),
            "actual_warm_predictions": sum(report["actual_warm_predictions"] for report in task_reports),
            "workload_shards": sum(report["workload_shards"] for report in task_reports),
            "cold_workloads": sum(report["cold_workloads"] for report in task_reports),
            "actual_warm_workloads": sum(report["actual_warm_workloads"] for report in task_reports),
        },
        "tasks": task_reports,
    }
