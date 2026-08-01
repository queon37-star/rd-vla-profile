"""Fail-closed validation for the 10-task, 3-episode calibration smoke gate."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from experiments.robot.libero.evaluation_protocol import (
    derive_paired_episode_seed,
    load_protocol_manifest,
)


SMOKE_TASK_IDS = tuple(range(10))
SMOKE_EPISODES_PER_TASK = 3


class SmokeValidationError(ValueError):
    """Raised when a smoke artifact violates the frozen collection contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeValidationError(message)


def _load_json(path: Path) -> Dict[str, Any]:
    _require(path.is_file(), f"missing JSON result: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SmokeValidationError(f"invalid JSON result {path}: {exc}") from exc
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
            raise SmokeValidationError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
        _require(isinstance(value, dict), f"JSONL record must be an object at {path}:{line_number}")
        records.append(value)
    _require(records, f"step log is empty: {path}")
    return records


def _validate_shadow_trace(record: Mapping[str, Any], context: str) -> None:
    _require(record.get("shadow_full_depth_enabled") is True, f"{context}: shadow_full_depth_enabled != true")
    _require(record.get("shadow_trace_complete") is True, f"{context}: shadow_trace_complete != true")
    _require(record.get("shadow_error") is None, f"{context}: shadow_error must be null")
    trace = record.get("shadow_trace")
    _require(isinstance(trace, list) and trace, f"{context}: shadow_trace must be non-empty")
    max_iter = record.get("max_recurrent_iteration")
    _require(isinstance(max_iter, int) and max_iter >= 2, f"{context}: invalid max_recurrent_iteration")
    _require(len(trace) == max_iter, f"{context}: shadow_trace must cover every recurrence iteration")
    baseline_k = record.get("K_t")
    _require(isinstance(baseline_k, int) and 1 <= baseline_k <= max_iter, f"{context}: invalid K_t")

    for expected_k, point in enumerate(trace, start=1):
        _require(isinstance(point, Mapping), f"{context}: shadow_trace[{expected_k}] must be an object")
        _require(point.get("k") == expected_k, f"{context}: shadow iterations must be contiguous")
        expected_phase = "production" if expected_k <= baseline_k else "shadow_tail"
        _require(point.get("phase") == expected_phase, f"{context}: shadow_trace[{expected_k}] phase mismatch")
        _require(point.get("state_finite") is True, f"{context}: non-finite shadow state at k={expected_k}")
        _require(point.get("output_finite") is True, f"{context}: non-finite shadow output at k={expected_k}")

    snapshot = record.get("shadow_production_snapshot")
    _require(isinstance(snapshot, Mapping), f"{context}: shadow_production_snapshot is required")
    _require(snapshot.get("K_t") == baseline_k, f"{context}: snapshot K_t mismatch")
    _require(snapshot.get("terminal_iteration") == baseline_k, f"{context}: snapshot terminal iteration mismatch")
    _require(
        snapshot.get("cached_final_output_reused") is True,
        f"{context}: snapshot cached_final_output_reused must be true",
    )


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
    _require(protocol.get("phase") == "smoke", f"task {task_id}: result phase must be smoke")
    _require(protocol.get("manifest_sha256") == manifest_sha256, f"task {task_id}: manifest SHA-256 mismatch")
    _require(protocol.get("task_suite_name") == "libero_spatial", f"task {task_id}: wrong task suite")
    _require(
        protocol.get("num_trials_per_task") == SMOKE_EPISODES_PER_TASK,
        f"task {task_id}: result must contain three trials",
    )
    _require(protocol.get("paired_rng") is True, f"task {task_id}: paired_rng must be true")
    _require(result.get("total_episodes") == SMOKE_EPISODES_PER_TASK, f"task {task_id}: total_episodes != 3")

    result_tasks = result.get("tasks")
    _require(isinstance(result_tasks, Mapping) and len(result_tasks) == 1, f"task {task_id}: expected one task result")
    episode_stats = next(iter(result_tasks.values()))
    _require(isinstance(episode_stats, list), f"task {task_id}: episode stats must be a list")
    _require(len(episode_stats) == SMOKE_EPISODES_PER_TASK, f"task {task_id}: expected three episode stats")

    records_by_episode: Dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        episode_id = record.get("episode_id")
        _require(
            isinstance(episode_id, int) and 0 <= episode_id < SMOKE_EPISODES_PER_TASK,
            f"task {task_id}: invalid step episode_id {episode_id!r}",
        )
        records_by_episode[episode_id].append(record)

    expected_state_ids = task_manifest["partitions"]["calibration"][:SMOKE_EPISODES_PER_TASK]
    successes = 0
    actual_warm_predictions = 0
    total_predictions = 0
    for paired_trial_id, episode_stat in enumerate(episode_stats):
        context = f"task {task_id} trial {paired_trial_id}"
        _require(isinstance(episode_stat, Mapping), f"{context}: episode stat must be an object")
        initial_state_id = expected_state_ids[paired_trial_id]
        expected_seed = derive_paired_episode_seed(
            base_seed=base_seed,
            phase="smoke",
            task_suite_name="libero_spatial",
            task_id=task_id,
            initial_state_id=initial_state_id,
            paired_trial_id=paired_trial_id,
        )
        expected_stat_fields = {
            "episode": paired_trial_id,
            "evaluation_protocol_phase": "smoke",
            "paired_trial_id": paired_trial_id,
            "initial_state_id": initial_state_id,
            "episode_seed": expected_seed,
            "initial_state_manifest_sha256": manifest_sha256,
            "smoke_excluded_from_fitting": True,
        }
        for field, expected in expected_stat_fields.items():
            _require(episode_stat.get(field) == expected, f"{context}: episode field {field!r} mismatch")

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
        if episode_stat.get("success") is True:
            successes += 1

        episode_warm_count = 0
        for prediction_step, record in enumerate(episode_records):
            prediction_context = f"{context} prediction {prediction_step}"
            expected_step_fields = {
                "task_id": task_id,
                "episode_id": paired_trial_id,
                "evaluation_protocol_phase": "smoke",
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
                "smoke_excluded_from_fitting": True,
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
            _validate_shadow_trace(record, prediction_context)
            warm_used = record.get("warm_start_used") is True
            if prediction_step == 0:
                _require(not warm_used, f"{prediction_context}: first prediction cannot use warm cache")
            elif warm_used:
                episode_warm_count += 1
                actual_warm_predictions += 1
            total_predictions += 1
        if len(episode_records) > 1:
            _require(episode_warm_count > 0, f"{context}: midpoint warm cache was never accepted")

    _require(result.get("total_successes") == successes, f"task {task_id}: total_successes mismatch")

    return {
        "task_id": task_id,
        "episodes": SMOKE_EPISODES_PER_TASK,
        "successes": successes,
        "predictions": total_predictions,
        "actual_warm_predictions": actual_warm_predictions,
        "initial_state_ids": list(expected_state_ids),
    }


def validate_smoke_run(
    run_root: str,
    manifest_path: str,
    *,
    base_seed: int = 7,
    task_ids: Iterable[int] = SMOKE_TASK_IDS,
) -> Dict[str, Any]:
    """Validate selected task artifacts and return a machine-readable report."""

    normalized_task_ids = tuple(int(task_id) for task_id in task_ids)
    _require(normalized_task_ids, "at least one task ID is required")
    _require(len(normalized_task_ids) == len(set(normalized_task_ids)), "task IDs must be unique")
    _require(all(task_id in SMOKE_TASK_IDS for task_id in normalized_task_ids), "task IDs must be in 0..9")
    manifest, manifest_sha256 = load_protocol_manifest(
        manifest_path, require_source_file_hashes=True
    )
    root = Path(run_root)
    _require(root.is_dir(), f"smoke run root does not exist: {root}")

    task_reports = [
        _validate_task(root, task_id, manifest, manifest_sha256, base_seed=base_seed)
        for task_id in normalized_task_ids
    ]
    return {
        "schema_version": 1,
        "valid": True,
        "phase": "smoke",
        "manifest_path": str(Path(manifest_path)),
        "manifest_sha256": manifest_sha256,
        "base_seed": int(base_seed),
        "task_ids": list(normalized_task_ids),
        "complete_10_task_gate": normalized_task_ids == SMOKE_TASK_IDS,
        "totals": {
            "tasks": len(task_reports),
            "episodes": sum(report["episodes"] for report in task_reports),
            "successes": sum(report["successes"] for report in task_reports),
            "predictions": sum(report["predictions"] for report in task_reports),
            "actual_warm_predictions": sum(report["actual_warm_predictions"] for report in task_reports),
        },
        "tasks": task_reports,
    }
