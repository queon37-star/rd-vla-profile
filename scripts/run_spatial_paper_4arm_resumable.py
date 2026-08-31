#!/usr/bin/env python3
"""Resumable launcher for the frozen LIBERO-Spatial 4-arm paper evaluation.

This companion launcher keeps the frozen method configuration from
run_spatial_paper_4arm.py and adds two operational safeguards:

1. task-level resume for the 10 x 50 paper run, and
2. a real runtime smoke mode (tasks 0 and 4, one episode each, all arms).

Task-level resume treats results.json as the completion checkpoint. If a run
stops inside a task, prediction records from that incomplete task are removed
before the task is rerun, while already completed tasks are preserved.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch

import experiments.robot.libero.evaluation_protocol as protocol
import experiments.robot.libero.run_libero_eval as base
import prismatic.models.action_heads as action_heads_module
import scripts.run_spatial_paper_4arm as frozen
from prismatic.models.action_delta_gate import (
    ACTION_DELTA_NONCONVERGENCE_THRESHOLD,
    load_action_delta_gate_artifact,
)


PAPER_PHASE = frozen.PAPER_PHASE
SMOKE_PHASE = "paper_smoke"
PAPER_ARMS = frozen.PAPER_ARMS
PAPER_TASK_IDS = frozen.PAPER_TASK_IDS
PAPER_EPISODES_PER_TASK = frozen.PAPER_EPISODES_PER_TASK
SMOKE_TASK_IDS = (0, 4)
SMOKE_EPISODES_PER_TASK = 1
PAPER_PROTOCOL_NAME = "libero-spatial-50x10-4arm-v1"
SMOKE_PROTOCOL_NAME = "libero-spatial-runtime-smoke-t0-t4-1ep-4arm-v1"


def _resolve_runner_trials(
    *,
    manifest,
    phase,
    task_id,
    initial_states,
    base_seed,
    initial_state_file_path=None,
):
    """Resolve frozen paper or smoke trials while retaining manifest checks."""
    if phase not in {PAPER_PHASE, SMOKE_PHASE}:
        return protocol.resolve_phase_trials(
            manifest=manifest,
            phase=phase,
            task_id=task_id,
            initial_states=initial_states,
            base_seed=base_seed,
            initial_state_file_path=initial_state_file_path,
        )

    # Reuse the validated calibration resolver once so the manifest schema,
    # initial-state tensor hash, and source-file hash remain authoritative.
    protocol.resolve_phase_trials(
        manifest=manifest,
        phase="calibration",
        task_id=task_id,
        initial_states=initial_states,
        base_seed=base_seed,
        initial_state_file_path=initial_state_file_path,
    )

    task_entry = manifest["tasks"][str(int(task_id))]
    all_state_ids = (
        list(task_entry["partitions"]["calibration"])
        + list(task_entry["partitions"]["screening"])
        + list(task_entry["partitions"]["final"])
    )
    if len(all_state_ids) != 50 or len(set(all_state_ids)) != 50:
        raise ValueError(
            f"task {task_id} must expose exactly 50 unique frozen official states"
        )

    if phase == SMOKE_PHASE:
        state_ids = all_state_ids[:SMOKE_EPISODES_PER_TASK]
        partition = "runtime_smoke_official_prefix"
    else:
        state_ids = all_state_ids
        partition = "all_official_50"

    trials = []
    for paired_trial_id, initial_state_id in enumerate(state_ids):
        trials.append(
            protocol.EpisodeTrial(
                phase=phase,
                partition=partition,
                paired_trial_id=paired_trial_id,
                initial_state_id=int(initial_state_id),
                episode_seed=protocol.derive_paired_episode_seed(
                    base_seed=base_seed,
                    phase=phase,
                    task_suite_name=manifest["task_suite_name"],
                    task_id=int(task_id),
                    initial_state_id=int(initial_state_id),
                    paired_trial_id=paired_trial_id,
                ),
                smoke_excluded_from_fitting=(phase == SMOKE_PHASE),
            )
        )
    return tuple(trials)


@contextmanager
def _runtime_patches(*, save_videos: bool):
    """Install paper-only protocol/runtime adapters and restore them afterward."""
    original_resolver = base.resolve_phase_trials
    original_video_stats = base.save_rollout_video_with_stats
    original_video = base.save_rollout_video
    original_deferred_validator = (
        action_heads_module.validate_action_delta_deferred_backfill_configuration
    )

    def deferred_validator_without_diagnostic_timing(**kwargs):
        # The development validator requires profile_coda_cost=True only for
        # synchronized diagnostic accounting. The scheduler does not consume
        # those timings, so keep every other check while the paper run remains
        # profile_coda_cost=False.
        forwarded = dict(kwargs)
        forwarded["profile_coda_cost"] = True
        return original_deferred_validator(**forwarded)

    base.resolve_phase_trials = _resolve_runner_trials
    action_heads_module.validate_action_delta_deferred_backfill_configuration = (
        deferred_validator_without_diagnostic_timing
    )

    if not save_videos:
        base.save_rollout_video_with_stats = lambda *args, **kwargs: None
        base.save_rollout_video = lambda *args, **kwargs: None

    try:
        yield
    finally:
        base.resolve_phase_trials = original_resolver
        base.save_rollout_video_with_stats = original_video_stats
        base.save_rollout_video = original_video
        action_heads_module.validate_action_delta_deferred_backfill_configuration = (
            original_deferred_validator
        )


def _build_arm_config(
    *,
    arm: str,
    checkpoint: Path,
    manifest_path: Path,
    artifact_path: Path | None,
    artifact_sha256: str | None,
    output_root: Path,
    seed: int,
    smoke: bool,
) -> base.GenerateConfig:
    cfg = frozen._build_arm_config(
        arm=arm,
        checkpoint=checkpoint,
        manifest_path=manifest_path,
        artifact_path=artifact_path,
        artifact_sha256=artifact_sha256,
        output_root=output_root,
        seed=seed,
    )
    if smoke:
        cfg.num_trials_per_task = SMOKE_EPISODES_PER_TASK
        cfg.evaluation_protocol_phase = SMOKE_PHASE
        cfg.run_id_note = f"paper_spatial_runtime_smoke_{arm}"
        cfg.save_version = f"paper-spatial-runtime-smoke-{arm}"
    return cfg


def _validate_arm_config(
    cfg: base.GenerateConfig,
    *,
    arm: str,
    smoke: bool,
) -> None:
    expected_phase = SMOKE_PHASE if smoke else PAPER_PHASE
    expected_trials = SMOKE_EPISODES_PER_TASK if smoke else PAPER_EPISODES_PER_TASK

    if cfg.task_suite_name != base.TaskSuite.LIBERO_SPATIAL:
        raise ValueError("paper runner is LIBERO Spatial-only")
    if cfg.evaluation_protocol_phase != expected_phase:
        raise ValueError(
            f"expected evaluation_protocol_phase={expected_phase!r}, "
            f"got {cfg.evaluation_protocol_phase!r}"
        )
    if int(cfg.num_trials_per_task) != expected_trials:
        raise ValueError(
            f"expected {expected_trials} episode(s) per task, "
            f"got {cfg.num_trials_per_task}"
        )
    if cfg.initial_states_path != "DEFAULT":
        raise ValueError("runner requires official DEFAULT initial states")
    if not Path(cfg.initial_state_manifest_path).is_file():
        raise ValueError("runner requires the frozen official-50 manifest")
    if not cfg.reset_rng_each_episode:
        raise ValueError("runner requires paired per-episode RNG reset")
    if cfg.recurrence_strategy != "adjacent_action_mse":
        raise ValueError("runner requires adjacent_action_mse recurrence")
    if (
        float(cfg.recurrence_kl_thresh) != 0.001
        or int(cfg.recurrence_max_iter) != 32
    ):
        raise ValueError("runner freezes action-MSE threshold=0.001 and max_iter=32")
    if not cfg.use_cached_final_output:
        raise ValueError("runner requires cached terminal output")
    if cfg.use_latent_precheck or cfg.latent_precheck_mode != "off":
        raise ValueError("legacy latent pre-check must stay disabled")
    if cfg.num_exec_actions != 5:
        raise ValueError("runner freezes num_exec_actions=5")
    if cfg.profile_coda_cost:
        raise ValueError("runner requires profile_coda_cost=False")

    if arm in {"warm_start", "combined"}:
        if not cfg.use_warm_start or cfg.warm_start_source != "midpoint":
            raise ValueError(f"{arm} requires midpoint warm-start")
        if int(cfg.warm_start_min_iter) != 2:
            raise ValueError(f"{arm} requires warm_start_min_iter=2")
    elif cfg.use_warm_start:
        raise ValueError(f"{arm} must use cold initialization")

    if arm in {"ldce", "combined"}:
        if not cfg.use_action_delta_deferred_backfill_filter:
            raise ValueError(f"{arm} requires LDCE deferred/backfill")
        if cfg.action_delta_deferred_runtime_policy != "lazy_prefix_exact":
            raise ValueError("paper LDCE freezes lazy_prefix_exact")
        if cfg.action_delta_deferred_scorer_backend != "eager":
            raise ValueError("paper LDCE freezes eager scorer")
        if int(cfg.action_delta_gate_min_terminal_iter) != 2:
            raise ValueError("paper LDCE freezes min_terminal_iter=2")
        if not Path(cfg.action_delta_gate_artifact_path).exists():
            raise ValueError("paper LDCE artifact path does not exist")
    elif cfg.use_action_delta_deferred_backfill_filter:
        raise ValueError(f"{arm} must not enable LDCE")

    if arm == "ldce" and not cfg.action_delta_deferred_apply_to_cold:
        raise ValueError("LDCE-only arm must apply the frozen predictor to cold inference")
    if arm == "combined" and cfg.action_delta_deferred_apply_to_cold:
        raise ValueError("Combined must not force the cold-only LDCE mode")


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _expected_total(task_ids: Sequence[int], episodes_per_task: int) -> int:
    return len(tuple(task_ids)) * int(episodes_per_task)


def _arm_is_complete(
    result_path: Path,
    *,
    expected_total_episodes: int,
    expected_task_count: int,
) -> bool:
    if not result_path.is_file():
        return False
    try:
        payload = _load_json(result_path)
    except Exception:
        return False
    return bool(
        payload.get("completed")
        and int(payload.get("total_episodes", -1)) == expected_total_episodes
        and len(payload.get("tasks", {})) == expected_task_count
    )


def _sanitize_step_log(
    step_log_path: Path,
    *,
    completed_task_ids: set[int],
) -> int:
    """Keep only completed-task records before rerunning an interrupted task."""
    if not completed_task_ids:
        if step_log_path.exists():
            step_log_path.unlink()
        return 0
    if not step_log_path.is_file():
        raise RuntimeError(
            "Cannot resume completed tasks because predictions.jsonl is missing. "
            "Use --overwrite to restart the arm."
        )

    kept_lines = []
    with step_log_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except Exception as exc:
                raise RuntimeError(
                    f"Malformed JSONL at {step_log_path}:{line_number}"
                ) from exc
            task_id = record.get("task_id")
            if task_id is None:
                raise RuntimeError(
                    f"Missing task_id at {step_log_path}:{line_number}; "
                    "cannot resume safely"
                )
            if int(task_id) in completed_task_ids:
                kept_lines.append(json.dumps(record, ensure_ascii=False))

    tmp = step_log_path.with_suffix(step_log_path.suffix + ".resume.tmp")
    tmp.write_text(
        ("\n".join(kept_lines) + "\n") if kept_lines else "",
        encoding="utf-8",
    )
    tmp.replace(step_log_path)
    return len(kept_lines)


def _build_protocol_record(
    *,
    cfg: base.GenerateConfig,
    arm: str,
    protocol_name: str,
    task_ids: Sequence[int],
    episodes_per_task: int,
    manifest_sha256: str,
    source_commit: str,
    checkpoint_meta,
    smoke: bool,
) -> dict:
    return {
        "name": protocol_name,
        "arm": arm,
        "phase": cfg.evaluation_protocol_phase,
        "task_ids": [int(value) for value in task_ids],
        "episodes_per_task": int(episodes_per_task),
        "expected_total_episodes": _expected_total(task_ids, episodes_per_task),
        "paired_rng": True,
        "runtime_smoke": bool(smoke),
        "excluded_from_paper_statistics": bool(smoke),
        "initial_state_manifest_path": cfg.initial_state_manifest_path,
        "initial_state_manifest_sha256": manifest_sha256,
        "source_commit": source_commit,
        "checkpoint": checkpoint_meta,
        "action_mse_threshold": 0.001,
        "recurrence_max_iter": 32,
        "num_exec_actions": 5,
        "warm_start": {
            "enabled": bool(cfg.use_warm_start),
            "source": cfg.warm_start_source if cfg.use_warm_start else None,
        },
        "ldce": {
            "enabled": bool(cfg.use_action_delta_deferred_backfill_filter),
            "shared_spatial_predictor": bool(
                cfg.use_action_delta_deferred_backfill_filter
            ),
            "nonconvergence_threshold": (
                float(ACTION_DELTA_NONCONVERGENCE_THRESHOLD)
                if cfg.use_action_delta_deferred_backfill_filter
                else None
            ),
            "runtime_policy": (
                cfg.action_delta_deferred_runtime_policy
                if cfg.use_action_delta_deferred_backfill_filter
                else None
            ),
            "apply_to_cold": (
                bool(cfg.action_delta_deferred_apply_to_cold)
                if cfg.use_action_delta_deferred_backfill_filter
                else None
            ),
            "artifact_sha256": (
                cfg.action_delta_gate_expected_sha256.lower()
                if cfg.use_action_delta_deferred_backfill_filter
                else None
            ),
        },
        "latency_scope_note": (
            "latency_ms is the existing CUDA-synchronized get_action timer; "
            "profile_coda_cost=False avoids profiling-only per-kernel synchronizations"
        ),
        "paper_runtime_adapter": {
            "bypasses_only_deferred_profile_coda_cost_requirement": True,
            "algorithmic_scheduler_logic_changed": False,
        },
    }


def _validate_resume_protocol(existing: Mapping, expected: Mapping) -> None:
    """Fail closed if a resume checkpoint belongs to a different experiment."""
    keys = (
        "name",
        "arm",
        "phase",
        "task_ids",
        "episodes_per_task",
        "expected_total_episodes",
        "initial_state_manifest_sha256",
        "source_commit",
        "action_mse_threshold",
        "recurrence_max_iter",
        "num_exec_actions",
        "runtime_smoke",
    )
    for key in keys:
        if existing.get(key) != expected.get(key):
            raise RuntimeError(
                f"Resume protocol mismatch for {key}: "
                f"existing={existing.get(key)!r}, expected={expected.get(key)!r}. "
                "Use --overwrite to start a fresh arm."
            )
    if existing.get("warm_start") != expected.get("warm_start"):
        raise RuntimeError("Resume warm-start configuration mismatch")
    if existing.get("ldce") != expected.get("ldce"):
        raise RuntimeError("Resume LDCE configuration mismatch")


def _resume_completed_tasks(
    *,
    full_results: dict,
    task_suite,
    task_ids: Sequence[int],
    episodes_per_task: int,
    step_log_path: Path,
) -> tuple[set[int], int, int]:
    tasks_payload = full_results.get("tasks")
    if not isinstance(tasks_payload, dict):
        raise RuntimeError("Resume results.json has no valid tasks object")

    allowed_names = {
        int(task_id): task_suite.get_task(int(task_id)).name for task_id in task_ids
    }
    reverse_names = {name: task_id for task_id, name in allowed_names.items()}
    unexpected = sorted(set(tasks_payload) - set(reverse_names))
    if unexpected:
        raise RuntimeError(
            f"Resume results contain tasks outside this run: {unexpected}"
        )

    completed_task_ids: set[int] = set()
    total_episodes = 0
    total_successes = 0
    for task_name, stats in tasks_payload.items():
        if not isinstance(stats, list) or len(stats) != int(episodes_per_task):
            raise RuntimeError(
                f"Resume task {task_name!r} is not an exact completed task: "
                f"expected {episodes_per_task} episode records"
            )
        task_id = int(reverse_names[task_name])
        completed_task_ids.add(task_id)
        total_episodes += len(stats)
        total_successes += sum(bool(record.get("success")) for record in stats)

    retained_prediction_records = _sanitize_step_log(
        step_log_path,
        completed_task_ids=completed_task_ids,
    )
    full_results["resume"] = {
        "task_level_resume": True,
        "completed_task_ids_at_resume": sorted(completed_task_ids),
        "retained_prediction_records": int(retained_prediction_records),
    }
    return completed_task_ids, total_episodes, total_successes


def _run_arm_resumable(
    cfg: base.GenerateConfig,
    *,
    arm: str,
    save_videos: bool,
    resume_existing: bool,
    smoke: bool,
    task_ids: Sequence[int],
    episodes_per_task: int,
    protocol_name: str,
) -> dict:
    _validate_arm_config(cfg, arm=arm, smoke=smoke)
    base.set_seed_everywhere(cfg.seed)

    action_delta_manifest = None
    action_delta_payload = None
    if cfg.use_action_delta_deferred_backfill_filter:
        action_delta_manifest, action_delta_payload = load_action_delta_gate_artifact(
            cfg.action_delta_gate_artifact_path,
            expected_sha256=cfg.action_delta_gate_expected_sha256.lower(),
        )

    model, action_head, proprio_projector, processor = base.initialize_model(cfg)
    resize_size = base.get_image_resize_size(cfg)
    log_file, _, run_id = base.setup_logging(cfg)
    base.RDVLAProfiler.set_enabled(False)
    base.RDVLAProfiler.set_timing_enabled(False)
    base.configure_recurrent_convergence_paths(cfg, run_id)

    result_path = Path(cfg.json_log_file)
    step_log_path = Path(base.get_step_log_file(cfg))
    summary_path = Path(cfg.recurrent_convergence_summary_file)
    progress_path = result_path.parent / "progress.json"

    _, manifest_sha256 = base.load_protocol_manifest(
        cfg.initial_state_manifest_path,
        require_source_file_hashes=True,
    )
    source_commit = base.current_source_commit()
    checkpoint_meta = base.checkpoint_identity(Path(cfg.pretrained_checkpoint))

    benchmark_dict = base.benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    if task_suite.n_tasks != 10:
        raise RuntimeError(f"Expected 10 LIBERO-Spatial tasks, got {task_suite.n_tasks}")

    expected_protocol = _build_protocol_record(
        cfg=cfg,
        arm=arm,
        protocol_name=protocol_name,
        task_ids=task_ids,
        episodes_per_task=episodes_per_task,
        manifest_sha256=manifest_sha256,
        source_commit=source_commit,
        checkpoint_meta=checkpoint_meta,
        smoke=smoke,
    )

    if resume_existing:
        if not result_path.is_file():
            raise RuntimeError("resume_existing=True but results.json is missing")
        full_results = _load_json(result_path)
        _validate_resume_protocol(
            full_results.get("paper_protocol", {}), expected_protocol
        )
        completed_task_ids, total_episodes, total_successes = (
            _resume_completed_tasks(
                full_results=full_results,
                task_suite=task_suite,
                task_ids=task_ids,
                episodes_per_task=episodes_per_task,
                step_log_path=step_log_path,
            )
        )
        if summary_path.exists():
            summary_path.unlink()
        base.log_message(
            "Task-level resume: "
            f"completed_tasks={sorted(completed_task_ids)}, "
            f"episodes={total_episodes}, successes={total_successes}",
            log_file,
        )
    else:
        for stale_path in (result_path, step_log_path, summary_path, progress_path):
            if stale_path.exists():
                stale_path.unlink()
        completed_task_ids = set()
        total_episodes = 0
        total_successes = 0
        full_results = {
            "schema_version": 2,
            "paper_protocol": expected_protocol,
            "config": asdict(cfg),
            "tasks": {},
        }
        if action_delta_manifest is not None:
            full_results["action_delta_artifact"] = {
                "artifact_sha256": action_delta_manifest["artifact_sha256"],
                "model_type": action_delta_manifest["model_type"],
                "outer_fold_metadata": action_delta_manifest["outer_fold"],
                "held_out_task_ids_metadata": action_delta_manifest[
                    "held_out_task_ids"
                ],
                "weights_shared_across_spatial_tasks": True,
            }

    expected_total = _expected_total(task_ids, episodes_per_task)
    base.log_message(
        f"Run arm={arm}: tasks={list(task_ids)}, "
        f"episodes_per_task={episodes_per_task}, expected_total={expected_total}",
        log_file,
    )
    base.log_message(
        f"Frozen manifest sha256={manifest_sha256}; source_commit={source_commit}",
        log_file,
    )

    try:
        with _runtime_patches(save_videos=save_videos):
            for task_id in task_ids:
                task_id = int(task_id)
                if task_id in completed_task_ids:
                    base.log_message(
                        f"[resume] skipping completed task {task_id}", log_file
                    )
                    continue

                action_head.clear_scalar_task_policy()
                if action_delta_payload is not None:
                    device = next(action_head.parameters()).device
                    prepared_gate = frozen._prepare_shared_spatial_predictor(
                        action_delta_payload,
                        device=device,
                        task_id=task_id,
                    )
                    prepared_scorer = action_head.configure_action_delta_gate(
                        prepared_gate,
                        deferred_scorer_backend=cfg.action_delta_deferred_scorer_backend,
                    )
                    base.log_message(
                        "Bound shared Spatial Action-Delta predictor: "
                        f"task={task_id}, "
                        f"artifact_fold_metadata={prepared_gate.outer_fold}, "
                        f"artifact_threshold={prepared_gate.threshold:.12f}, "
                        f"LDCE_q={ACTION_DELTA_NONCONVERGENCE_THRESHOLD:.12f}, "
                        f"runtime_policy={cfg.action_delta_deferred_runtime_policy}, "
                        "compile_setup_ms="
                        f"{prepared_scorer.compile_setup_ms if prepared_scorer is not None else 0.0:.3f}",
                        log_file,
                    )
                else:
                    action_head.clear_action_delta_gate()

                total_episodes, total_successes, task_stats = base.run_task(
                    cfg,
                    task_suite,
                    task_id,
                    model,
                    resize_size,
                    processor,
                    action_head,
                    proprio_projector,
                    total_episodes,
                    total_successes,
                    log_file,
                    cfg.save_version,
                    None,
                    None,
                    None,
                    None,
                    source_commit=source_commit,
                )
                task = task_suite.get_task(task_id)
                full_results["tasks"][task.name] = task_stats
                completed_task_ids.add(task_id)
                full_results["total_episodes"] = total_episodes
                full_results["total_successes"] = total_successes
                full_results["overall_success_rate"] = (
                    float(total_successes) / float(total_episodes)
                    if total_episodes
                    else 0.0
                )
                full_results["completed"] = False
                _write_json(result_path, full_results)
                _write_json(
                    progress_path,
                    {
                        "schema_version": 1,
                        "arm": arm,
                        "runtime_smoke": bool(smoke),
                        "completed_task_ids": sorted(completed_task_ids),
                        "remaining_task_ids": [
                            int(value)
                            for value in task_ids
                            if int(value) not in completed_task_ids
                        ],
                        "total_episodes": total_episodes,
                        "expected_total_episodes": expected_total,
                    },
                )

        if total_episodes != expected_total:
            raise RuntimeError(
                f"Arm {arm} completed {total_episodes} episodes; "
                f"expected {expected_total}"
            )
        if completed_task_ids != {int(value) for value in task_ids}:
            raise RuntimeError(
                f"Arm {arm} completed tasks {sorted(completed_task_ids)}; "
                f"expected {list(task_ids)}"
            )

        convergence_summary = base.save_recurrent_convergence_summary(
            cfg, full_results, log_file
        )
        if convergence_summary is not None:
            full_results["recurrent_convergence_summary"] = convergence_summary

        full_results["total_episodes"] = total_episodes
        full_results["total_successes"] = total_successes
        full_results["overall_success_rate"] = (
            float(total_successes) / float(total_episodes)
        )
        full_results["completed"] = True
        _write_json(result_path, full_results)
        _write_json(
            progress_path,
            {
                "schema_version": 1,
                "arm": arm,
                "runtime_smoke": bool(smoke),
                "completed": True,
                "completed_task_ids": sorted(completed_task_ids),
                "remaining_task_ids": [],
                "total_episodes": total_episodes,
                "expected_total_episodes": expected_total,
            },
        )
        base.log_message(
            f"Completed arm={arm}: successes={total_successes}/{total_episodes} "
            f"({100.0 * full_results['overall_success_rate']:.1f}%)",
            log_file,
        )
        return full_results
    finally:
        if log_file:
            log_file.close()
        del processor, proprio_projector, action_head, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _parse_arms(values: Iterable[str]) -> tuple[str, ...]:
    return frozen._parse_arms(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen Baseline/Warm/LDCE/Combined LIBERO-Spatial "
            "evaluation with task-level resume."
        )
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/12_24-24_24_Spatial_40k",
    )
    parser.add_argument(
        "--initial-state-manifest",
        default=(
            "experiments/robot/libero/manifests/"
            "libero_spatial_official_50_v1.json"
        ),
    )
    parser.add_argument(
        "--action-delta-artifact",
        default=os.environ.get("RDVLA_ACTION_DELTA_ARTIFACT", ""),
        help=(
            "Frozen Spatial Action-Delta artifact file or directory. "
            "Can also be supplied via RDVLA_ACTION_DELTA_ARTIFACT."
        ),
    )
    parser.add_argument(
        "--action-delta-sha256",
        default=os.environ.get("RDVLA_ACTION_DELTA_SHA256", ""),
        help="Optional artifact SHA-256; otherwise read adjacent manifest.json.",
    )
    parser.add_argument(
        "--output-root",
        default="benchmark_results/paper_spatial_50x10",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--arms",
        nargs="+",
        default=list(PAPER_ARMS),
        help="Subset/order of arms. Default: baseline warm_start ldce combined.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an incomplete arm at task granularity. A partially written "
            "current task is discarded and rerun; completed tasks are retained."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard existing outputs for requested incomplete arms.",
    )
    parser.add_argument(
        "--save-videos",
        action="store_true",
        help="Save rollout MP4s; disabled by default.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the execution plan/configuration only.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run a real runtime smoke: tasks 0 and 4, one episode each per arm "
            "(8 episodes for all four arms). Outputs go under OUTPUT_ROOT/smoke "
            "and are excluded from paper statistics."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.resume and args.overwrite:
        raise ValueError("Choose only one of --resume or --overwrite")

    arms = _parse_arms(args.arms)
    checkpoint = Path(args.checkpoint)
    manifest_path = Path(args.initial_state_manifest)
    base_output_root = Path(args.output_root)
    output_root = base_output_root / "smoke" if args.smoke else base_output_root

    task_ids = SMOKE_TASK_IDS if args.smoke else PAPER_TASK_IDS
    episodes_per_task = (
        SMOKE_EPISODES_PER_TASK if args.smoke else PAPER_EPISODES_PER_TASK
    )
    protocol_name = SMOKE_PROTOCOL_NAME if args.smoke else PAPER_PROTOCOL_NAME
    expected_episodes_per_arm = _expected_total(task_ids, episodes_per_task)

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Initial-state manifest does not exist: {manifest_path}"
        )

    needs_ldce = any(arm in {"ldce", "combined"} for arm in arms)
    artifact_path = (
        Path(args.action_delta_artifact) if args.action_delta_artifact else None
    )
    artifact_sha256 = None
    if needs_ldce:
        if artifact_path is None or not artifact_path.exists():
            raise FileNotFoundError(
                "LDCE/Combined requires --action-delta-artifact or "
                "RDVLA_ACTION_DELTA_ARTIFACT"
            )
        artifact_sha256 = frozen._artifact_sha256(
            artifact_path,
            args.action_delta_sha256 or None,
        )
        # Dry-run should still verify the artifact bytes and tensor contract.
        load_action_delta_gate_artifact(
            artifact_path,
            expected_sha256=artifact_sha256,
        )

    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "launcher_state.json"
    launcher_state = {
        "schema_version": 2,
        "protocol": protocol_name,
        "runtime_smoke": bool(args.smoke),
        "arms_requested": list(arms),
        "task_ids": [int(value) for value in task_ids],
        "episodes_per_task": int(episodes_per_task),
        "expected_episodes_per_arm": expected_episodes_per_arm,
        "expected_total_episodes": expected_episodes_per_arm * len(arms),
        "completed_arms": [],
        "failed_arm": None,
    }
    if state_path.is_file() and args.resume:
        try:
            previous = _load_json(state_path)
            launcher_state["completed_arms"] = list(
                previous.get("completed_arms", [])
            )
        except Exception:
            pass

    configs = {}
    for arm in arms:
        cfg = _build_arm_config(
            arm=arm,
            checkpoint=checkpoint,
            manifest_path=manifest_path,
            artifact_path=artifact_path,
            artifact_sha256=artifact_sha256,
            output_root=output_root,
            seed=args.seed,
            smoke=args.smoke,
        )
        _validate_arm_config(cfg, arm=arm, smoke=args.smoke)
        configs[arm] = cfg

    if args.dry_run:
        print(
            json.dumps(
                {
                    "execution_plan": {
                        "protocol": protocol_name,
                        "runtime_smoke": bool(args.smoke),
                        "task_ids": list(task_ids),
                        "episodes_per_task": episodes_per_task,
                        "episodes_per_arm": expected_episodes_per_arm,
                        "total_requested_episodes": (
                            expected_episodes_per_arm * len(arms)
                        ),
                        "output_root": str(output_root),
                        "task_level_resume": True,
                    },
                    "configs": {arm: asdict(cfg) for arm, cfg in configs.items()},
                },
                indent=2,
            )
        )
        return 0

    for arm in arms:
        result_path = output_root / arm / "results.json"
        if args.resume and _arm_is_complete(
            result_path,
            expected_total_episodes=expected_episodes_per_arm,
            expected_task_count=len(task_ids),
        ):
            if arm not in launcher_state["completed_arms"]:
                launcher_state["completed_arms"].append(arm)
            _write_json(state_path, launcher_state)
            print(f"[resume] skipping completed arm: {arm}")
            continue

        if result_path.exists() and not (args.overwrite or args.resume):
            raise FileExistsError(
                f"{result_path} already exists. Use --resume or --overwrite."
            )

        launcher_state["failed_arm"] = None
        launcher_state["active_arm"] = arm
        _write_json(state_path, launcher_state)

        try:
            _run_arm_resumable(
                configs[arm],
                arm=arm,
                save_videos=args.save_videos,
                resume_existing=(args.resume and result_path.exists()),
                smoke=args.smoke,
                task_ids=task_ids,
                episodes_per_task=episodes_per_task,
                protocol_name=protocol_name,
            )
        except Exception as exc:
            launcher_state["failed_arm"] = arm
            launcher_state["failure"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            _write_json(state_path, launcher_state)
            raise

        if arm not in launcher_state["completed_arms"]:
            launcher_state["completed_arms"].append(arm)
        launcher_state["active_arm"] = None
        launcher_state["failed_arm"] = None
        launcher_state.pop("failure", None)
        _write_json(state_path, launcher_state)

    launcher_state["completed"] = (
        len(launcher_state["completed_arms"]) == len(arms)
    )
    _write_json(state_path, launcher_state)
    print(
        "Completed evaluation: "
        f"{len(arms)} arms x {expected_episodes_per_arm} episodes"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
