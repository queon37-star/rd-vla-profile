#!/usr/bin/env python3
"""Profile synchronized RD-VLA action-head wall-clock latency for the paper arms.

This is a dedicated component-timing run. It reuses the exact frozen method
configuration from ``run_spatial_paper_4arm.py`` but measures only the
``action_head.predict_action(...)`` call, i.e. after VLM hidden states have
already been produced.

Timer boundary per policy query:

    torch.cuda.synchronize()
    start = time.perf_counter()
    action_head.predict_action(...)
    torch.cuda.synchronize()
    stop = time.perf_counter()

Thus the reported action-head latency includes Prelude, recurrent refinement,
warm-start handling, LDCE scorer/scheduler, Coda/backfill, stopping logic,
output projection, and Python control overhead inside predict_action. VLM
forward, hidden-state extraction, action unnormalization, image preprocessing,
and environment stepping are excluded.

The default paper profile uses 10 tasks x 10 paired official states per arm
(100 episodes/arm, 400 measured episodes total), plus one unmeasured warm-up
rollout per arm. The ten states are selected at frozen positions
0,5,...,45 from the same 50-state manifest ordering, preserving the original
10/10/30 partition proportions (2 calibration, 2 screening, 6 final states).

Outputs are separate from the 2,000-episode success evaluation. Task-level JSON
files make the run resumable without mixing partial-task timing samples.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

import experiments.robot.libero.evaluation_protocol as protocol
import experiments.robot.libero.run_libero_eval as base
import prismatic.models.action_heads as action_heads_module
import scripts.run_spatial_paper_4arm as frozen
from prismatic.models.action_delta_gate import load_action_delta_gate_artifact


PROFILE_PHASE = "paper_action_head_profile"
WARMUP_PHASE = "paper_action_head_warmup"
SMOKE_PHASE = "paper_action_head_smoke"
PAPER_ARMS = frozen.PAPER_ARMS
PROFILE_TASK_IDS = tuple(range(10))
PROFILE_STATE_POSITIONS = tuple(range(0, 50, 5))  # 10 positions: 0,5,...,45
PROFILE_EPISODES_PER_TASK = len(PROFILE_STATE_POSITIONS)
SMOKE_TASK_IDS = (0, 4)
SMOKE_STATE_POSITIONS = (0,)
PROFILE_PROTOCOL = "libero-spatial-action-head-10x10-4arm-v1"
SMOKE_PROTOCOL = "libero-spatial-action-head-smoke-t0-t4-1ep-4arm-v1"


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
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _all_state_ids(manifest, task_id: int) -> list[int]:
    entry = manifest["tasks"][str(int(task_id))]
    ids = (
        list(entry["partitions"]["calibration"])
        + list(entry["partitions"]["screening"])
        + list(entry["partitions"]["final"])
    )
    if len(ids) != 50 or len(set(ids)) != 50:
        raise ValueError(f"task {task_id} must expose exactly 50 unique official states")
    return [int(value) for value in ids]


def _resolve_profile_trials(
    *,
    manifest,
    phase,
    task_id,
    initial_states,
    base_seed,
    initial_state_file_path=None,
):
    if phase not in {PROFILE_PHASE, WARMUP_PHASE, SMOKE_PHASE}:
        return protocol.resolve_phase_trials(
            manifest=manifest,
            phase=phase,
            task_id=task_id,
            initial_states=initial_states,
            base_seed=base_seed,
            initial_state_file_path=initial_state_file_path,
        )

    # Keep the existing manifest/tensor/source-file validation authoritative.
    protocol.resolve_phase_trials(
        manifest=manifest,
        phase="calibration",
        task_id=task_id,
        initial_states=initial_states,
        base_seed=base_seed,
        initial_state_file_path=initial_state_file_path,
    )

    all_ids = _all_state_ids(manifest, int(task_id))
    if phase == PROFILE_PHASE:
        positions = PROFILE_STATE_POSITIONS
        partition = "profile_stratified_10_of_50"
    elif phase == SMOKE_PHASE:
        positions = SMOKE_STATE_POSITIONS
        partition = "profile_smoke_official_state"
    else:
        positions = (0,)
        partition = "profile_warmup_official_state"

    trials = []
    for paired_trial_id, position in enumerate(positions):
        initial_state_id = all_ids[int(position)]
        trials.append(
            protocol.EpisodeTrial(
                phase=phase,
                partition=partition,
                paired_trial_id=paired_trial_id,
                initial_state_id=initial_state_id,
                episode_seed=protocol.derive_paired_episode_seed(
                    base_seed=base_seed,
                    phase=phase,
                    task_suite_name=manifest["task_suite_name"],
                    task_id=int(task_id),
                    initial_state_id=initial_state_id,
                    paired_trial_id=paired_trial_id,
                ),
                smoke_excluded_from_fitting=(phase != PROFILE_PHASE),
            )
        )
    return tuple(trials)


@contextmanager
def _runtime_patches():
    original_resolver = base.resolve_phase_trials
    original_video_stats = base.save_rollout_video_with_stats
    original_video = base.save_rollout_video
    original_deferred_validator = (
        action_heads_module.validate_action_delta_deferred_backfill_configuration
    )

    def deferred_validator_without_diagnostic_coda_timing(**kwargs):
        # Preserve every algorithmic validation condition. Only satisfy the old
        # development-only requirement that profile_coda_cost be True; actual
        # execution remains profile_coda_cost=False, so no per-Coda syncs are
        # introduced into the component measurement.
        forwarded = dict(kwargs)
        forwarded["profile_coda_cost"] = True
        return original_deferred_validator(**forwarded)

    base.resolve_phase_trials = _resolve_profile_trials
    base.save_rollout_video_with_stats = lambda *args, **kwargs: None
    base.save_rollout_video = lambda *args, **kwargs: None
    action_heads_module.validate_action_delta_deferred_backfill_configuration = (
        deferred_validator_without_diagnostic_coda_timing
    )
    try:
        yield
    finally:
        base.resolve_phase_trials = original_resolver
        base.save_rollout_video_with_stats = original_video_stats
        base.save_rollout_video = original_video
        action_heads_module.validate_action_delta_deferred_backfill_configuration = (
            original_deferred_validator
        )


class ActionHeadWallClockTimer:
    """Temporarily wrap action_head.predict_action with two CUDA syncs."""

    def __init__(self, action_head):
        self.action_head = action_head
        self.original = action_head.predict_action
        self.samples: list[dict] = []
        self.current_task_id: int | None = None

    def __enter__(self):
        original = self.original
        samples = self.samples
        owner = self

        def timed_predict_action(*args, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            failed = False
            try:
                result = original(*args, **kwargs)
                return result
            except Exception:
                failed = True
                raise
            finally:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                samples.append(
                    {
                        "sequence_id": len(samples),
                        "task_id": owner.current_task_id,
                        "action_head_latency_ms": float(elapsed_ms),
                        "failed_call": bool(failed),
                    }
                )

        self.action_head.predict_action = timed_predict_action
        return self

    def __exit__(self, exc_type, exc, tb):
        self.action_head.predict_action = self.original


def _stats(values: Sequence[float]) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot summarize zero action-head latency samples")
    return {
        "count": int(array.size),
        "mean_ms": float(array.mean()),
        "median_ms": float(np.median(array)),
        "std_ms": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "p90_ms": float(np.percentile(array, 90)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
    }


def _validate_cfg(cfg: base.GenerateConfig, *, arm: str, smoke: bool) -> None:
    if cfg.task_suite_name != base.TaskSuite.LIBERO_SPATIAL:
        raise ValueError("Action-head profile is LIBERO-Spatial only")
    expected_phase = SMOKE_PHASE if smoke else PROFILE_PHASE
    expected_trials = 1 if smoke else PROFILE_EPISODES_PER_TASK
    if cfg.evaluation_protocol_phase != expected_phase:
        raise ValueError(f"Unexpected profile phase: {cfg.evaluation_protocol_phase}")
    if int(cfg.num_trials_per_task) != expected_trials:
        raise ValueError(f"Expected {expected_trials} trials/task")
    if cfg.recurrence_strategy != "adjacent_action_mse":
        raise ValueError("Profile must use adjacent_action_mse")
    if float(cfg.recurrence_kl_thresh) != 0.001 or int(cfg.recurrence_max_iter) != 32:
        raise ValueError("Profile freezes threshold=0.001 and max_iter=32")
    if int(cfg.num_exec_actions) != 5:
        raise ValueError("Profile freezes num_exec_actions=5")
    if cfg.profile_coda_cost:
        raise ValueError("profile_coda_cost must stay False")
    if cfg.profile_pytorch or cfg.profile_timing_summary:
        raise ValueError("Other profilers must stay disabled")
    if cfg.use_latent_precheck or cfg.latent_precheck_mode != "off":
        raise ValueError("Legacy latent pre-check must stay disabled")
    if not cfg.use_cached_final_output:
        raise ValueError("Cached terminal output must stay enabled")

    expect_warm = arm in {"warm_start", "combined"}
    expect_ldce = arm in {"ldce", "combined"}
    if bool(cfg.use_warm_start) != expect_warm:
        raise ValueError(f"Warm-start mismatch for {arm}")
    if expect_warm and cfg.warm_start_source != "midpoint":
        raise ValueError(f"{arm} must use midpoint warm-start")
    if bool(cfg.use_action_delta_deferred_backfill_filter) != expect_ldce:
        raise ValueError(f"LDCE mismatch for {arm}")
    if expect_ldce:
        if cfg.action_delta_deferred_runtime_policy != "lazy_prefix_exact":
            raise ValueError("LDCE must use lazy_prefix_exact")
        if cfg.action_delta_deferred_scorer_backend != "eager":
            raise ValueError("LDCE must use eager scorer")
        if int(cfg.action_delta_gate_min_terminal_iter) != 2:
            raise ValueError("LDCE must use min_terminal_iter=2")
    if arm == "ldce" and not cfg.action_delta_deferred_apply_to_cold:
        raise ValueError("LDCE-only must apply to cold inference")
    if arm == "combined" and cfg.action_delta_deferred_apply_to_cold:
        raise ValueError("Combined must not force cold-only LDCE")


def _build_cfg(
    *,
    arm: str,
    checkpoint: Path,
    manifest: Path,
    artifact: Path | None,
    artifact_sha: str | None,
    output_root: Path,
    seed: int,
    smoke: bool,
) -> base.GenerateConfig:
    cfg = frozen._build_arm_config(
        arm=arm,
        checkpoint=checkpoint,
        manifest_path=manifest,
        artifact_path=artifact,
        artifact_sha256=artifact_sha,
        output_root=output_root,
        seed=seed,
    )
    cfg.evaluation_protocol_phase = SMOKE_PHASE if smoke else PROFILE_PHASE
    cfg.num_trials_per_task = 1 if smoke else PROFILE_EPISODES_PER_TASK
    cfg.run_id_note = f"paper_action_head_{'smoke' if smoke else 'profile'}_{arm}"
    cfg.save_version = f"paper-action-head-{'smoke' if smoke else 'profile'}-{arm}"
    arm_dir = output_root / arm
    cfg.json_log_file = str(arm_dir / "debug_results.json")
    cfg.step_log_file = str(arm_dir / "debug_predictions.jsonl")
    cfg.recurrent_convergence_dir = str(arm_dir)
    cfg.recurrent_convergence_log_file = str(arm_dir / "debug_predictions.jsonl")
    cfg.recurrent_convergence_summary_file = str(arm_dir / "debug_summary.json")
    _validate_cfg(cfg, arm=arm, smoke=smoke)
    return cfg


def _bind_predictor(action_head, payload, cfg, task_id: int) -> None:
    action_head.clear_scalar_task_policy()
    if payload is None:
        action_head.clear_action_delta_gate()
        return
    device = next(action_head.parameters()).device
    prepared = frozen._prepare_shared_spatial_predictor(
        payload,
        device=device,
        task_id=int(task_id),
    )
    action_head.configure_action_delta_gate(
        prepared,
        deferred_scorer_backend=cfg.action_delta_deferred_scorer_backend,
    )


def _run_one_task(
    *,
    cfg,
    task_suite,
    task_id: int,
    model,
    resize_size,
    processor,
    action_head,
    proprio_projector,
    log_file,
    source_commit: str,
    total_episodes: int,
    total_successes: int,
):
    return base.run_task(
        cfg,
        task_suite,
        int(task_id),
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


def _profile_arm(
    cfg: base.GenerateConfig,
    *,
    arm: str,
    output_root: Path,
    task_ids: Sequence[int],
    smoke: bool,
    resume: bool,
) -> dict:
    arm_dir = output_root / arm
    arm_dir.mkdir(parents=True, exist_ok=True)

    artifact_manifest = None
    artifact_payload = None
    if cfg.use_action_delta_deferred_backfill_filter:
        artifact_manifest, artifact_payload = load_action_delta_gate_artifact(
            cfg.action_delta_gate_artifact_path,
            expected_sha256=cfg.action_delta_gate_expected_sha256.lower(),
        )

    base.set_seed_everywhere(cfg.seed)
    model, action_head, proprio_projector, processor = base.initialize_model(cfg)
    resize_size = base.get_image_resize_size(cfg)
    log_file, _, run_id = base.setup_logging(cfg)
    base.RDVLAProfiler.set_enabled(False)
    base.RDVLAProfiler.set_timing_enabled(False)
    base.configure_recurrent_convergence_paths(cfg, run_id)

    _, manifest_sha = base.load_protocol_manifest(
        cfg.initial_state_manifest_path,
        require_source_file_hashes=True,
    )
    source_commit = base.current_source_commit()
    checkpoint_meta = base.checkpoint_identity(Path(cfg.pretrained_checkpoint))
    task_suite = base.benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    if task_suite.n_tasks != 10:
        raise RuntimeError(f"Expected 10 Spatial tasks, got {task_suite.n_tasks}")

    protocol_record = {
        "schema_version": 1,
        "protocol": SMOKE_PROTOCOL if smoke else PROFILE_PROTOCOL,
        "arm": arm,
        "source_commit": source_commit,
        "checkpoint": checkpoint_meta,
        "manifest_sha256": manifest_sha,
        "task_ids": [int(value) for value in task_ids],
        "episodes_per_task": 1 if smoke else PROFILE_EPISODES_PER_TASK,
        "profile_state_positions": (
            list(SMOKE_STATE_POSITIONS) if smoke else list(PROFILE_STATE_POSITIONS)
        ),
        "measurement_scope": {
            "name": "action_head.predict_action synchronized wall-clock",
            "start": "after pre-call torch.cuda.synchronize, immediately before action_head.predict_action",
            "end": "after action_head.predict_action returns and post-call torch.cuda.synchronize completes",
            "includes": [
                "Prelude",
                "recurrent core",
                "warm-start state handling",
                "LDCE predictor and scheduler when enabled",
                "Coda/backfill and stopping logic",
                "action-head output projection",
                "Python control overhead inside predict_action",
            ],
            "excludes": [
                "image preprocessing",
                "vision/VLM forward",
                "VLM hidden-state extraction",
                "action unnormalization after predict_action",
                "environment stepping",
            ],
            "inner_coda_profiling_enabled": False,
            "e2e_latency_from_this_profile_is_reportable": False,
        },
        "warmup": "one unmeasured task-0 rollout after model initialization per process/arm",
        "artifact_sha256": (
            artifact_manifest.get("artifact_sha256") if artifact_manifest else None
        ),
    }

    protocol_path = arm_dir / "protocol.json"
    if resume and protocol_path.is_file():
        old = _load_json(protocol_path)
        for key in ("protocol", "arm", "source_commit", "manifest_sha256", "task_ids", "profile_state_positions"):
            if old.get(key) != protocol_record.get(key):
                raise RuntimeError(
                    f"Resume protocol mismatch for {key}; use --overwrite for a fresh profile"
                )
    else:
        _write_json(protocol_path, protocol_record)

    try:
        with _runtime_patches():
            # Unmeasured warm-up on task 0. This pays one-time CUDA/kernel/cache
            # setup before any paper latency sample is collected.
            _bind_predictor(action_head, artifact_payload, cfg, 0)
            saved_phase = cfg.evaluation_protocol_phase
            saved_trials = cfg.num_trials_per_task
            cfg.evaluation_protocol_phase = WARMUP_PHASE
            cfg.num_trials_per_task = 1
            _run_one_task(
                cfg=cfg,
                task_suite=task_suite,
                task_id=0,
                model=model,
                resize_size=resize_size,
                processor=processor,
                action_head=action_head,
                proprio_projector=proprio_projector,
                log_file=log_file,
                source_commit=source_commit,
                total_episodes=0,
                total_successes=0,
            )
            cfg.evaluation_protocol_phase = saved_phase
            cfg.num_trials_per_task = saved_trials

            total_episodes = 0
            total_successes = 0
            timer = ActionHeadWallClockTimer(action_head)
            with timer:
                for task_id in task_ids:
                    task_id = int(task_id)
                    task_path = arm_dir / f"task_{task_id:02d}.json"
                    if resume and task_path.is_file():
                        existing = _load_json(task_path)
                        if existing.get("completed"):
                            total_episodes += int(existing["episodes"])
                            total_successes += int(existing["successes"])
                            print(f"[resume] {arm}: skip completed task {task_id}")
                            continue

                    _bind_predictor(action_head, artifact_payload, cfg, task_id)
                    timer.current_task_id = task_id
                    sample_start = len(timer.samples)
                    before_episodes = total_episodes

                    total_episodes, total_successes, task_stats = _run_one_task(
                        cfg=cfg,
                        task_suite=task_suite,
                        task_id=task_id,
                        model=model,
                        resize_size=resize_size,
                        processor=processor,
                        action_head=action_head,
                        proprio_projector=proprio_projector,
                        log_file=log_file,
                        source_commit=source_commit,
                        total_episodes=total_episodes,
                        total_successes=total_successes,
                    )
                    task_samples = [dict(value) for value in timer.samples[sample_start:]]
                    expected_predictions = sum(
                        int(ep.get("num_predictions", 0)) for ep in task_stats
                    )
                    if len(task_samples) != expected_predictions:
                        raise RuntimeError(
                            f"task {task_id}: timing samples={len(task_samples)} but "
                            f"run_task reports predictions={expected_predictions}"
                        )
                    if any(sample.get("failed_call") for sample in task_samples):
                        raise RuntimeError(f"task {task_id}: failed predict_action call was timed")

                    cursor = 0
                    weighted_iters = 0.0
                    for episode_record in task_stats:
                        n = int(episode_record.get("num_predictions", 0))
                        for sample in task_samples[cursor : cursor + n]:
                            sample["episode"] = int(episode_record.get("episode", -1))
                            sample["paired_trial_id"] = episode_record.get("paired_trial_id")
                            sample["initial_state_id"] = episode_record.get("initial_state_id")
                            sample["success"] = bool(episode_record.get("success"))
                        if episode_record.get("avg_iters") is not None:
                            weighted_iters += float(episode_record["avg_iters"]) * n
                        cursor += n
                    if cursor != len(task_samples):
                        raise RuntimeError(f"task {task_id}: failed to align timing samples to episodes")

                    values = [float(s["action_head_latency_ms"]) for s in task_samples]
                    task_payload = {
                        "schema_version": 1,
                        "completed": True,
                        "arm": arm,
                        "task_id": task_id,
                        "task_name": task_suite.get_task(task_id).name,
                        "episodes": int(total_episodes - before_episodes),
                        "successes": int(sum(bool(ep.get("success")) for ep in task_stats)),
                        "num_predictions": len(task_samples),
                        "prediction_weighted_avg_iters": (
                            weighted_iters / len(task_samples) if task_samples else None
                        ),
                        "latency": _stats(values),
                        "episode_stats": task_stats,
                        "samples": task_samples,
                    }
                    _write_json(task_path, task_payload)
                    print(
                        f"{arm} task={task_id}: n={len(values)} "
                        f"action_head_mean={task_payload['latency']['mean_ms']:.3f} ms"
                    )
    finally:
        if log_file:
            log_file.close()
        del processor, proprio_projector, action_head, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    task_payloads = []
    for task_id in task_ids:
        path = arm_dir / f"task_{int(task_id):02d}.json"
        if not path.is_file():
            raise RuntimeError(f"Missing completed task profile: {path}")
        payload = _load_json(path)
        if not payload.get("completed"):
            raise RuntimeError(f"Incomplete task profile: {path}")
        task_payloads.append(payload)

    all_samples = [sample for task in task_payloads for sample in task["samples"]]
    all_values = [float(sample["action_head_latency_ms"]) for sample in all_samples]
    episode_means = []
    for task in task_payloads:
        samples = task["samples"]
        by_episode: dict[int, list[float]] = {}
        for sample in samples:
            episode = int(sample.get("episode", -1))
            by_episode.setdefault(episode, []).append(float(sample["action_head_latency_ms"]))
        episode_means.extend(float(np.mean(values)) for values in by_episode.values() if values)

    task_means = [float(task["latency"]["mean_ms"]) for task in task_payloads]
    total_predictions = sum(int(task["num_predictions"]) for task in task_payloads)
    weighted_iters_numerator = sum(
        float(task["prediction_weighted_avg_iters"]) * int(task["num_predictions"])
        for task in task_payloads
        if task.get("prediction_weighted_avg_iters") is not None
    )
    summary = {
        "schema_version": 1,
        "completed": True,
        "arm": arm,
        "protocol": protocol_record,
        "episodes": sum(int(task["episodes"]) for task in task_payloads),
        "successes": sum(int(task["successes"]) for task in task_payloads),
        "num_predictions": total_predictions,
        "prediction_weighted_avg_iters": (
            weighted_iters_numerator / total_predictions if total_predictions else None
        ),
        "action_head_latency_prediction_weighted": _stats(all_values),
        "action_head_latency_episode_balanced": _stats(episode_means),
        "action_head_latency_task_balanced_mean_ms": float(np.mean(task_means)),
        "per_task": {
            str(task["task_id"]): {
                "task_name": task["task_name"],
                "num_predictions": task["num_predictions"],
                "mean_ms": task["latency"]["mean_ms"],
                "median_ms": task["latency"]["median_ms"],
                "p95_ms": task["latency"]["p95_ms"],
                "avg_iters": task["prediction_weighted_avg_iters"],
            }
            for task in task_payloads
        },
    }
    _write_json(arm_dir / "summary.json", summary)
    return summary


def _parse_arms(values: Iterable[str]) -> tuple[str, ...]:
    return frozen._parse_arms(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile synchronized action_head.predict_action latency for the frozen paper arms."
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/12_24-24_24_Spatial_40k",
    )
    parser.add_argument(
        "--initial-state-manifest",
        default="experiments/robot/libero/manifests/libero_spatial_official_50_v1.json",
    )
    parser.add_argument(
        "--action-delta-artifact",
        default=os.environ.get("RDVLA_ACTION_DELTA_ARTIFACT", ""),
    )
    parser.add_argument(
        "--action-delta-sha256",
        default=os.environ.get("RDVLA_ACTION_DELTA_SHA256", ""),
    )
    parser.add_argument(
        "--output-root",
        default="benchmark_results/paper_spatial_action_head_latency_10x10",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--arms", nargs="+", default=list(PAPER_ARMS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Measure one episode on tasks 0 and 4 per arm; output is isolated under /smoke.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.resume and args.overwrite:
        raise ValueError("Choose only one of --resume or --overwrite")

    arms = _parse_arms(args.arms)
    checkpoint = Path(args.checkpoint)
    manifest = Path(args.initial_state_manifest)
    base_output_root = Path(args.output_root)
    output_root = base_output_root / "smoke" if args.smoke else base_output_root
    task_ids = SMOKE_TASK_IDS if args.smoke else PROFILE_TASK_IDS

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest}")

    needs_ldce = any(arm in {"ldce", "combined"} for arm in arms)
    artifact = Path(args.action_delta_artifact) if args.action_delta_artifact else None
    artifact_sha = None
    if needs_ldce:
        if artifact is None or not artifact.exists():
            raise FileNotFoundError("LDCE/Combined requires --action-delta-artifact")
        artifact_sha = frozen._artifact_sha256(
            artifact,
            args.action_delta_sha256 or None,
        )
        # Verify bytes + payload contract even in dry-run.
        load_action_delta_gate_artifact(
            artifact,
            expected_sha256=artifact_sha,
        )

    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    configs = {}
    for arm in arms:
        configs[arm] = _build_cfg(
            arm=arm,
            checkpoint=checkpoint,
            manifest=manifest,
            artifact=artifact,
            artifact_sha=artifact_sha,
            output_root=output_root,
            seed=args.seed,
            smoke=args.smoke,
        )

    plan = {
        "schema_version": 1,
        "protocol": SMOKE_PROTOCOL if args.smoke else PROFILE_PROTOCOL,
        "arms": list(arms),
        "task_ids": list(task_ids),
        "episodes_per_task": 1 if args.smoke else PROFILE_EPISODES_PER_TASK,
        "measured_episodes_per_arm": len(task_ids) * (1 if args.smoke else PROFILE_EPISODES_PER_TASK),
        "measured_total_episodes": len(arms) * len(task_ids) * (1 if args.smoke else PROFILE_EPISODES_PER_TASK),
        "unmeasured_warmup_episodes": len(arms),
        "profile_state_positions": list(SMOKE_STATE_POSITIONS if args.smoke else PROFILE_STATE_POSITIONS),
        "timer": "CUDA-sync + perf_counter around action_head.predict_action only",
        "inner_coda_profiling": False,
        "e2e_latency_from_this_run": "do not report; use the completed 2,000-episode paper run",
        "task_level_resume": True,
    }
    _write_json(output_root / "plan.json", plan)

    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    launcher_state = {
        **plan,
        "completed_arms": [],
        "active_arm": None,
        "failed_arm": None,
        "completed": False,
    }
    state_path = output_root / "launcher_state.json"
    if args.resume and state_path.is_file():
        previous = _load_json(state_path)
        launcher_state["completed_arms"] = list(previous.get("completed_arms", []))

    summaries = {}
    for arm in arms:
        arm_summary_path = output_root / arm / "summary.json"
        if args.resume and arm_summary_path.is_file():
            existing = _load_json(arm_summary_path)
            if existing.get("completed"):
                print(f"[resume] skip completed arm: {arm}")
                summaries[arm] = existing
                if arm not in launcher_state["completed_arms"]:
                    launcher_state["completed_arms"].append(arm)
                _write_json(state_path, launcher_state)
                continue

        launcher_state["active_arm"] = arm
        launcher_state["failed_arm"] = None
        _write_json(state_path, launcher_state)
        try:
            summary = _profile_arm(
                configs[arm],
                arm=arm,
                output_root=output_root,
                task_ids=task_ids,
                smoke=args.smoke,
                resume=args.resume,
            )
            summaries[arm] = summary
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
        _write_json(state_path, launcher_state)

    if "baseline" in summaries:
        baseline_mean = float(
            summaries["baseline"]["action_head_latency_prediction_weighted"]["mean_ms"]
        )
        comparison = {}
        for arm, summary in summaries.items():
            mean_ms = float(summary["action_head_latency_prediction_weighted"]["mean_ms"])
            comparison[arm] = {
                "mean_action_head_latency_ms": mean_ms,
                "speedup_vs_baseline_percent": (
                    100.0 * (baseline_mean - mean_ms) / baseline_mean
                ),
                "median_ms": summary["action_head_latency_prediction_weighted"]["median_ms"],
                "p95_ms": summary["action_head_latency_prediction_weighted"]["p95_ms"],
                "num_predictions": summary["num_predictions"],
                "profile_avg_iters": summary["prediction_weighted_avg_iters"],
            }
        _write_json(
            output_root / "comparison.json",
            {
                "schema_version": 1,
                "protocol": plan["protocol"],
                "primary_latency_aggregation": "prediction-weighted mean action-head wall-clock",
                "comparison": comparison,
            },
        )

    launcher_state["active_arm"] = None
    launcher_state["failed_arm"] = None
    launcher_state.pop("failure", None)
    launcher_state["completed"] = len(launcher_state["completed_arms"]) == len(arms)
    _write_json(state_path, launcher_state)
    print("Completed action-head latency profile")
    return 0


if __name__ == "__main__":
    sys.exit(main())
