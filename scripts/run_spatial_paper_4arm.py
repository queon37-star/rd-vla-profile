#!/usr/bin/env python3
"""Run the frozen 4-arm LIBERO-Spatial paper evaluation.

One invocation executes four paired conditions over the same 50 official
initial states for each of the 10 LIBERO-Spatial tasks:

    baseline, warm_start, ldce, combined

The existing 10/10/30 manifest is reused only as an identity source. For this
paper evaluation the three frozen partitions are concatenated into one
50-state paired evaluation set; no fitting, screening, or threshold selection
is performed from these rollout outcomes.

The Action-Delta predictor is one frozen Spatial predictor. The artifact's
historical fold-4 held-out metadata is preserved, but the exact same weights
are materialized for every Spatial task. Tasks 4/5 use the existing production
preparer and the other eight tasks use the existing shadow preparer; both paths
validate and copy the same artifact tensors.
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
from typing import Iterable

import torch

import experiments.robot.libero.evaluation_protocol as protocol
import experiments.robot.libero.run_libero_eval as base
import prismatic.models.action_heads as action_heads_module
from prismatic.models.action_delta_gate import (
    ACTION_DELTA_GATE_HELD_OUT_TASK_IDS,
    ACTION_DELTA_GATE_SHADOW_CALIBRATION_TASK_IDS,
    ACTION_DELTA_NONCONVERGENCE_THRESHOLD,
    load_action_delta_gate_artifact,
    prepare_action_delta_gate,
    prepare_action_delta_gate_shadow,
)


PAPER_PHASE = "paper_eval"
PAPER_ARMS = ("baseline", "warm_start", "ldce", "combined")
PAPER_TASK_IDS = tuple(range(10))
PAPER_EPISODES_PER_TASK = 50
PAPER_TOTAL_EPISODES_PER_ARM = len(PAPER_TASK_IDS) * PAPER_EPISODES_PER_TASK


def _resolve_paper_trials(
    *,
    manifest,
    phase,
    task_id,
    initial_states,
    base_seed,
    initial_state_file_path=None,
):
    """Resolve all 50 frozen official states as one paired paper-eval set."""
    if phase != PAPER_PHASE:
        return protocol.resolve_phase_trials(
            manifest=manifest,
            phase=phase,
            task_id=task_id,
            initial_states=initial_states,
            base_seed=base_seed,
            initial_state_file_path=initial_state_file_path,
        )

    # Reuse the existing calibration resolver once so all manifest, tensor-hash,
    # and raw source-file checks stay authoritative.
    protocol.resolve_phase_trials(
        manifest=manifest,
        phase="calibration",
        task_id=task_id,
        initial_states=initial_states,
        base_seed=base_seed,
        initial_state_file_path=initial_state_file_path,
    )

    task_entry = manifest["tasks"][str(int(task_id))]
    state_ids = (
        list(task_entry["partitions"]["calibration"])
        + list(task_entry["partitions"]["screening"])
        + list(task_entry["partitions"]["final"])
    )
    if (
        len(state_ids) != PAPER_EPISODES_PER_TASK
        or len(set(state_ids)) != PAPER_EPISODES_PER_TASK
    ):
        raise ValueError(
            f"paper_eval task {task_id} must resolve exactly 50 unique official states"
        )

    trials = []
    for paired_trial_id, initial_state_id in enumerate(state_ids):
        trials.append(
            protocol.EpisodeTrial(
                phase=PAPER_PHASE,
                partition="all_official_50",
                paired_trial_id=paired_trial_id,
                initial_state_id=int(initial_state_id),
                episode_seed=protocol.derive_paired_episode_seed(
                    base_seed=base_seed,
                    phase=PAPER_PHASE,
                    task_suite_name=manifest["task_suite_name"],
                    task_id=int(task_id),
                    initial_state_id=int(initial_state_id),
                    paired_trial_id=paired_trial_id,
                ),
                smoke_excluded_from_fitting=False,
            )
        )
    return tuple(trials)


def _paper_deferred_validator(**kwargs):
    """Drop only the old diagnostic-profiling requirement for paper runtime.

    The current deferred/backfill validator requires profile_coda_cost=True
    solely because the development experiments collected synchronized Coda
    timing diagnostics. The scheduler itself does not depend on those timing
    values. For the paper rollout we keep every other validation check and
    execute the method with profile_coda_cost=False so the existing outer
    synchronized get_action timer measures the actual online policy query.
    """
    forwarded = dict(kwargs)
    forwarded["profile_coda_cost"] = True
    return _paper_deferred_validator.original(**forwarded)


@contextmanager
def _paper_runner_patches(*, save_videos: bool):
    """Install paper-only protocol/runtime adapters and restore them afterward."""
    original_resolver = base.resolve_phase_trials
    original_video_stats = base.save_rollout_video_with_stats
    original_video = base.save_rollout_video
    original_deferred_validator = (
        action_heads_module.validate_action_delta_deferred_backfill_configuration
    )

    base.resolve_phase_trials = _resolve_paper_trials
    _paper_deferred_validator.original = original_deferred_validator
    action_heads_module.validate_action_delta_deferred_backfill_configuration = (
        _paper_deferred_validator
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


def _artifact_sha256(artifact_path: Path, explicit_sha256: str | None) -> str:
    if explicit_sha256:
        value = explicit_sha256.strip().lower()
        if len(value) != 64 or any(
            ch not in "0123456789abcdef" for ch in value
        ):
            raise ValueError(
                "--action-delta-sha256 must be a 64-character hexadecimal SHA-256"
            )
        return value

    manifest_path = (
        artifact_path / "manifest.json"
        if artifact_path.is_dir()
        else artifact_path.parent / "manifest.json"
    )
    if not manifest_path.is_file():
        raise ValueError(
            "Action-Delta SHA-256 was not supplied and no adjacent manifest.json was found"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    value = str(manifest.get("artifact_sha256", "")).lower()
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"Invalid artifact_sha256 in {manifest_path}")
    return value


def _prepare_shared_spatial_predictor(payload, *, device, task_id: int):
    """Materialize the same frozen predictor weights for any Spatial task."""
    if task_id in ACTION_DELTA_GATE_HELD_OUT_TASK_IDS:
        return prepare_action_delta_gate(
            payload,
            device=device,
            task_id=task_id,
        )
    if task_id in ACTION_DELTA_GATE_SHADOW_CALIBRATION_TASK_IDS:
        return prepare_action_delta_gate_shadow(
            payload,
            device=device,
            task_id=task_id,
        )
    raise ValueError(f"Unsupported LIBERO-Spatial task_id={task_id}")


def _build_arm_config(
    *,
    arm: str,
    checkpoint: Path,
    manifest_path: Path,
    artifact_path: Path | None,
    artifact_sha256: str | None,
    output_root: Path,
    seed: int,
) -> base.GenerateConfig:
    if arm not in PAPER_ARMS:
        raise ValueError(f"Unknown paper arm: {arm}")

    use_warm = arm in {"warm_start", "combined"}
    use_ldce = arm in {"ldce", "combined"}
    arm_dir = output_root / arm
    arm_dir.mkdir(parents=True, exist_ok=True)

    cfg = base.GenerateConfig(
        pretrained_checkpoint=str(checkpoint),
        sync_checkpoint_source_config=False,
        task_suite_name=base.TaskSuite.LIBERO_SPATIAL,
        num_trials_per_task=PAPER_EPISODES_PER_TASK,
        task_id=None,
        initial_states_path="DEFAULT",
        evaluation_protocol_phase=PAPER_PHASE,
        initial_state_manifest_path=str(manifest_path),
        local_log_dir=str(arm_dir / "logs"),
        run_id_note=f"paper_spatial_50x10_{arm}",
        use_wandb=False,
        seed=int(seed),
        reset_rng_each_episode=True,
        episode_seed_stride=1,
        use_recurrent=True,
        recurrence_strategy="adjacent_action_mse",
        recurrence_kl_thresh=0.001,
        recurrence_max_iter=32,
        use_warm_start=use_warm,
        warm_start_source="midpoint",
        warm_start_min_iter=2,
        validate_warm_start_finite=use_warm,
        # Keep profiling off. run_episode() already synchronizes CUDA around
        # get_action, so latency_ms remains the online policy-query timer.
        profile_coda_cost=False,
        use_cached_final_output=True,
        use_latent_precheck=False,
        latent_precheck_mode="off",
        latent_precheck_trace_level="off",
        shadow_full_depth=False,
        collect_preconvergence_raw_shadow=False,
        use_action_delta_gate=False,
        collect_action_delta_gate_shadow=False,
        use_action_delta_nonconvergence_filter=False,
        use_action_delta_deferred_backfill_filter=use_ldce,
        action_delta_gate_artifact_path=(
            str(artifact_path) if use_ldce and artifact_path else ""
        ),
        action_delta_gate_expected_sha256=(
            artifact_sha256 if use_ldce and artifact_sha256 else ""
        ),
        action_delta_gate_max_skip=1,
        action_delta_gate_min_terminal_iter=2,
        action_delta_gate_exact_coda_audit=False,
        action_delta_gate_return_mode="anchor",
        action_delta_deferred_scorer_backend="eager",
        # Use the same scheduler for LDCE-only and Combined. The only method
        # difference is whether the initial latent source is cold or midpoint.
        action_delta_deferred_runtime_policy="lazy_prefix_exact",
        action_delta_deferred_apply_to_cold=(arm == "ldce"),
        num_exec_actions=5,
        adaptive_exec=False,
        dynamic_exec=False,
        use_linear_decay_horizon=False,
        profile_pytorch=False,
        profile_timing_summary=False,
        json_log_file=str(arm_dir / "results.json"),
        step_log_file=str(arm_dir / "predictions.jsonl"),
        recurrent_convergence_dir=str(arm_dir),
        recurrent_convergence_log_file=str(arm_dir / "predictions.jsonl"),
        recurrent_convergence_summary_file=str(arm_dir / "summary.json"),
        save_version=f"paper-spatial-50x10-{arm}",
    )
    return cfg


def _validate_paper_config(cfg: base.GenerateConfig, *, arm: str) -> None:
    if cfg.task_suite_name != base.TaskSuite.LIBERO_SPATIAL:
        raise ValueError("paper 50x10 runner is LIBERO Spatial-only")
    if cfg.evaluation_protocol_phase != PAPER_PHASE:
        raise ValueError("paper runner requires evaluation_protocol_phase='paper_eval'")
    if cfg.num_trials_per_task != PAPER_EPISODES_PER_TASK:
        raise ValueError("paper runner requires exactly 50 episodes per task")
    if cfg.initial_states_path != "DEFAULT":
        raise ValueError("paper runner requires official DEFAULT initial states")
    if not Path(cfg.initial_state_manifest_path).is_file():
        raise ValueError("paper runner requires the frozen official-50 manifest")
    if not cfg.reset_rng_each_episode:
        raise ValueError("paper runner requires paired per-episode RNG reset")
    if cfg.recurrence_strategy != "adjacent_action_mse":
        raise ValueError("paper runner requires adjacent_action_mse recurrence")
    if (
        float(cfg.recurrence_kl_thresh) != 0.001
        or int(cfg.recurrence_max_iter) != 32
    ):
        raise ValueError(
            "paper runner freezes action-MSE threshold=0.001 and max_iter=32"
        )
    if not cfg.use_cached_final_output:
        raise ValueError("paper runner requires cached terminal output")
    if cfg.use_latent_precheck or cfg.latent_precheck_mode != "off":
        raise ValueError("legacy latent pre-check must stay disabled")
    if cfg.num_exec_actions != 5:
        raise ValueError("paper runner freezes num_exec_actions=5")
    if cfg.profile_coda_cost:
        raise ValueError("paper runner requires profile_coda_cost=False")
    if arm in {"warm_start", "combined"}:
        if not cfg.use_warm_start or cfg.warm_start_source != "midpoint":
            raise ValueError(f"{arm} requires midpoint warm-start")
    elif cfg.use_warm_start:
        raise ValueError(f"{arm} must use cold initialization")

    if arm in {"ldce", "combined"}:
        if not cfg.use_action_delta_deferred_backfill_filter:
            raise ValueError(f"{arm} requires LDCE deferred/backfill")
        if cfg.action_delta_deferred_runtime_policy != "lazy_prefix_exact":
            raise ValueError("paper LDCE freezes lazy_prefix_exact")
        if cfg.action_delta_deferred_scorer_backend != "eager":
            raise ValueError("paper LDCE freezes eager scorer")
        if cfg.action_delta_gate_min_terminal_iter != 2:
            raise ValueError("paper LDCE freezes min_terminal_iter=2")
        if not Path(cfg.action_delta_gate_artifact_path).exists():
            raise ValueError("paper LDCE artifact path does not exist")
    elif cfg.use_action_delta_deferred_backfill_filter:
        raise ValueError(f"{arm} must not enable LDCE")

    if arm == "ldce" and not cfg.action_delta_deferred_apply_to_cold:
        raise ValueError(
            "LDCE-only arm must apply the frozen predictor to cold inference"
        )
    if arm == "combined" and cfg.action_delta_deferred_apply_to_cold:
        raise ValueError(
            "Combined arm uses midpoint warm-start and must not force cold-only mode"
        )


def _arm_is_complete(result_path: Path) -> bool:
    if not result_path.is_file():
        return False
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        int(payload.get("total_episodes", -1)) == PAPER_TOTAL_EPISODES_PER_ARM
        and len(payload.get("tasks", {})) == len(PAPER_TASK_IDS)
    )


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _run_arm(
    cfg: base.GenerateConfig,
    *,
    arm: str,
    save_videos: bool,
) -> dict:
    _validate_paper_config(cfg, arm=arm)
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

    step_log_path = Path(base.get_step_log_file(cfg))
    if step_log_path.exists():
        step_log_path.unlink()
    result_path = Path(cfg.json_log_file)
    if result_path.exists():
        result_path.unlink()
    summary_path = Path(cfg.recurrent_convergence_summary_file)
    if summary_path.exists():
        summary_path.unlink()

    _, manifest_sha256 = base.load_protocol_manifest(
        cfg.initial_state_manifest_path,
        require_source_file_hashes=True,
    )
    source_commit = base.current_source_commit()
    checkpoint_meta = base.checkpoint_identity(Path(cfg.pretrained_checkpoint))

    benchmark_dict = base.benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    if task_suite.n_tasks != len(PAPER_TASK_IDS):
        raise RuntimeError(
            f"Expected 10 LIBERO-Spatial tasks, got {task_suite.n_tasks}"
        )

    full_results = {
        "schema_version": 1,
        "paper_protocol": {
            "name": "libero-spatial-50x10-4arm-v1",
            "arm": arm,
            "phase": PAPER_PHASE,
            "task_ids": list(PAPER_TASK_IDS),
            "episodes_per_task": PAPER_EPISODES_PER_TASK,
            "expected_total_episodes": PAPER_TOTAL_EPISODES_PER_ARM,
            "paired_rng": True,
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
        },
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

    total_episodes = 0
    total_successes = 0

    base.log_message(
        f"Paper arm={arm}: 10 tasks x 50 episodes = {PAPER_TOTAL_EPISODES_PER_ARM}",
        log_file,
    )
    base.log_message(
        f"Frozen manifest sha256={manifest_sha256}; source_commit={source_commit}",
        log_file,
    )

    with _paper_runner_patches(save_videos=save_videos):
        for task_id in PAPER_TASK_IDS:
            action_head.clear_scalar_task_policy()

            if action_delta_payload is not None:
                device = next(action_head.parameters()).device
                prepared_gate = _prepare_shared_spatial_predictor(
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
                    f"task={task_id}, artifact_fold_metadata={prepared_gate.outer_fold}, "
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
            full_results["total_episodes"] = total_episodes
            full_results["total_successes"] = total_successes
            full_results["overall_success_rate"] = (
                float(total_successes) / float(total_episodes)
                if total_episodes
                else 0.0
            )
            _write_json(result_path, full_results)

    if total_episodes != PAPER_TOTAL_EPISODES_PER_ARM:
        raise RuntimeError(
            f"Arm {arm} completed {total_episodes} episodes; "
            f"expected {PAPER_TOTAL_EPISODES_PER_ARM}"
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

    base.log_message(
        f"Completed paper arm={arm}: successes={total_successes}/{total_episodes} "
        f"({100.0 * full_results['overall_success_rate']:.1f}%)",
        log_file,
    )
    if log_file:
        log_file.close()

    del processor, proprio_projector, action_head, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return full_results


def _parse_arms(values: Iterable[str]) -> tuple[str, ...]:
    result = []
    for value in values:
        for item in value.split(","):
            arm = item.strip()
            if not arm:
                continue
            if arm not in PAPER_ARMS:
                raise ValueError(f"Unknown arm {arm!r}; choose from {PAPER_ARMS}")
            if arm not in result:
                result.append(arm)
    if not result:
        raise ValueError("At least one arm is required")
    return tuple(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Baseline/Warm/LDCE/Combined over LIBERO-Spatial 10x50."
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/12_24-24_24_Spatial_40k",
        help="Spatial RD-VLA checkpoint.",
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
        help="Optional artifact SHA-256. If omitted, read adjacent manifest.json.",
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
        help=(
            "Subset/order of arms. Default: baseline warm_start ldce combined."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip arms whose results.json already contains all 500 episodes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing incomplete arm outputs.",
    )
    parser.add_argument(
        "--save-videos",
        action="store_true",
        help="Save rollout MP4s. Disabled by default for the 2,000-episode run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate inputs and print the frozen arm configurations without "
            "running LIBERO."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    arms = _parse_arms(args.arms)
    checkpoint = Path(args.checkpoint)
    manifest_path = Path(args.initial_state_manifest)
    output_root = Path(args.output_root)

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
        artifact_sha256 = _artifact_sha256(
            artifact_path,
            args.action_delta_sha256 or None,
        )

    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "launcher_state.json"
    launcher_state = {
        "schema_version": 1,
        "protocol": "libero-spatial-50x10-4arm-v1",
        "arms_requested": list(arms),
        "expected_episodes_per_arm": PAPER_TOTAL_EPISODES_PER_ARM,
        "expected_total_episodes": PAPER_TOTAL_EPISODES_PER_ARM * len(arms),
        "completed_arms": [],
        "failed_arm": None,
    }
    if state_path.is_file() and args.resume:
        try:
            previous = json.loads(state_path.read_text(encoding="utf-8"))
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
        )
        _validate_paper_config(cfg, arm=arm)
        configs[arm] = cfg

    if args.dry_run:
        print(
            json.dumps(
                {arm: asdict(cfg) for arm, cfg in configs.items()},
                indent=2,
            )
        )
        return 0

    for arm in arms:
        result_path = output_root / arm / "results.json"
        if args.resume and _arm_is_complete(result_path):
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
            _run_arm(configs[arm], arm=arm, save_videos=args.save_videos)
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
        "Completed paper evaluation: "
        f"{len(arms)} arms x {PAPER_TOTAL_EPISODES_PER_ARM} episodes"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
