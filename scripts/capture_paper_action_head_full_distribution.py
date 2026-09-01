#!/usr/bin/env python3
"""Capture the full paper prediction distribution for Action-head replay.

This is a *source workload capture* run, not a latency measurement.  It reruns
only the two source trajectories needed by the paper Action-head replay:

    baseline   -> source distribution for Baseline and LDCE replay
    warm_start -> source distribution for Warm-start and Combined replay

Each source arm executes the frozen LIBERO-Spatial 10 x 50 paper protocol.  A
post-VLM Action-head workload is captured for every policy prediction.  To keep
storage practical, the exact production boundary workload is compacted to the
unique VLM hidden layers consumed by the frozen Spatial Action head (11, 23)
before it is written to disk.  Replay reconstructs the original layer axis
outside the timed region and calls the unmodified ActionHeadRecurrent.

Important measurement rule
--------------------------
The source rollout's latency_ms is invalid for paper latency reporting because
capturing a workload performs GPU -> CPU copies before get_action returns.  The
captured files are only inputs to the separate replay benchmark.

By default the script also verifies every source prediction against the already
completed 2,000-episode paper rollout (identity, K, warm-start use, and episode
success) so capture is fail-closed if the source distribution drifts.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import experiments.robot.libero.run_libero_eval as base  # noqa: E402
import scripts.run_spatial_paper_4arm as frozen  # noqa: E402
from scripts.paper_action_head_full_distribution_lib import (  # noqa: E402
    COMPACT_WORKLOAD_SCHEMA_VERSION,
    COMPACT_WORKLOAD_TYPE,
    append_jsonl,
    load_jsonl,
    save_compact_workload,
    validate_frozen_spatial_action_head,
)


SOURCE_ARMS = ("baseline", "warm_start")
PROTOCOL_NAME = "paper-action-head-full-distribution-capture-v1"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "benchmark_results/paper_action_head_full_distribution"
)
DEFAULT_REFERENCE_ROOT = REPO_ROOT / "benchmark_results/paper_spatial_50x10"
DEFAULT_MAX_STORAGE_GIB = 80.0
CAPTURE_ALL_PREDICTIONS_LIMIT = 1_000_000


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _parse_arms(values: Iterable[str]) -> tuple[str, ...]:
    arms: list[str] = []
    for value in values:
        for raw in value.split(","):
            arm = raw.strip()
            if not arm:
                continue
            if arm not in SOURCE_ARMS:
                raise ValueError(f"Unknown source arm {arm!r}; choose from {SOURCE_ARMS}")
            if arm not in arms:
                arms.append(arm)
    if not arms:
        raise ValueError("At least one source arm is required")
    return tuple(arms)


def _prediction_identity(record: Mapping[str, Any]) -> tuple[int, int, int, int, int]:
    return (
        int(record["task_id"]),
        int(record["paired_trial_id"]),
        int(record["prediction_step"]),
        int(record["initial_state_id"]),
        int(record["episode_seed"]),
    )


def _minimal_reference_index(path: Path) -> dict[int, list[dict[str, Any]]]:
    """Stream a large paper JSONL once and retain only parity-critical fields."""

    by_task = {task_id: [] for task_id in frozen.PAPER_TASK_IDS}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid reference JSONL at {path}:{line_number}") from exc
            task_id = int(record["task_id"])
            by_task[task_id].append(
                {
                    "identity": _prediction_identity(record),
                    "K_t": int(record["K_t"]),
                    "warm_start_used": bool(record.get("warm_start_used", False)),
                    "success": bool(record["success"]),
                }
            )
    for task_id in by_task:
        by_task[task_id].sort(key=lambda item: item["identity"])
    return by_task


def _load_reference_indices(
    reference_root: Path,
    arms: Iterable[str],
) -> dict[str, dict[int, list[dict[str, Any]]]]:
    result = {}
    for arm in arms:
        path = reference_root / arm / "predictions.jsonl"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing completed-paper reference for {arm}: {path}. "
                "Use --skip-reference-parity only for a deliberate non-formal capture."
            )
        print(f"Indexing reference predictions: {path}", flush=True)
        result[arm] = _minimal_reference_index(path)
    return result


def _estimated_compact_tensor_bytes(
    reference_index: Mapping[int, list[Mapping[str, Any]]]
) -> dict[str, int]:
    # Frozen Spatial dimensions.  These are independently checked against the
    # loaded Action head before capture starts.
    hidden_bytes = 1 * 2 * (512 + 8) * 896 * 2
    proprio_input_bytes = 1 * 8 * 2
    proprio_feature_bytes = 1 * 1 * 896 * 2
    selected_state_bytes = 1 * 8 * 896 * 2
    all_records = [record for rows in reference_index.values() for record in rows]
    warm_records = sum(bool(record["warm_start_used"]) for record in all_records)
    count = len(all_records)
    total = count * (
        hidden_bytes
        + proprio_input_bytes
        + proprio_feature_bytes
        + selected_state_bytes
    ) + warm_records * selected_state_bytes
    return {
        "prediction_count": int(count),
        "actual_warm_prediction_count": int(warm_records),
        "raw_compact_tensor_bytes": int(total),
    }


def _validate_source_config(cfg: base.GenerateConfig, *, arm: str) -> None:
    _require(arm in SOURCE_ARMS, f"unsupported source arm: {arm}")
    _require(cfg.task_suite_name == base.TaskSuite.LIBERO_SPATIAL, "capture is Spatial-only")
    _require(cfg.num_trials_per_task == frozen.PAPER_EPISODES_PER_TASK, "capture requires 50 episodes/task")
    _require(cfg.initial_states_path == "DEFAULT", "capture requires official initial states")
    _require(cfg.reset_rng_each_episode, "capture requires paired per-episode RNG reset")
    _require(cfg.use_recurrent, "capture requires recurrent inference")
    _require(cfg.recurrence_strategy == "adjacent_action_mse", "capture requires adjacent action MSE")
    _require(float(cfg.recurrence_kl_thresh) == 0.001, "capture freezes action-MSE threshold=0.001")
    _require(int(cfg.recurrence_max_iter) == 32, "capture freezes max_iter=32")
    _require(cfg.use_cached_final_output, "capture requires cached final output")
    _require(not cfg.use_latent_precheck and cfg.latent_precheck_mode == "off", "legacy pre-check must be off")
    _require(not cfg.use_action_delta_deferred_backfill_filter, "source capture must not run LDCE")
    _require(not cfg.profile_coda_cost and not cfg.profile_timing_summary, "profiling must stay off during capture")
    _require(cfg.num_exec_actions == 5, "capture freezes num_exec_actions=5")
    _require(cfg.evaluation_protocol_phase == "calibration", "internal capture trigger must use calibration phase")
    _require(
        int(cfg.calibration_workload_predictions_per_episode) == CAPTURE_ALL_PREDICTIONS_LIMIT,
        "capture-all prediction limit changed",
    )
    if arm == "baseline":
        _require(not cfg.use_warm_start, "baseline source must be cold")
    else:
        _require(cfg.use_warm_start and cfg.warm_start_source == "midpoint", "warm source requires midpoint warm-start")
        _require(int(cfg.warm_start_min_iter) == 2, "warm source freezes min_iter=2")


def _paper_trials_through_calibration_switch(
    *,
    manifest,
    phase,
    task_id,
    initial_states,
    base_seed,
    initial_state_file_path=None,
):
    """Return the exact paper_eval 50 trials while the capture switch says calibration."""

    _require(phase == "calibration", f"unexpected capture phase: {phase}")
    return frozen._resolve_paper_trials(
        manifest=manifest,
        phase=frozen.PAPER_PHASE,
        task_id=task_id,
        initial_states=initial_states,
        base_seed=base_seed,
        initial_state_file_path=initial_state_file_path,
    )


def _compact_workload_saver(
    cfg,
    workload,
    *,
    capture_requested: bool,
    identity: Mapping[str, Any] | None,
):
    fields = {
        "action_head_workload_requested": bool(capture_requested),
        "action_head_workload_captured": False,
        "action_head_workload_file": None,
        "action_head_workload_sha256": None,
        "action_head_workload_schema_version": None,
        "action_head_workload_tensor_fields": [],
        "action_head_workload_capture_in_action_latency": bool(capture_requested),
        "action_head_workload_type": None,
    }
    if not capture_requested:
        _require(workload is None, "workload produced without capture request")
        return fields
    _require(isinstance(workload, Mapping), "requested workload payload is missing")
    _require(isinstance(identity, Mapping), "requested workload identity is missing")

    state = getattr(cfg, "_paper_full_distribution_capture_state", None)
    _require(isinstance(state, dict), "full-distribution capture state is missing")
    task_dir = Path(state["task_dir"])
    layer_indices = tuple(int(value) for value in state["layer_indices"])
    source_arm = str(state["source_arm"])
    filename = (
        f"task{int(identity['task_id'])}_trial{int(identity['paired_trial_id'])}_"
        f"pred{int(identity['prediction_step'])}.pt"
    )
    output_path = task_dir / "workloads" / filename
    descriptor = save_compact_workload(
        output_path,
        workload,
        identity=identity,
        layer_indices=layer_indices,
        source_arm=source_arm,
    )
    state["workload_count"] += 1
    state["written_file_bytes"] += int(descriptor["file_bytes"])
    state["compact_tensor_bytes"] += int(descriptor["compact_tensor_bytes"])
    state["source_tensor_bytes"] += int(descriptor["source_tensor_bytes"])
    max_bytes = int(state["max_storage_bytes"])
    if state["written_file_bytes"] > max_bytes:
        raise RuntimeError(
            "Action-head workload storage guard exceeded: "
            f"written={state['written_file_bytes'] / (1024**3):.2f} GiB > "
            f"limit={max_bytes / (1024**3):.2f} GiB"
        )

    step_log_dir = Path(base.get_step_log_file(cfg)).resolve().parent
    try:
        recorded_path = str(output_path.resolve().relative_to(step_log_dir))
    except ValueError:
        recorded_path = str(output_path.resolve())
    fields.update(
        {
            "action_head_workload_captured": True,
            "action_head_workload_file": recorded_path,
            "action_head_workload_sha256": descriptor["sha256"],
            "action_head_workload_schema_version": COMPACT_WORKLOAD_SCHEMA_VERSION,
            "action_head_workload_tensor_fields": [
                "selected_actions_hidden_states",
                "proprio_input",
                "proprio_features",
                "incoming_warm_start_state",
                "selected_initial_state",
            ],
            "action_head_workload_type": COMPACT_WORKLOAD_TYPE,
            "action_head_workload_compact_selected_layer_indices": list(layer_indices),
            "action_head_workload_file_bytes": int(descriptor["file_bytes"]),
            "action_head_workload_compact_tensor_bytes": int(descriptor["compact_tensor_bytes"]),
            "action_head_workload_source_tensor_bytes": int(descriptor["source_tensor_bytes"]),
            "action_head_workload_source_actions_hidden_states_bytes": int(
                descriptor["source_actions_hidden_states_bytes"]
            ),
        }
    )
    return fields


@contextmanager
def _capture_patches():
    original_resolver = base.resolve_phase_trials
    original_saver = base.save_prediction_action_head_workload
    original_video_stats = base.save_rollout_video_with_stats
    original_video = base.save_rollout_video
    base.resolve_phase_trials = _paper_trials_through_calibration_switch
    base.save_prediction_action_head_workload = _compact_workload_saver
    base.save_rollout_video_with_stats = lambda *args, **kwargs: None
    base.save_rollout_video = lambda *args, **kwargs: None
    try:
        yield
    finally:
        base.resolve_phase_trials = original_resolver
        base.save_prediction_action_head_workload = original_saver
        base.save_rollout_video_with_stats = original_video_stats
        base.save_rollout_video = original_video


def _validate_task_parity(
    captured_path: Path,
    reference_rows: list[Mapping[str, Any]],
    *,
    task_id: int,
    arm: str,
) -> dict[str, Any]:
    captured = load_jsonl(captured_path)
    captured_rows = [
        {
            "identity": _prediction_identity(record),
            "K_t": int(record["K_t"]),
            "warm_start_used": bool(record.get("warm_start_used", False)),
            "success": bool(record["success"]),
        }
        for record in captured
    ]
    captured_rows.sort(key=lambda item: item["identity"])
    _require(
        len(captured_rows) == len(reference_rows),
        f"{arm} task{task_id}: prediction count mismatch: "
        f"capture={len(captured_rows)}, reference={len(reference_rows)}",
    )
    mismatches = []
    for index, (actual, expected) in enumerate(zip(captured_rows, reference_rows)):
        differences = {
            field: {"capture": actual[field], "reference": expected[field]}
            for field in ("identity", "K_t", "warm_start_used", "success")
            if actual[field] != expected[field]
        }
        if differences:
            mismatches.append({"index": index, "differences": differences})
            if len(mismatches) >= 10:
                break
    _require(not mismatches, f"{arm} task{task_id}: source parity failed: {mismatches[:3]}")
    return {
        "passed": True,
        "prediction_count": len(captured_rows),
        "checked_fields": ["identity", "K_t", "warm_start_used", "success"],
    }


def _task_is_complete(task_dir: Path, *, reference_count: int | None) -> bool:
    summary_path = task_dir / "capture_summary.json"
    if not summary_path.is_file():
        return False
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if payload.get("completed") is not True:
        return False
    if reference_count is not None and int(payload.get("prediction_count", -1)) != reference_count:
        return False
    return True


def _run_source_arm(
    *,
    arm: str,
    checkpoint: Path,
    manifest_path: Path,
    output_root: Path,
    seed: int,
    max_storage_bytes: int,
    reference_index: dict[int, list[dict[str, Any]]] | None,
    resume: bool,
) -> dict[str, Any]:
    arm_root = output_root / "capture" / arm
    cfg = frozen._build_arm_config(
        arm=arm,
        checkpoint=checkpoint,
        manifest_path=manifest_path,
        artifact_path=None,
        artifact_sha256=None,
        output_root=arm_root / "_config_scratch",
        seed=seed,
    )
    # Existing run_episode() uses these two legacy fields as its workload-capture
    # switch.  We deliberately do not call base.validate_config(); our strict
    # validator below freezes the paper algorithm while the resolver patch maps
    # the switch back to the exact paper_eval 50-state trial set.
    cfg.evaluation_protocol_phase = "calibration"
    cfg.calibration_workload_predictions_per_episode = CAPTURE_ALL_PREDICTIONS_LIMIT
    cfg.calibration_workload_dir = str(arm_root / "workloads")
    cfg.local_log_dir = str(arm_root / "logs")
    cfg.run_id_note = f"paper_action_head_full_distribution_capture_{arm}"
    _validate_source_config(cfg, arm=arm)

    base.set_seed_everywhere(cfg.seed)
    model, action_head, proprio_projector, processor = base.initialize_model(cfg)
    resize_size = base.get_image_resize_size(cfg)
    base.RDVLAProfiler.set_enabled(False)
    base.RDVLAProfiler.set_timing_enabled(False)
    layer_indices = validate_frozen_spatial_action_head(action_head)
    source_commit = _git_commit()

    benchmark_dict = base.benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    _require(task_suite.n_tasks == 10, f"Expected 10 Spatial tasks, got {task_suite.n_tasks}")

    arm_summary: dict[str, Any] = {
        "schema_version": 1,
        "protocol": PROTOCOL_NAME,
        "source_arm": arm,
        "source_commit": source_commit,
        "checkpoint": str(checkpoint.resolve()),
        "initial_state_manifest": str(manifest_path.resolve()),
        "selected_vlm_layers": list(layer_indices),
        "capture_all_predictions_limit": CAPTURE_ALL_PREDICTIONS_LIMIT,
        "source_latency_valid_for_paper": False,
        "source_latency_exclusion_reason": (
            "GPU-to-CPU workload capture occurs before get_action returns"
        ),
        "tasks": {},
        "completed": False,
    }

    total_episodes = 0
    total_successes = 0
    with _capture_patches():
        for task_id in frozen.PAPER_TASK_IDS:
            task_dir = arm_root / f"task{task_id}"
            ref_rows = None if reference_index is None else reference_index[task_id]
            if resume and _task_is_complete(
                task_dir,
                reference_count=None if ref_rows is None else len(ref_rows),
            ):
                task_summary = json.loads(
                    (task_dir / "capture_summary.json").read_text(encoding="utf-8")
                )
                arm_summary["tasks"][str(task_id)] = task_summary
                print(f"[resume] {arm} task{task_id} already captured", flush=True)
                continue

            if task_dir.exists() and any(task_dir.iterdir()):
                raise FileExistsError(
                    f"Incomplete capture output exists: {task_dir}. "
                    "Remove it before rerunning that task."
                )
            task_dir.mkdir(parents=True, exist_ok=True)
            cfg.step_log_file = str(task_dir / "predictions.jsonl")
            cfg.recurrent_convergence_log_file = cfg.step_log_file
            cfg.recurrent_convergence_summary_file = str(task_dir / "summary.json")
            cfg.calibration_workload_dir = str(task_dir / "workloads")
            capture_state = {
                "task_dir": str(task_dir),
                "layer_indices": list(layer_indices),
                "source_arm": arm,
                "workload_count": 0,
                "written_file_bytes": 0,
                "compact_tensor_bytes": 0,
                "source_tensor_bytes": 0,
                "max_storage_bytes": int(max_storage_bytes),
            }
            cfg._paper_full_distribution_capture_state = capture_state
            task_log_path = arm_root / "logs" / f"task{task_id}.log"
            task_log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = task_log_path.open("w", encoding="utf-8")
            try:
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
                    f"paper-ah-full-distribution-{arm}",
                    None,
                    None,
                    None,
                    None,
                    source_commit=source_commit,
                )
            finally:
                log_file.close()

            prediction_count = int(capture_state["workload_count"])
            parity = None
            if ref_rows is not None:
                parity = _validate_task_parity(
                    Path(cfg.step_log_file),
                    ref_rows,
                    task_id=task_id,
                    arm=arm,
                )
                _require(prediction_count == len(ref_rows), "capture saver/reference count mismatch")
            task_summary = {
                "schema_version": 1,
                "protocol": PROTOCOL_NAME,
                "source_arm": arm,
                "task_id": int(task_id),
                "episodes": len(task_stats),
                "successes": sum(bool(row["success"]) for row in task_stats),
                "prediction_count": prediction_count,
                "actual_warm_prediction_count": sum(
                    bool(record.get("warm_start_used", False))
                    for record in load_jsonl(Path(cfg.step_log_file))
                ),
                "workload_file_bytes": int(capture_state["written_file_bytes"]),
                "compact_tensor_bytes": int(capture_state["compact_tensor_bytes"]),
                "source_tensor_bytes_before_compaction": int(capture_state["source_tensor_bytes"]),
                "compaction_tensor_byte_ratio": (
                    float(capture_state["compact_tensor_bytes"])
                    / float(capture_state["source_tensor_bytes"])
                    if capture_state["source_tensor_bytes"]
                    else None
                ),
                "reference_parity": parity,
                "completed": True,
            }
            _write_json(task_dir / "capture_summary.json", task_summary)
            arm_summary["tasks"][str(task_id)] = task_summary
            _write_json(arm_root / "capture_summary.json", arm_summary)
            print(
                f"Captured {arm} task{task_id}: {prediction_count} predictions, "
                f"files={capture_state['written_file_bytes'] / (1024**3):.2f} GiB",
                flush=True,
            )

    del cfg._paper_full_distribution_capture_state
    del processor, proprio_projector, action_head, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    all_task_summaries = list(arm_summary["tasks"].values())
    arm_summary.update(
        {
            "prediction_count": int(sum(row["prediction_count"] for row in all_task_summaries)),
            "actual_warm_prediction_count": int(
                sum(row["actual_warm_prediction_count"] for row in all_task_summaries)
            ),
            "workload_file_bytes": int(
                sum(row["workload_file_bytes"] for row in all_task_summaries)
            ),
            "compact_tensor_bytes": int(
                sum(row["compact_tensor_bytes"] for row in all_task_summaries)
            ),
            "source_tensor_bytes_before_compaction": int(
                sum(row["source_tensor_bytes_before_compaction"] for row in all_task_summaries)
            ),
            "completed": len(all_task_summaries) == 10,
        }
    )
    _write_json(arm_root / "capture_summary.json", arm_summary)
    return arm_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "outputs/12_24-24_24_Spatial_40k",
    )
    parser.add_argument(
        "--initial-state-manifest",
        type=Path,
        default=(
            REPO_ROOT
            / "experiments/robot/libero/manifests/libero_spatial_official_50_v1.json"
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--reference-paper-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--arms", nargs="+", default=list(SOURCE_ARMS))
    parser.add_argument("--max-storage-gib", type=float, default=DEFAULT_MAX_STORAGE_GIB)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--skip-reference-parity",
        action="store_true",
        help="Development only: capture without matching the completed paper rollout.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print storage estimates without running LIBERO.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    arms = _parse_arms(args.arms)
    checkpoint = args.checkpoint.resolve()
    manifest_path = args.initial_state_manifest.resolve()
    output_root = args.output_root.resolve()
    _require(checkpoint.exists(), f"Checkpoint does not exist: {checkpoint}")
    _require(manifest_path.is_file(), f"Manifest does not exist: {manifest_path}")
    _require(args.max_storage_gib > 0, "--max-storage-gib must be positive")
    max_storage_bytes = int(float(args.max_storage_gib) * (1024**3))

    reference_indices = None
    if not args.skip_reference_parity:
        reference_indices = _load_reference_indices(args.reference_paper_root.resolve(), arms)
        for arm in arms:
            estimate = _estimated_compact_tensor_bytes(reference_indices[arm])
            safety_estimate = int(estimate["raw_compact_tensor_bytes"] * 1.10)
            print(
                f"{arm}: reference predictions={estimate['prediction_count']}, "
                f"warm={estimate['actual_warm_prediction_count']}, "
                f"raw compact tensors~{estimate['raw_compact_tensor_bytes'] / (1024**3):.2f} GiB",
                flush=True,
            )
            _require(
                safety_estimate <= max_storage_bytes,
                f"{arm}: projected compact capture with 10% file-overhead guard "
                f"({safety_estimate / (1024**3):.2f} GiB) exceeds --max-storage-gib",
            )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "protocol": PROTOCOL_NAME,
                    "arms": list(arms),
                    "checkpoint": str(checkpoint),
                    "manifest": str(manifest_path),
                    "output_root": str(output_root),
                    "max_storage_gib": args.max_storage_gib,
                    "reference_parity": not args.skip_reference_parity,
                },
                indent=2,
            )
        )
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    run_summary: dict[str, Any] = {
        "schema_version": 1,
        "protocol": PROTOCOL_NAME,
        "source_commit": _git_commit(),
        "arms": {},
        "completed": False,
    }
    for arm in arms:
        summary = _run_source_arm(
            arm=arm,
            checkpoint=checkpoint,
            manifest_path=manifest_path,
            output_root=output_root,
            seed=args.seed,
            max_storage_bytes=max_storage_bytes,
            reference_index=None if reference_indices is None else reference_indices[arm],
            resume=args.resume,
        )
        run_summary["arms"][arm] = summary
        _write_json(output_root / "capture_summary.json", run_summary)

    run_summary["completed"] = all(
        value.get("completed") is True for value in run_summary["arms"].values()
    )
    run_summary["prediction_count"] = int(
        sum(value.get("prediction_count", 0) for value in run_summary["arms"].values())
    )
    run_summary["workload_file_bytes"] = int(
        sum(value.get("workload_file_bytes", 0) for value in run_summary["arms"].values())
    )
    _write_json(output_root / "capture_summary.json", run_summary)
    print(
        f"Full-distribution capture complete: {run_summary['prediction_count']} workloads, "
        f"{run_summary['workload_file_bytes'] / (1024**3):.2f} GiB",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
