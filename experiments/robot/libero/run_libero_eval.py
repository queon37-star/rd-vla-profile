import json
import logging
import os
import random
import shutil
import sys
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import torch
import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

import wandb

sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK
from prismatic.utils.rdvla_profiler import RDVLAProfiler


class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)



@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    use_minivlm: bool = True                         # If True, uses minivlm
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy
    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    task_id: Optional[int] = None                    # If set, only run this specific task ID
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)
    reset_rng_each_episode: bool = False             # Reset policy/environment RNG once per episode
    episode_seed_stride: int = 1                     # Offset between consecutive episode seeds

    # fmt: on
    save_version: str = "rd-vla"                     # Version tag for saved videos
    phase: str = "Inference"

    use_recurrent: bool = False

    # Recurrence strategy: "fixed", "kl_divergence", or "cosine_similarity"
    recurrence_strategy: str = "fixed"
    recurrent_num_iter: int = 12
    recurrence_kl_thresh: float = 0.001
    recurrence_cos_thresh: float = 0.999
    recurrence_max_iter: int = 32

    # Disabled-by-default warm-start inference settings.
    use_warm_start: bool = False
    warm_start_source: str = "s1"
    warm_start_min_iter: int = 2
    validate_warm_start_finite: bool = False

    # Adaptive recurrence Coda profiling and final-output cache comparison.
    profile_coda_cost: bool = False
    use_cached_final_output: bool = False

    # Optional PyTorch Profiler instrumentation for RD-VLA action inference.
    profile_pytorch: bool = False
    profile_steps: int = 1
    profile_trace_path: str = "./profiles/rdvla_pytorch_trace.json"
    profile_record_shapes: bool = False
    profile_memory: bool = False
    profile_timing_summary: bool = False
    profile_timing_summary_path: str = "./benchmark_results/profiler/rdvla_timing_summary.jsonl"
    profile_timing_steps: int = 1
    profile_timing_cuda_sync: bool = False

    # Optional latent-state pre-check before running Coda in adaptive recurrence.
    use_latent_precheck: bool = False
    latent_precheck_thresh: float = 0.12
    latent_precheck_min_iter: int = 2
    latent_precheck_force_interval: int = 0

    # Fixed execution: always use first N actions
    num_exec_actions: int = 5

    # Adaptive execution strategy (fixed threshold)
    adaptive_exec: bool = False
    adaptive_exec_threshold: int = 4  # Iteration threshold for switching
    adaptive_exec_low: int = 4        # Actions when iters <= threshold (fast/uncertain)
    adaptive_exec_high: int = 8       # Actions when iters > threshold (slow/confident)

    # Linear decay horizon execution strategy
    use_linear_decay_horizon: bool = False  # Map iters to action count via linear decay

    # Dynamic execution strategy (mean/std based, 4 buckets: 2, 4, 6, 8 actions)
    dynamic_exec: bool = False        # Use dynamic mean/std based action count
    dynamic_exec_warmup: int = 5      # Episodes before using dynamic thresholds

    # JSON results logging
    json_log_file: str = ""           # Path to save JSON results (empty = disabled)

    # Step-level recurrence metric log.
    # If empty, the path will be derived from json_log_file.
    step_log_file: str = ""

    # Recurrent convergence metric outputs.
    recurrent_convergence_dir: str = "./benchmark_results/recurrent_convergence"
    recurrent_convergence_log_file: str = ""
    recurrent_convergence_summary_file: str = ""



def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"
    supported_warm_start_sources = {"s1", "midpoint", "final"}
    if cfg.warm_start_source not in supported_warm_start_sources:
        raise ValueError(
            f"Unsupported warm_start_source: {cfg.warm_start_source}. "
            f"Expected one of {sorted(supported_warm_start_sources)}"
        )
    if cfg.use_warm_start:
        assert cfg.use_recurrent, "Warm-start requires recurrent inference"
    if cfg.warm_start_min_iter < 2:
        raise ValueError("warm_start_min_iter must be at least 2")
    if cfg.episode_seed_stride < 1:
        raise ValueError("episode_seed_stride must be at least 1")

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"

    if cfg.profile_pytorch:
        assert cfg.profile_steps >= 0, "profile_steps must be non-negative!"
        assert cfg.profile_trace_path, "profile_trace_path must be non-empty when profile_pytorch is enabled!"

    if cfg.profile_timing_summary:
        assert cfg.profile_timing_steps >= 0, "profile_timing_steps must be non-negative!"
        assert cfg.profile_timing_summary_path, (
            "profile_timing_summary_path must be non-empty when profile_timing_summary is enabled!"
        )


def calculate_linear_decay_horizon(actual_iters: int) -> int:
    """Map recurrence iterations to number of actions via linear decay.

    Few iterations (easy) → execute all 8 actions.
    Many iterations (hard) → execute fewer actions (minimum 2).
    """
    if actual_iters <= 6:
        return 8
    elif actual_iters <= 8:
        return 7
    else:
        # Linear decay from 7 down to minimum 2
        return max(2, 7 - (actual_iters - 8))


def save_rollout_video_with_stats(
    rollout_images, replay_stats, idx, success, task_description, log_file=None, save_version=None
):
    """Save rollout video with thinking steps overlaid on frames."""
    from experiments.robot.robot_utils import DATE, DATE_TIME

    rollout_dir = f"./rollouts/{save_version}/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task}.mp4"

    h, w = rollout_images[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(mp4_path, fourcc, 30, (w, h))

    frame_stats = {}
    frame_cursor = 0
    for iters, num_actions in replay_stats:
        for f in range(frame_cursor, min(frame_cursor + num_actions, len(rollout_images))):
            frame_stats[f] = iters
        frame_cursor += num_actions

    for i, img in enumerate(rollout_images):
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if i in frame_stats:
            cv2.putText(frame, f"Thinking Steps: {frame_stats[i]}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        writer.write(frame)

    writer.release()
    msg = f"Saved rollout MP4 (with stats) at path {mp4_path}"
    print(msg)
    if log_file is not None:
        log_file.write(msg + "\n")
    return mp4_path


def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    model = get_model(cfg)
    model.set_version(cfg.save_version)

    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)

    action_head = get_action_head(cfg, model.llm_dim)

    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, processor


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    if cfg.unnorm_key:
        unnorm_key = cfg.unnorm_key
    else:
        unnorm_key = cfg.task_suite_name
        if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
            unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"
    cfg.unnorm_key = unnorm_key



def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id



def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def _pytorch_profiler_activities():
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return activities


def _profile_trace_path_for_call(base_path: str, profile_call_index: int, profile_steps: int) -> Path:
    path = Path(base_path)
    if profile_steps <= 1:
        return path

    suffix = path.suffix or ".json"
    return path.with_name(f"{path.stem}_step{profile_call_index:04d}{suffix}")


def run_with_optional_pytorch_profile(cfg, profile_state, log_file, action_fn):
    """Run one action inference call, optionally under torch.profiler."""
    profile_steps = max(0, int(getattr(cfg, "profile_steps", 1)))
    if (
        not getattr(cfg, "profile_pytorch", False)
        or profile_state is None
        or profile_state.get("profiled_calls", 0) >= profile_steps
    ):
        return action_fn()

    profile_call_index = profile_state.get("profiled_calls", 0) + 1
    RDVLAProfiler.set_enabled(True)
    try:
        with torch.profiler.profile(
            activities=_pytorch_profiler_activities(),
            record_shapes=getattr(cfg, "profile_record_shapes", False),
            profile_memory=getattr(cfg, "profile_memory", False),
        ) as prof:
            result = action_fn()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    finally:
        RDVLAProfiler.set_enabled(False)

    profile_state["profiled_calls"] = profile_call_index
    sort_by = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
    table = prof.key_averages().table(sort_by=sort_by, row_limit=100)
    header = f"\nPyTorch profiler action inference {profile_call_index}/{profile_steps}"
    print(header)
    print(table)
    if log_file:
        log_file.write(header + "\n")
        log_file.write(table + "\n")
        log_file.flush()

    base_trace_path = Path(getattr(cfg, "profile_trace_path", "") or "./profiles/rdvla_pytorch_trace.json")
    trace_path = _profile_trace_path_for_call(str(base_trace_path), profile_call_index, profile_steps)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(trace_path))

    if trace_path != base_trace_path:
        base_trace_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(trace_path, base_trace_path)
        log_message(f"Saved PyTorch profiler trace to {trace_path}; latest trace copied to {base_trace_path}", log_file)
    else:
        log_message(f"Saved PyTorch profiler trace to {trace_path}", log_file)

    return result


def _timing_ms(timings: Dict[str, float], *names: str) -> Optional[float]:
    for name in names:
        if name in timings:
            return float(timings[name])
    return None


def _build_timing_summary_record(timing_record, result):
    timings = timing_record.get("timings_ms", {})
    counts = timing_record.get("counts", {})
    metadata = dict(timing_record.get("metadata", {}))

    actual_iters = None
    final_kl = None
    recurrence_debug = {}
    if isinstance(result, tuple):
        if len(result) > 1:
            actual_iters = result[1]
        if len(result) > 2:
            final_kl = result[2]
        if len(result) > 3 and result[3] is not None:
            payload = result[3]
            if isinstance(payload, dict):
                recurrence_debug = payload.get("recurrence_debug", payload) or {}
                if not isinstance(recurrence_debug, dict):
                    recurrence_debug = {}

    final_mse = recurrence_debug.get("final_mse")
    if final_mse is None:
        final_mse = recurrence_debug.get("final_conv_score", final_kl)

    record = {
        **metadata,
        "total_ms": _timing_ms(timings, "RDVLA/get_vla_action_total"),
        "prepare_images_ms": _timing_ms(timings, "RDVLA/get_vla_action/prepare_images"),
        "processor_primary_ms": _timing_ms(timings, "RDVLA/get_vla_action/processor_primary"),
        "vla_predict_action_ms": _timing_ms(timings, "RDVLA/get_vla_action/vla_predict_action"),
        "vision_backbone_ms": _timing_ms(timings, "RDVLA/vlm/vision_backbone"),
        "projector_ms": _timing_ms(timings, "RDVLA/vlm/projector"),
        "language_model_forward_ms": _timing_ms(timings, "RDVLA/vlm/language_model_forward"),
        "extract_hidden_states_ms": _timing_ms(
            timings,
            "RDVLA/vlm/extract_hidden_states_total",
            "RDVLA/vlm/extract_hidden_states",
        ),
        "action_head_total_ms": _timing_ms(
            timings,
            "RDVLA/action_head/predict_action_total",
            "RDVLA/action_head/wrapper_total",
        ),
        "init_state_ms": _timing_ms(
            timings,
            "RDVLA/action_head/init_state_total",
            "RDVLA/action_head/init_state",
        ),
        "prelude_ms": _timing_ms(timings, "RDVLA/action_head/prelude_total"),
        "recurrent_loop_ms": _timing_ms(timings, "RDVLA/action_head/recurrent_loop_total"),
        "get_output_each_iter_ms": _timing_ms(timings, "RDVLA/action_head/get_output_each_iter"),
        "final_get_output_ms": _timing_ms(timings, "RDVLA/action_head/final_get_output"),
        "actual_iters": _as_int(actual_iters),
        "final_mse": _as_float(final_mse),
        "final_kl": _as_float(final_kl),
        "stop_reason": recurrence_debug.get("stop_reason"),
        "used_latent_precheck": bool(recurrence_debug.get("use_latent_precheck", False)),
        "used_coda_stop": bool(recurrence_debug.get("adaptive_stop", False)),
        "timings_ms": timings,
        "timing_counts": counts,
    }
    return record


def run_action_with_optional_profiles(
    cfg,
    profile_state,
    timing_state,
    log_file,
    action_fn,
    timing_metadata,
):
    timing_steps = max(0, int(getattr(cfg, "profile_timing_steps", 1)))
    should_time = (
        getattr(cfg, "profile_timing_summary", False)
        and timing_state is not None
        and timing_state.get("timed_calls", 0) < timing_steps
    )

    if not should_time and not getattr(cfg, "profile_pytorch", False):
        return action_fn()

    if should_time:
        RDVLAProfiler.set_timing_cuda_sync(getattr(cfg, "profile_timing_cuda_sync", False))
        RDVLAProfiler.set_timing_enabled(True)
        RDVLAProfiler.start_timing_record(timing_metadata)

    timing_record = None
    try:
        if getattr(cfg, "profile_pytorch", False):
            result = run_with_optional_pytorch_profile(cfg, profile_state, log_file, action_fn)
        else:
            result = action_fn()
    finally:
        if should_time:
            timing_record = RDVLAProfiler.finish_timing_record()
            RDVLAProfiler.set_timing_enabled(False)

    if timing_record is not None:
        timing_state["timed_calls"] = timing_state.get("timed_calls", 0) + 1
        timing_summary = _build_timing_summary_record(timing_record, result)
        append_jsonl(getattr(cfg, "profile_timing_summary_path", None), [timing_summary])

    return result


def append_jsonl(path, records):
    """Append a list of dict records to a JSONL file."""
    if not path or not records:
        return

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_step_log_file(cfg):
    if getattr(cfg, "step_log_file", None):
        return cfg.step_log_file

    json_log_file = getattr(cfg, "json_log_file", None)
    if json_log_file:
        root, _ = os.path.splitext(json_log_file)
        return root + "_step_log.jsonl"

    return None


def configure_recurrent_convergence_paths(cfg, run_id: str):
    """Derive stable JSONL/JSON paths for recurrence convergence metrics."""
    output_dir = getattr(cfg, "recurrent_convergence_dir", "") or "./benchmark_results/recurrent_convergence"
    os.makedirs(output_dir, exist_ok=True)

    if not getattr(cfg, "recurrent_convergence_log_file", None):
        cfg.recurrent_convergence_log_file = os.path.join(output_dir, f"{run_id}_predictions.jsonl")
    if not getattr(cfg, "recurrent_convergence_summary_file", None):
        cfg.recurrent_convergence_summary_file = os.path.join(output_dir, f"{run_id}_summary.json")
    if not getattr(cfg, "step_log_file", None):
        cfg.step_log_file = cfg.recurrent_convergence_log_file

    return cfg.recurrent_convergence_log_file, cfg.recurrent_convergence_summary_file


def _as_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _numeric_stats(values):
    values = [_as_float(v) for v in values]
    values = [v for v in values if v is not None]
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None, "std": None}

    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
    }


def summarize_episode_convergence(records: List[Dict[str, Any]], success: bool) -> Dict[str, Any]:
    iterations = [r.get("recurrent_iteration_count") for r in records]
    final_mse = [r.get("final_mse") for r in records]
    adaptive_stops = sum(1 for r in records if r.get("adaptive_stop"))

    return {
        "success": bool(success),
        "num_action_predictions": len(records),
        "avg_iteration": _numeric_stats(iterations)["mean"],
        "max_iteration": _numeric_stats(iterations)["max"],
        "min_iteration": _numeric_stats(iterations)["min"],
        "iteration_stats": _numeric_stats(iterations),
        "final_mse_stats": _numeric_stats(final_mse),
        "adaptive_stop_count": int(adaptive_stops),
    }


def summarize_coda_profiling(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    profiled_records = [r for r in records if r.get("profiling_enabled")]
    return {
        "enabled_prediction_count": len(profiled_records),
        "get_output_call_count_stats": _numeric_stats([r.get("get_output_call_count") for r in profiled_records]),
        "coda_ms_total_stats": _numeric_stats([r.get("coda_ms_total") for r in profiled_records]),
        "get_output_ms_total_stats": _numeric_stats([r.get("get_output_ms_total") for r in profiled_records]),
        "run_one_iteration_ms_total_stats": _numeric_stats(
            [r.get("run_one_iteration_ms_total") for r in profiled_records]
        ),
        "output_proj_ms_total_stats": _numeric_stats([r.get("output_proj_ms_total") for r in profiled_records]),
        "coda_time_ratio_total_stats": _numeric_stats([r.get("coda_time_ratio_total") for r in profiled_records]),
    }


def summarize_latent_precheck(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    enabled_records = [r for r in records if r.get("use_latent_precheck")]
    return {
        "enabled_prediction_count": len(enabled_records),
        "skip_count_stats": _numeric_stats([r.get("latent_precheck_skip_count") for r in enabled_records]),
        "call_count_stats": _numeric_stats([r.get("latent_precheck_call_count") for r in enabled_records]),
        "skip_ratio_stats": _numeric_stats([r.get("latent_precheck_skip_ratio") for r in enabled_records]),
    }


def save_recurrent_convergence_summary(cfg, full_results, log_file=None):
    """Write run-level success/failure comparison for recurrent convergence metrics."""
    step_log_path = get_step_log_file(cfg)
    summary_path = getattr(cfg, "recurrent_convergence_summary_file", None)
    if not step_log_path or not summary_path or not os.path.exists(step_log_path):
        return None

    prediction_records = []
    with open(step_log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prediction_records.append(json.loads(line))

    success_records = [r for r in prediction_records if r.get("success") is True]
    failure_records = [r for r in prediction_records if r.get("success") is False]

    episode_records = []
    for task_stats in full_results.get("tasks", {}).values():
        episode_records.extend(task_stats)

    success_episodes = [r for r in episode_records if r.get("success") is True]
    failure_episodes = [r for r in episode_records if r.get("success") is False]

    summary = {
        "schema_version": 1,
        "prediction_log_file": step_log_path,
        "total_prediction_records": len(prediction_records),
        "total_episode_records": len(episode_records),
        "prediction_level": {
            "all": {
                "iteration_stats": _numeric_stats([r.get("recurrent_iteration_count") for r in prediction_records]),
                "final_mse_stats": _numeric_stats([r.get("final_mse") for r in prediction_records]),
                "adaptive_stop_count": sum(1 for r in prediction_records if r.get("adaptive_stop")),
                "coda_profiling": summarize_coda_profiling(prediction_records),
                "latent_precheck": summarize_latent_precheck(prediction_records),
            },
            "success": {
                "iteration_stats": _numeric_stats([r.get("recurrent_iteration_count") for r in success_records]),
                "final_mse_stats": _numeric_stats([r.get("final_mse") for r in success_records]),
                "adaptive_stop_count": sum(1 for r in success_records if r.get("adaptive_stop")),
                "coda_profiling": summarize_coda_profiling(success_records),
                "latent_precheck": summarize_latent_precheck(success_records),
            },
            "failure": {
                "iteration_stats": _numeric_stats([r.get("recurrent_iteration_count") for r in failure_records]),
                "final_mse_stats": _numeric_stats([r.get("final_mse") for r in failure_records]),
                "adaptive_stop_count": sum(1 for r in failure_records if r.get("adaptive_stop")),
                "coda_profiling": summarize_coda_profiling(failure_records),
                "latent_precheck": summarize_latent_precheck(failure_records),
            },
        },
        "rollout_level": {
            "all": {
                "avg_iteration_stats": _numeric_stats([r.get("avg_iters") for r in episode_records]),
                "max_iteration_stats": _numeric_stats([r.get("max_iters") for r in episode_records]),
                "min_iteration_stats": _numeric_stats([r.get("min_iters") for r in episode_records]),
            },
            "success": {
                "avg_iteration_stats": _numeric_stats([r.get("avg_iters") for r in success_episodes]),
                "max_iteration_stats": _numeric_stats([r.get("max_iters") for r in success_episodes]),
                "min_iteration_stats": _numeric_stats([r.get("min_iters") for r in success_episodes]),
            },
            "failure": {
                "avg_iteration_stats": _numeric_stats([r.get("avg_iters") for r in failure_episodes]),
                "max_iteration_stats": _numeric_stats([r.get("max_iters") for r in failure_episodes]),
                "min_iteration_stats": _numeric_stats([r.get("min_iters") for r in failure_episodes]),
            },
        },
    }

    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log_message(f"Saved recurrent convergence summary to {summary_path}", log_file)
    return summary



def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    initial_states = task_suite.get_task_init_states(task_id)

    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None



def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, img



def process_action(action, model_family):
    """Process action before sending to environment."""
    action = normalize_gripper_action(action, binarize=True)

    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action



# 원본 run_episode 함수 시그니처
# def run_episode(
#     cfg: GenerateConfig,
#     env,
#     task_description: str,
#     model,
#     resize_size,
#     processor=None,
#     action_head=None,
#     proprio_projector=None,
#
#     initial_state=None,
#     log_file=None,
#     global_iters=None,
# ):


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,

    initial_state=None,
    log_file=None,
    global_iters=None,
    task_id=None,
    episode_idx=None,
    profile_state=None,
    timing_state=None,
):
    """Run a single episode in the environment."""
    episode_seed = None
    if getattr(cfg, "reset_rng_each_episode", False):
        episode_id_for_seed = int(episode_idx) if episode_idx is not None else 0
        episode_seed = int(cfg.seed) + episode_id_for_seed * int(cfg.episode_seed_stride)
        random.seed(episode_seed)
        np.random.seed(episode_seed)
        torch.manual_seed(episode_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(episode_seed)

    env.reset()

    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    t = 0
    replay_images = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    action_queue = deque()
    warm_start_state = None
    warm_start_cache_age = 0
    episode_iters = []
    replay_stats = []  # (iters, num_actions) per prediction
    episode_action_latencies_ms = []
    episode_step_logs = []
    prediction_step = 0
    prev_action_vec = None
    prev_proprio_vec = None

    success = False
    try:
        while t < max_steps + cfg.num_steps_wait:
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)

            if len(action_queue) == 0:
                cache_age_for_prediction = warm_start_cache_age
                proprio_before_pred = None
                if "state" in observation:
                    proprio_before_pred = np.array(observation["state"], dtype=np.float32).reshape(-1).copy()

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                action_start = time.perf_counter()

                # 원본 action prediction 호출 코드
                # actions, actual_iters, final_kl = get_action(
                #     cfg,
                #     model,
                #     observation,
                #     task_description,
                #     processor=processor,
                #     action_head=action_head,
                #     proprio_projector=proprio_projector,
                #     use_film=cfg.use_film,
                #     use_minivlm=cfg.use_minivlm
                # )

                # recurrence debug metric 전달을 위해 수정한 호출 코드
                def predict_action_once():
                    return get_action(
                        cfg,
                        model,
                        observation,
                        task_description,
                        processor=processor,
                        action_head=action_head,
                        proprio_projector=proprio_projector,
                        use_film=cfg.use_film,
                        use_minivlm=cfg.use_minivlm,
                        warm_start_state=warm_start_state,
                    )

                timing_metadata = {
                    "task_id": task_id,
                    "episode_id": episode_idx,
                    "timestep": int(t),
                    "action_prediction_index": prediction_step,
                    "prediction_step": prediction_step,
                }
                actions, actual_iters, final_kl, inference_metadata = run_action_with_optional_profiles(
                    cfg,
                    profile_state,
                    timing_state,
                    log_file,
                    predict_action_once,
                    timing_metadata,
                )
                inference_metadata = inference_metadata or {}
                if isinstance(inference_metadata, dict):
                    recurrence_debug = inference_metadata.get("recurrence_debug", inference_metadata)
                    warm_start_metadata = dict(inference_metadata.get("warm_start") or {})
                    next_warm_start_state = inference_metadata.get("next_warm_start_state")
                else:
                    recurrence_debug = {}
                    warm_start_metadata = {}
                    next_warm_start_state = None

                warm_start_enabled = bool(
                    warm_start_metadata.get("enabled", getattr(cfg, "use_warm_start", False))
                )
                warm_start_state_provided = bool(warm_start_metadata.get("state_provided", False))
                warm_start_used = bool(warm_start_metadata.get("state_used", False))
                warm_start_source = warm_start_metadata.get("source")
                warm_start_source_index = warm_start_metadata.get("source_index")
                warm_start_source_iteration = warm_start_metadata.get("source_iteration")
                warm_start_source_K = warm_start_metadata.get("source_K")
                warm_start_candidate_state_count = warm_start_metadata.get(
                    "candidate_state_count"
                )
                initial_state_origin = warm_start_metadata.get("initial_state_origin", "random")
                warm_start_reset = bool(warm_start_metadata.get("reset", False))
                warm_start_reset_reason = warm_start_metadata.get("reset_reason")

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                action_latency_ms = (time.perf_counter() - action_start) * 1000.0
                episode_action_latencies_ms.append(action_latency_ms)
                log_message(
                    f"  Action inference latency: {action_latency_ms:.2f} ms, iters={actual_iters}",
                    log_file,
                )

                if getattr(cfg, "use_warm_start", False):
                    if warm_start_reset:
                        warm_start_cache_age = 0
                    if torch.is_tensor(next_warm_start_state):
                        warm_start_state = next_warm_start_state
                        warm_start_cache_age += 1
                    else:
                        warm_start_state = None
                        warm_start_cache_age = 0
                        if not warm_start_reset:
                            warm_start_reset = True
                            warm_start_reset_reason = "next_warm_start_state_missing"
                else:
                    warm_start_state = None
                    warm_start_cache_age = 0

                if actual_iters is not None:
                    episode_iters.append(actual_iters)

                curr_action_vec = None
                if actions is not None and len(actions) > 0:
                    curr_action_vec = np.asarray(actions, dtype=np.float32).reshape(-1)

                prev_action_delta = None
                if prev_action_vec is not None and curr_action_vec is not None:
                    prev_action_delta = float(np.linalg.norm(curr_action_vec - prev_action_vec))

                proprio_delta = None
                if prev_proprio_vec is not None and proprio_before_pred is not None:
                    proprio_delta = float(np.linalg.norm(proprio_before_pred - prev_proprio_vec))

                debug = recurrence_debug or {}
                recurrence_strategy = getattr(cfg, "recurrence_strategy", None)
                recurrent_num_iter = getattr(cfg, "recurrent_num_iter", None)
                fixed_k = int(recurrent_num_iter) if recurrence_strategy == "fixed" and recurrent_num_iter is not None else None
                max_recurrent_iteration = debug.get("max_iter")
                if max_recurrent_iteration is None:
                    max_recurrent_iteration = fixed_k if fixed_k is not None else getattr(cfg, "recurrence_max_iter", None)
                threshold = debug.get("threshold")
                if threshold is None and recurrence_strategy == "kl_divergence":
                    threshold = float(getattr(cfg, "recurrence_kl_thresh", 0.001))
                elif threshold is None and recurrence_strategy == "cosine_similarity":
                    threshold = float(getattr(cfg, "recurrence_cos_thresh", 0.999))

                iteration_mse = debug.get("iteration_mse", debug.get("conv_score_list", []))
                recurrent_iteration_count = _as_int(debug.get("K_t", actual_iters))
                final_mse = debug.get("final_mse")
                if final_mse is None and iteration_mse:
                    final_mse = iteration_mse[-1]

                step_record = {
                    "schema_version": 1,
                    "task_id": task_id,
                    "task_name": task_description,
                    "episode_id": episode_idx,
                    "timestep": int(t),
                    "action_prediction_index": prediction_step,
                    "prediction_step": prediction_step,
                    "reset_rng_each_episode": bool(getattr(cfg, "reset_rng_each_episode", False)),
                    "episode_seed": episode_seed,
                    "method": debug.get("strategy", recurrence_strategy),
                    "recurrence_strategy": debug.get("strategy", recurrence_strategy),
                    "threshold": threshold,
                    "fixed_K": fixed_k,
                    "K_t": recurrent_iteration_count,
                    "recurrent_iteration_count": recurrent_iteration_count,
                    "max_recurrent_iteration": _as_int(max_recurrent_iteration),
                    "adaptive_stop": bool(debug.get("adaptive_stop", False)),
                    "metric_name": debug.get("metric_name", "mse_between_action_outputs"),
                    "iteration_mse": iteration_mse,
                    "iteration_metric_values": debug.get("iteration_metric_values", iteration_mse),
                    "final_mse": _as_float(final_mse),
                    "final_conv_score": debug.get("final_conv_score", final_kl),
                    "conv_score_list": debug.get("conv_score_list", []),
                    "action_delta_list": debug.get("action_delta_list", []),
                    "latent_mse_list": debug.get("latent_mse_list", []),
                    "latent_l2_list": debug.get("latent_l2_list", []),
                    "latent_action_mse_pairs": debug.get("latent_action_mse_pairs", []),
                    "latent_action_pair_count": debug.get("latent_action_pair_count", 0),
                    "use_latent_precheck": bool(debug.get("use_latent_precheck", False)),
                    "latent_precheck_thresh": debug.get("latent_precheck_thresh"),
                    "latent_precheck_min_iter": debug.get("latent_precheck_min_iter"),
                    "latent_precheck_force_interval": debug.get("latent_precheck_force_interval"),
                    "latent_precheck_coda_call_mask": debug.get("latent_precheck_coda_call_mask", []),
                    "latent_precheck_skipped_iters": debug.get("latent_precheck_skipped_iters", []),
                    "latent_precheck_called_iters": debug.get("latent_precheck_called_iters", []),
                    "latent_precheck_skip_count": debug.get("latent_precheck_skip_count", 0),
                    "latent_precheck_call_count": debug.get("latent_precheck_call_count", 0),
                    "latent_precheck_skip_ratio": debug.get("latent_precheck_skip_ratio", 0.0),
                    "latent_precheck_decisions": debug.get("latent_precheck_decisions", []),
                    "first_converged_k_1e_4": debug.get("first_converged_k_1e_4", None),
                    "first_converged_k_5e_4": debug.get("first_converged_k_5e_4", None),
                    "warm_start_enabled": warm_start_enabled,
                    "warm_start_used": warm_start_used,
                    "warm_start_state_provided": warm_start_state_provided,
                    "warm_start_source": warm_start_source,
                    "warm_start_source_index": _as_int(warm_start_source_index),
                    "warm_start_source_iteration": _as_int(warm_start_source_iteration),
                    "warm_start_source_K": _as_int(warm_start_source_K),
                    "warm_start_candidate_state_count": _as_int(
                        warm_start_candidate_state_count
                    ),
                    "warm_start_cache_age": int(cache_age_for_prediction),
                    "warm_start_reset": warm_start_reset,
                    "warm_start_reset_reason": warm_start_reset_reason,
                    "initial_state_origin": initial_state_origin,
                    "warm_start_min_iter": _as_int(
                        debug.get("warm_start_min_iter_configured", getattr(cfg, "warm_start_min_iter", 2))
                    ),
                    "effective_min_iter": _as_int(debug.get("effective_min_iter", 2)),
                    "min_iter_gate_block_count": _as_int(debug.get("min_iter_gate_block_count", 0)),
                    "first_threshold_satisfied_k": _as_int(debug.get("first_threshold_satisfied_k")),
                    "prev_action_delta": prev_action_delta,
                    "proprio_delta": proprio_delta,
                    "latency_ms": action_latency_ms,
                    "rollout_avg_iteration": None,
                    "rollout_max_iteration": None,
                    "rollout_min_iteration": None,
                    "success": None,
                }
                step_record["profiling_enabled"] = bool(debug.get("profiling_enabled", False))
                step_record["use_cached_final_output"] = bool(
                    debug.get("use_cached_final_output", getattr(cfg, "use_cached_final_output", False))
                )
                if step_record["profiling_enabled"]:
                    for timing_key in (
                        "run_one_iteration_ms_list",
                        "get_output_ms_list",
                        "coda_ms_list",
                        "output_proj_ms_list",
                        "convergence_check_ms_list",
                        "get_output_call_count",
                        "coda_ms_total",
                        "get_output_ms_total",
                        "run_one_iteration_ms_total",
                        "output_proj_ms_total",
                        "coda_time_ratio_total",
                    ):
                        step_record[timing_key] = debug.get(timing_key)

                prediction_step += 1
                prev_action_vec = curr_action_vec
                prev_proprio_vec = proprio_before_pred

                if cfg.use_linear_decay_horizon and actual_iters is not None:
                    num_actions = calculate_linear_decay_horizon(actual_iters)
                elif cfg.dynamic_exec and actual_iters is not None:
                    all_observed = (global_iters or []) + episode_iters

                    if len(all_observed) >= cfg.dynamic_exec_warmup:
                        mean_iters = np.mean(all_observed)
                        std_iters = np.std(all_observed) if len(all_observed) > 1 else 1.0

                        if actual_iters < mean_iters - std_iters:
                            num_actions = 2
                        elif actual_iters < mean_iters:
                            num_actions = 4
                        elif actual_iters < mean_iters + std_iters:
                            num_actions = 6
                        else:
                            num_actions = 8
                    else:
                        num_actions = cfg.num_exec_actions
                elif cfg.adaptive_exec and actual_iters is not None:
                    if actual_iters > cfg.adaptive_exec_threshold:
                        num_actions = cfg.adaptive_exec_high
                    else:
                        num_actions = cfg.adaptive_exec_low
                else:
                    num_actions = cfg.num_exec_actions

                # 원본 action queue 삽입 코드
                # replay_stats.append((actual_iters or 0, num_actions))
                # for i in range(num_actions):
                #     action_queue.append(actions[i])

                # num_actions가 반환된 action chunk 길이를 넘지 않도록 수정한 코드
                num_actions = min(num_actions, len(actions))
                step_record["executed_actions_from_prediction"] = int(num_actions)
                episode_step_logs.append(step_record)
                replay_stats.append((actual_iters or 0, num_actions))
                for i in range(num_actions):
                    action_queue.append(actions[i])

            action = action_queue.popleft()
            action = process_action(action, cfg.model_family)
            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)

    if episode_action_latencies_ms:
        lat = np.array(episode_action_latencies_ms, dtype=np.float64)
        warmup_skip = min(2, len(lat))
        steady_lat = lat[warmup_skip:] if len(lat) > warmup_skip else lat

        log_message(
            f"  Action latency summary: {len(lat)} preds, "
            f"raw_avg={np.mean(lat):.2f} ms, "
            f"raw_p95={np.percentile(lat, 95):.2f} ms, "
            f"first={lat[0]:.2f} ms, "
            f"second={(lat[1] if len(lat) > 1 else float('nan')):.2f} ms, "
            f"steady_skip={warmup_skip}, "
            f"steady_avg={np.mean(steady_lat):.2f} ms, "
            f"steady_median={np.median(steady_lat):.2f} ms, "
            f"steady_p90={np.percentile(steady_lat, 90):.2f} ms, "
            f"steady_p95={np.percentile(steady_lat, 95):.2f} ms",
            log_file,
        )

    rollout_summary = summarize_episode_convergence(episode_step_logs, success)
    for record in episode_step_logs:
        record["success"] = bool(success)
        record["rollout_avg_iteration"] = rollout_summary["avg_iteration"]
        record["rollout_max_iteration"] = rollout_summary["max_iteration"]
        record["rollout_min_iteration"] = rollout_summary["min_iteration"]

    append_jsonl(get_step_log_file(cfg), episode_step_logs)

    return success, replay_images, episode_iters, replay_stats, rollout_summary




def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,

    total_episodes=0,
    total_successes=0,
    log_file=None,
    save_version=None,
    profile_state=None,
    timing_state=None,
):
    """Run evaluation for a single task."""
    task = task_suite.get_task(task_id)

    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    task_episodes, task_successes = 0, 0
    all_iters_success, all_iters_failure, all_iters = [], [], []
    task_episode_stats = []
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)

        if cfg.initial_states_path == "DEFAULT":
            initial_state = initial_states[episode_idx]
        else:
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        # 원본 run_episode 호출 코드
        # success, replay_images, episode_iters, replay_stats = run_episode(
        #     cfg, env, task_description, model, resize_size, processor,
        #     action_head, proprio_projector, initial_state, log_file,
        #     global_iters=all_iters,
        # )

        # step-level recurrence log를 위해 task/episode id를 전달하는 호출 코드
        success, replay_images, episode_iters, replay_stats, rollout_summary = run_episode(
            cfg, env, task_description, model, resize_size, processor,
            action_head, proprio_projector, initial_state, log_file,
            global_iters=all_iters,
            task_id=task_id,
            episode_idx=episode_idx,
            profile_state=profile_state,
            timing_state=timing_state,
        )

        if episode_iters:
            ep_avg = np.mean(episode_iters)
            all_iters.extend(episode_iters)
            if success:
                all_iters_success.extend(episode_iters)
            else:
                all_iters_failure.extend(episode_iters)
            log_message(f"  Episode iters: {len(episode_iters)} preds, avg={ep_avg:.1f}", log_file)

        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        task_episode_stats.append({
            "episode": episode_idx,
            "success": success,
            "num_predictions": len(episode_iters),
            "avg_iters": float(np.mean(episode_iters)) if episode_iters else None,
            "max_iters": float(np.max(episode_iters)) if episode_iters else None,
            "min_iters": float(np.min(episode_iters)) if episode_iters else None,
            "recurrent_convergence": rollout_summary,
        })

        if replay_stats:
            save_rollout_video_with_stats(
                replay_images, replay_stats, total_episodes, success=success,
                task_description=task_description, log_file=log_file, save_version=save_version,
            )
        else:
            save_rollout_video(
                replay_images, total_episodes, success=success, task_description=task_description,
                log_file=log_file, save_version=save_version,
            )

        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    if all_iters:
        log_message(f"\n=== Task {task_id} Iteration Stats ===", log_file)
        log_message(f"  Total predictions: {len(all_iters)}", log_file)
        log_message(f"  Avg iters (all): {np.mean(all_iters):.2f} +/- {np.std(all_iters):.2f}", log_file)
        if all_iters_success:
            log_message(f"  Avg iters (success): {np.mean(all_iters_success):.2f} +/- {np.std(all_iters_success):.2f}", log_file)
        if all_iters_failure:
            log_message(f"  Avg iters (failure): {np.mean(all_iters_failure):.2f} +/- {np.std(all_iters_failure):.2f}", log_file)
    
    env.close()
    del env

    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{task_description}": task_success_rate,
                f"num_episodes/{task_description}": task_episodes,
            }
        )

    return total_episodes, total_successes, task_episode_stats



@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Evaluate a trained policy on LIBERO benchmark tasks."""
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)

    model, action_head, proprio_projector, processor = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    RDVLAProfiler.set_enabled(False)
    RDVLAProfiler.set_timing_enabled(False)
    profile_state = {"profiled_calls": 0} if cfg.profile_pytorch else None
    timing_state = {"timed_calls": 0} if cfg.profile_timing_summary else None
    if cfg.profile_pytorch:
        log_message(
            f"PyTorch profiler enabled for first {cfg.profile_steps} action inference calls; "
            f"trace path: {cfg.profile_trace_path}",
            log_file,
        )
    if cfg.profile_timing_summary:
        timing_summary_path = cfg.profile_timing_summary_path
        os.makedirs(os.path.dirname(timing_summary_path) or ".", exist_ok=True)
        if os.path.exists(timing_summary_path):
            os.remove(timing_summary_path)
            log_message(f"Removed existing PyTorch timing summary file: {timing_summary_path}", log_file)
        log_message(
            f"PyTorch timing summary enabled for first {cfg.profile_timing_steps} action inference calls; "
            f"summary path: {timing_summary_path}",
            log_file,
        )

    convergence_log_path, convergence_summary_path = configure_recurrent_convergence_paths(cfg, run_id)
    log_message(f"Recurrent convergence prediction log: {convergence_log_path}", log_file)
    log_message(f"Recurrent convergence summary file: {convergence_summary_path}", log_file)

    step_log_path = get_step_log_file(cfg)
    if step_log_path and os.path.exists(step_log_path):
        os.remove(step_log_path)
        log_message(f"Removed existing step log file: {step_log_path}", log_file)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)

    total_episodes, total_successes = 0, 0
    full_results = {"config": str(cfg), "tasks": {}}

    if cfg.task_id is not None:
        task_ids = [cfg.task_id]
        log_message(f"Running only task {cfg.task_id}", log_file)
    else:
        start_task = getattr(cfg, 'start_task_id', 0)
        task_ids = range(start_task, num_tasks)
        if start_task > 0:
            log_message(f"Starting from task {start_task}", log_file)

    for task_id in tqdm.tqdm(task_ids):
        total_episodes, total_successes, task_stats = run_task(
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
            profile_state,
            timing_state,
        )
        task = task_suite.get_task(task_id)
        full_results["tasks"][task.name] = task_stats

        if cfg.json_log_file:
            with open(cfg.json_log_file, "w") as jf:
                json.dump(full_results, jf, indent=2)

    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Save final JSON results
    if cfg.json_log_file:
        full_results["overall_success_rate"] = final_success_rate
        full_results["total_episodes"] = total_episodes
        full_results["total_successes"] = total_successes
        full_results["recurrent_convergence_log_file"] = get_step_log_file(cfg)
        full_results["recurrent_convergence_summary_file"] = getattr(cfg, "recurrent_convergence_summary_file", None)
        with open(cfg.json_log_file, "w") as jf:
            json.dump(full_results, jf, indent=2)
        log_message(f"Saved JSON results to {cfg.json_log_file}", log_file)

    convergence_summary = save_recurrent_convergence_summary(cfg, full_results, log_file)
    if convergence_summary is not None:
        full_results["recurrent_convergence_summary"] = convergence_summary

    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": final_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

    RDVLAProfiler.set_enabled(False)
    RDVLAProfiler.set_timing_enabled(False)

    if log_file:
        log_file.close()

    return final_success_rate



if __name__ == "__main__":
    eval_libero()
