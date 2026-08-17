import json
import hashlib
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
from typing import Any, Dict, List, Mapping, Optional, Union

import cv2
import torch
import draccus
import numpy as np
import tqdm
from libero.libero import benchmark, get_libero_path

import wandb
from configs.rdvla_precheck import (
    canonicalize_recurrence_strategy,
    validate_fixed_terminal_only_configuration,
    validate_latent_only_configuration,
    validate_latent_precheck_configuration,
)
from experiments.robot.libero.evaluation_protocol import (
    load_protocol_manifest,
    reset_episode_environment,
    resolve_phase_trials,
    validate_protocol_configuration,
)
from experiments.robot.libero.latent_metric_trace import (
    build_action_head_workload_identity,
    build_latent_metric_trace_records,
    build_stop_reason_fields,
    require_prediction_id,
)
from experiments.robot.libero.raw_preconvergence_trace import (
    RAW_PRECONVERGENCE_SCHEMA_VERSION,
    RawPreconvergenceShardWriter,
    build_prediction_payload,
    checkpoint_identity,
    current_source_commit,
)
from experiments.robot.libero.action_delta_gate_shadow_collection import (
    ACTION_DELTA_GATE_SHADOW_COLLECTION_MODE,
    ActionDeltaGateShadowWriter,
    build_shadow_prediction_payload,
)

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
from prismatic.models.action_head_workload import (
    ACTION_HEAD_WORKLOAD_SCHEMA_VERSION,
    ACTION_HEAD_WORKLOAD_TENSORS,
    save_action_head_workload,
)
from prismatic.models.numerical_retry import NumericalInferenceAbort
from prismatic.models.scalar_stopping_policy import (
    SUPPORTED_SCALAR_EXECUTION_MODES,
    load_scalar_policy_artifact,
    prepare_scalar_task_policy,
)
from prismatic.models.action_delta_gate import (
    ACTION_DELTA_NONCONVERGENCE_MIN_TERMINAL_ITER,
    ACTION_DELTA_NONCONVERGENCE_THRESHOLD,
    ACTION_DELTA_GATE_RETURN_MODES,
    load_action_delta_gate_artifact,
    prepare_action_delta_gate,
    prepare_action_delta_gate_shadow,
)
from prismatic.models.action_delta_gate_shadow import (
    ACTION_DELTA_GATE_SHADOW_DEVELOPMENT_TASK_IDS,
    ACTION_DELTA_GATE_SHADOW_SCHEMA_VERSION,
)
from prismatic.models.action_delta_deferred_scorer import (
    ACTION_DELTA_DEFERRED_COMPILED_NUMERICAL_EQUIVALENCE,
    ACTION_DELTA_DEFERRED_SCORER_BACKENDS,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK
from prismatic.utils.rdvla_profiler import RDVLAProfiler


class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


PARITY_HASH_SCHEMA = {
    "schema_version": 1,
    "algorithm": "sha256",
    "dtype": "preserved and included in the canonical header",
    "shape": "preserved and included in the canonical header",
    "byte_order": "little-endian",
    "memory_order": "C-contiguous",
}


def _tensor_or_array_sha256(value) -> Optional[str]:
    """Hash dtype, shape, and canonical little-endian C-order payload bytes."""

    if value is None:
        return None
    if torch.is_tensor(value):
        tensor = value.detach().to(device="cpu", copy=True).contiguous()
        element_size = tensor.element_size()
        array = tensor.view(torch.uint8).numpy().reshape(-1)
        if sys.byteorder == "big" and element_size > 1:
            array = array.reshape(-1, element_size)[:, ::-1].reshape(-1).copy()
        header = {
            "schema_version": PARITY_HASH_SCHEMA["schema_version"],
            "kind": "torch",
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "byte_order": "little",
            "memory_order": "C",
        }
    else:
        source = np.asarray(value)
        if source.dtype.hasobject:
            raise ValueError("parity hashing does not support object arrays")
        canonical_dtype = (
            source.dtype.newbyteorder("<")
            if source.dtype.itemsize > 1
            else source.dtype
        )
        canonical = np.ascontiguousarray(source.astype(canonical_dtype, copy=False))
        array = canonical.view(np.uint8).reshape(-1)
        header = {
            "schema_version": PARITY_HASH_SCHEMA["schema_version"],
            "kind": "numpy",
            "dtype": canonical.dtype.str,
            "shape": list(canonical.shape),
            "byte_order": "little",
            "memory_order": "C",
        }
    digest = hashlib.sha256()
    digest.update(b"rd-vla-prediction-parity-v1\0")
    digest.update(
        json.dumps(
            header, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _rng_state_sha256() -> str:
    """Hash Python, NumPy, CPU torch, and available CUDA RNG states."""
    digest = hashlib.sha256()
    digest.update(repr(random.getstate()).encode("utf-8"))
    numpy_state = np.random.get_state()
    digest.update(str(numpy_state[0]).encode("ascii"))
    digest.update(np.ascontiguousarray(numpy_state[1]).view(np.uint8).tobytes())
    digest.update(repr(numpy_state[2:]).encode("ascii"))
    digest.update(torch.get_rng_state().contiguous().numpy().tobytes())
    if torch.cuda.is_available():
        for state in torch.cuda.get_rng_state_all():
            digest.update(state.contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


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
    sync_checkpoint_source_config: bool = True        # Synchronize local checkpoint source/config mirrors
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
    evaluation_protocol_phase: str = "legacy"        # legacy | smoke | calibration | screening | final_holdout
    initial_state_manifest_path: str = ""             # Frozen official-state manifest for non-legacy phases
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

    # "kl_divergence" is a legacy alias for adjacent action-output MSE.
    recurrence_strategy: str = "fixed"
    recurrent_num_iter: int = 12
    recurrence_kl_thresh: float = 0.001
    recurrence_cos_thresh: float = 0.999
    recurrence_max_iter: int = 32

    # Independent latent-only recurrence stopping (inactive unless selected).
    latent_only_metric: str = "raw_mse"
    latent_only_cold_threshold: float = 0.0
    latent_only_warm_threshold: float = 0.0
    latent_only_min_iter: int = 2
    latent_only_eps: float = 1e-8

    # Task-level OOF scalar stopping policy.
    scalar_policy_artifact_path: str = ""
    scalar_policy_expected_sha256: str = ""
    scalar_policy_execution_mode: str = "direct"

    # Fold-4 Phase-B Action-Delta Gate (LIBERO Spatial tasks 4 and 5 only).
    use_action_delta_gate: bool = False
    action_delta_gate_artifact_path: str = ""
    action_delta_gate_expected_sha256: str = ""
    action_delta_gate_max_skip: int = 1
    action_delta_gate_min_terminal_iter: int = 2
    action_delta_gate_exact_coda_audit: bool = False
    action_delta_gate_return_mode: str = "anchor"
    collect_action_delta_gate_shadow: bool = False
    # Development-only high-side predictor filter; never a convergence gate.
    use_action_delta_nonconvergence_filter: bool = False
    use_action_delta_deferred_backfill_filter: bool = False
    action_delta_deferred_scorer_backend: str = "eager"
    action_delta_gate_shadow_dir: str = ""
    action_delta_gate_shadow_shard_size: int = 64

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
    latent_precheck_mode: str = "legacy"
    latent_precheck_trace_level: str = "off"
    latent_precheck_thresh: float = 0.12
    latent_precheck_min_iter: int = 2
    latent_precheck_force_interval: int = 0
    latent_precheck_warm_thresh: Optional[float] = None
    latent_precheck_max_skip_iters: int = 0
    latent_precheck_confirmation_mode: str = "next_iter"
    nonfinite_policy: str = "legacy"
    shadow_full_depth: bool = False
    collect_preconvergence_raw_shadow: bool = False
    preconvergence_raw_shadow_dir: str = ""
    preconvergence_raw_shadow_max_depth: int = 32
    preconvergence_raw_shadow_shard_size: int = 32
    calibration_workload_dir: str = ""
    calibration_workload_predictions_per_episode: int = 0

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
    canonical_recurrence_strategy = canonicalize_recurrence_strategy(
        cfg.recurrence_strategy
    )
    validate_fixed_terminal_only_configuration(
        canonical_recurrence_strategy,
        recurrent_num_iter=cfg.recurrent_num_iter,
        recurrence_max_iter=cfg.recurrence_max_iter,
    )

    if canonical_recurrence_strategy == "scalar_policy":
        if not cfg.use_recurrent:
            raise ValueError(
                "scalar_policy requires use_recurrent=True"
            )
        if cfg.task_suite_name != TaskSuite.LIBERO_SPATIAL:
            raise ValueError(
                "the exported scalar OOF artifact is LIBERO Spatial-only"
            )
        if not cfg.scalar_policy_artifact_path:
            raise ValueError(
                "scalar_policy_artifact_path is required"
            )
        if not Path(cfg.scalar_policy_artifact_path).exists():
            raise ValueError(
                "scalar policy artifact path does not exist: "
                f"{cfg.scalar_policy_artifact_path}"
            )

        expected_hash = cfg.scalar_policy_expected_sha256
        if (
            len(expected_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_hash.lower()
            )
        ):
            raise ValueError(
                "scalar_policy_expected_sha256 must be "
                "a 64-character hexadecimal SHA-256"
            )

        if (
            cfg.scalar_policy_execution_mode
            not in SUPPORTED_SCALAR_EXECUTION_MODES
        ):
            raise ValueError(
                "scalar_policy_execution_mode must be direct "
                "or confirm_next"
            )
        if not cfg.use_warm_start:
            raise ValueError(
                "scalar_policy requires use_warm_start=True"
            )
        if cfg.warm_start_source != "midpoint":
            raise ValueError(
                "scalar_policy requires warm_start_source='midpoint'"
            )
        if cfg.use_latent_precheck:
            raise ValueError(
                "scalar_policy cannot use latent pre-check"
            )
        if cfg.latent_precheck_mode != "off":
            raise ValueError(
                "scalar_policy requires latent_precheck_mode='off'"
            )
        if cfg.latent_precheck_trace_level != "off":
            raise ValueError(
                "scalar_policy requires "
                "latent_precheck_trace_level='off'"
            )
        if cfg.shadow_full_depth:
            raise ValueError(
                "scalar_policy cannot enable shadow_full_depth"
            )
        if cfg.collect_preconvergence_raw_shadow:
            raise ValueError(
                "scalar_policy cannot collect raw shadow trajectories"
            )
        if cfg.use_cached_final_output:
            raise ValueError(
                "scalar_policy cannot use cached final output"
            )
    else:
        if (
            cfg.scalar_policy_artifact_path
            or cfg.scalar_policy_expected_sha256
        ):
            raise ValueError(
                "scalar policy artifact settings require "
                "recurrence_strategy='scalar_policy'"
            )

    action_delta_mode_count = sum(
        bool(value)
        for value in (
            cfg.use_action_delta_gate,
            cfg.collect_action_delta_gate_shadow,
            cfg.use_action_delta_nonconvergence_filter,
            cfg.use_action_delta_deferred_backfill_filter,
        )
    )
    if action_delta_mode_count > 1:
        raise ValueError(
            "production Action-Delta Gate, diagnostic shadow collection, and "
            "both diagnostic non-convergence filters are mutually exclusive"
        )
    if (
        cfg.action_delta_deferred_scorer_backend
        not in ACTION_DELTA_DEFERRED_SCORER_BACKENDS
    ):
        raise ValueError(
            "action_delta_deferred_scorer_backend must be one of "
            f"{ACTION_DELTA_DEFERRED_SCORER_BACKENDS}"
        )
    if (
        cfg.action_delta_deferred_scorer_backend != "eager"
        and not cfg.use_action_delta_deferred_backfill_filter
    ):
        raise ValueError(
            "compile_default scorer backend is deferred/backfill-only"
        )

    if cfg.use_action_delta_gate:
        if not cfg.use_recurrent:
            raise ValueError("Action-Delta Gate requires use_recurrent=True")
        if canonical_recurrence_strategy != "adjacent_action_mse":
            raise ValueError(
                "Action-Delta Gate requires recurrence_strategy="
                "'adjacent_action_mse' or legacy alias 'kl_divergence'"
            )
        if cfg.task_suite_name != TaskSuite.LIBERO_SPATIAL:
            raise ValueError("fold-4 Action-Delta Gate is LIBERO Spatial-only")
        if cfg.task_id not in {4, 5}:
            raise ValueError("fold-4 Action-Delta Gate requires explicit task_id 4 or 5")
        if not cfg.action_delta_gate_artifact_path:
            raise ValueError("action_delta_gate_artifact_path is required")
        if not Path(cfg.action_delta_gate_artifact_path).exists():
            raise ValueError(
                "Action-Delta Gate artifact path does not exist: "
                f"{cfg.action_delta_gate_artifact_path}"
            )
        expected_hash = cfg.action_delta_gate_expected_sha256
        if (
            len(expected_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_hash.lower()
            )
        ):
            raise ValueError(
                "action_delta_gate_expected_sha256 must be a "
                "64-character hexadecimal SHA-256"
            )
        if not cfg.use_warm_start or cfg.warm_start_source != "midpoint":
            raise ValueError("Action-Delta Gate requires midpoint warm-start")
        if cfg.warm_start_min_iter != 2:
            raise ValueError("Action-Delta Gate requires warm_start_min_iter=2")
        if cfg.use_latent_precheck:
            raise ValueError("Action-Delta Gate cannot use latent pre-check")
        if cfg.latent_precheck_mode != "off":
            raise ValueError("Action-Delta Gate requires latent_precheck_mode='off'")
        if cfg.latent_precheck_trace_level != "off":
            raise ValueError(
                "Action-Delta Gate requires latent_precheck_trace_level='off'"
            )
        if cfg.shadow_full_depth or cfg.collect_preconvergence_raw_shadow:
            raise ValueError("Action-Delta Gate cannot collect shadow trajectories")
        if cfg.action_delta_gate_max_skip != 1:
            raise ValueError("Action-Delta Gate Phase B requires max_skip=1")
        if (
            not isinstance(cfg.action_delta_gate_min_terminal_iter, int)
            or isinstance(cfg.action_delta_gate_min_terminal_iter, bool)
            or cfg.action_delta_gate_min_terminal_iter < 2
        ):
            raise ValueError(
                "Action-Delta Gate minimum terminal iteration must be "
                "an integer >= 2"
            )
        if not isinstance(cfg.action_delta_gate_exact_coda_audit, bool):
            raise ValueError(
                "Action-Delta Gate exact Coda audit must be boolean"
            )
        if cfg.action_delta_gate_return_mode not in ACTION_DELTA_GATE_RETURN_MODES:
            raise ValueError(
                "Action-Delta Gate return mode must be 'anchor', "
                "'predicted_correction', 'exact_terminal', or "
                "'oracle_confirm'"
            )
        if not cfg.use_cached_final_output:
            raise ValueError(
                "Action-Delta Gate Phase B requires use_cached_final_output=True"
            )
    elif not (
        cfg.collect_action_delta_gate_shadow
        or cfg.use_action_delta_nonconvergence_filter
        or cfg.use_action_delta_deferred_backfill_filter
    ) and (
        cfg.action_delta_gate_artifact_path
        or cfg.action_delta_gate_expected_sha256
    ):
        raise ValueError(
            "Action-Delta Gate artifact settings require use_action_delta_gate=True"
        )

    if cfg.collect_action_delta_gate_shadow:
        if not cfg.use_recurrent:
            raise ValueError("Action-Delta Gate shadow collection requires use_recurrent=True")
        if canonical_recurrence_strategy != "adjacent_action_mse":
            raise ValueError(
                "Action-Delta Gate shadow collection requires "
                "adjacent action-MSE recurrence"
            )
        if cfg.task_suite_name != TaskSuite.LIBERO_SPATIAL:
            raise ValueError("Action-Delta Gate shadow collection is LIBERO Spatial-only")
        if cfg.task_id is not None and cfg.task_id not in ACTION_DELTA_GATE_SHADOW_DEVELOPMENT_TASK_IDS:
            raise ValueError(
                "Action-Delta Gate shadow calibration permits only development "
                f"tasks {ACTION_DELTA_GATE_SHADOW_DEVELOPMENT_TASK_IDS}; Task 4/5 are forbidden"
            )
        if cfg.evaluation_protocol_phase != "calibration":
            raise ValueError(
                "Phase-A Action-Delta Gate shadow collection requires the "
                "10-state official calibration manifest partition"
            )
        if not cfg.action_delta_gate_artifact_path:
            raise ValueError("shadow collection requires action_delta_gate_artifact_path")
        if not Path(cfg.action_delta_gate_artifact_path).exists():
            raise ValueError(
                "Action-Delta Gate artifact path does not exist: "
                f"{cfg.action_delta_gate_artifact_path}"
            )
        expected_hash = cfg.action_delta_gate_expected_sha256
        if (
            len(expected_hash) != 64
            or any(character not in "0123456789abcdef" for character in expected_hash.lower())
        ):
            raise ValueError(
                "action_delta_gate_expected_sha256 must be a 64-character "
                "hexadecimal SHA-256"
            )
        if not cfg.action_delta_gate_shadow_dir:
            raise ValueError("action_delta_gate_shadow_dir is required")
        if (
            isinstance(cfg.action_delta_gate_shadow_shard_size, bool)
            or not isinstance(cfg.action_delta_gate_shadow_shard_size, int)
            or cfg.action_delta_gate_shadow_shard_size < 1
        ):
            raise ValueError("action_delta_gate_shadow_shard_size must be an integer >= 1")
        if not cfg.use_warm_start or cfg.warm_start_source != "midpoint":
            raise ValueError("shadow collection requires midpoint warm-start")
        if cfg.warm_start_min_iter != 2:
            raise ValueError("shadow collection requires warm_start_min_iter=2")
        if cfg.use_latent_precheck or cfg.latent_precheck_mode != "off":
            raise ValueError("shadow collection requires latent pre-check off")
        if cfg.latent_precheck_trace_level != "off":
            raise ValueError("shadow collection requires latent_precheck_trace_level='off'")
        if cfg.shadow_full_depth or cfg.collect_preconvergence_raw_shadow:
            raise ValueError("shadow collection cannot enable post-production shadow recurrence")
        if not cfg.use_cached_final_output:
            raise ValueError("shadow collection requires exact terminal-output reuse")
        if (
            isinstance(cfg.action_delta_gate_min_terminal_iter, bool)
            or not isinstance(cfg.action_delta_gate_min_terminal_iter, int)
            or cfg.action_delta_gate_min_terminal_iter < 2
        ):
            raise ValueError(
                "deployment-matched Phase-A collection requires "
                "action_delta_gate_min_terminal_iter to be an integer >= 2"
            )
        if float(cfg.recurrence_kl_thresh) != 0.001:
            raise ValueError(
                "deployment-matched Phase-A collection requires "
                "recurrence_kl_thresh=0.001"
            )
        if cfg.action_delta_gate_exact_coda_audit:
            raise ValueError("shadow collection cannot enable exact-Coda trigger audit")
        if cfg.action_delta_gate_return_mode != "anchor":
            raise ValueError("shadow collection does not accept a production return policy")
    elif cfg.action_delta_gate_shadow_dir:
        raise ValueError(
            "action_delta_gate_shadow_dir requires "
            "collect_action_delta_gate_shadow=True"
        )

    if cfg.use_action_delta_nonconvergence_filter:
        if not cfg.use_recurrent:
            raise ValueError("non-convergence filter requires use_recurrent=True")
        if canonical_recurrence_strategy != "adjacent_action_mse":
            raise ValueError(
                "non-convergence filter requires adjacent action-MSE recurrence"
            )
        if cfg.task_suite_name != TaskSuite.LIBERO_SPATIAL:
            raise ValueError("non-convergence filter is LIBERO Spatial-only")
        if (
            cfg.task_id is not None
            and cfg.task_id not in ACTION_DELTA_GATE_SHADOW_DEVELOPMENT_TASK_IDS
        ):
            raise ValueError(
                "non-convergence filter permits only development tasks "
                f"{ACTION_DELTA_GATE_SHADOW_DEVELOPMENT_TASK_IDS}; Task 4/5 are forbidden"
            )
        if not cfg.action_delta_gate_artifact_path:
            raise ValueError("non-convergence filter requires action_delta_gate_artifact_path")
        if not Path(cfg.action_delta_gate_artifact_path).exists():
            raise ValueError(
                "Action-Delta artifact path does not exist: "
                f"{cfg.action_delta_gate_artifact_path}"
            )
        expected_hash = cfg.action_delta_gate_expected_sha256
        if (
            len(expected_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_hash.lower()
            )
        ):
            raise ValueError(
                "action_delta_gate_expected_sha256 must be a 64-character "
                "hexadecimal SHA-256"
            )
        if not cfg.use_warm_start or cfg.warm_start_source != "midpoint":
            raise ValueError("non-convergence filter requires midpoint warm-start")
        if cfg.warm_start_min_iter != 2:
            raise ValueError("non-convergence filter requires warm_start_min_iter=2")
        if cfg.use_latent_precheck or cfg.latent_precheck_mode != "off":
            raise ValueError("non-convergence filter requires latent pre-check off")
        if cfg.latent_precheck_trace_level != "off":
            raise ValueError(
                "non-convergence filter requires latent_precheck_trace_level='off'"
            )
        if cfg.shadow_full_depth or cfg.collect_preconvergence_raw_shadow:
            raise ValueError("non-convergence filter cannot collect shadow recurrence")
        if not cfg.use_cached_final_output:
            raise ValueError("non-convergence filter requires exact terminal-output reuse")
        if not cfg.profile_coda_cost:
            raise ValueError("non-convergence filter requires profile_coda_cost=True")
        if (
            cfg.action_delta_gate_min_terminal_iter
            != ACTION_DELTA_NONCONVERGENCE_MIN_TERMINAL_ITER
        ):
            raise ValueError(
                "non-convergence filter requires action_delta_gate_min_terminal_iter="
                f"{ACTION_DELTA_NONCONVERGENCE_MIN_TERMINAL_ITER}"
            )
        if cfg.action_delta_gate_max_skip != 1:
            raise ValueError("non-convergence filter requires max_skip=1")
        if float(cfg.recurrence_kl_thresh) != 0.001:
            raise ValueError("non-convergence filter requires recurrence_kl_thresh=0.001")
        if cfg.action_delta_gate_exact_coda_audit:
            raise ValueError("non-convergence filter cannot enable exact-Coda gate audit")
        if cfg.action_delta_gate_return_mode != "anchor":
            raise ValueError("non-convergence filter does not use an Action-Delta return mode")

    if cfg.use_action_delta_deferred_backfill_filter:
        if not cfg.use_recurrent:
            raise ValueError("deferred/backfill filter requires use_recurrent=True")
        if canonical_recurrence_strategy != "adjacent_action_mse":
            raise ValueError(
                "deferred/backfill filter requires adjacent action-MSE recurrence"
            )
        if cfg.task_suite_name != TaskSuite.LIBERO_SPATIAL:
            raise ValueError("deferred/backfill filter is LIBERO Spatial-only")
        deferred_phase = cfg.evaluation_protocol_phase
        if deferred_phase == "calibration":
            if (
                cfg.task_id is not None
                and cfg.task_id
                not in ACTION_DELTA_GATE_SHADOW_DEVELOPMENT_TASK_IDS
            ):
                raise ValueError(
                    "deferred/backfill calibration permits only development "
                    f"tasks {ACTION_DELTA_GATE_SHADOW_DEVELOPMENT_TASK_IDS}; "
                    "task 4 requires screening and task 5 requires final_holdout"
                )
        elif deferred_phase == "screening":
            if cfg.task_id != 4:
                raise ValueError(
                    "deferred/backfill screening requires explicit task_id=4"
                )
        elif deferred_phase == "final_holdout":
            if cfg.task_id != 5:
                raise ValueError(
                    "deferred/backfill final_holdout requires explicit task_id=5"
                )
        else:
            raise ValueError(
                "deferred/backfill evaluation requires phase calibration, "
                "screening, or final_holdout"
            )
        if deferred_phase in {"screening", "final_holdout"}:
            if cfg.action_delta_deferred_scorer_backend != "eager":
                raise ValueError(
                    f"{deferred_phase} requires "
                    "action_delta_deferred_scorer_backend='eager'"
                )
            if cfg.action_delta_gate_min_terminal_iter != 2:
                raise ValueError(
                    f"{deferred_phase} requires "
                    "action_delta_gate_min_terminal_iter=2"
                )
        if not cfg.action_delta_gate_artifact_path:
            raise ValueError("deferred/backfill filter requires action_delta_gate_artifact_path")
        if not Path(cfg.action_delta_gate_artifact_path).exists():
            raise ValueError(
                "Action-Delta artifact path does not exist: "
                f"{cfg.action_delta_gate_artifact_path}"
            )
        expected_hash = cfg.action_delta_gate_expected_sha256
        if (
            len(expected_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_hash.lower()
            )
        ):
            raise ValueError(
                "action_delta_gate_expected_sha256 must be a 64-character "
                "hexadecimal SHA-256"
            )
        if not cfg.use_warm_start or cfg.warm_start_source != "midpoint":
            raise ValueError("deferred/backfill filter requires midpoint warm-start")
        if cfg.warm_start_min_iter != 2:
            raise ValueError("deferred/backfill filter requires warm_start_min_iter=2")
        if cfg.use_latent_precheck or cfg.latent_precheck_mode != "off":
            raise ValueError("deferred/backfill filter requires latent pre-check off")
        if cfg.latent_precheck_trace_level != "off":
            raise ValueError(
                "deferred/backfill filter requires latent_precheck_trace_level='off'"
            )
        if cfg.shadow_full_depth or cfg.collect_preconvergence_raw_shadow:
            raise ValueError("deferred/backfill filter cannot collect shadow recurrence")
        if not cfg.use_cached_final_output:
            raise ValueError("deferred/backfill filter requires exact terminal-output reuse")
        if not cfg.profile_coda_cost:
            raise ValueError("deferred/backfill filter requires profile_coda_cost=True")
        if (
            isinstance(cfg.action_delta_gate_min_terminal_iter, bool)
            or not isinstance(cfg.action_delta_gate_min_terminal_iter, int)
            or cfg.action_delta_gate_min_terminal_iter < 2
        ):
            raise ValueError(
                "deferred/backfill filter requires action_delta_gate_min_terminal_iter "
                "to be an integer >= 2"
            )
        if (
            cfg.action_delta_deferred_scorer_backend == "compile_default"
            and cfg.action_delta_gate_min_terminal_iter != 2
        ):
            raise ValueError(
                "compile_default deferred scorer runtime trial requires "
                "action_delta_gate_min_terminal_iter=2"
            )
        if float(cfg.recurrence_kl_thresh) != 0.001:
            raise ValueError("deferred/backfill filter requires recurrence_kl_thresh=0.001")
        if cfg.action_delta_gate_exact_coda_audit:
            raise ValueError("deferred/backfill filter cannot enable exact-Coda gate audit")
        if cfg.action_delta_gate_return_mode != "anchor":
            raise ValueError("deferred/backfill filter does not use an Action-Delta return mode")

    validate_latent_only_configuration(
        cfg.recurrence_strategy,
        metric=cfg.latent_only_metric,
        cold_threshold=cfg.latent_only_cold_threshold,
        warm_threshold=cfg.latent_only_warm_threshold,
        min_iter=cfg.latent_only_min_iter,
        eps=cfg.latent_only_eps,
        use_latent_precheck=cfg.use_latent_precheck,
        latent_precheck_mode=cfg.latent_precheck_mode,
        shadow_full_depth=cfg.shadow_full_depth,
        use_cached_final_output=cfg.use_cached_final_output,
    )
    validate_latent_precheck_configuration(
        cfg.latent_precheck_mode,
        cfg.latent_precheck_trace_level,
        cfg.use_latent_precheck,
        origin_aware_implemented=True,
        warm_threshold=cfg.latent_precheck_warm_thresh,
        max_skip_iters=cfg.latent_precheck_max_skip_iters,
        confirmation_mode=cfg.latent_precheck_confirmation_mode,
        warm_start_source=cfg.warm_start_source,
        recurrence_strategy=cfg.recurrence_strategy,
        use_warm_start=cfg.use_warm_start,
        min_iter=cfg.latent_precheck_min_iter,
        nonfinite_policy=cfg.nonfinite_policy,
        shadow_full_depth=cfg.shadow_full_depth,
    )
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
    validate_protocol_configuration(
        phase=cfg.evaluation_protocol_phase,
        task_suite_name=cfg.task_suite_name,
        num_trials_per_task=cfg.num_trials_per_task,
        initial_states_path=cfg.initial_states_path,
        manifest_path=cfg.initial_state_manifest_path,
        reset_rng_each_episode=cfg.reset_rng_each_episode,
    )
    if cfg.evaluation_protocol_phase != "legacy":
        protocol_manifest, _ = load_protocol_manifest(
            cfg.initial_state_manifest_path, require_source_file_hashes=True
        )
        if protocol_manifest["task_suite_name"] != cfg.task_suite_name:
            raise ValueError(
                "Initial-state protocol manifest task_suite_name does not match the runner configuration"
            )
    if cfg.evaluation_protocol_phase == "calibration":
        if (
            cfg.task_id is not None
            and cfg.task_id
            not in ACTION_DELTA_GATE_SHADOW_DEVELOPMENT_TASK_IDS
        ):
            raise ValueError(
                "calibration permits only development tasks "
                f"{ACTION_DELTA_GATE_SHADOW_DEVELOPMENT_TASK_IDS}; task 4 "
                "requires screening and task 5 requires final_holdout"
            )
    elif cfg.evaluation_protocol_phase == "screening":
        if cfg.task_id != 4:
            raise ValueError("screening requires explicit task_id=4")
    elif cfg.evaluation_protocol_phase == "final_holdout":
        if cfg.task_id != 5:
            raise ValueError("final_holdout requires explicit task_id=5")

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"

    if cfg.profile_pytorch:
        assert cfg.profile_steps >= 0, "profile_steps must be non-negative!"
        assert cfg.profile_trace_path, "profile_trace_path must be non-empty when profile_pytorch is enabled!"

    workload_count = cfg.calibration_workload_predictions_per_episode
    if isinstance(workload_count, bool) or not isinstance(workload_count, int) or workload_count < 0:
        raise ValueError("calibration_workload_predictions_per_episode must be an integer >= 0")
    if workload_count > 0:
        if cfg.evaluation_protocol_phase != "calibration":
            raise ValueError("action-head workload capture is calibration-only")
        if not cfg.calibration_workload_dir:
            raise ValueError(
                "calibration_workload_dir is required when action-head workload capture is enabled"
            )
        if not cfg.shadow_full_depth:
            raise ValueError("action-head workload capture requires shadow_full_depth=True")
        if cfg.latent_precheck_mode != "off" or cfg.use_latent_precheck:
            raise ValueError("action-head workload capture requires clean pre-check off mode")
        if not cfg.use_warm_start or cfg.warm_start_source != "midpoint":
            raise ValueError("action-head workload capture requires midpoint warm-start")
    elif cfg.calibration_workload_dir:
        raise ValueError(
            "calibration_workload_dir requires calibration_workload_predictions_per_episode > 0"
        )

    raw_shard_size = cfg.preconvergence_raw_shadow_shard_size
    if (
        isinstance(raw_shard_size, bool)
        or not isinstance(raw_shard_size, int)
        or raw_shard_size < 1
    ):
        raise ValueError("preconvergence_raw_shadow_shard_size must be an integer >= 1")
    if cfg.collect_preconvergence_raw_shadow:
        if cfg.evaluation_protocol_phase not in {"smoke", "calibration"}:
            raise ValueError("raw preconvergence collection is smoke/calibration-only")
        if not cfg.preconvergence_raw_shadow_dir:
            raise ValueError(
                "preconvergence_raw_shadow_dir is required when raw collection is enabled"
            )
        if not cfg.shadow_full_depth:
            raise ValueError("raw preconvergence collection requires shadow_full_depth=True")
        if cfg.latent_precheck_mode != "off" or cfg.use_latent_precheck:
            raise ValueError("raw preconvergence collection requires clean pre-check off mode")
        if canonicalize_recurrence_strategy(cfg.recurrence_strategy) != "adjacent_action_mse":
            raise ValueError("raw preconvergence collection requires adjacent action-MSE stopping")
        if not cfg.use_warm_start or cfg.warm_start_source != "midpoint":
            raise ValueError("raw preconvergence collection requires midpoint warm-start")
        if not cfg.use_cached_final_output:
            raise ValueError("raw preconvergence collection requires cached final output")
        if cfg.preconvergence_raw_shadow_max_depth != cfg.recurrence_max_iter:
            raise ValueError(
                "preconvergence_raw_shadow_max_depth must equal recurrence_max_iter"
            )
    elif cfg.preconvergence_raw_shadow_dir:
        raise ValueError(
            "preconvergence_raw_shadow_dir requires collect_preconvergence_raw_shadow=True"
        )

    if cfg.profile_timing_summary:
        assert cfg.profile_timing_steps >= 0, "profile_timing_steps must be non-negative!"
        assert cfg.profile_timing_summary_path, (
            "profile_timing_summary_path must be non-empty when profile_timing_summary is enabled!"
        )


def _prepare_action_delta_gate_for_evaluation(
    cfg: GenerateConfig,
    payload,
    *,
    device: torch.device,
    task_id: int,
):
    if cfg.evaluation_protocol_phase == "calibration":
        return prepare_action_delta_gate_shadow(
            payload,
            device=device,
            task_id=task_id,
        )
    if cfg.evaluation_protocol_phase in {"screening", "final_holdout"}:
        return prepare_action_delta_gate(
            payload,
            device=device,
            task_id=task_id,
        )
    if (
        cfg.collect_action_delta_gate_shadow
        or cfg.use_action_delta_nonconvergence_filter
        or cfg.use_action_delta_deferred_backfill_filter
    ):
        return prepare_action_delta_gate_shadow(
            payload,
            device=device,
            task_id=task_id,
        )
    return prepare_action_delta_gate(
        payload,
        device=device,
        task_id=task_id,
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


def build_scalar_policy_log_fields(debug):
    """Extract compact JSON-safe scalar-policy runtime metadata."""

    debug = debug or {}
    applied = bool(
        debug.get("scalar_policy_applied", False)
    )

    return {
        "scalar_policy_requested": bool(
            debug.get("scalar_policy_requested", False)
        ),
        "scalar_policy_applied": applied,
        "scalar_policy_cold_fallback": bool(
            debug.get(
                "scalar_policy_cold_fallback",
                False,
            )
        ),
        "scalar_policy_execution_mode": debug.get(
            "scalar_policy_execution_mode"
        ),
        "scalar_policy_task_id": _as_int(
            debug.get("scalar_policy_task_id")
        ),
        "scalar_policy_outer_fold": _as_int(
            debug.get("scalar_policy_outer_fold")
        ),
        "scalar_policy_threshold": _as_float(
            debug.get("scalar_policy_threshold")
        ),
        "scalar_policy_gate_iteration": _as_int(
            debug.get(
                "scalar_policy_gate_iteration"
            )
        ),
        "scalar_policy_terminal_iteration": _as_int(
            debug.get(
                "scalar_policy_terminal_iteration"
            )
        ),
        "scalar_policy_score_call_count": _as_int(
            debug.get(
                "scalar_policy_score_call_count"
            )
        ),
        "scalar_policy_final_score": (
            _as_float(debug.get("final_conv_score"))
            if applied
            else None
        ),
        "scalar_policy_score_trace": debug.get(
            "scalar_policy_score_trace",
            [],
        ),
    }


def build_action_delta_gate_log_fields(debug):
    """Extract compact JSON-safe Action-Delta Gate runtime metadata."""

    debug = debug or {}
    return {
        "use_action_delta_gate": bool(
            debug.get("use_action_delta_gate", False)
        ),
        "action_delta_gate_requested": bool(
            debug.get("action_delta_gate_requested", False)
        ),
        "action_delta_gate_applied": bool(
            debug.get("action_delta_gate_applied", False)
        ),
        "action_delta_gate_threshold": _as_float(
            debug.get("action_delta_gate_threshold")
        ),
        "action_delta_gate_outer_fold": _as_int(
            debug.get("action_delta_gate_outer_fold")
        ),
        "action_delta_gate_held_out_task_ids": debug.get(
            "action_delta_gate_held_out_task_ids", []
        ),
        "action_delta_gate_score_call_count": _as_int(
            debug.get("action_delta_gate_score_call_count")
        ),
        "action_delta_gate_predicted_trigger_count": _as_int(
            debug.get("action_delta_gate_predicted_trigger_count")
        ),
        "action_delta_gate_exact_terminal_accepted_trigger_count": _as_int(
            debug.get(
                "action_delta_gate_exact_terminal_accepted_trigger_count"
            )
        ),
        "action_delta_gate_oracle_confirm_accepted_count": _as_int(
            debug.get("action_delta_gate_oracle_confirm_accepted_count")
        ),
        "action_delta_gate_oracle_confirm_rejected_false_safe_count": _as_int(
            debug.get(
                "action_delta_gate_oracle_confirm_rejected_false_safe_count"
            )
        ),
        "action_delta_gate_exact_confirmation_trace": debug.get(
            "action_delta_gate_exact_confirmation_trace", []
        ),
        "action_delta_gate_diagnostic_coda_call_count": _as_int(
            debug.get("action_delta_gate_diagnostic_coda_call_count")
        ),
        "action_delta_gate_diagnostic_coda_iterations": debug.get(
            "action_delta_gate_diagnostic_coda_iterations", []
        ),
        "action_delta_gate_diagnostic_get_output_ms_list": debug.get(
            "action_delta_gate_diagnostic_get_output_ms_list", []
        ),
        "action_delta_gate_diagnostic_get_output_ms_total": _as_float(
            debug.get("action_delta_gate_diagnostic_get_output_ms_total")
        ),
        "action_delta_gate_mode_is_diagnostic": bool(
            debug.get("action_delta_gate_mode_is_diagnostic", False)
        ),
        "action_delta_gate_efficiency_eligible": bool(
            debug.get("action_delta_gate_efficiency_eligible", True)
        ),
        "action_delta_gate_min_terminal_iter": _as_int(
            debug.get("action_delta_gate_min_terminal_iter")
        ),
        "action_delta_gate_return_mode": debug.get(
            "action_delta_gate_return_mode", "anchor"
        ),
        "action_delta_gate_first_eligible_terminal_iteration": _as_int(
            debug.get(
                "action_delta_gate_first_eligible_terminal_iteration"
            )
        ),
        "action_delta_gate_exact_audit_enabled": bool(
            debug.get("action_delta_gate_exact_audit_enabled", False)
        ),
        "action_delta_gate_exact_audit_performed": bool(
            debug.get("action_delta_gate_exact_audit_performed", False)
        ),
        "action_delta_gate_exact_audit_anchor_iteration": _as_int(
            debug.get("action_delta_gate_exact_audit_anchor_iteration")
        ),
        "action_delta_gate_exact_audit_terminal_iteration": _as_int(
            debug.get("action_delta_gate_exact_audit_terminal_iteration")
        ),
        "action_delta_gate_exact_audit_full_mse": _as_float(
            debug.get("action_delta_gate_exact_audit_full_mse")
        ),
        "action_delta_gate_exact_audit_l2": _as_float(
            debug.get("action_delta_gate_exact_audit_l2")
        ),
        "action_delta_gate_exact_audit_max_abs": _as_float(
            debug.get("action_delta_gate_exact_audit_max_abs")
        ),
        "action_delta_gate_exact_audit_per_step_mse": debug.get(
            "action_delta_gate_exact_audit_per_step_mse"
        ),
        "action_delta_gate_exact_audit_per_step_max_abs": debug.get(
            "action_delta_gate_exact_audit_per_step_max_abs"
        ),
        "action_delta_gate_exact_audit_per_dim_mse": debug.get(
            "action_delta_gate_exact_audit_per_dim_mse"
        ),
        "action_delta_gate_exact_audit_per_dim_max_abs": debug.get(
            "action_delta_gate_exact_audit_per_dim_max_abs"
        ),
        "action_delta_gate_exact_audit_anchor_action": debug.get(
            "action_delta_gate_exact_audit_anchor_action"
        ),
        "action_delta_gate_exact_audit_terminal_action": debug.get(
            "action_delta_gate_exact_audit_terminal_action"
        ),
        "action_delta_gate_exact_audit_delta_action": debug.get(
            "action_delta_gate_exact_audit_delta_action"
        ),
        "action_delta_gate_exact_audit_predicted_delta_action": debug.get(
            "action_delta_gate_exact_audit_predicted_delta_action"
        ),
        "action_delta_gate_exact_audit_predicted_corrected_action": debug.get(
            "action_delta_gate_exact_audit_predicted_corrected_action"
        ),
        "action_delta_gate_exact_audit_correction_full_mse": _as_float(
            debug.get("action_delta_gate_exact_audit_correction_full_mse")
        ),
        "action_delta_gate_exact_audit_correction_l2": _as_float(
            debug.get("action_delta_gate_exact_audit_correction_l2")
        ),
        "action_delta_gate_exact_audit_correction_max_abs": _as_float(
            debug.get("action_delta_gate_exact_audit_correction_max_abs")
        ),
        "action_delta_gate_exact_audit_correction_per_step_mse": debug.get(
            "action_delta_gate_exact_audit_correction_per_step_mse"
        ),
        "action_delta_gate_exact_audit_correction_per_step_max_abs": debug.get(
            "action_delta_gate_exact_audit_correction_per_step_max_abs"
        ),
        "action_delta_gate_exact_audit_correction_per_dim_mse": debug.get(
            "action_delta_gate_exact_audit_correction_per_dim_mse"
        ),
        "action_delta_gate_exact_audit_correction_per_dim_max_abs": debug.get(
            "action_delta_gate_exact_audit_correction_per_dim_max_abs"
        ),
        "action_delta_gate_exact_audit_prefix_step_count": _as_int(
            debug.get("action_delta_gate_exact_audit_prefix_step_count")
        ),
        "action_delta_gate_exact_audit_anchor_reuse_prefix_mse": _as_float(
            debug.get("action_delta_gate_exact_audit_anchor_reuse_prefix_mse")
        ),
        "action_delta_gate_exact_audit_correction_prefix_mse": _as_float(
            debug.get("action_delta_gate_exact_audit_correction_prefix_mse")
        ),
        "action_delta_gate_exact_audit_correction_full_mse_ratio": _as_float(
            debug.get("action_delta_gate_exact_audit_correction_full_mse_ratio")
        ),
        "action_delta_gate_exact_audit_correction_prefix_mse_ratio": _as_float(
            debug.get("action_delta_gate_exact_audit_correction_prefix_mse_ratio")
        ),
        "action_delta_gate_exact_audit_action_shape": debug.get(
            "action_delta_gate_exact_audit_action_shape"
        ),
        "action_delta_gate_exact_audit_metric_action_shape": debug.get(
            "action_delta_gate_exact_audit_metric_action_shape"
        ),
        "action_delta_gate_exact_audit_leading_batch_dim_squeezed": (
            debug.get(
                "action_delta_gate_exact_audit_leading_batch_dim_squeezed"
            )
        ),
        "action_delta_gate_exact_audit_get_output_ms": _as_float(
            debug.get("action_delta_gate_exact_audit_get_output_ms")
        ),
        "action_delta_gate_exact_audit_get_output_call_count": _as_int(
            debug.get(
                "action_delta_gate_exact_audit_get_output_call_count"
            )
        ),
        "action_delta_gate_exact_audit_error": debug.get(
            "action_delta_gate_exact_audit_error"
        ),
        "action_delta_gate_score_trace": debug.get(
            "action_delta_gate_score_trace", []
        ),
        "action_delta_gate_triggered": bool(
            debug.get("action_delta_gate_triggered", False)
        ),
        "action_delta_gate_anchor_iteration": _as_int(
            debug.get("action_delta_gate_anchor_iteration")
        ),
        "action_delta_gate_terminal_iteration": _as_int(
            debug.get("action_delta_gate_terminal_iteration")
        ),
        "action_delta_gate_returned_action_source_iteration": _as_int(
            debug.get("action_delta_gate_returned_action_source_iteration")
        ),
        "action_delta_gate_skipped_coda_count": _as_int(
            debug.get("action_delta_gate_skipped_coda_count")
        ),
        "action_delta_gate_fallback_reason": debug.get(
            "action_delta_gate_fallback_reason"
        ),
        "action_delta_gate_predictor_ms_list": debug.get(
            "action_delta_gate_predictor_ms_list", []
        ),
        "action_delta_gate_predictor_ms_total": _as_float(
            debug.get("action_delta_gate_predictor_ms_total")
        ),
        "action_delta_gate_returned_previous_coda": bool(
            debug.get("action_delta_gate_returned_previous_coda", False)
        ),
        "action_delta_gate_returned_predicted_correction": bool(
            debug.get(
                "action_delta_gate_returned_predicted_correction",
                False,
            )
        ),
        "action_delta_gate_returned_anchor": bool(
            debug.get("action_delta_gate_returned_anchor", False)
        ),
    }


def build_decode_call_log_fields(debug):
    """Extract JSON-safe terminal decode metadata independent of profiling."""

    debug = debug or {}
    return {
        "coda_call_count": _as_int(debug.get("coda_call_count")),
        "get_output_call_count": _as_int(debug.get("get_output_call_count")),
        "final_state_coda_executed": debug.get("final_state_coda_executed"),
        "returned_cached_final_output": debug.get("returned_cached_final_output"),
    }


def build_action_delta_nonconvergence_log_fields(debug):
    """Extract development-only high-side filter accounting."""

    debug = debug or {}
    prefix = "action_delta_nonconvergence_filter_"
    names = (
        "requested",
        "applied",
        "development_only",
        "efficiency_eligible",
        "threshold",
        "min_terminal_iter",
        "max_skip",
        "score_call_count",
        "score_trace",
        "predicted_event_count",
        "actual_coda_skip_count",
        "forced_next_coda_call_count",
        "consecutive_skip_prevention_count",
        "max_iter_skip_prevention_count",
        "exact_coda_call_count",
        "recurrent_K",
        "first_trajectory_divergence_terminal_iteration",
        "events",
        "fallback_reason",
        "predictor_ms_list",
        "predictor_ms_total",
        "recurrent_ms_total",
        "coda_ms_total",
        "get_output_ms_total",
        "estimate_scorer_cost_ms_per_call",
        "estimate_coda_cost_ms_per_call",
        "estimated_gross_coda_savings_ms",
        "estimated_scorer_cost_ms",
        "estimated_net_savings_ms",
        "measured_gross_coda_savings_proxy_ms",
        "measured_net_savings_proxy_ms",
        "measured_net_savings_ms",
        "measured_net_savings_status",
    )
    fields = {
        "use_action_delta_nonconvergence_filter": bool(
            debug.get("use_action_delta_nonconvergence_filter", False)
        )
    }
    fields.update({prefix + name: debug.get(prefix + name) for name in names})
    return fields


def build_action_delta_deferred_backfill_log_fields(
    debug, prediction_identity=None
):
    """Extract adjacent-history deferred/backfill diagnostic accounting."""

    debug = debug or {}
    prefix = "action_delta_deferred_backfill_filter_"
    names = (
        "requested",
        "applied",
        "development_only",
        "efficiency_eligible",
        "threshold",
        "min_terminal_iter",
        "scorer_backend",
        "scorer_numerical_equivalence",
        "compile_setup_ms",
        "compile_setup_in_predictor_timing",
        "development_decision_parity",
        "score_call_count",
        "score_trace",
        "predictor_ms_list",
        "predictor_ms_total",
        "high_score_deferred_call_count",
        "consecutive_run_lengths",
        "runs",
        "backfill_coda_call_count",
        "backfill_get_output_ms_list",
        "backfill_get_output_ms_total",
        "backfill_coda_ms_list",
        "backfill_coda_ms_total",
        "current_state_coda_call_count",
        "current_get_output_ms_list",
        "current_get_output_ms_total",
        "current_coda_ms_list",
        "current_coda_ms_total",
        "truly_eliminated_coda_call_count",
        "total_exact_coda_call_count",
        "recurrent_K",
        "exact_stop_mse_trace",
        "unresolved_max_iter_fallback_count",
        "fallback_reason",
        "recurrent_ms_total",
        "coda_ms_total",
        "get_output_ms_total",
        "actual_inference_component_ms_total",
        "fixed_scorer_cost_ms_per_call",
        "fixed_coda_cost_ms_per_call",
        "fixed_estimated_scorer_cost_ms",
        "fixed_estimated_coda_savings_ms",
        "fixed_estimated_net_savings_ms",
    )
    fields = {
        "use_action_delta_deferred_backfill_filter": bool(
            debug.get("use_action_delta_deferred_backfill_filter", False)
        )
    }
    fields.update({prefix + name: debug.get(prefix + name) for name in names})
    if prediction_identity is not None:
        fields[prefix + "runs"] = [
            {**run, "prediction_identity": dict(prediction_identity)}
            for run in (fields[prefix + "runs"] or [])
        ]
    return fields


def resolve_fixed_k_log_value(debug, recurrence_strategy, recurrent_num_iter):
    """Resolve fixed depth for legacy and terminal-only step records."""

    debug = debug or {}
    fixed_k = _as_int(debug.get("fixed_K"))
    if (
        fixed_k is None
        and recurrence_strategy in {"fixed", "fixed_terminal_only"}
        and recurrent_num_iter is not None
    ):
        fixed_k = int(recurrent_num_iter)
    return fixed_k


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
    if (
        final_mse is None
        and recurrence_debug.get("canonical_recurrence_strategy") != "latent_only"
    ):
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
        "canonical_stop_reason": recurrence_debug.get("canonical_stop_reason"),
        "canonical_recurrence_strategy": recurrence_debug.get("canonical_recurrence_strategy"),
        "metric_name": recurrence_debug.get("metric_name"),
        "actual_origin": recurrence_debug.get("actual_origin"),
        "effective_threshold": recurrence_debug.get("effective_threshold"),
        "coda_call_count": recurrence_debug.get("coda_call_count"),
        **build_scalar_policy_log_fields(
            recurrence_debug
        ),
        **build_action_delta_gate_log_fields(recurrence_debug),
        **build_action_delta_nonconvergence_log_fields(recurrence_debug),
        **build_action_delta_deferred_backfill_log_fields(recurrence_debug),
        "latent_metric_call_count": recurrence_debug.get("latent_metric_call_count"),
        "latent_precheck_mode": recurrence_debug.get("latent_precheck_mode", "legacy"),
        "latent_precheck_trace_level_requested": recurrence_debug.get("latent_precheck_trace_level_requested", "off"),
        "latent_precheck_trace_collected": recurrence_debug.get("latent_precheck_trace_collected"),
        "latent_precheck_origin": recurrence_debug.get("latent_precheck_origin"),
        "latent_precheck_active_threshold": recurrence_debug.get("latent_precheck_active_threshold"),
        "latent_precheck_max_skip_iters": recurrence_debug.get("latent_precheck_max_skip_iters"),
        "latent_precheck_confirmation_mode": recurrence_debug.get("latent_precheck_confirmation_mode"),
        "latent_precheck_call_count": recurrence_debug.get("latent_precheck_call_count"),
        "coda_reason_counts": recurrence_debug.get("coda_reason_counts", {}),
        "adjacent_comparison_pair_count": recurrence_debug.get("adjacent_comparison_pair_count"),
        "final_state_coda_executed": recurrence_debug.get("final_state_coda_executed"),
        "returned_cached_final_output": recurrence_debug.get("returned_cached_final_output"),
        "max_iteration_convergence_evaluable": recurrence_debug.get("max_iteration_convergence_evaluable"),
        "final_convergence_evaluable": recurrence_debug.get("final_convergence_evaluable"),
        "execution_path": recurrence_debug.get("execution_path"),
        "nonfinite_policy": recurrence_debug.get("nonfinite_policy"),
        "numerical_retry_attempted": recurrence_debug.get("numerical_retry_attempted"),
        "numerical_retry_succeeded": recurrence_debug.get("numerical_retry_succeeded"),
        "first_attempt_origin": recurrence_debug.get("first_attempt_origin"),
        "first_attempt_failure": recurrence_debug.get("first_attempt_failure"),
        "first_attempt_coda_attempt_count": recurrence_debug.get("first_attempt_coda_attempt_count"),
        "retry_coda_call_count": recurrence_debug.get("retry_coda_call_count"),
        "get_output_attempt_count_intent_to_treat": recurrence_debug.get(
            "get_output_attempt_count_intent_to_treat"
        ),
        "shadow_full_depth_enabled": recurrence_debug.get("shadow_full_depth_enabled", False),
        "shadow_trace_complete": recurrence_debug.get("shadow_trace_complete"),
        "shadow_tail_start_iteration": recurrence_debug.get("shadow_tail_start_iteration"),
        "shadow_tail_iteration_count": recurrence_debug.get("shadow_tail_iteration_count"),
        "shadow_error": recurrence_debug.get("shadow_error"),
        "shadow_production_snapshot": recurrence_debug.get("shadow_production_snapshot"),
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


def save_prediction_action_head_workload(
    cfg,
    workload,
    *,
    capture_requested: bool,
    identity: Optional[Mapping[str, Any]],
):
    fields = {
        "action_head_workload_requested": bool(capture_requested),
        "action_head_workload_captured": False,
        "action_head_workload_file": None,
        "action_head_workload_sha256": None,
        "action_head_workload_schema_version": None,
        "action_head_workload_tensor_fields": [],
        "action_head_workload_capture_in_action_latency": bool(capture_requested),
    }
    if not capture_requested:
        if workload is not None:
            raise RuntimeError("action-head workload was produced without a capture request")
        return fields
    if not isinstance(workload, Mapping):
        raise RuntimeError("requested action-head workload metadata is missing")
    if not isinstance(identity, Mapping):
        raise RuntimeError("requested action-head workload identity is missing")

    output_dir = Path(cfg.calibration_workload_dir)
    filename = (
        f"task{int(identity['task_id'])}_trial{int(identity['paired_trial_id'])}_"
        f"pred{int(identity['prediction_step'])}.pt"
    )
    output_path = output_dir / filename
    digest = save_action_head_workload(output_path, workload, identity=identity)
    step_log_file = get_step_log_file(cfg)
    if not step_log_file:
        raise RuntimeError("action-head workload capture requires a step log file")
    step_log_dir = Path(step_log_file).resolve().parent
    resolved_output = output_path.resolve()
    try:
        recorded_path = str(resolved_output.relative_to(step_log_dir))
    except ValueError:
        recorded_path = str(resolved_output)
    fields.update(
        {
            "action_head_workload_captured": True,
            "action_head_workload_file": recorded_path,
            "action_head_workload_sha256": digest,
            "action_head_workload_schema_version": ACTION_HEAD_WORKLOAD_SCHEMA_VERSION,
            "action_head_workload_tensor_fields": list(ACTION_HEAD_WORKLOAD_TENSORS),
        }
    )
    return fields


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


def summarize_action_delta_gate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    requested = [r for r in records if r.get("action_delta_gate_requested")]
    efficiency_eligible = [
        r
        for r in requested
        if r.get("action_delta_gate_efficiency_eligible", True)
    ]
    return {
        "requested_prediction_count": len(requested),
        "applied_prediction_count": sum(
            1 for r in requested if r.get("action_delta_gate_applied")
        ),
        "trigger_count": sum(
            1
            for r in efficiency_eligible
            if r.get("action_delta_gate_triggered")
        ),
        "predicted_gate_trigger_count": int(
            sum(
                _as_int(r.get("action_delta_gate_predicted_trigger_count"))
                or 0
                for r in requested
            )
        ),
        "exact_terminal_accepted_trigger_count": int(
            sum(
                _as_int(
                    r.get(
                        "action_delta_gate_exact_terminal_accepted_trigger_count"
                    )
                )
                or 0
                for r in requested
            )
        ),
        "oracle_confirm_accepted_count": int(
            sum(
                _as_int(
                    r.get("action_delta_gate_oracle_confirm_accepted_count")
                )
                or 0
                for r in requested
            )
        ),
        "oracle_confirm_rejected_false_safe_count": int(
            sum(
                _as_int(
                    r.get(
                        "action_delta_gate_oracle_confirm_rejected_false_safe_count"
                    )
                )
                or 0
                for r in requested
            )
        ),
        "diagnostic_mode_prediction_count": len(requested)
        - len(efficiency_eligible),
        "diagnostic_coda_call_count": int(
            sum(
                _as_int(r.get("action_delta_gate_diagnostic_coda_call_count"))
                or 0
                for r in requested
            )
        ),
        "skipped_coda_count": int(
            sum(
                _as_int(r.get("action_delta_gate_skipped_coda_count")) or 0
                for r in efficiency_eligible
            )
        ),
        "score_call_count": int(
            sum(
                _as_int(r.get("action_delta_gate_score_call_count")) or 0
                for r in requested
            )
        ),
        "predictor_ms_total_stats": _numeric_stats(
            [r.get("action_delta_gate_predictor_ms_total") for r in requested]
        ),
        "exact_audit_enabled_prediction_count": sum(
            1
            for r in requested
            if r.get("action_delta_gate_exact_audit_enabled")
        ),
        "exact_audit_performed_count": sum(
            1
            for r in requested
            if r.get("action_delta_gate_exact_audit_performed")
        ),
        "exact_audit_get_output_call_count": int(
            sum(
                _as_int(
                    r.get(
                        "action_delta_gate_exact_audit_get_output_call_count"
                    )
                )
                or 0
                for r in requested
            )
        ),
        "exact_audit_get_output_ms_total": float(
            sum(
                _as_float(
                    r.get("action_delta_gate_exact_audit_get_output_ms")
                )
                or 0.0
                for r in requested
            )
        ),
        "exact_audit_error_count": sum(
            1
            for r in requested
            if r.get("action_delta_gate_exact_audit_error") is not None
        ),
    }


def summarize_action_delta_nonconvergence_filter(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    requested = [
        record
        for record in records
        if record.get("action_delta_nonconvergence_filter_requested")
    ]
    prefix = "action_delta_nonconvergence_filter_"

    def total(name):
        return sum(_as_float(record.get(prefix + name)) or 0.0 for record in requested)

    first_divergence_record = next(
        (
            record
            for record in requested
            if (_as_int(record.get(prefix + "actual_coda_skip_count")) or 0) > 0
        ),
        None,
    )
    first_divergence = None
    if first_divergence_record is not None:
        first_divergence = {
            "task_id": _as_int(first_divergence_record.get("task_id")),
            "episode_id": _as_int(first_divergence_record.get("episode_id")),
            "episode_seed": _as_int(first_divergence_record.get("episode_seed")),
            "action_prediction_index": _as_int(
                first_divergence_record.get("action_prediction_index")
            ),
            "environment_timestep": _as_int(
                first_divergence_record.get("timestep")
            ),
            "terminal_iteration": _as_int(
                first_divergence_record.get(
                    prefix + "first_trajectory_divergence_terminal_iteration"
                )
            ),
        }

    return {
        "development_only": True,
        "excluded_from_production_efficiency_claims": True,
        "requested_prediction_count": len(requested),
        "applied_prediction_count": sum(
            bool(record.get(prefix + "applied")) for record in requested
        ),
        "score_call_count": int(total("score_call_count")),
        "predicted_nonconvergence_event_count": int(total("predicted_event_count")),
        "actual_coda_skip_count": int(total("actual_coda_skip_count")),
        "forced_next_coda_call_count": int(total("forced_next_coda_call_count")),
        "consecutive_skip_prevention_count": int(
            total("consecutive_skip_prevention_count")
        ),
        "exact_coda_call_count": int(total("exact_coda_call_count")),
        "first_trajectory_divergence_point": first_divergence,
        "predictor_ms_total": float(total("predictor_ms_total")),
        "recurrent_ms_total": float(total("recurrent_ms_total")),
        "coda_ms_total": float(total("coda_ms_total")),
        "get_output_ms_total": float(total("get_output_ms_total")),
        "estimated_gross_coda_savings_ms": float(
            total("estimated_gross_coda_savings_ms")
        ),
        "estimated_scorer_cost_ms": float(total("estimated_scorer_cost_ms")),
        "estimated_net_savings_ms": float(total("estimated_net_savings_ms")),
        "measured_net_savings_proxy_ms": float(
            total("measured_net_savings_proxy_ms")
        ),
        "measured_net_savings_status": "requires_paired_warm_only_counterfactual",
    }


def summarize_action_delta_deferred_backfill_filter(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    prefix = "action_delta_deferred_backfill_filter_"
    requested = [record for record in records if record.get(prefix + "requested")]

    def total(name):
        return sum(_as_float(record.get(prefix + name)) or 0.0 for record in requested)

    run_lengths = [
        int(length)
        for record in requested
        for length in (record.get(prefix + "consecutive_run_lengths") or [])
    ]
    scorer_backends = sorted(
        {
            str(record.get(prefix + "scorer_backend"))
            for record in requested
            if record.get(prefix + "scorer_backend") is not None
        }
    )
    return {
        "development_only": True,
        "excluded_from_production_efficiency_claims": True,
        "requested_prediction_count": len(requested),
        "scorer_backends": scorer_backends,
        "compile_setup_ms_one_time_per_task": max(
            (
                _as_float(record.get(prefix + "compile_setup_ms")) or 0.0
                for record in requested
            ),
            default=0.0,
        ),
        "compile_setup_in_predictor_timing": False,
        "applied_prediction_count": sum(
            bool(record.get(prefix + "applied")) for record in requested
        ),
        "score_call_count": int(total("score_call_count")),
        "high_score_deferred_call_count": int(
            total("high_score_deferred_call_count")
        ),
        "backfill_coda_call_count": int(total("backfill_coda_call_count")),
        "current_state_coda_call_count": int(
            total("current_state_coda_call_count")
        ),
        "truly_eliminated_coda_call_count": int(
            total("truly_eliminated_coda_call_count")
        ),
        "total_exact_coda_call_count": int(total("total_exact_coda_call_count")),
        "unresolved_max_iter_fallback_count": int(
            total("unresolved_max_iter_fallback_count")
        ),
        "consecutive_run_lengths": run_lengths,
        "predictor_ms_total": float(total("predictor_ms_total")),
        "backfill_get_output_ms_total": float(
            total("backfill_get_output_ms_total")
        ),
        "current_get_output_ms_total": float(
            total("current_get_output_ms_total")
        ),
        "recurrent_ms_total": float(total("recurrent_ms_total")),
        "coda_ms_total": float(total("coda_ms_total")),
        "get_output_ms_total": float(total("get_output_ms_total")),
        "fixed_estimated_scorer_cost_ms": float(
            total("fixed_estimated_scorer_cost_ms")
        ),
        "fixed_estimated_coda_savings_ms": float(
            total("fixed_estimated_coda_savings_ms")
        ),
        "fixed_estimated_net_savings_ms": float(
            total("fixed_estimated_net_savings_ms")
        ),
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
                "action_delta_gate": summarize_action_delta_gate(prediction_records),
                "action_delta_nonconvergence_filter": summarize_action_delta_nonconvergence_filter(
                    prediction_records
                ),
                "action_delta_deferred_backfill_filter": summarize_action_delta_deferred_backfill_filter(
                    prediction_records
                ),
            },
            "success": {
                "iteration_stats": _numeric_stats([r.get("recurrent_iteration_count") for r in success_records]),
                "final_mse_stats": _numeric_stats([r.get("final_mse") for r in success_records]),
                "adaptive_stop_count": sum(1 for r in success_records if r.get("adaptive_stop")),
                "coda_profiling": summarize_coda_profiling(success_records),
                "latent_precheck": summarize_latent_precheck(success_records),
                "action_delta_gate": summarize_action_delta_gate(success_records),
                "action_delta_nonconvergence_filter": summarize_action_delta_nonconvergence_filter(
                    success_records
                ),
                "action_delta_deferred_backfill_filter": summarize_action_delta_deferred_backfill_filter(
                    success_records
                ),
            },
            "failure": {
                "iteration_stats": _numeric_stats([r.get("recurrent_iteration_count") for r in failure_records]),
                "final_mse_stats": _numeric_stats([r.get("final_mse") for r in failure_records]),
                "adaptive_stop_count": sum(1 for r in failure_records if r.get("adaptive_stop")),
                "coda_profiling": summarize_coda_profiling(failure_records),
                "latent_precheck": summarize_latent_precheck(failure_records),
                "action_delta_gate": summarize_action_delta_gate(failure_records),
                "action_delta_nonconvergence_filter": summarize_action_delta_nonconvergence_filter(
                    failure_records
                ),
                "action_delta_deferred_backfill_filter": summarize_action_delta_deferred_backfill_filter(
                    failure_records
                ),
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
    episode_protocol=None,
    profile_state=None,
    timing_state=None,
    raw_shadow_writer=None,
    action_delta_shadow_writer=None,
):
    """Run a single episode in the environment."""
    episode_protocol = dict(episode_protocol or {})
    evaluation_protocol_phase = episode_protocol.get("phase", "legacy")
    paired_rng = bool(episode_protocol.get("paired_rng", False))
    episode_seed = episode_protocol.get("episode_seed")
    episode_seed_source = "paired_protocol" if episode_seed is not None else None
    if episode_seed is None and getattr(cfg, "reset_rng_each_episode", False):
        episode_id_for_seed = int(episode_idx) if episode_idx is not None else 0
        episode_seed = int(cfg.seed) + episode_id_for_seed * int(cfg.episode_seed_stride)
        episode_seed_source = "legacy_stride"

    obs, environment_seed_applied = reset_episode_environment(
        env,
        initial_state,
        episode_seed=episode_seed,
        torch_module=torch,
        seed_environment=paired_rng,
    )

    protocol_log_metadata = {
        "evaluation_protocol_phase": evaluation_protocol_phase,
        "source_commit": episode_protocol.get("source_commit"),
        "initial_state_partition": episode_protocol.get("partition"),
        "paired_trial_id": episode_protocol.get("paired_trial_id"),
        "initial_state_id": episode_protocol.get("initial_state_id"),
        "initial_states_sha256": episode_protocol.get("initial_states_sha256"),
        "initial_states_file": episode_protocol.get("initial_states_file"),
        "initial_states_file_sha256": episode_protocol.get("initial_states_file_sha256"),
        "initial_state_manifest_sha256": episode_protocol.get("manifest_sha256"),
        "paired_rng": paired_rng,
        "episode_seed_source": episode_seed_source,
        "environment_seed_applied": environment_seed_applied,
        "smoke_excluded_from_fitting": bool(episode_protocol.get("smoke_excluded_from_fitting", False)),
    }
    parity_hash_logging_enabled = bool(
        cfg.shadow_full_depth
        or evaluation_protocol_phase in {"screening", "final_holdout"}
    )

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
    action_delta_shadow_predictions = []
    action_delta_shadow_episode_error = None

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
                prediction_id = require_prediction_id(prediction_step)
                cache_age_for_prediction = warm_start_cache_age
                proprio_before_pred = None
                if "state" in observation:
                    proprio_before_pred = np.array(observation["state"], dtype=np.float32).reshape(-1).copy()

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                rng_state_before_sha256 = (
                    _rng_state_sha256() if parity_hash_logging_enabled else None
                )
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
                workload_capture_limit = int(
                    getattr(cfg, "calibration_workload_predictions_per_episode", 0)
                )
                capture_action_head_workload = (
                    cfg.evaluation_protocol_phase == "calibration"
                    and prediction_id < workload_capture_limit
                )
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
                        capture_action_head_workload=capture_action_head_workload,
                    )

                timing_metadata = {
                    "task_id": task_id,
                    "episode_id": episode_idx,
                    "timestep": int(t),
                    "action_prediction_index": prediction_id,
                    "prediction_step": prediction_id,
                    **protocol_log_metadata,
                }
                try:
                    actions, actual_iters, final_kl, inference_metadata = run_action_with_optional_profiles(
                        cfg,
                        profile_state,
                        timing_state,
                        log_file,
                        predict_action_once,
                        timing_metadata,
                    )
                except NumericalInferenceAbort as numerical_abort:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    action_latency_ms = (time.perf_counter() - action_start) * 1000.0
                    episode_action_latencies_ms.append(action_latency_ms)
                    abort_rng_state_after_sha256 = (
                        _rng_state_sha256()
                        if parity_hash_logging_enabled
                        else None
                    )
                    abort_debug = (
                        getattr(getattr(action_head, "model", None), "last_recurrence_debug", None)
                        or {}
                    )
                    warm_start_state = None
                    warm_start_cache_age = 0
                    episode_step_logs.append(
                        {
                            "schema_version": 1,
                            "task_id": task_id,
                            "task_name": task_description,
                            "episode_id": episode_idx,
                            "episode_seed": episode_seed,
                            "timestep": int(t),
                            "action_prediction_index": prediction_id,
                            "prediction_step": prediction_id,
                            **protocol_log_metadata,
                            "action_delta_gate_artifact_sha256": (
                                cfg.action_delta_gate_expected_sha256.lower()
                                if cfg.use_action_delta_deferred_backfill_filter
                                else None
                            ),
                            "action_delta_deferred_scorer_backend": (
                                cfg.action_delta_deferred_scorer_backend
                                if cfg.use_action_delta_deferred_backfill_filter
                                else None
                            ),
                            "action_delta_deferred_threshold": (
                                float(ACTION_DELTA_NONCONVERGENCE_THRESHOLD)
                                if cfg.use_action_delta_deferred_backfill_filter
                                else None
                            ),
                            "action_delta_deferred_min_terminal_iter": (
                                int(cfg.action_delta_gate_min_terminal_iter)
                                if cfg.use_action_delta_deferred_backfill_filter
                                else None
                            ),
                            "recurrent_iteration_count": None,
                            "final_mse": None,
                            "adaptive_stop": False,
                            **build_stop_reason_fields(abort_debug),
                            "execution_path": "numerical_abort",
                            "nonfinite_policy": getattr(cfg, "nonfinite_policy", "legacy"),
                            "numerical_retry_attempted": True,
                            "numerical_retry_succeeded": False,
                            "numerical_abort": numerical_abort.to_dict(),
                            "first_attempt_origin": abort_debug.get("first_attempt_origin"),
                            "latency_ms": action_latency_ms,
                            "success": None,
                            "parity_hash_schema": PARITY_HASH_SCHEMA,
                            "returned_action_sha256": None,
                            "next_warm_start_state_sha256": None,
                            "rng_state_before_action_sha256": (
                                rng_state_before_sha256
                            ),
                            "rng_state_after_action_sha256": (
                                abort_rng_state_after_sha256
                            ),
                        }
                    )
                    raise
                inference_metadata = inference_metadata or {}
                if isinstance(inference_metadata, dict):
                    recurrence_debug = inference_metadata.get("recurrence_debug", inference_metadata)
                    warm_start_metadata = dict(inference_metadata.get("warm_start") or {})
                    next_warm_start_state = inference_metadata.get("next_warm_start_state")
                    action_head_workload = inference_metadata.get("action_head_workload")
                    preconvergence_raw_shadow = inference_metadata.get(
                        "preconvergence_raw_shadow"
                    )
                    action_delta_gate_shadow = inference_metadata.get(
                        "action_delta_gate_shadow"
                    )
                else:
                    recurrence_debug = {}
                    warm_start_metadata = {}
                    next_warm_start_state = None
                    action_head_workload = None
                    preconvergence_raw_shadow = None
                    action_delta_gate_shadow = None

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
                # Parity hashing is logging-only and intentionally begins only
                # after inference latency has been finalized.
                rng_state_after_sha256 = (
                    _rng_state_sha256() if parity_hash_logging_enabled else None
                )
                returned_action_sha256 = (
                    _tensor_or_array_sha256(actions)
                    if parity_hash_logging_enabled
                    else None
                )
                next_warm_start_state_sha256 = (
                    _tensor_or_array_sha256(next_warm_start_state)
                    if parity_hash_logging_enabled
                    else None
                )
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
                fixed_k = resolve_fixed_k_log_value(
                    debug, recurrence_strategy, recurrent_num_iter
                )
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
                final_convergence_evaluable = debug.get("final_convergence_evaluable")
                if final_mse is None and iteration_mse and final_convergence_evaluable is not False:
                    final_mse = iteration_mse[-1]
                actual_origin = "ACTUAL_WARM" if warm_start_used else "COLD"
                latent_metric_trace = build_latent_metric_trace_records(
                    debug,
                    task_id=task_id,
                    episode_id=episode_idx,
                    prediction_id=prediction_id,
                    actual_origin=actual_origin,
                )

                step_record = {
                    "schema_version": 1,
                    "task_id": task_id,
                    "task_name": task_description,
                    "episode_id": episode_idx,
                    "timestep": int(t),
                    "action_prediction_index": prediction_id,
                    "prediction_step": prediction_id,
                    "reset_rng_each_episode": bool(getattr(cfg, "reset_rng_each_episode", False)),
                    "episode_seed": episode_seed,
                    **protocol_log_metadata,
                    "action_delta_gate_artifact_sha256": (
                        cfg.action_delta_gate_expected_sha256.lower()
                        if cfg.use_action_delta_deferred_backfill_filter
                        else None
                    ),
                    "action_delta_deferred_scorer_backend": (
                        cfg.action_delta_deferred_scorer_backend
                        if cfg.use_action_delta_deferred_backfill_filter
                        else None
                    ),
                    "action_delta_deferred_threshold": (
                        float(ACTION_DELTA_NONCONVERGENCE_THRESHOLD)
                        if cfg.use_action_delta_deferred_backfill_filter
                        else None
                    ),
                    "action_delta_deferred_min_terminal_iter": (
                        int(cfg.action_delta_gate_min_terminal_iter)
                        if cfg.use_action_delta_deferred_backfill_filter
                        else None
                    ),
                    "method": debug.get("strategy", recurrence_strategy),
                    "recurrence_strategy": debug.get("strategy", recurrence_strategy),
                    "requested_recurrence_strategy": debug.get("requested_recurrence_strategy", recurrence_strategy),
                    "canonical_recurrence_strategy": debug.get("canonical_recurrence_strategy"),
                    "canonical_metric_name": debug.get("canonical_metric_name"),
                    "action_mse_threshold": debug.get("action_mse_threshold"),
                    "threshold": threshold,
                    "fixed_K": fixed_k,
                    "K_t": recurrent_iteration_count,
                    "recurrent_iteration_count": recurrent_iteration_count,
                    "max_recurrent_iteration": _as_int(max_recurrent_iteration),
                    "adaptive_stop": bool(debug.get("adaptive_stop", False)),
                    **build_stop_reason_fields(debug),
                    "metric_name": debug.get("metric_name", "mse_between_action_outputs"),
                    "iteration_mse": iteration_mse,
                    "iteration_metric_values": debug.get("iteration_metric_values", iteration_mse),
                    "actual_origin": debug.get("actual_origin", actual_origin),
                    "configured_cold_threshold": debug.get("configured_cold_threshold"),
                    "configured_warm_threshold": debug.get("configured_warm_threshold"),
                    "effective_threshold": debug.get("effective_threshold"),
                    "latent_only_metric": debug.get("latent_only_metric"),
                    "latent_only_min_iter": debug.get("latent_only_min_iter"),
                    "latent_only_eps": debug.get("latent_only_eps"),
                    "latent_only_trace": debug.get("latent_only_trace", []),
                    **build_decode_call_log_fields(debug),
                    **build_scalar_policy_log_fields(
                        debug
                    ),
                    **build_action_delta_gate_log_fields(debug),
                    **build_action_delta_nonconvergence_log_fields(debug),
                    **build_action_delta_deferred_backfill_log_fields(
                        debug,
                        {
                            "task_id": int(task_id),
                            "episode_id": int(episode_idx),
                            "episode_seed": int(episode_seed),
                            "action_prediction_index": int(prediction_id),
                            "environment_timestep": int(t),
                            "trajectory_id": protocol_log_metadata.get(
                                "trajectory_id"
                            ),
                            "initial_state_id": protocol_log_metadata.get(
                                "initial_state_id"
                            ),
                        },
                    ),
                    "latent_metric_call_count": debug.get("latent_metric_call_count"),
                    "latent_metric_trace_enabled": debug.get(
                        "latent_metric_trace_enabled", False
                    ),
                    "latent_dynamics_trace_enabled": debug.get(
                        "latent_dynamics_trace_enabled", False
                    ),
                    "latent_dynamics_warm_anchor_available": debug.get(
                        "latent_dynamics_warm_anchor_available", False
                    ),
                    "latent_metric_trace_eps": debug.get("latent_metric_trace_eps"),
                    "latent_metric_trace": latent_metric_trace,
                    "final_mse": _as_float(final_mse),
                    "final_convergence_evaluable": final_convergence_evaluable,
                    "final_conv_score": debug.get("final_conv_score", final_kl),
                    "conv_score_list": debug.get("conv_score_list", []),
                    "action_delta_list": debug.get("action_delta_list", []),
                    "latent_mse_list": debug.get("latent_mse_list", []),
                    "latent_l2_list": debug.get("latent_l2_list", []),
                    "latent_action_mse_pairs": debug.get("latent_action_mse_pairs", []),
                    "latent_action_pair_count": debug.get("latent_action_pair_count", 0),
                    "use_latent_precheck": bool(debug.get("use_latent_precheck", False)),
                    "latent_precheck_mode": debug.get("latent_precheck_mode", getattr(cfg, "latent_precheck_mode", "legacy")),
                    "latent_precheck_trace_level_requested": debug.get("latent_precheck_trace_level_requested", getattr(cfg, "latent_precheck_trace_level", "off")),
                    "latent_precheck_trace_level_applied": debug.get("latent_precheck_trace_level_applied"),
                    "latent_precheck_trace_collected": debug.get("latent_precheck_trace_collected"),
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
                    "latent_precheck_warm_thresh": debug.get("latent_precheck_warm_thresh"),
                    "latent_precheck_cold_thresh": debug.get("latent_precheck_cold_thresh"),
                    "latent_precheck_active_threshold": debug.get("latent_precheck_active_threshold"),
                    "latent_precheck_origin": debug.get("latent_precheck_origin"),
                    "latent_precheck_max_skip_iters": debug.get("latent_precheck_max_skip_iters"),
                    "latent_precheck_confirmation_mode": debug.get("latent_precheck_confirmation_mode"),
                    "latent_metric_count": debug.get("latent_metric_count"),
                    "coda_reason_counts": debug.get("coda_reason_counts", {}),
                    "coda_call_records": debug.get("coda_call_records", []),
                    "adjacent_comparison_pairs": debug.get("adjacent_comparison_pairs", []),
                    "adjacent_comparison_pair_count": debug.get("adjacent_comparison_pair_count", 0),
                    "origin_aware_scheduler_state": debug.get("origin_aware_scheduler_state"),
                    "cached_final_matches_returned": debug.get("cached_final_matches_returned"),
                    "max_iteration_convergence_evaluable": debug.get("max_iteration_convergence_evaluable"),
                    "execution_path": debug.get("execution_path"),
                    "nonfinite_policy": debug.get(
                        "nonfinite_policy", getattr(cfg, "nonfinite_policy", "legacy")
                    ),
                    "numerical_retry_attempted": debug.get("numerical_retry_attempted", False),
                    "numerical_retry_succeeded": debug.get("numerical_retry_succeeded"),
                    "numerical_retry_count": debug.get("numerical_retry_count", 0),
                    "first_attempt_origin": debug.get("first_attempt_origin"),
                    "first_attempt_failure": debug.get("first_attempt_failure"),
                    "first_attempt_coda_attempt_count": debug.get("first_attempt_coda_attempt_count", 0),
                    "retry_coda_call_count": debug.get("retry_coda_call_count", 0),
                    "get_output_attempt_count_intent_to_treat": debug.get(
                        "get_output_attempt_count_intent_to_treat"
                    ),
                    "shadow_full_depth_enabled": debug.get("shadow_full_depth_enabled", False),
                    "shadow_trace_complete": debug.get("shadow_trace_complete"),
                    "shadow_tail_start_iteration": debug.get("shadow_tail_start_iteration"),
                    "shadow_tail_iteration_count": debug.get("shadow_tail_iteration_count"),
                    "shadow_error": debug.get("shadow_error"),
                    "shadow_production_snapshot": debug.get("shadow_production_snapshot"),
                    "shadow_trace": debug.get("shadow_trace", []),
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
                    "warm_start_first_attempt_used": warm_start_metadata.get(
                        "first_attempt_state_used"
                    ),
                    "warm_start_first_attempt_origin": warm_start_metadata.get(
                        "first_attempt_initial_state_origin"
                    ),
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
                    "parity_hash_schema": PARITY_HASH_SCHEMA,
                    "returned_action_sha256": returned_action_sha256,
                    "next_warm_start_state_sha256": next_warm_start_state_sha256,
                    "rng_state_before_action_sha256": rng_state_before_sha256,
                    "rng_state_after_action_sha256": rng_state_after_sha256,
                }
                workload_identity = build_action_head_workload_identity(
                    capture_requested=capture_action_head_workload,
                    task_id=task_id,
                    episode_id=episode_idx,
                    paired_trial_id=protocol_log_metadata["paired_trial_id"],
                    prediction_id=prediction_id,
                    initial_state_id=protocol_log_metadata["initial_state_id"],
                    episode_seed=episode_seed,
                )
                step_record.update(
                    save_prediction_action_head_workload(
                        cfg,
                        action_head_workload,
                        capture_requested=capture_action_head_workload,
                        identity=workload_identity,
                    )
                )
                if cfg.collect_preconvergence_raw_shadow:
                    if raw_shadow_writer is None:
                        raise RuntimeError("raw collection enabled without a shard writer")
                    if preconvergence_raw_shadow is None:
                        raise RuntimeError(
                            "raw collection was requested but the action head returned no trajectory"
                        )
                    expected_origin = "ACTUAL_WARM" if warm_start_used else "COLD_PRIMARY"
                    if preconvergence_raw_shadow.get("actual_origin") != expected_origin:
                        raise RuntimeError("raw trajectory origin does not match rollout warm-start metadata")
                    raw_prediction = build_prediction_payload(
                        preconvergence_raw_shadow,
                        task_id=int(task_id),
                        task_name=task_description,
                        episode_id=int(episode_idx),
                        timestep=int(t),
                        prediction_id=prediction_id,
                        protocol_identity=protocol_log_metadata,
                        warm_start_metadata=warm_start_metadata,
                        checkpoint=raw_shadow_writer.checkpoint,
                        source_commit=raw_shadow_writer.source_commit,
                        run_identity=raw_shadow_writer.run_identity,
                        returned_action_sha256=step_record["returned_action_sha256"],
                        rng_state_before_sha256=rng_state_before_sha256,
                        rng_state_after_sha256=rng_state_after_sha256,
                    )
                    raw_shadow_writer.add(raw_prediction)
                    step_record.update(
                        {
                            "preconvergence_raw_shadow_collected": True,
                            "preconvergence_raw_shadow_schema_version": RAW_PRECONVERGENCE_SCHEMA_VERSION,
                            "preconvergence_raw_shadow_tensor_sha256": raw_prediction[
                                "tensor_sha256"
                            ],
                        }
                    )
                else:
                    step_record["preconvergence_raw_shadow_collected"] = False
                if cfg.collect_action_delta_gate_shadow:
                    if action_delta_shadow_writer is None:
                        raise RuntimeError(
                            "Action-Delta shadow collection enabled without a writer"
                        )
                    if action_delta_gate_shadow is None:
                        raise RuntimeError(
                            "Action-Delta shadow collection was requested but "
                            "the action head returned no payload"
                        )
                    action_delta_shadow_prediction = build_shadow_prediction_payload(
                        action_delta_gate_shadow,
                        task_id=int(task_id),
                        task_name=task_description,
                        episode_id=int(episode_idx),
                        initial_state_id=int(
                            protocol_log_metadata["initial_state_id"]
                        ),
                        paired_trial_id=int(
                            protocol_log_metadata["paired_trial_id"]
                        ),
                        episode_seed=int(episode_seed),
                        prediction_id=prediction_id,
                        environment_timestep=int(t),
                        initial_state_manifest_sha256=str(
                            protocol_log_metadata[
                                "initial_state_manifest_sha256"
                            ]
                        ),
                        protocol_identity=protocol_log_metadata,
                        warm_start_metadata=warm_start_metadata,
                        returned_action=np.asarray(actions),
                    )
                    action_delta_shadow_predictions.append(
                        action_delta_shadow_prediction
                    )
                    step_record.update(
                        {
                            "action_delta_gate_shadow_collected": True,
                            "action_delta_gate_shadow_schema_version": (
                                ACTION_DELTA_GATE_SHADOW_SCHEMA_VERSION
                            ),
                            "action_delta_gate_shadow_eligible_row_count": len(
                                action_delta_shadow_prediction["transitions"]
                            ),
                            "action_delta_gate_shadow_prediction_id": (
                                action_delta_shadow_prediction["prediction_id"]
                            ),
                            "action_delta_gate_shadow_influenced_control": False,
                        }
                    )
                else:
                    step_record["action_delta_gate_shadow_collected"] = False
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
                        "coda_ms_total",
                        "get_output_ms_total",
                        "run_one_iteration_ms_total",
                        "output_proj_ms_total",
                        "coda_time_ratio_total",
                    ):
                        step_record[timing_key] = debug.get(timing_key)

                prediction_step = prediction_id + 1
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
        if cfg.collect_action_delta_gate_shadow:
            action_delta_shadow_episode_error = f"{type(e).__name__}: {e}"

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
    if raw_shadow_writer is not None:
        raw_shadow_writer.flush()
    if action_delta_shadow_writer is not None:
        if action_delta_shadow_episode_error is not None:
            raise RuntimeError(
                "Action-Delta shadow Phase-A episode did not complete: "
                f"{action_delta_shadow_episode_error}"
            )
        action_delta_shadow_writer.add_episode(
            action_delta_shadow_predictions,
            success=success,
        )
        action_delta_shadow_writer.flush()

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
    raw_shadow_writer=None,
    action_delta_shadow_writer=None,
    source_commit=None,
):
    """Run evaluation for a single task."""
    task = task_suite.get_task(task_id)

    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)
    evaluation_protocol_phase = getattr(cfg, "evaluation_protocol_phase", "legacy")
    protocol_trials = None
    manifest_sha256 = None
    protocol_task_entry = None
    if evaluation_protocol_phase != "legacy":
        protocol_manifest, manifest_sha256 = load_protocol_manifest(
            cfg.initial_state_manifest_path, require_source_file_hashes=True
        )
        initial_state_file_path = (
            Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
        )
        protocol_trials = resolve_phase_trials(
            manifest=protocol_manifest,
            phase=evaluation_protocol_phase,
            task_id=task_id,
            initial_states=initial_states,
            base_seed=cfg.seed,
            initial_state_file_path=str(initial_state_file_path),
        )
        if len(protocol_trials) != cfg.num_trials_per_task:
            raise RuntimeError(
                f"Resolved {len(protocol_trials)} protocol trials, expected {cfg.num_trials_per_task}"
            )
        protocol_task_entry = protocol_manifest["tasks"][str(task_id)]
        log_message(
            f"Evaluation protocol {evaluation_protocol_phase}: manifest={manifest_sha256}, "
            f"state_ids={[trial.initial_state_id for trial in protocol_trials]}",
            log_file,
        )
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    task_episodes, task_successes = 0, 0
    all_iters_success, all_iters_failure, all_iters = [], [], []
    task_episode_stats = []
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)

        episode_protocol = None
        initial_state_id = episode_idx
        if protocol_trials is not None:
            trial = protocol_trials[episode_idx]
            initial_state_id = trial.initial_state_id
            initial_state = initial_states[initial_state_id]
            episode_protocol = {
                **trial.to_dict(),
                "paired_rng": True,
                "source_commit": source_commit,
                "manifest_sha256": manifest_sha256,
                "initial_states_sha256": protocol_task_entry["initial_states_sha256"],
                "initial_states_file": protocol_task_entry.get("initial_states_file"),
                "initial_states_file_sha256": protocol_task_entry.get("initial_states_file_sha256"),
            }
        elif cfg.initial_states_path == "DEFAULT":
            initial_state = initial_states[initial_state_id]
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
            episode_protocol=episode_protocol,
            profile_state=profile_state,
            timing_state=timing_state,
            raw_shadow_writer=raw_shadow_writer,
            action_delta_shadow_writer=action_delta_shadow_writer,
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
            "evaluation_protocol_phase": evaluation_protocol_phase,
            "paired_trial_id": None if episode_protocol is None else episode_protocol["paired_trial_id"],
            "initial_state_id": initial_state_id,
            "episode_seed": None if episode_protocol is None else episode_protocol["episode_seed"],
            "initial_state_manifest_sha256": manifest_sha256,
            "smoke_excluded_from_fitting": bool(
                episode_protocol is not None and episode_protocol["smoke_excluded_from_fitting"]
            ),
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

    scalar_policy_manifest = None
    scalar_policy_payload = None
    action_delta_gate_manifest = None
    action_delta_gate_payload = None

    if (
        canonicalize_recurrence_strategy(
            cfg.recurrence_strategy
        )
        == "scalar_policy"
    ):
        (
            scalar_policy_manifest,
            scalar_policy_payload,
        ) = load_scalar_policy_artifact(
            cfg.scalar_policy_artifact_path,
            expected_sha256=(
                cfg.scalar_policy_expected_sha256.lower()
            ),
        )

    if (
        cfg.use_action_delta_gate
        or cfg.collect_action_delta_gate_shadow
        or cfg.use_action_delta_nonconvergence_filter
        or cfg.use_action_delta_deferred_backfill_filter
    ):
        (
            action_delta_gate_manifest,
            action_delta_gate_payload,
        ) = load_action_delta_gate_artifact(
            cfg.action_delta_gate_artifact_path,
            expected_sha256=cfg.action_delta_gate_expected_sha256.lower(),
        )

    model, action_head, proprio_projector, processor = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    if scalar_policy_manifest is not None:
        log_message(
            "Loaded scalar OOF policy artifact: "
            f"sha256={scalar_policy_manifest['artifact_sha256']}, "
            f"mode={cfg.scalar_policy_execution_mode}",
            log_file,
        )
    if action_delta_gate_manifest is not None:
        log_message(
            (
                "Loaded frozen fold-4 Action-Delta predictor for "
                "development shadow collection: "
                if cfg.collect_action_delta_gate_shadow
                else "Loaded fold-4 Action-Delta Gate artifact: "
            )
            + f"sha256={action_delta_gate_manifest['artifact_sha256']}, "
            f"tasks={action_delta_gate_manifest['held_out_task_ids']}",
            log_file,
        )

    raw_shadow_writer = None
    action_delta_shadow_writer = None
    shared_source_commit = None
    shared_checkpoint_identity = None
    if cfg.evaluation_protocol_phase in {"screening", "final_holdout"}:
        shared_source_commit = current_source_commit()
    if cfg.collect_preconvergence_raw_shadow:
        shared_source_commit = current_source_commit()
        shared_checkpoint_identity = checkpoint_identity(
            Path(cfg.pretrained_checkpoint)
        )
        raw_shadow_writer = RawPreconvergenceShardWriter(
            Path(cfg.preconvergence_raw_shadow_dir),
            shard_size=cfg.preconvergence_raw_shadow_shard_size,
            maximum_shadow_depth=cfg.preconvergence_raw_shadow_max_depth,
            source_commit=shared_source_commit,
            checkpoint=shared_checkpoint_identity,
            run_identity={
                "run_id": run_id,
                "evaluation_protocol_phase": cfg.evaluation_protocol_phase,
                "task_suite_name": cfg.task_suite_name,
                "seed": int(cfg.seed),
            },
        )
        log_message(
            f"Raw preconvergence shadow collection enabled: {raw_shadow_writer.output_dir}",
            log_file,
        )
    if cfg.collect_action_delta_gate_shadow:
        if shared_source_commit is None:
            shared_source_commit = current_source_commit()
        if shared_checkpoint_identity is None:
            shared_checkpoint_identity = checkpoint_identity(
                Path(cfg.pretrained_checkpoint)
            )
        protocol_manifest, protocol_manifest_sha256 = load_protocol_manifest(
            cfg.initial_state_manifest_path,
            require_source_file_hashes=True,
        )
        expected_shadow_tasks = (
            (int(cfg.task_id),)
            if cfg.task_id is not None
            else ACTION_DELTA_GATE_SHADOW_DEVELOPMENT_TASK_IDS
        )
        action_delta_shadow_writer = ActionDeltaGateShadowWriter(
            Path(cfg.action_delta_gate_shadow_dir),
            shard_size=cfg.action_delta_gate_shadow_shard_size,
            expected_task_ids=expected_shadow_tasks,
            expected_trajectories_per_task=10,
            source_commit=shared_source_commit,
            artifact_identity={
                "path": str(Path(cfg.action_delta_gate_artifact_path).resolve()),
                "sha256": action_delta_gate_manifest["artifact_sha256"],
                "outer_fold": action_delta_gate_manifest["outer_fold"],
                "model_type": action_delta_gate_manifest["model_type"],
                "calibration_method": action_delta_gate_manifest[
                    "calibration_method"
                ],
                "threshold": action_delta_gate_manifest["threshold"],
                "held_out_task_ids": action_delta_gate_manifest[
                    "held_out_task_ids"
                ],
            },
            checkpoint_identity=shared_checkpoint_identity,
            initial_state_manifest_identity={
                "path": str(Path(cfg.initial_state_manifest_path).resolve()),
                "sha256": protocol_manifest_sha256,
                "protocol": protocol_manifest["protocol"],
                "partition": "calibration",
            },
            configuration={
                "task_suite_name": cfg.task_suite_name,
                "task_ids": list(expected_shadow_tasks),
                "evaluation_protocol_phase": cfg.evaluation_protocol_phase,
                "official_manifest_states_per_task": 10,
                "seed": int(cfg.seed),
                "reset_rng_each_episode": bool(cfg.reset_rng_each_episode),
                "warm_start": {
                    "enabled": bool(cfg.use_warm_start),
                    "source": cfg.warm_start_source,
                    "minimum_iteration": int(cfg.warm_start_min_iter),
                },
                "recurrence": {
                    "strategy": canonicalize_recurrence_strategy(
                        cfg.recurrence_strategy
                    ),
                    "adjacent_action_mse_threshold": float(
                        cfg.recurrence_kl_thresh
                    ),
                    "maximum_iteration": int(cfg.recurrence_max_iter),
                    "use_cached_final_output": bool(
                        cfg.use_cached_final_output
                    ),
                },
                "gate": {
                    "production_enabled": False,
                    "shadow_only": True,
                    "min_terminal_iteration": int(
                        cfg.action_delta_gate_min_terminal_iter
                    ),
                    "threshold": float(
                        action_delta_gate_manifest["threshold"]
                    ),
                    "artifact_sha256": action_delta_gate_manifest[
                        "artifact_sha256"
                    ],
                },
                "collection_mode": ACTION_DELTA_GATE_SHADOW_COLLECTION_MODE,
                "collector_schema_version": (
                    ACTION_DELTA_GATE_SHADOW_SCHEMA_VERSION
                ),
            },
            run_identity={
                "run_id": run_id,
                "evaluation_protocol_phase": cfg.evaluation_protocol_phase,
                "task_suite_name": cfg.task_suite_name,
                "seed": int(cfg.seed),
            },
        )
        log_message(
            "Deployment-matched Action-Delta shadow collection enabled: "
            f"{action_delta_shadow_writer.output_dir}; tasks={list(expected_shadow_tasks)}",
            log_file,
        )

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

    if scalar_policy_manifest is not None:
        full_results["scalar_policy"] = {
            "artifact_path": cfg.scalar_policy_artifact_path,
            "artifact_sha256": scalar_policy_manifest[
                "artifact_sha256"
            ],
            "policy_name": scalar_policy_manifest[
                "policy_name"
            ],
            "target_reference": scalar_policy_manifest[
                "target_reference"
            ],
            "execution_mode": (
                cfg.scalar_policy_execution_mode
            ),
            "task_level_oof": True,
        }
    if action_delta_gate_manifest is not None:
        full_results["action_delta_gate"] = {
            "artifact_path": cfg.action_delta_gate_artifact_path,
            "artifact_sha256": action_delta_gate_manifest["artifact_sha256"],
            "outer_fold": action_delta_gate_manifest["outer_fold"],
            "held_out_task_ids": action_delta_gate_manifest["held_out_task_ids"],
            "calibration_method": action_delta_gate_manifest["calibration_method"],
            "threshold": action_delta_gate_manifest["threshold"],
            "max_skip": cfg.action_delta_gate_max_skip,
            "min_terminal_iter": cfg.action_delta_gate_min_terminal_iter,
            "exact_coda_audit": cfg.action_delta_gate_exact_coda_audit,
            "return_mode": cfg.action_delta_gate_return_mode,
        }
        if cfg.collect_action_delta_gate_shadow:
            full_results["action_delta_gate"]["shadow_collection_only"] = True
            full_results["action_delta_gate"]["production_gate_enabled"] = False
        if cfg.use_action_delta_nonconvergence_filter:
            full_results["action_delta_nonconvergence_filter"] = {
                "development_only": True,
                "excluded_from_production_efficiency_claims": True,
                "production_convergence_gate_unchanged": True,
                "threshold": ACTION_DELTA_NONCONVERGENCE_THRESHOLD,
                "min_terminal_iter": ACTION_DELTA_NONCONVERGENCE_MIN_TERMINAL_ITER,
                "max_skip": 1,
                "allowed_task_ids": list(
                    (
                        (4,)
                        if cfg.evaluation_protocol_phase == "screening"
                        else (5,)
                        if cfg.evaluation_protocol_phase == "final_holdout"
                        else ACTION_DELTA_GATE_SHADOW_DEVELOPMENT_TASK_IDS
                    )
                ),
            }
        if cfg.use_action_delta_deferred_backfill_filter:
            full_results["action_delta_deferred_backfill_filter"] = {
                "development_only": True,
                "excluded_from_production_efficiency_claims": True,
                "adjacent_exact_stopping_semantics": True,
                "threshold": ACTION_DELTA_NONCONVERGENCE_THRESHOLD,
                "min_terminal_iter": cfg.action_delta_gate_min_terminal_iter,
                "scorer_backend": cfg.action_delta_deferred_scorer_backend,
                "compiled_configuration": (
                    {
                        "fullgraph": True,
                        "dynamic": False,
                        "mode": None,
                    }
                    if cfg.action_delta_deferred_scorer_backend
                    == "compile_default"
                    else None
                ),
                "compiled_numerical_equivalence": (
                    ACTION_DELTA_DEFERRED_COMPILED_NUMERICAL_EQUIVALENCE
                    if cfg.action_delta_deferred_scorer_backend
                    == "compile_default"
                    else "authoritative_eager"
                ),
                "allowed_task_ids": list(
                    ACTION_DELTA_GATE_SHADOW_DEVELOPMENT_TASK_IDS
                ),
            }

    if cfg.evaluation_protocol_phase != "legacy":
        protocol_manifest, manifest_sha256 = load_protocol_manifest(
            cfg.initial_state_manifest_path, require_source_file_hashes=True
        )
        full_results["evaluation_protocol"] = {
            "phase": cfg.evaluation_protocol_phase,
            "source_commit": shared_source_commit,
            "manifest_path": cfg.initial_state_manifest_path,
            "manifest_sha256": manifest_sha256,
            "task_suite_name": protocol_manifest["task_suite_name"],
            "num_trials_per_task": cfg.num_trials_per_task,
            "paired_rng": True,
            "prediction_parity_hash_schema": PARITY_HASH_SCHEMA,
            "action_head_workload_schema_version": ACTION_HEAD_WORKLOAD_SCHEMA_VERSION,
            "calibration_workload_predictions_per_episode": int(
                cfg.calibration_workload_predictions_per_episode
            ),
        }
        if cfg.use_action_delta_deferred_backfill_filter:
            full_results["evaluation_protocol"]["frozen_deferred_policy"] = {
                "warm_start_source": cfg.warm_start_source,
                "warm_start_min_iter": int(cfg.warm_start_min_iter),
                "recurrence_threshold": float(cfg.recurrence_kl_thresh),
                "action_delta_threshold": float(
                    ACTION_DELTA_NONCONVERGENCE_THRESHOLD
                ),
                "min_terminal_iter": int(
                    cfg.action_delta_gate_min_terminal_iter
                ),
                "scorer_backend": cfg.action_delta_deferred_scorer_backend,
                "artifact_sha256": (
                    cfg.action_delta_gate_expected_sha256.lower()
                ),
            }
        log_message(
            f"Frozen evaluation protocol: phase={cfg.evaluation_protocol_phase}, "
            f"manifest_sha256={manifest_sha256}",
            log_file,
        )

    if cfg.task_id is not None:
        task_ids = [cfg.task_id]
        log_message(f"Running only task {cfg.task_id}", log_file)
    elif cfg.evaluation_protocol_phase == "calibration":
        task_ids = ACTION_DELTA_GATE_SHADOW_DEVELOPMENT_TASK_IDS
        log_message(
            "Running frozen calibration development tasks only: "
            f"{list(task_ids)}",
            log_file,
        )
    elif (
        cfg.collect_action_delta_gate_shadow
        or cfg.use_action_delta_nonconvergence_filter
        or cfg.use_action_delta_deferred_backfill_filter
    ):
        task_ids = ACTION_DELTA_GATE_SHADOW_DEVELOPMENT_TASK_IDS
        log_message(
            "Running Action-Delta development tasks only: "
            f"{list(task_ids)}",
            log_file,
        )
    else:
        start_task = getattr(cfg, 'start_task_id', 0)
        task_ids = range(start_task, num_tasks)
        if start_task > 0:
            log_message(f"Starting from task {start_task}", log_file)

    for task_id in tqdm.tqdm(task_ids):
        if scalar_policy_payload is not None:
            action_head_device = next(
                action_head.parameters()
            ).device
            prepared_scalar_policy = (
                prepare_scalar_task_policy(
                    scalar_policy_payload,
                    int(task_id),
                    device=action_head_device,
                )
            )
            action_head.configure_scalar_task_policy(
                prepared_scalar_policy,
                cfg.scalar_policy_execution_mode,
            )
            log_message(
                "Bound scalar OOF policy: "
                f"task={task_id}, "
                f"fold={prepared_scalar_policy.outer_fold}, "
                f"threshold={prepared_scalar_policy.threshold:.9f}, "
                f"mode={cfg.scalar_policy_execution_mode}",
                log_file,
            )
        else:
            action_head.clear_scalar_task_policy()

        if action_delta_gate_payload is not None:
            action_head_device = next(action_head.parameters()).device
            prepared_action_delta_gate = (
                _prepare_action_delta_gate_for_evaluation(
                    cfg,
                    action_delta_gate_payload,
                    device=action_head_device,
                    task_id=int(task_id),
                )
            )
            prepared_deferred_scorer = action_head.configure_action_delta_gate(
                prepared_action_delta_gate,
                deferred_scorer_backend=(
                    cfg.action_delta_deferred_scorer_backend
                    if cfg.use_action_delta_deferred_backfill_filter
                    else "eager"
                ),
            )
            log_message(
                "Bound Action-Delta Gate: "
                f"task={task_id}, fold={prepared_action_delta_gate.outer_fold}, "
                f"threshold={prepared_action_delta_gate.threshold:.12f}, "
                f"nonconvergence_q={ACTION_DELTA_NONCONVERGENCE_THRESHOLD:.12f}, "
                f"deferred_scorer_backend={cfg.action_delta_deferred_scorer_backend}, "
                "compile_setup_ms="
                f"{prepared_deferred_scorer.compile_setup_ms if prepared_deferred_scorer is not None else 0.0:.3f}",
                log_file,
            )
        else:
            action_head.clear_action_delta_gate()

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
            raw_shadow_writer,
            action_delta_shadow_writer,
            source_commit=shared_source_commit,
        )
        task = task_suite.get_task(task_id)
        full_results["tasks"][task.name] = task_stats

        if cfg.json_log_file:
            with open(cfg.json_log_file, "w") as jf:
                json.dump(full_results, jf, indent=2)

    if raw_shadow_writer is not None:
        raw_manifest_path = raw_shadow_writer.finalize()
        full_results["preconvergence_raw_shadow_manifest"] = str(raw_manifest_path)
        log_message(f"Finalized raw shadow manifest: {raw_manifest_path}", log_file)
    if action_delta_shadow_writer is not None:
        action_delta_shadow_manifest_path = action_delta_shadow_writer.finalize()
        full_results["action_delta_gate_shadow_manifest"] = str(
            action_delta_shadow_manifest_path
        )
        action_delta_shadow_summary = json.loads(
            action_delta_shadow_manifest_path.read_text(encoding="utf-8")
        )["summary"]
        full_results["action_delta_gate_shadow_summary"] = (
            action_delta_shadow_summary
        )
        for shadow_task_id, task_summary in action_delta_shadow_summary[
            "by_task"
        ].items():
            log_message(
                "Action-Delta shadow Phase-A summary: "
                f"task={shadow_task_id}, trajectories={task_summary['trajectories']}, "
                f"eligible_rows={task_summary['eligible_rows']}, "
                f"predicted_triggers={task_summary['predicted_triggers']}, "
                f"exact_safe_triggers={task_summary['exact_safe_triggers']}, "
                f"false_safe_triggers={task_summary['false_safe_triggers']}, "
                "false_safe_rate="
                f"{task_summary['false_safe_rate_among_predicted_triggers']}",
                log_file,
            )
        aggregate_shadow = action_delta_shadow_summary["aggregate"]
        log_message(
            "Action-Delta shadow Phase-A aggregate: "
            f"trajectories={aggregate_shadow['trajectories']}, "
            f"eligible_rows={aggregate_shadow['eligible_rows']}, "
            f"predicted_triggers={aggregate_shadow['predicted_triggers']}, "
            f"exact_safe_triggers={aggregate_shadow['exact_safe_triggers']}, "
            f"false_safe_triggers={aggregate_shadow['false_safe_triggers']}, "
            "false_safe_rate="
            f"{aggregate_shadow['false_safe_rate_among_predicted_triggers']}",
            log_file,
        )
        log_message(
            "Finalized deployment-matched Action-Delta shadow manifest: "
            f"{action_delta_shadow_manifest_path}",
            log_file,
        )

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
