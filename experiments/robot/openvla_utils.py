import filecmp
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import json_numpy
import numpy as np
import requests
import tensorflow as tf
import torch
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

json_numpy.patch()

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import ActionHeadRecurrent, RecurrentConfigInternal
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.projectors import ProprioProjector
from prismatic.utils.rdvla_profiler import rdvla_range
from prismatic.vla.constants import ACTION_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType

DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
OPENVLA_IMAGE_SIZE = 224

np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})


def model_is_on_hf_hub(model_path: str) -> bool:
    try:
        HfApi().model_info(model_path)
        return True
    except Exception:
        return False


def update_auto_map(pretrained_checkpoint: str) -> None:
    if not os.path.isdir(pretrained_checkpoint):
        return

    config_path = os.path.join(pretrained_checkpoint, "config.json")
    if not os.path.exists(config_path):
        print(f"Warning: No config.json found at {config_path}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(pretrained_checkpoint, f"config.json.back.{timestamp}")
    shutil.copy2(config_path, backup_path)

    with open(config_path, "r") as f:
        config = json.load(f)

    config["auto_map"] = {
        "AutoConfig": "configuration_prismatic.OpenVLAConfig",
        "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction",
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def check_identical_files(path1: Union[str, Path], path2: Union[str, Path]) -> bool:
    path1, path2 = Path(path1), Path(path2)
    if path1.stat().st_size != path2.stat().st_size:
        return False
    return filecmp.cmp(path1, path2, shallow=False)


def _handle_file_sync(curr_filepath: str, checkpoint_filepath: str, file_type: str) -> None:
    if os.path.exists(checkpoint_filepath):
        match = check_identical_files(curr_filepath, checkpoint_filepath)
        if not match:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{checkpoint_filepath}.back.{timestamp}"
            shutil.copy2(checkpoint_filepath, backup_path)
            shutil.copy2(curr_filepath, checkpoint_filepath)
    else:
        shutil.copy2(curr_filepath, checkpoint_filepath)


def check_model_logic_mismatch(pretrained_checkpoint: str) -> None:
    if not os.path.isdir(pretrained_checkpoint):
        return

    curr_files = {"modeling_prismatic.py": None, "configuration_prismatic.py": None}
    for root, _, files in os.walk("./prismatic/"):
        for filename in curr_files.keys():
            if filename in files and curr_files[filename] is None:
                curr_files[filename] = os.path.join(root, filename)

    for filename, curr_filepath in curr_files.items():
        if curr_filepath is None:
            continue
        checkpoint_filepath = os.path.join(pretrained_checkpoint, filename)
        _handle_file_sync(curr_filepath, checkpoint_filepath, filename)


def prepare_checkpoint_source_config(cfg: Any) -> bool:
    """Synchronize local checkpoint source/config files when explicitly enabled."""

    if not getattr(cfg, "sync_checkpoint_source_config", True):
        return False

    update_auto_map(cfg.pretrained_checkpoint)
    check_model_logic_mismatch(cfg.pretrained_checkpoint)
    return True


def find_checkpoint_file(pretrained_checkpoint: str, file_pattern: str) -> str:
    assert os.path.isdir(pretrained_checkpoint)
    checkpoint_files = []
    for filename in os.listdir(pretrained_checkpoint):
        if file_pattern in filename and "checkpoint" in filename and filename.endswith('.pt'):
            checkpoint_files.append(os.path.join(pretrained_checkpoint, filename))
    assert len(checkpoint_files) == 1, (
        f"Expected exactly 1 {file_pattern} checkpoint but found {len(checkpoint_files)} in: {pretrained_checkpoint}"
    )
    return checkpoint_files[0]


def load_component_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    state_dict = torch.load(checkpoint_path, weights_only=True)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def get_vla(cfg: Any) -> torch.nn.Module:
    print("Instantiating pretrained VLA policy...")

    if not model_is_on_hf_hub(cfg.pretrained_checkpoint):
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

        prepare_checkpoint_source_config(cfg)

    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        torch_dtype=torch.bfloat16,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
        low_cpu_mem_usage=False,
        trust_remote_code=False,
    )

    if cfg.use_film:
        vla = _apply_film_to_vla(vla, cfg)

    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
    vla.eval()

    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        vla = vla.to(DEVICE)

    _load_dataset_stats(vla, cfg.pretrained_checkpoint)
    return vla


def _apply_film_to_vla(vla: torch.nn.Module, cfg: Any) -> torch.nn.Module:
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=32, lora_alpha=16, lora_dropout=0.0,
        target_modules="all-linear", init_lora_weights="gaussian",
    )
    vla = get_peft_model(vla, lora_config)

    new_vision_backbone = FiLMedPrismaticVisionBackbone(
        vision_backbone=vla.vision_backbone, llm_dim=vla.llm_dim,
    )
    vla.model.vision_backbone = new_vision_backbone

    checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, "vision_backbone")
    state_dict = torch.load(checkpoint_path, weights_only=True)
    vla.model.vision_backbone.load_state_dict(state_dict)

    vla = vla.model
    vla.vision_backbone = vla.vision_backbone.to(torch.bfloat16)
    return vla


def _load_dataset_stats(vla: torch.nn.Module, checkpoint_path: str) -> None:
    if model_is_on_hf_hub(checkpoint_path):
        dataset_statistics_path = hf_hub_download(
            repo_id=checkpoint_path, filename="dataset_statistics.json",
        )
    else:
        dataset_statistics_path = os.path.join(checkpoint_path, "dataset_statistics.json")

    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            vla.norm_stats = json.load(f)
    else:
        print("WARNING: No dataset_statistics.json found for checkpoint.")


def get_processor(cfg: Any) -> AutoProcessor:
    return AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=False)


def get_proprio_projector(cfg: Any, llm_dim: int, proprio_dim: int) -> ProprioProjector:
    proprio_projector = ProprioProjector(llm_dim=llm_dim, proprio_dim=proprio_dim)
    proprio_projector = proprio_projector.to(torch.bfloat16).to(DEVICE)
    proprio_projector.eval()

    if model_is_on_hf_hub(cfg.pretrained_checkpoint):
        proprio_projector_path = hf_hub_download(
            repo_id=cfg.pretrained_checkpoint, filename="proprio_projector--checkpoint.pt"
        )
        state_dict = load_component_state_dict(proprio_projector_path)
    else:
        checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, "proprio_projector")
        state_dict = load_component_state_dict(checkpoint_path)

    proprio_projector.load_state_dict(state_dict)
    return proprio_projector


def get_action_head(cfg: Any, llm_dim: int):
    config_path = None
    if not model_is_on_hf_hub(cfg.pretrained_checkpoint):
        checkpoint_dir = Path(cfg.pretrained_checkpoint)
        config_files = list(checkpoint_dir.glob("action_head_config--*.json"))
        if config_files:
            config_path = config_files[0]

    if config_path and config_path.exists():
        with open(config_path, 'r') as f:
            saved_cfg = json.load(f)

        head_type = saved_cfg.pop('_type', None)
        print(f"Auto-detected action head type: {head_type}")

        for key in ['prelude_vlm_layers', 'recurrent_vlm_layers', 'coda_vlm_layers']:
            if key in saved_cfg:
                saved_cfg[key] = tuple(saved_cfg[key])
        head_cfg = RecurrentConfigInternal(**saved_cfg)
        action_head = ActionHeadRecurrent(hidden_dim=llm_dim, cfg=head_cfg)
    else:
        raise ValueError(
            f"No action_head_config JSON found in checkpoint: {cfg.pretrained_checkpoint}"
        )

    action_head = action_head.to(torch.bfloat16).to(DEVICE)
    action_head.eval()

    if model_is_on_hf_hub(cfg.pretrained_checkpoint):
        action_head_path = hf_hub_download(
            repo_id=cfg.pretrained_checkpoint, filename="action_head--checkpoint.pt"
        )
        state_dict = load_component_state_dict(action_head_path)
    else:
        checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, "action_head")
        state_dict = load_component_state_dict(checkpoint_path)

    action_head.load_state_dict(state_dict)
    return action_head


def resize_image_for_policy(img: np.ndarray, resize_size: Union[int, Tuple[int, int]]) -> np.ndarray:
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = tf.image.encode_jpeg(img)
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    return img.numpy()


def crop_and_resize(image: tf.Tensor, crop_scale: float, batch_size: int) -> tf.Tensor:
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack([height_offsets, width_offsets, height_offsets + new_heights, width_offsets + new_widths], axis=1)
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE))

    if expanded_dims:
        image = image[0]
    return image


def center_crop_image(image: Union[np.ndarray, Image.Image]) -> Image.Image:
    if not isinstance(image, tf.Tensor):
        image = tf.convert_to_tensor(np.array(image))
    orig_dtype = image.dtype
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = crop_and_resize(image, 0.9, 1)
    image = tf.clip_by_value(image, 0, 1)
    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)
    return Image.fromarray(image.numpy()).convert("RGB")


def check_image_format(image: Any) -> None:
    assert (isinstance(image, np.ndarray) and len(image.shape) == 3
            and image.shape[-1] == 3 and image.dtype == np.uint8), \
        "Image must be numpy array with shape (H, W, 3) and dtype np.uint8"


def normalize_proprio(proprio: np.ndarray, norm_stats: Dict[str, Any]) -> np.ndarray:
    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["min"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["max"]), np.array(norm_stats["min"])
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["q01"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["q99"]), np.array(norm_stats["q01"])
    else:
        raise ValueError("Unsupported normalization type")

    return np.clip(
        np.where(mask, 2 * (proprio - proprio_low) / (proprio_high - proprio_low + 1e-8) - 1, proprio),
        a_min=-1.0, a_max=1.0,
    )


def prepare_images_for_vla(images: List[np.ndarray], cfg: Any) -> List[Image.Image]:
    processed_images = []
    for image in images:
        check_image_format(image)
        if image.shape != (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE, 3):
            image = resize_image_for_policy(image, OPENVLA_IMAGE_SIZE)
        pil_image = Image.fromarray(image).convert("RGB")
        if cfg.center_crop:
            pil_image = center_crop_image(pil_image)
        processed_images.append(pil_image)
    return processed_images


def get_vla_action(
    cfg: Any, vla: torch.nn.Module, processor: Any, obs: Dict[str, Any], task_label: str,
    action_head: Optional[torch.nn.Module] = None,
    proprio_projector: Optional[torch.nn.Module] = None,
    use_film: bool = False, use_minivlm: bool = False,
    warm_start_state=None,
    capture_action_head_workload: bool = False,
) -> List[np.ndarray]:
    effective_warm_start_state = (
        warm_start_state if getattr(cfg, "use_warm_start", False) else None
    )
    with rdvla_range("RDVLA/get_vla_action_total"):
        with torch.inference_mode():
            with rdvla_range("RDVLA/get_vla_action/collect_images"):
                all_images = [obs["full_image"]]
                if cfg.num_images_in_input > 1:
                    all_images.extend([obs[k] for k in obs.keys() if "wrist" in k])

            with rdvla_range("RDVLA/get_vla_action/prepare_images"):
                all_images = prepare_images_for_vla(all_images, cfg)
                primary_image = all_images.pop(0)

            with rdvla_range("RDVLA/get_vla_action/build_prompt"):
                if not use_minivlm:
                    prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
                else:
                    prompt = f'<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat action should the robot take to {task_label.lower()}?<|im_end|>\n<|im_start|>assistant\n'

            with rdvla_range("RDVLA/get_vla_action/processor_primary"):
                inputs = processor(prompt, primary_image).to(DEVICE, dtype=torch.bfloat16)

            if all_images:
                with rdvla_range("RDVLA/get_vla_action/processor_wrist"):
                    all_wrist_inputs = [
                        processor(prompt, image_wrist).to(DEVICE, dtype=torch.bfloat16) for image_wrist in all_images
                    ]
                    primary_pixel_values = inputs["pixel_values"]
                    all_wrist_pixel_values = [wrist_inputs["pixel_values"] for wrist_inputs in all_wrist_inputs]
                    inputs["pixel_values"] = torch.cat([primary_pixel_values] + all_wrist_pixel_values, dim=1)

            proprio = None
            if cfg.use_proprio:
                with rdvla_range("RDVLA/get_vla_action/normalize_proprio"):
                    proprio = obs["state"]
                    proprio_norm_stats = vla.norm_stats[cfg.unnorm_key]["proprio"]
                    obs["state"] = normalize_proprio(proprio, proprio_norm_stats)
                    proprio = obs["state"]

            # 원본 recurrence 호출/반환 코드
            # actual_iters = None
            # final_kl = None
            # if action_head is None:
            #     action, _, actual_iters, final_kl = vla.predict_action(**inputs, unnorm_key=cfg.unnorm_key, do_sample=False)
            # else:
            #     convergence_strategy = getattr(cfg, 'recurrence_strategy', 'fixed')
            #     if convergence_strategy == 'fixed':
            #         convergence_strategy = None
            #     action, _, actual_iters, final_kl = vla.predict_action(
            #         **inputs, unnorm_key=cfg.unnorm_key, do_sample=False,
            #         proprio=proprio, proprio_projector=proprio_projector,
            #         action_head=action_head, use_film=use_film,
            #         num_iter=getattr(cfg, 'recurrent_num_iter', None),
            #         convergence_strategy=convergence_strategy,
            #         kl_thresh=getattr(cfg, 'recurrence_kl_thresh', 0.001),
            #         cos_thresh=getattr(cfg, 'recurrence_cos_thresh', 0.999),
            #         max_iter=getattr(cfg, 'recurrence_max_iter', 32),
            #     )

            # recurrence debug metric 전달을 위해 수정한 코드
            actual_iters = None
            final_kl = None
            recurrence_debug = None
            action_head_inference_metadata = {}
            with rdvla_range("RDVLA/get_vla_action/vla_predict_action"):
                if action_head is None:
                    action, _, actual_iters, final_kl = vla.predict_action(
                        **inputs, unnorm_key=cfg.unnorm_key, do_sample=False
                    )
                else:
                    convergence_strategy = getattr(cfg, "recurrence_strategy", "fixed")
                    if convergence_strategy == "fixed":
                        convergence_strategy = None
                    action, _, actual_iters, final_kl = vla.predict_action(
                        **inputs, unnorm_key=cfg.unnorm_key, do_sample=False,
                        proprio=proprio, proprio_projector=proprio_projector,
                        action_head=action_head, use_film=use_film,
                        num_iter=getattr(cfg, "recurrent_num_iter", None),
                        convergence_strategy=convergence_strategy,
                        kl_thresh=getattr(cfg, "recurrence_kl_thresh", 0.001),
                        cos_thresh=getattr(cfg, "recurrence_cos_thresh", 0.999),
                        max_iter=getattr(cfg, "recurrence_max_iter", 32),
                        warm_start_state=effective_warm_start_state,
                        enable_warm_start=bool(getattr(cfg, "use_warm_start", False)),
                        warm_start_source=getattr(cfg, "warm_start_source", "s1"),
                        warm_start_min_iter=getattr(cfg, "warm_start_min_iter", 2),
                        validate_warm_start_finite=bool(
                            getattr(cfg, "validate_warm_start_finite", False)
                        ),
                        profile_coda_cost=getattr(cfg, "profile_coda_cost", False),
                        use_cached_final_output=getattr(cfg, "use_cached_final_output", False),
                        use_latent_precheck=getattr(cfg, "use_latent_precheck", False),
                        latent_precheck_mode=getattr(cfg, "latent_precheck_mode", "legacy"),
                        latent_precheck_trace_level=getattr(cfg, "latent_precheck_trace_level", "off"),
                        latent_precheck_thresh=getattr(cfg, "latent_precheck_thresh", 0.12),
                        latent_precheck_min_iter=getattr(cfg, "latent_precheck_min_iter", 2),
                        latent_precheck_force_interval=getattr(cfg, "latent_precheck_force_interval", 0),
                        latent_precheck_warm_thresh=getattr(cfg, "latent_precheck_warm_thresh", None),
                        latent_precheck_max_skip_iters=getattr(cfg, "latent_precheck_max_skip_iters", 0),
                        latent_precheck_confirmation_mode=getattr(cfg, "latent_precheck_confirmation_mode", "next_iter"),
                        nonfinite_policy=getattr(cfg, "nonfinite_policy", "legacy"),
                        shadow_full_depth=getattr(cfg, "shadow_full_depth", False),
                        collect_preconvergence_raw_shadow=getattr(
                            cfg, "collect_preconvergence_raw_shadow", False
                        ),
                        preconvergence_raw_shadow_max_depth=getattr(
                            cfg, "preconvergence_raw_shadow_max_depth", 32
                        ),
                        capture_action_head_workload=capture_action_head_workload,
                        latent_only_metric=getattr(cfg, "latent_only_metric", "raw_mse"),
                        latent_only_cold_threshold=getattr(
                            cfg, "latent_only_cold_threshold", 0.0
                        ),
                        latent_only_warm_threshold=getattr(
                            cfg, "latent_only_warm_threshold", 0.0
                        ),
                        latent_only_min_iter=getattr(cfg, "latent_only_min_iter", 2),
                        latent_only_eps=getattr(cfg, "latent_only_eps", 1e-8),
                        use_action_delta_gate=bool(
                            getattr(cfg, "use_action_delta_gate", False)
                        ),
                        action_delta_gate_max_skip=getattr(
                            cfg, "action_delta_gate_max_skip", 1
                        ),
                        action_delta_gate_min_terminal_iter=getattr(
                            cfg, "action_delta_gate_min_terminal_iter", 2
                        ),
                        action_delta_gate_exact_coda_audit=bool(
                            getattr(
                                cfg,
                                "action_delta_gate_exact_coda_audit",
                                False,
                            )
                        ),
                        action_delta_gate_return_mode=getattr(
                            cfg,
                            "action_delta_gate_return_mode",
                            "anchor",
                        ),
                        collect_action_delta_gate_shadow=bool(
                            getattr(
                                cfg,
                                "collect_action_delta_gate_shadow",
                                False,
                            )
                        ),
                    )
                    recurrence_debug = getattr(getattr(action_head, "model", None), "last_recurrence_debug", None)
                    action_head_inference_metadata = (
                        getattr(getattr(action_head, "model", None), "last_inference_metadata", None) or {}
                    )

        with rdvla_range("RDVLA/get_vla_action/action_postprocess"):
            actions = [action[i] for i in range(min(len(action), cfg.num_open_loop_steps))]
        warm_start_metadata = action_head_inference_metadata.get("warm_start", {})
        inference_metadata = {
            "recurrence_debug": recurrence_debug,
            "next_warm_start_state": action_head_inference_metadata.get("next_warm_start_state"),
            "action_head_workload": action_head_inference_metadata.get("action_head_workload"),
            "preconvergence_raw_shadow": action_head_inference_metadata.get(
                "preconvergence_raw_shadow"
            ),
            "action_delta_gate_shadow": action_head_inference_metadata.get(
                "action_delta_gate_shadow"
            ),
            "warm_start": {
                "enabled": bool(getattr(cfg, "use_warm_start", False)),
                "source": warm_start_metadata.get("source"),
                "source_index": warm_start_metadata.get("source_index"),
                "source_iteration": warm_start_metadata.get("source_iteration"),
                "source_K": warm_start_metadata.get("source_K"),
                "candidate_state_count": warm_start_metadata.get("candidate_state_count"),
                "state_provided": bool(
                    warm_start_metadata.get("state_provided", effective_warm_start_state is not None)
                ),
                "state_used": bool(warm_start_metadata.get("state_used", False)),
                "initial_state_origin": warm_start_metadata.get("initial_state_origin", "random"),
                "reset": bool(warm_start_metadata.get("reset", False)),
                "reset_reason": warm_start_metadata.get("reset_reason"),
                "first_attempt_state_used": warm_start_metadata.get("first_attempt_state_used"),
                "first_attempt_initial_state_origin": warm_start_metadata.get(
                    "first_attempt_initial_state_origin"
                ),
            },
        }
        return actions, actual_iters, final_kl, inference_metadata


def get_action_from_server(
    observation: Dict[str, Any], server_endpoint: str = "http://0.0.0.0:8777/act"
) -> Dict[str, Any]:
    return requests.post(server_endpoint, json=observation).json()
