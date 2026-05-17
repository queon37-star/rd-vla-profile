import yaml
from pathlib import Path
from dataclasses import fields, is_dataclass
from typing import TypeVar, Type, Any, Dict, get_type_hints, get_origin
from enum import Enum
import argparse

from configs.base import TrainConfig, EvalConfig, ActionHeadType, TaskSuite


T = TypeVar('T')


def deep_update(base: Dict, updates: Dict) -> Dict:
    """Recursively update nested dict."""
    result = base.copy()
    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _convert_value(value: Any, target_type: type) -> Any:
    """Convert value to target type, handling string representations of primitives."""
    if value is None:
        return None
    if isinstance(value, target_type):
        return value
    if target_type == float and isinstance(value, (int, str)):
        return float(value)
    if target_type == int and isinstance(value, (float, str)):
        return int(value)
    if target_type == bool and isinstance(value, str):
        return value.lower() in ('true', '1', 'yes')
    if target_type == str:
        return str(value)
    return value


def dict_to_dataclass(cls: Type[T], data: Dict[str, Any]) -> T:
    """Convert nested dict to nested dataclass."""
    if not is_dataclass(cls):
        if isinstance(data, cls):
            return data
        if issubclass(cls, Enum) and isinstance(data, str):
            return cls(data)
        return data

    hints = get_type_hints(cls)
    kwargs = {}

    for f in fields(cls):
        if f.name not in data:
            continue

        field_type = hints[f.name]
        value = data[f.name]

        origin = get_origin(field_type)
        if origin is not None:
            if value is None:
                kwargs[f.name] = None
            else:
                kwargs[f.name] = value
        elif is_dataclass(field_type):
            if isinstance(value, dict):
                kwargs[f.name] = dict_to_dataclass(field_type, value)
            else:
                kwargs[f.name] = value
        elif isinstance(field_type, type) and issubclass(field_type, Enum):
            kwargs[f.name] = field_type(value) if isinstance(value, str) else value
        elif isinstance(field_type, type) and field_type in (float, int, bool, str):
            kwargs[f.name] = _convert_value(value, field_type)
        else:
            kwargs[f.name] = value

    return cls(**kwargs)


def dataclass_to_dict(obj) -> Dict:
    """Convert dataclass to dict, handling enums."""
    if is_dataclass(obj):
        result = {}
        for f in fields(obj):
            value = getattr(obj, f.name)
            result[f.name] = dataclass_to_dict(value)
        return result
    elif isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, list):
        return [dataclass_to_dict(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return obj


def load_yaml_config(path: str) -> Dict:
    """Load YAML config file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def parse_cli_overrides(args: list) -> Dict:
    """Parse CLI args like --model.lora_rank=32 into nested dict."""
    overrides = {}
    for arg in args:
        if not arg.startswith('--'):
            continue
        if '=' in arg:
            key, value = arg[2:].split('=', 1)
        else:
            continue

        try:
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif '.' in value and value.replace('.', '').replace('-', '').isdigit():
                value = float(value)
            elif value.lstrip('-').isdigit():
                value = int(value)
        except:
            pass

        parts = key.split('.')
        current = overrides
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    return overrides


def load_config(config_class: Type[T], yaml_path: str = None, cli_args: list = None) -> T:
    """Load config from YAML with CLI overrides."""
    base = dataclass_to_dict(config_class())

    if yaml_path:
        yaml_config = load_yaml_config(yaml_path)
        base = deep_update(base, yaml_config)

    if cli_args:
        overrides = parse_cli_overrides(cli_args)
        base = deep_update(base, overrides)

    return dict_to_dataclass(config_class, base)


def save_config(config, path: str):
    """Save config to YAML file."""
    data = dataclass_to_dict(config)
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def create_arg_parser():
    """Create argument parser for run.py."""
    parser = argparse.ArgumentParser(description='RD-VLA Training and Evaluation')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], required=True)
    return parser


def get_legacy_config(cfg, mode: str):
    """Convert new config format to legacy format for backward compatibility."""
    if mode == 'train':
        return _convert_train_config(cfg)
    else:
        return _convert_eval_config(cfg)


def _convert_train_config(cfg: TrainConfig):
    """Convert TrainConfig to legacy FinetuneConfig format."""
    from dataclasses import dataclass as dc

    @dc
    class LegacyConfig:
        pass

    legacy = LegacyConfig()

    legacy.config_file_path = cfg.model.config_path
    legacy.vlm_path = cfg.model.vlm_path
    legacy.use_minivlm = cfg.model.use_minivlm
    legacy.resum_vla_path = cfg.resume_path or cfg.model.vlm_path

    legacy.data_root_dir = Path(cfg.data.data_root_dir)
    legacy.dataset_name = cfg.data.dataset_name
    legacy.dataset_fraction = cfg.data.dataset_fraction
    legacy.run_root_dir = Path(cfg.output_dir)
    legacy.shuffle_buffer_size = cfg.data.shuffle_buffer_size

    legacy.use_film = cfg.model.use_film
    legacy.num_images_in_input = cfg.model.num_images_in_input
    legacy.use_proprio = cfg.model.use_proprio
    legacy.phase1_path = "None"

    legacy.batch_size = cfg.batch_size
    legacy.learning_rate = cfg.optimizer.learning_rate
    legacy.lr_warmup_steps = cfg.optimizer.lr_warmup_steps
    legacy.num_steps_before_decay = cfg.optimizer.num_steps_before_decay
    legacy.grad_accumulation_steps = cfg.grad_accumulation_steps
    legacy.max_steps = cfg.max_steps
    legacy.use_val_set = cfg.use_val_set
    legacy.val_freq = cfg.val_freq
    legacy.val_time_limit = cfg.val_time_limit
    legacy.save_freq = cfg.save_freq
    legacy.save_latest_checkpoint_only = cfg.save_latest_only
    legacy.resume = cfg.resume
    legacy.resume_step = cfg.resume_step
    legacy.image_aug = cfg.data.image_aug
    legacy.diffusion_sample_freq = 50

    legacy.use_lora = cfg.model.use_lora
    legacy.lora_rank = cfg.model.lora_rank
    legacy.lora_dropout = cfg.model.lora_dropout
    legacy.merge_lora_during_training = cfg.merge_lora
    legacy.use_fz = False

    legacy.use_wandb = cfg.use_wandb
    legacy.wandb_entity = cfg.wandb_entity
    legacy.wandb_project = cfg.wandb_project
    legacy.run_id_note = cfg.run_name
    legacy.run_id_override = None
    legacy.wandb_log_freq = cfg.wandb_log_freq

    legacy.phase = "Training"

    legacy.use_recurrent = False
    legacy.recurrent_cfg = None

    legacy.action_head = cfg.action_head

    legacy.use_muon = cfg.optimizer.use_muon
    legacy.muon_lr = cfg.optimizer.muon_lr
    legacy.muon_momentum = cfg.optimizer.muon_momentum
    legacy.muon_weight_decay = cfg.optimizer.muon_weight_decay
    legacy.muon_ns_steps = cfg.optimizer.muon_ns_steps
    legacy.muon_matched_adamw_rms = 0.2

    return legacy


def _convert_eval_config(cfg: EvalConfig):
    """Convert EvalConfig to legacy GenerateConfig format."""
    from dataclasses import dataclass as dc

    @dc
    class LegacyConfig:
        pass

    legacy = LegacyConfig()

    legacy.model_family = "openvla"
    legacy.pretrained_checkpoint = cfg.checkpoint_path
    legacy.use_minivlm = cfg.model.use_minivlm
    legacy.use_film = cfg.model.use_film
    legacy.num_images_in_input = cfg.model.num_images_in_input
    legacy.use_proprio = cfg.model.use_proprio

    legacy.center_crop = cfg.center_crop
    legacy.num_open_loop_steps = cfg.num_open_loop_steps
    legacy.unnorm_key = ""

    legacy.load_in_8bit = False
    legacy.load_in_4bit = False

    legacy.task_suite_name = cfg.tasks.suite.value
    legacy.num_steps_wait = 10
    legacy.num_trials_per_task = cfg.tasks.num_rollouts
    legacy.initial_states_path = "DEFAULT"
    legacy.env_img_res = cfg.env_resolution

    legacy.run_id_note = None
    legacy.local_log_dir = cfg.log_dir

    legacy.use_wandb = cfg.use_wandb
    legacy.wandb_entity = cfg.wandb_entity
    legacy.wandb_project = cfg.wandb_project

    legacy.seed = cfg.seed

    legacy.save_version = "rd-vla"
    legacy.phase = "Inference"

    legacy.use_recurrent = cfg.action_head.type == ActionHeadType.RECURRENT
    legacy.recurrent_num_iter = cfg.action_head.recurrent.mean_recurrence

    num_patches_per_image = 256
    legacy.num_task_tokens = num_patches_per_image * cfg.model.num_images_in_input

    return legacy
