from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


class ActionHeadType(str, Enum):
    RECURRENT = "recurrent"


class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


@dataclass
class ModelConfig:
    vlm_path: str = "pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b"
    config_path: str = "pretrained_models/configs"
    use_minivlm: bool = True
    num_images_in_input: int = 2
    use_proprio: bool = True
    use_film: bool = False
    use_lora: bool = True
    lora_rank: int = 64
    lora_dropout: float = 0.0


@dataclass
class RecurrentConfig:
    hidden_dim: int = 896
    num_heads: int = 8
    prelude_vlm_layers: List[int] = field(default_factory=list)
    recurrent_vlm_layers: List[int] = field(default_factory=lambda: [6, 23])
    coda_vlm_layers: List[int] = field(default_factory=list)
    action_chunk_len: int = 8
    action_dim: int = 7
    mean_recurrence: int = 12
    backprop_depth: int = 8
    random_iterations: bool = True
    init_std: float = 0.632
    rms_norm_eps: float = 1e-6
    rope_base: float = 10000.0


@dataclass
class ActionHeadConfig:
    type: ActionHeadType = ActionHeadType.RECURRENT
    recurrent: RecurrentConfig = field(default_factory=RecurrentConfig)


@dataclass
class OptimizerConfig:
    learning_rate: float = 2e-4
    lr_warmup_steps: int = 0
    num_steps_before_decay: int = 200000
    use_muon: bool = False
    muon_lr: float = 2e-2
    muon_momentum: float = 0.95
    muon_weight_decay: float = 0.1
    muon_ns_steps: int = 5


@dataclass
class DataConfig:
    data_root_dir: str = "data/libero"
    dataset_name: str = "libero_spatial_no_noops"
    dataset_fraction: float = 1.0
    shuffle_buffer_size: int = 100_000
    image_aug: bool = True


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    action_head: ActionHeadConfig = field(default_factory=ActionHeadConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    data: DataConfig = field(default_factory=DataConfig)

    batch_size: int = 16
    grad_accumulation_steps: int = 2
    max_steps: int = 200000
    save_freq: int = 5000
    save_latest_only: bool = False
    merge_lora: bool = True
    use_val_set: bool = False
    val_freq: int = 10_000
    val_time_limit: int = 180

    output_dir: str = "outputs"
    run_name: Optional[str] = None

    use_wandb: bool = True
    wandb_project: str = "rd-vla"
    wandb_entity: Optional[str] = None
    wandb_log_freq: int = 10

    resume: bool = False
    resume_path: Optional[str] = None
    resume_step: Optional[int] = None

    seed: int = 42


@dataclass
class EvalTaskConfig:
    suite: TaskSuite = TaskSuite.LIBERO_SPATIAL
    num_tasks: Optional[int] = None
    num_rollouts: int = 50
    task_ids: Optional[List[int]] = None


@dataclass
class EvalConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    action_head: ActionHeadConfig = field(default_factory=ActionHeadConfig)

    checkpoint_path: str = ""

    tasks: EvalTaskConfig = field(default_factory=EvalTaskConfig)

    video_dir: str = "rollouts"
    log_dir: str = "eval_logs"

    num_open_loop_steps: int = 8
    center_crop: bool = True
    env_resolution: int = 256

    use_wandb: bool = False
    wandb_project: str = "rd-vla-eval"

    seed: int = 7
