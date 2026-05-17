import os
import sys
import time

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from collections import deque
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from accelerate import PartialState
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from prismatic.training.muon import Muon
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
import wandb

from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map
)
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import ActionHeadRecurrent, RecurrentConfigInternal
from configs.base import ActionHeadType
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.projectors import ProprioProjector
from prismatic.training.train_utils import (
    get_current_action_mask,
    get_next_actions_mask
)
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
    NUM_TOKENS
)
from prismatic.vla.datasets import RLDSDataset, RLDSBatchTransform
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from prismatic.models import load, load_vla


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def remove_ddp_in_checkpoint(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k[:7] == "module.":
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def get_run_id(cfg):
    if hasattr(cfg, 'run_id_override') and cfg.run_id_override is not None:
        return cfg.run_id_override
    elif cfg.resume:
        run_id = cfg.config_file_path.split("/")[-1]
        if "chkpt" in run_id.split("--")[-1]:
            run_id = "--".join(run_id.split("--")[:-1])
        return run_id
    else:
        run_id = f"{cfg.config_file_path.split('/')[-1]}+{cfg.dataset_name}+b{cfg.batch_size * cfg.grad_accumulation_steps}+lr-{cfg.learning_rate}"
        if cfg.use_lora:
            run_id += f"+lora-r{cfg.lora_rank}"
        if cfg.image_aug:
            run_id += "--image_aug"
        if hasattr(cfg, 'run_id_note') and cfg.run_id_note:
            run_id += f"--{cfg.run_id_note}"
        return run_id


def load_checkpoint(module_name, path, step, device="cpu"):
    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)


def wrap_ddp(module, device_id, find_unused=False):
    return DDP(module, device_ids=[device_id], find_unused_parameters=find_unused, gradient_as_bucket_view=True)


def count_parameters(module, name):
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"# trainable params in {name}: {num_params}")


def init_module(module_class, module_name, cfg, device_id, module_args, to_bf16=False, find_unused_params=False):
    module = module_class(**module_args)
    count_parameters(module, module_name)

    if cfg.resume:
        state_dict = load_checkpoint(module_name, cfg.resum_vla_path, cfg.resume_step)
        module.load_state_dict(state_dict)
        print('Loaded checkpoint!')

    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device_id)

    return wrap_ddp(module, device_id, find_unused_params)


def run_forward_pass(vla, action_head, proprio_projector, batch, action_tokenizer, device_id,
                     use_proprio, use_film, num_patches, cfg=None):
    metrics = {}
    ground_truth_actions = batch["actions"].to(device_id).to(torch.bfloat16)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output: CausalLMOutputWithPast = vla(
            input_ids=batch["input_ids"].to(device_id),
            attention_mask=batch["attention_mask"].to(device_id),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
            labels=batch["labels"],
            output_hidden_states=True,
            proprio=batch["proprio"] if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
            use_film=use_film,
        )

    ground_truth_token_ids = batch["labels"][:, 1:].to(device_id)
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)

    multi_layer_hidden_states = []
    for item in output.hidden_states[0:]:
        text_hidden_states = item[:, num_patches:-1]
        batch_size = batch["input_ids"].shape[0]
        actions_hidden_states = text_hidden_states[current_action_mask | next_actions_mask].reshape(
            batch_size, 1, NUM_TOKENS, -1).to(torch.bfloat16)
        task_latten_states = item[:, :num_patches].reshape(batch_size, 1, num_patches, -1)
        all_hidden_states = torch.cat((task_latten_states, actions_hidden_states), 2)
        multi_layer_hidden_states.append(all_hidden_states)
    multi_layer_hidden_states = torch.cat(multi_layer_hidden_states, dim=1)

    result = action_head.module.predict_action(
        multi_layer_hidden_states,
        proprio=batch["proprio"] if use_proprio else None,
        proprio_projector=proprio_projector if use_proprio else None,
        phase=cfg.phase,
    )

    predicted_actions = result

    # Deep supervision: predicted_actions may be a list of predictions
    if isinstance(predicted_actions, list):
        n = len(predicted_actions)
        weights = [(i + 1) for i in range(n)]
        total_weight = sum(weights)

        losses = [torch.nn.L1Loss()(pred, ground_truth_actions) for pred in predicted_actions]
        loss = sum(w * l for w, l in zip(weights, losses)) / total_weight

        final_pred = predicted_actions[-1]
        metrics.update({
            "loss_value": loss.item(),
            "curr_action_l1_loss": torch.nn.L1Loss()(final_pred[:, 0], ground_truth_actions[:, 0]).item(),
            "next_actions_l1_loss": torch.nn.L1Loss()(final_pred[:, 1:], ground_truth_actions[:, 1:]).item(),
            "deep_supervision_n": len(predicted_actions),
            "iter_0_loss": losses[0].item() if len(losses) > 0 else 0.0,
            "iter_final_loss": losses[-1].item() if len(losses) > 0 else 0.0,
        })
    else:
        loss = torch.nn.L1Loss()(predicted_actions, ground_truth_actions)
        metrics.update({
            "loss_value": loss.item(),
            "curr_action_l1_loss": torch.nn.L1Loss()(predicted_actions[:, 0], ground_truth_actions[:, 0]).item(),
            "next_actions_l1_loss": torch.nn.L1Loss()(predicted_actions[:, 1:], ground_truth_actions[:, 1:]).item(),
        })

    return loss, metrics


def save_training_checkpoint(cfg, run_dir, log_step, vla, processor, proprio_projector,
                             action_head, train_dataset, distributed_state, new_state_dict):
    if cfg.save_latest_checkpoint_only:
        checkpoint_dir = run_dir
        checkpoint_name_suffix = "latest_checkpoint.pt"
    else:
        checkpoint_dir = Path(str(run_dir) + f"--{log_step}_chkpt")
        checkpoint_name_suffix = f"{log_step}_checkpoint.pt"

    adapter_dir = checkpoint_dir / "lora_adapter"

    if distributed_state.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(adapter_dir, exist_ok=True)
        save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)
        print(f"Saving Model Checkpoint for Step {log_step}")

    dist.barrier()

    if distributed_state.is_main_process:
        processor.save_pretrained(checkpoint_dir)
        if hasattr(cfg, 'use_fz') and cfg.use_fz:
            vla.module.save_pretrained(checkpoint_dir)
        else:
            vla.module.save_pretrained(adapter_dir)

        if cfg.use_proprio and proprio_projector is not None:
            torch.save(proprio_projector.state_dict(), checkpoint_dir / f"proprio_projector--{checkpoint_name_suffix}")

        if action_head is not None:
            action_head_unwrapped = getattr(action_head, 'module', action_head)
            torch.save(action_head_unwrapped.state_dict(), checkpoint_dir / f"action_head--{checkpoint_name_suffix}")
            if hasattr(action_head_unwrapped, 'cfg'):
                import json
                config_path = checkpoint_dir / f"action_head_config--{checkpoint_name_suffix.replace('.pt', '.json')}"
                cfg_dict = {k: list(v) if isinstance(v, tuple) else v for k, v in action_head_unwrapped.cfg.__dict__.items()}
                cfg_dict['_type'] = type(action_head_unwrapped).__name__
                with open(config_path, 'w') as f:
                    json.dump(cfg_dict, f, indent=2)

        if cfg.use_film:
            torch.save(vla.module.vision_backbone.state_dict(), checkpoint_dir / f"vision_backbone--{checkpoint_name_suffix}")

        import shutil
        config_path = Path(cfg.config_file_path)
        for py_file in ["configuration_prismatic.py", "modeling_prismatic.py"]:
            src_file = config_path / py_file
            if src_file.exists():
                shutil.copy(src_file, checkpoint_dir / py_file)

    dist.barrier()

    if cfg.use_lora and cfg.merge_lora_during_training:
        if cfg.use_minivlm:
            config = AutoConfig.from_pretrained(os.path.join(cfg.config_file_path, "config.json"))
            base_vla = AutoModelForVision2Seq.from_config(config, torch_dtype=torch.bfloat16)
            new_state_dict['action_queries.weight'] = vla.state_dict()['module.base_model.model.action_queries.weight'].cpu()
            base_vla.load_state_dict(new_state_dict, strict=False)
        else:
            base_vla = AutoModelForVision2Seq.from_pretrained(
                cfg.config_file_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False, trust_remote_code=False
            )

        merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
        merged_vla = merged_vla.merge_and_unload()

        if distributed_state.is_main_process:
            merged_vla.save_pretrained(checkpoint_dir)
            print(f"Saved merged model for Step {log_step} at: {checkpoint_dir}")

        dist.barrier()


def run_validation(
    vla, action_head, proprio_projector, val_dataloader, action_tokenizer,
    device_id, cfg, num_patches, log_step, distributed_state, val_time_limit,
):
    val_start_time = time.time()
    vla.eval()
    all_val_metrics = []

    with torch.no_grad():
        for batch in val_dataloader:
            _, metrics = run_forward_pass(
                vla=vla, action_head=action_head, proprio_projector=proprio_projector,
                batch=batch, action_tokenizer=action_tokenizer, device_id=device_id,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film, num_patches=num_patches, cfg=cfg,
            )
            all_val_metrics.append(metrics)
            if time.time() - val_start_time > val_time_limit:
                break

    if not all_val_metrics:
        vla.train()
        return

    avg_val_metrics = {}
    for metric_name in all_val_metrics[0].keys():
        values = [m[metric_name] for m in all_val_metrics if metric_name in m]
        if values:
            avg_val_metrics[metric_name] = sum(values) / len(values)

    if distributed_state.is_main_process:
        if cfg.use_wandb:
            wandb.log({f"VLA Val/{k}": v for k, v in avg_val_metrics.items()}, step=log_step)
        else:
            print(f"[val step {log_step}] " + ", ".join(f"{k}={v:.6f}" for k, v in avg_val_metrics.items()))

    vla.train()


def finetune(cfg):
    global RAW_STATE_DICT

    cfg.config_file_path = cfg.config_file_path.rstrip("/")
    print(f"Fine-tuning on `{cfg.dataset_name}`")

    run_id = get_run_id(cfg)
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)

    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    if distributed_state.is_main_process:
        if cfg.use_wandb:
            if not cfg.wandb_entity:
                wandb.init(project=cfg.wandb_project, name=f"ft+{run_id}", mode="online")
            else:
                wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity, name=f"ft+{run_id}", mode="online")

    print(f"NUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}, ACTION_DIM: {ACTION_DIM}")

    if model_is_on_hf_hub(cfg.config_file_path):
        vla_download_path = snapshot_download(repo_id=cfg.config_file_path)
        cfg.config_file_path = vla_download_path
    else:
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    if distributed_state.is_main_process:
        update_auto_map(cfg.config_file_path)
        check_model_logic_mismatch(cfg.config_file_path)

    dist.barrier()

    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    processor = AutoProcessor.from_pretrained(cfg.config_file_path, trust_remote_code=True)

    if cfg.use_minivlm:
        hf_token = os.environ.get("HF_TOKEN", "")
        if 'prism-qwen25-extra-dinosiglip-224px-0_5b' in cfg.vlm_path:
            vlm = load(cfg.vlm_path, hf_token=hf_token, load_for_training=True)
        else:
            vlm = load_vla(cfg.vlm_path, hf_token=hf_token, load_for_training=True)
        config = AutoConfig.from_pretrained(os.path.join(cfg.config_file_path, "config.json"))
        vla = AutoModelForVision2Seq.from_config(config, torch_dtype=torch.bfloat16).to(device_id)

        replace_map = [
            ("vision_backbone.dino_featurizer", "vision_backbone.featurizer"),
            ("vision_backbone.siglip_featurizer", "vision_backbone.fused_featurizer"),
            ("llm_backbone.llm", "language_model"),
            ("projector.projector.0", "projector.fc1"),
            ("projector.projector.2", "projector.fc2"),
            ("projector.projector.4", "projector.fc3"),
            ("gamma", "scale_factor"),
        ]

        def rename_state_dict_keys(state_dict, replace_map):
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k
                for old, new in replace_map:
                    if old in new_k:
                        new_k = new_k.replace(old, new)
                new_state_dict[new_k] = v
            return new_state_dict

        old_state_dict = vlm.state_dict()
        RAW_STATE_DICT = rename_state_dict_keys(old_state_dict, replace_map)
        vla.load_state_dict(RAW_STATE_DICT, strict=False)
        del old_state_dict
    else:
        RAW_STATE_DICT = {}
        vla = AutoModelForVision2Seq.from_pretrained(
            cfg.config_file_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False, trust_remote_code=False
        ).to(device_id)

    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=2 * cfg.lora_rank,
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        for name, param in vla.named_parameters():
            if "action_queries" in name:
                param.requires_grad = True
        vla.print_trainable_parameters()
    else:
        for name, param in vla.named_parameters():
            if "action_queries" in name:
                param.requires_grad = True

    vla = wrap_ddp(vla, device_id, find_unused=True)

    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = init_module(
            ProprioProjector, "proprio_projector", cfg, device_id,
            {"llm_dim": vla.module.llm_dim, "proprio_dim": PROPRIO_DIM},
            to_bf16=True,
        )

    llm_dim = vla.module.llm_dim
    action_head = None
    action_head_type = getattr(cfg, 'action_head', None)
    if action_head_type is not None and hasattr(action_head_type, 'type'):
        rec_cfg = action_head_type.recurrent
        internal_cfg = RecurrentConfigInternal(
            hidden_dim=rec_cfg.hidden_dim,
            num_heads=rec_cfg.num_heads,
            prelude_vlm_layers=tuple(rec_cfg.prelude_vlm_layers),
            recurrent_vlm_layers=tuple(rec_cfg.recurrent_vlm_layers),
            coda_vlm_layers=tuple(rec_cfg.coda_vlm_layers),
            action_chunk_len=rec_cfg.action_chunk_len,
            action_dim=rec_cfg.action_dim,
            mean_recurrence=rec_cfg.mean_recurrence,
            backprop_depth=rec_cfg.backprop_depth,
            random_iterations=rec_cfg.random_iterations,
            init_std=rec_cfg.init_std,
            rms_norm_eps=rec_cfg.rms_norm_eps,
            rope_base=rec_cfg.rope_base,
        )
        action_head = init_module(
            ActionHeadRecurrent, "action_head", cfg, device_id,
            {"hidden_dim": llm_dim, "cfg": internal_cfg},
            to_bf16=True,
        )

    if action_head is None:
        raise ValueError("action_head config is required")

    NUM_PATCHES = vla.module.vision_backbone.get_num_patches() * vla.module.vision_backbone.get_num_images_in_input()

    vla_params = [param for param in vla.parameters() if param.requires_grad]
    action_head_params = [param for param in action_head.parameters() if param.requires_grad] if action_head else []
    proprio_params = [param for param in proprio_projector.parameters() if param.requires_grad] if cfg.use_proprio else []

    total_params = sum(p.numel() for p in vla_params + action_head_params + proprio_params)
    print(f"# total trainable params: {total_params}")

    if cfg.use_muon:
        print("Using MUON optimizer for action head")
        action_head_muon_params = [p for p in action_head_params if p.ndim == 2]
        action_head_adam_params = [p for p in action_head_params if p.ndim != 2]

        optimizer = AdamW(vla_params + proprio_params, lr=cfg.learning_rate)
        muon_optimizer = Muon([
            {"params": action_head_muon_params, "use_muon": True, "lr": cfg.muon_lr,
             "momentum": cfg.muon_momentum, "weight_decay": cfg.muon_weight_decay,
             "ns_steps": cfg.muon_ns_steps, "matched_adamw_rms": cfg.muon_matched_adamw_rms},
            {"params": action_head_adam_params, "use_muon": False, "lr": cfg.learning_rate,
             "weight_decay": 0.0, "adamw_betas": (0.9, 0.95), "adamw_eps": 1e-8},
        ])
        optimizers = [optimizer, muon_optimizer]
    else:
        trainable_params = vla_params + action_head_params + proprio_params
        optimizer = AdamW(trainable_params, lr=cfg.learning_rate)
        optimizers = [optimizer]

    original_lr = optimizer.param_groups[0]["lr"]
    scheduler = MultiStepLR(optimizer, milestones=[cfg.num_steps_before_decay], gamma=0.1)

    action_tokenizer = ActionTokenizer(processor.tokenizer)

    use_wrist_image = cfg.num_images_in_input > 1
    batch_transform = RLDSBatchTransform(
        action_tokenizer, processor.tokenizer, image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder, use_wrist_image=use_wrist_image,
        use_proprio=cfg.use_proprio, use_minivlm=cfg.use_minivlm
    )

    dataset_fraction = getattr(cfg, 'dataset_fraction', 1.0)
    train_dataset = RLDSDataset(
        cfg.data_root_dir, cfg.dataset_name, batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size, image_aug=cfg.image_aug,
        dataset_fraction=dataset_fraction,
    )

    if cfg.use_val_set:
        val_dataset = RLDSDataset(
            cfg.data_root_dir, cfg.dataset_name, batch_transform,
            resize_resolution=tuple(vla.module.config.image_sizes),
            shuffle_buffer_size=cfg.shuffle_buffer_size // 10, image_aug=cfg.image_aug,
            train=False,
        )

    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, sampler=None, collate_fn=collator, num_workers=0)
    if distributed_state.is_main_process:
        print(f'Dataset size: {len(dataloader)} batches')

    if cfg.use_val_set:
        val_dataloader = DataLoader(
            val_dataset, batch_size=cfg.batch_size, sampler=None, collate_fn=collator, num_workers=0,
        )

    recent_metrics = {
        "loss_value": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
    }

    with tqdm.tqdm(total=cfg.max_steps, leave=False, disable=not distributed_state.is_main_process) as progress:
        vla.train()
        for opt in optimizers:
            opt.zero_grad()

        batch_idx = 0
        max_batch_idx = cfg.max_steps * cfg.grad_accumulation_steps

        while batch_idx < max_batch_idx:
            for batch in dataloader:
                if batch_idx >= max_batch_idx:
                    break

                loss, metrics = run_forward_pass(
                    vla=vla, action_head=action_head, proprio_projector=proprio_projector if cfg.use_proprio else None,
                    batch=batch, action_tokenizer=action_tokenizer, device_id=device_id,
                    use_proprio=cfg.use_proprio,
                    use_film=cfg.use_film, num_patches=NUM_PATCHES, cfg=cfg,
                )

                normalized_loss = loss / cfg.grad_accumulation_steps
                normalized_loss.backward()

                for metric_name, value in metrics.items():
                    if metric_name in recent_metrics:
                        recent_metrics[metric_name].append(value)

                gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
                log_step = gradient_step_idx if not cfg.resume else cfg.resume_step + gradient_step_idx

                if distributed_state.is_main_process and log_step % cfg.wandb_log_freq == 0 and cfg.use_wandb:
                    smoothened = {k: sum(v) / len(v) for k, v in recent_metrics.items() if v}
                    wandb.log({f"VLA Train/{k}": v for k, v in smoothened.items()}, step=log_step)

                if cfg.lr_warmup_steps > 0:
                    lr_progress = min((gradient_step_idx + 1) / cfg.lr_warmup_steps, 1.0)
                    current_lr = original_lr * (0.1 + 0.9 * lr_progress)
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = current_lr

                if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                    all_params = vla_params + action_head_params + proprio_params
                    torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)

                    for opt in optimizers:
                        opt.step()
                    scheduler.step()
                    for opt in optimizers:
                        opt.zero_grad()
                    progress.update()

                if gradient_step_idx > 0 and log_step % cfg.save_freq == 0:
                    save_training_checkpoint(
                        cfg=cfg, run_dir=run_dir, log_step=log_step, vla=vla, processor=processor,
                        proprio_projector=proprio_projector if cfg.use_proprio else None,
                        action_head=action_head, train_dataset=train_dataset,
                        distributed_state=distributed_state, new_state_dict=RAW_STATE_DICT,
                    )

                if cfg.use_val_set and log_step > 0 and log_step % cfg.val_freq == 0:
                    run_validation(
                        vla=vla, action_head=action_head,
                        proprio_projector=proprio_projector if cfg.use_proprio else None,
                        val_dataloader=val_dataloader, action_tokenizer=action_tokenizer,
                        device_id=device_id, cfg=cfg, num_patches=NUM_PATCHES,
                        log_step=log_step, distributed_state=distributed_state,
                        val_time_limit=cfg.val_time_limit,
                    )

                batch_idx += 1

                if log_step == cfg.max_steps:
                    print(f"Max step {cfg.max_steps} reached! Stopping training...")
                    break

            if batch_idx >= max_batch_idx:
                break

        final_step = cfg.resume_step + gradient_step_idx if cfg.resume else gradient_step_idx
        if distributed_state.is_main_process:
            print(f"Training complete at step {final_step}. Saving final checkpoint...")
        save_training_checkpoint(
            cfg=cfg, run_dir=run_dir, log_step=final_step, vla=vla, processor=processor,
            proprio_projector=proprio_projector if cfg.use_proprio else None,
            action_head=action_head, train_dataset=train_dataset,
            distributed_state=distributed_state, new_state_dict=RAW_STATE_DICT,
        )


if __name__ == "__main__":
    print("Use run.py with --mode train instead")
