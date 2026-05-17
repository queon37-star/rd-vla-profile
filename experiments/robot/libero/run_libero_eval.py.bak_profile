import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import cv2
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



def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"


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
):
    """Run a single episode in the environment."""
    env.reset()

    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    t = 0
    replay_images = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    action_queue = deque()
    episode_iters = []
    replay_stats = []  # (iters, num_actions) per prediction

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
                actions, actual_iters, final_kl = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    use_film=cfg.use_film,
                    use_minivlm=cfg.use_minivlm
                )

                if actual_iters is not None:
                    episode_iters.append(actual_iters)

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

    return success, replay_images, episode_iters, replay_stats




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
    save_version=None
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

        success, replay_images, episode_iters, replay_stats = run_episode(
            cfg, env, task_description, model, resize_size, processor,
            action_head, proprio_projector, initial_state, log_file,
            global_iters=all_iters,
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
            cfg.save_version
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
        with open(cfg.json_log_file, "w") as jf:
            json.dump(full_results, jf, indent=2)
        log_message(f"Saved JSON results to {cfg.json_log_file}", log_file)

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

    if log_file:
        log_file.close()

    return final_success_rate



if __name__ == "__main__":
    eval_libero()
