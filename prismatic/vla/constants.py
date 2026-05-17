import sys
from typing import Optional
from enum import Enum

# Qwen2.5-0.5B token constants
IGNORE_INDEX = -100
ACTION_TOKEN_BEGIN_IDX = 151386
STOP_INDEX = 2  # '</s>'
NUM_TOKENS = 64


# Defines supported normalization schemes for action and proprioceptive state.
class NormalizationType(str, Enum):
    # fmt: off
    NORMAL = "normal"               # Normalize to Mean = 0, Stdev = 1
    BOUNDS = "bounds"               # Normalize to Interval = [-1, 1]
    BOUNDS_Q99 = "bounds_q99"       # Normalize [quantile_01, ..., quantile_99] --> [-1, ..., 1]
    # fmt: on


# Define constants for each robot platform
YAM_BIMANUAL_TASK_NAMES = [
    "put_the_cube_into_the_bowl",
]

LIBERO_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

CALVIN_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

ALOHA_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 25,
    "ACTION_DIM": 14,
    "PROPRIO_DIM": 14,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
}

BRIDGE_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 5,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 7,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

YAM_BIMANUAL_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 14,
    "PROPRIO_DIM": 14,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}


# Function to detect robot platform from command line arguments
def detect_robot_platform():
    cmd_args = " ".join(sys.argv).lower()

    if "libero" in cmd_args:
        return "LIBERO"
    elif "aloha" in cmd_args:
        return "ALOHA"
    elif "bridge" in cmd_args:
        return "BRIDGE"
    elif "calvin" in cmd_args:
        return "CALVIN"
    elif any(task_name in cmd_args for task_name in YAM_BIMANUAL_TASK_NAMES):
        return "YAM_BIMANUAL"
    config_dataset_name = _get_dataset_name_from_config()
    if config_dataset_name:
        dataset_name = config_dataset_name.lower()
        if "aloha" in dataset_name:
            return "ALOHA"
        if "calvin" in dataset_name:
            return "CALVIN"
        if "libero" in dataset_name:
            return "LIBERO"
        if "bridge" in dataset_name:
            return "BRIDGE"
        if  any(task_name in dataset_name for task_name in YAM_BIMANUAL_TASK_NAMES):
            return "YAM_BIMANUAL"
    # Default to LIBERO if unclear
    return "LIBERO"


def _get_dataset_name_from_config() -> Optional[str]:
    if "--config" not in sys.argv:
        return None
    try:
        config_idx = sys.argv.index("--config")
        config_path = sys.argv[config_idx + 1]
    except (ValueError, IndexError):
        return None
    try:
        import yaml
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        if isinstance(cfg, dict):
            if "data" in cfg and isinstance(cfg["data"], dict):
                return cfg["data"].get("dataset_name")
            return cfg.get("dataset_name")
    except Exception:
        return None
    return None


# Determine which robot platform to use
ROBOT_PLATFORM = detect_robot_platform()

# Set the appropriate constants based on the detected platform
if ROBOT_PLATFORM == "LIBERO":
    constants = LIBERO_CONSTANTS
elif ROBOT_PLATFORM == "ALOHA":
    constants = ALOHA_CONSTANTS
elif ROBOT_PLATFORM == "BRIDGE":
    constants = BRIDGE_CONSTANTS
elif ROBOT_PLATFORM == "CALVIN":
    constants = CALVIN_CONSTANTS
elif ROBOT_PLATFORM == "YAM_BIMANUAL":
    constants = YAM_BIMANUAL_CONSTANTS

# Assign constants to global variables
NUM_ACTIONS_CHUNK = constants["NUM_ACTIONS_CHUNK"]
ACTION_DIM = constants["ACTION_DIM"]
PROPRIO_DIM = constants["PROPRIO_DIM"]
ACTION_PROPRIO_NORMALIZATION_TYPE = constants["ACTION_PROPRIO_NORMALIZATION_TYPE"]

# Print which robot platform constants are being used (for debugging)
print(f"Using {ROBOT_PLATFORM} constants:")
print(f"  NUM_ACTIONS_CHUNK = {NUM_ACTIONS_CHUNK}")
print(f"  ACTION_DIM = {ACTION_DIM}")
print(f"  PROPRIO_DIM = {PROPRIO_DIM}")
print(f"  ACTION_PROPRIO_NORMALIZATION_TYPE = {ACTION_PROPRIO_NORMALIZATION_TYPE}")
print("If needed, manually set the correct constants in `prismatic/vla/constants.py`!")
