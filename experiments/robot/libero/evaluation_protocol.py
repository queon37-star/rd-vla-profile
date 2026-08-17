"""Deterministic online-evaluation protocol for the LIBERO Spatial study.

The legacy runner remains the default.  Opting into this protocol freezes the
official 50 initial states into disjoint calibration, screening, and final
partitions and derives condition-independent RNG seeds for paired comparisons.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


PROTOCOL_SCHEMA_VERSION = 1
PROTOCOL_NAME = "libero-spatial-origin-aware-v1"
PARTITION_ALGORITHM = "sha256-ranked-v1"
PAIRED_SEED_NAMESPACE = "rd-vla-libero-paired-v1"
OFFICIAL_INITIAL_STATE_COUNT = 50
PARTITION_SIZES = {"calibration": 10, "screening": 10, "final": 30}
PHASE_TRIAL_COUNTS = {
    "smoke": 3,
    "calibration": PARTITION_SIZES["calibration"],
    "screening": PARTITION_SIZES["screening"],
    "final_holdout": PARTITION_SIZES["final"],
}
SUPPORTED_PROTOCOL_PHASES = ("legacy", *PHASE_TRIAL_COUNTS)


@dataclass(frozen=True)
class EpisodeTrial:
    """One phase-local paired trial and its frozen official initial state."""

    phase: str
    partition: str
    paired_trial_id: int
    initial_state_id: int
    episode_seed: int
    smoke_excluded_from_fitting: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_file(path: str) -> str:
    """Hash the exact serialized initial-state file used by LIBERO."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_initial_states(initial_states: Sequence[Any]) -> str:
    """Hash an ordered initial-state collection without lossy JSON conversion."""

    digest = hashlib.sha256()
    digest.update(b"rd-vla-libero-initial-states-v1\0")
    digest.update(len(initial_states).to_bytes(8, byteorder="big", signed=False))
    for state_id, state in enumerate(initial_states):
        array = np.asarray(state)
        if array.dtype.hasobject:
            raise ValueError(f"Initial state {state_id} has unsupported object dtype")
        contiguous = np.ascontiguousarray(array)
        header = _canonical_json_bytes(
            {"state_id": state_id, "dtype": contiguous.dtype.str, "shape": list(contiguous.shape)}
        )
        payload = contiguous.tobytes(order="C")
        digest.update(len(header).to_bytes(8, byteorder="big", signed=False))
        digest.update(header)
        digest.update(len(payload).to_bytes(8, byteorder="big", signed=False))
        digest.update(payload)
    return digest.hexdigest()


def partition_initial_state_ids(task_suite_name: str, task_id: int) -> Dict[str, Tuple[int, ...]]:
    """Return the frozen, deterministic 10/10/30 partition for one task."""

    ranked_ids = sorted(
        range(OFFICIAL_INITIAL_STATE_COUNT),
        key=lambda state_id: hashlib.sha256(
            _canonical_json_bytes(
                {
                    "algorithm": PARTITION_ALGORITHM,
                    "protocol": PROTOCOL_NAME,
                    "task_suite_name": str(task_suite_name),
                    "task_id": int(task_id),
                    "initial_state_id": state_id,
                }
            )
        ).digest(),
    )
    calibration_end = PARTITION_SIZES["calibration"]
    screening_end = calibration_end + PARTITION_SIZES["screening"]
    return {
        "calibration": tuple(ranked_ids[:calibration_end]),
        "screening": tuple(ranked_ids[calibration_end:screening_end]),
        "final": tuple(ranked_ids[screening_end:]),
    }


def build_protocol_manifest(
    task_suite_name: str,
    task_initial_states: Mapping[int, Sequence[Any]],
    *,
    task_names: Optional[Mapping[int, str]] = None,
    task_initial_state_files: Optional[Mapping[int, str]] = None,
) -> Dict[str, Any]:
    """Build a deterministic manifest from the states returned by LIBERO."""

    tasks: Dict[str, Any] = {}
    for task_id in sorted(task_initial_states):
        initial_states = task_initial_states[task_id]
        if len(initial_states) != OFFICIAL_INITIAL_STATE_COUNT:
            raise ValueError(
                f"Task {task_id} exposes {len(initial_states)} initial states; "
                f"expected exactly {OFFICIAL_INITIAL_STATE_COUNT}"
            )
        partitions = partition_initial_state_ids(task_suite_name, task_id)
        entry: Dict[str, Any] = {
            "initial_states_sha256": hash_initial_states(initial_states),
            "state_count": len(initial_states),
            "partitions": {name: list(ids) for name, ids in partitions.items()},
        }
        if task_initial_state_files is not None:
            if task_id not in task_initial_state_files:
                raise ValueError(f"Missing initial-state source file for task {task_id}")
            source_path = Path(task_initial_state_files[task_id])
            entry["initial_states_file"] = source_path.name
            entry["initial_states_file_sha256"] = sha256_file(str(source_path))
        if task_names is not None and task_id in task_names:
            entry["task_name"] = str(task_names[task_id])
        tasks[str(task_id)] = entry

    manifest = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol": PROTOCOL_NAME,
        "partition_algorithm": PARTITION_ALGORITHM,
        "task_suite_name": str(task_suite_name),
        "expected_state_count": OFFICIAL_INITIAL_STATE_COUNT,
        "partition_sizes": dict(PARTITION_SIZES),
        "tasks": tasks,
    }
    validate_protocol_manifest(manifest)
    return manifest


def validate_protocol_manifest(manifest: Mapping[str, Any]) -> None:
    """Fail closed when a manifest does not exactly match this protocol."""

    expected_header = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol": PROTOCOL_NAME,
        "partition_algorithm": PARTITION_ALGORITHM,
        "expected_state_count": OFFICIAL_INITIAL_STATE_COUNT,
        "partition_sizes": PARTITION_SIZES,
    }
    for key, expected in expected_header.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Protocol manifest field {key!r} must equal {expected!r}")

    task_suite_name = manifest.get("task_suite_name")
    if not isinstance(task_suite_name, str) or not task_suite_name:
        raise ValueError("Protocol manifest task_suite_name must be a non-empty string")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, Mapping) or not tasks:
        raise ValueError("Protocol manifest tasks must be a non-empty mapping")
    if task_suite_name == "libero_spatial" and set(tasks) != {str(task_id) for task_id in range(10)}:
        raise ValueError("The LIBERO Spatial protocol manifest must contain exactly task IDs 0..9")

    expected_ids = set(range(OFFICIAL_INITIAL_STATE_COUNT))
    for task_id_text, entry in tasks.items():
        try:
            task_id = int(task_id_text)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid protocol manifest task ID: {task_id_text!r}") from exc
        if not isinstance(entry, Mapping):
            raise ValueError(f"Protocol manifest task {task_id} must be a mapping")
        if entry.get("state_count") != OFFICIAL_INITIAL_STATE_COUNT:
            raise ValueError(f"Protocol manifest task {task_id} must contain exactly 50 states")
        state_hash = entry.get("initial_states_sha256")
        if not isinstance(state_hash, str) or len(state_hash) != 64:
            raise ValueError(f"Protocol manifest task {task_id} has an invalid initial-state SHA-256")
        source_name = entry.get("initial_states_file")
        source_hash = entry.get("initial_states_file_sha256")
        if (source_name is None) != (source_hash is None):
            raise ValueError(
                f"Protocol manifest task {task_id} must provide both initial-state file name and SHA-256"
            )
        if source_name is not None:
            if not isinstance(source_name, str) or not source_name:
                raise ValueError(f"Protocol manifest task {task_id} has an invalid initial-state file name")
            if not isinstance(source_hash, str) or len(source_hash) != 64:
                raise ValueError(f"Protocol manifest task {task_id} has an invalid initial-state file SHA-256")

        partitions = entry.get("partitions")
        if not isinstance(partitions, Mapping):
            raise ValueError(f"Protocol manifest task {task_id} is missing partitions")
        expected_partitions = partition_initial_state_ids(task_suite_name, task_id)
        observed_ids = []
        for partition, expected_size in PARTITION_SIZES.items():
            ids = partitions.get(partition)
            if not isinstance(ids, list) or len(ids) != expected_size:
                raise ValueError(
                    f"Protocol manifest task {task_id} partition {partition!r} must contain {expected_size} IDs"
                )
            if tuple(ids) != expected_partitions[partition]:
                raise ValueError(
                    f"Protocol manifest task {task_id} partition {partition!r} does not match "
                    f"{PARTITION_ALGORITHM}"
                )
            if any(not isinstance(state_id, int) for state_id in ids):
                raise ValueError(f"Protocol manifest task {task_id} contains a non-integer state ID")
            observed_ids.extend(ids)
        if len(observed_ids) != len(set(observed_ids)) or set(observed_ids) != expected_ids:
            raise ValueError(f"Protocol manifest task {task_id} partitions must be disjoint and cover IDs 0..49")


def validate_protocol_source_file_hashes(manifest: Mapping[str, Any]) -> None:
    """Require a raw source-file identity for every task in a production manifest."""

    for task_id, entry in manifest["tasks"].items():
        if entry.get("initial_states_file") is None or entry.get("initial_states_file_sha256") is None:
            raise ValueError(
                f"Protocol manifest task {task_id} is missing the required raw initial-state file SHA-256"
            )


def load_protocol_manifest(
    path: str, *, require_source_file_hashes: bool = False
) -> Tuple[Dict[str, Any], str]:
    """Load, validate, and hash the exact manifest bytes used by a run."""

    manifest_path = Path(path)
    raw = manifest_path.read_bytes()
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid protocol manifest JSON: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Protocol manifest root must be a JSON object")
    validate_protocol_manifest(manifest)
    if require_source_file_hashes:
        validate_protocol_source_file_hashes(manifest)
    return manifest, hashlib.sha256(raw).hexdigest()


def write_protocol_manifest(path: str, manifest: Mapping[str, Any], *, overwrite: bool = False) -> str:
    """Write a canonical manifest and return the file SHA-256."""

    validate_protocol_manifest(manifest)
    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing protocol manifest: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
    output_path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def derive_paired_episode_seed(
    *,
    base_seed: int,
    phase: str,
    task_suite_name: str,
    task_id: int,
    initial_state_id: int,
    paired_trial_id: int,
) -> int:
    """Derive a stable uint32 seed; condition/arm identity is intentionally absent."""

    payload = _canonical_json_bytes(
        {
            "namespace": PAIRED_SEED_NAMESPACE,
            "base_seed": int(base_seed),
            "phase": str(phase),
            "task_suite_name": str(task_suite_name),
            "task_id": int(task_id),
            "initial_state_id": int(initial_state_id),
            "paired_trial_id": int(paired_trial_id),
        }
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], byteorder="big", signed=False)


def validate_protocol_configuration(
    *,
    phase: str,
    task_suite_name: str,
    num_trials_per_task: int,
    initial_states_path: str,
    manifest_path: str,
    reset_rng_each_episode: bool,
) -> None:
    """Validate only opt-in protocol settings; legacy behavior remains untouched."""

    if phase not in SUPPORTED_PROTOCOL_PHASES:
        raise ValueError(f"Unsupported evaluation_protocol_phase: {phase!r}")
    if phase == "legacy":
        return
    if task_suite_name != "libero_spatial":
        raise ValueError("The frozen online protocol is defined only for the 10 LIBERO Spatial tasks")
    if initial_states_path != "DEFAULT":
        raise ValueError("The frozen online protocol requires LIBERO's official DEFAULT initial states")
    if not manifest_path:
        raise ValueError("initial_state_manifest_path is required for a non-legacy evaluation protocol")
    if not Path(manifest_path).is_file():
        raise ValueError(f"Initial-state protocol manifest does not exist: {manifest_path}")
    if not reset_rng_each_episode:
        raise ValueError("A non-legacy evaluation protocol requires reset_rng_each_episode=True")
    expected_trials = PHASE_TRIAL_COUNTS[phase]
    if int(num_trials_per_task) != expected_trials:
        raise ValueError(
            f"evaluation_protocol_phase={phase!r} requires num_trials_per_task={expected_trials}, "
            f"got {num_trials_per_task}"
        )


def resolve_phase_trials(
    *,
    manifest: Mapping[str, Any],
    phase: str,
    task_id: int,
    initial_states: Sequence[Any],
    base_seed: int,
    initial_state_file_path: Optional[str] = None,
) -> Tuple[EpisodeTrial, ...]:
    """Validate runtime states and resolve phase-local paired trials."""

    validate_protocol_manifest(manifest)
    if phase not in PHASE_TRIAL_COUNTS:
        raise ValueError(f"Cannot resolve trials for evaluation protocol phase {phase!r}")
    if len(initial_states) != OFFICIAL_INITIAL_STATE_COUNT:
        raise ValueError(
            f"Task {task_id} exposes {len(initial_states)} initial states; "
            f"expected exactly {OFFICIAL_INITIAL_STATE_COUNT}"
        )
    task_key = str(int(task_id))
    if task_key not in manifest["tasks"]:
        raise ValueError(f"Task {task_id} is absent from the initial-state protocol manifest")
    task_entry = manifest["tasks"][task_key]
    actual_hash = hash_initial_states(initial_states)
    expected_hash = task_entry["initial_states_sha256"]
    if actual_hash != expected_hash:
        raise ValueError(
            f"Task {task_id} initial-state hash mismatch: expected {expected_hash}, got {actual_hash}"
        )
    expected_file_hash = task_entry.get("initial_states_file_sha256")
    if expected_file_hash is not None:
        if initial_state_file_path is None:
            raise ValueError(f"Task {task_id} requires an initial-state source file for hash validation")
        actual_file = Path(initial_state_file_path)
        expected_file_name = task_entry["initial_states_file"]
        if actual_file.name != expected_file_name:
            raise ValueError(
                f"Task {task_id} initial-state file mismatch: expected {expected_file_name}, got {actual_file.name}"
            )
        actual_file_hash = sha256_file(str(actual_file))
        if actual_file_hash != expected_file_hash:
            raise ValueError(
                f"Task {task_id} initial-state file hash mismatch: expected {expected_file_hash}, "
                f"got {actual_file_hash}"
            )

    if phase == "smoke":
        partition = "calibration"
    elif phase == "final_holdout":
        partition = "final"
    else:
        partition = phase
    state_ids = list(task_entry["partitions"][partition])
    if phase == "smoke":
        state_ids = state_ids[: PHASE_TRIAL_COUNTS["smoke"]]
    trials = []
    for paired_trial_id, initial_state_id in enumerate(state_ids):
        trials.append(
            EpisodeTrial(
                phase=phase,
                partition=partition,
                paired_trial_id=paired_trial_id,
                initial_state_id=initial_state_id,
                episode_seed=derive_paired_episode_seed(
                    base_seed=base_seed,
                    phase=phase,
                    task_suite_name=manifest["task_suite_name"],
                    task_id=task_id,
                    initial_state_id=initial_state_id,
                    paired_trial_id=paired_trial_id,
                ),
                smoke_excluded_from_fitting=phase == "smoke",
            )
        )
    return tuple(trials)


def reset_episode_environment(
    env,
    initial_state,
    *,
    episode_seed: Optional[int],
    torch_module,
    seed_environment: bool,
):
    """Seed paired RNGs, then reset, then apply the frozen initial state."""

    if episode_seed is not None:
        seed = int(episode_seed)
        random.seed(seed)
        np.random.seed(seed)
        torch_module.manual_seed(seed)
        if torch_module.cuda.is_available():
            torch_module.cuda.manual_seed_all(seed)

    environment_seed_applied = False
    if seed_environment:
        if episode_seed is None:
            raise ValueError("seed_environment=True requires an episode_seed")
        env_seed = getattr(env, "seed", None)
        if callable(env_seed):
            env_seed(int(episode_seed))
            environment_seed_applied = True

    env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()
    return obs, environment_seed_applied
