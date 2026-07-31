import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pytest

from experiments.robot.libero.evaluation_protocol import (
    OFFICIAL_INITIAL_STATE_COUNT,
    PARTITION_SIZES,
    build_protocol_manifest,
    derive_paired_episode_seed,
    hash_initial_states,
    load_protocol_manifest,
    partition_initial_state_ids,
    reset_episode_environment,
    resolve_phase_trials,
    validate_protocol_configuration,
    validate_protocol_manifest,
    write_protocol_manifest,
)


def _states(task_id):
    return [np.array([task_id, state_id, state_id / 10.0], dtype=np.float32) for state_id in range(50)]


def _manifest():
    states = {task_id: _states(task_id) for task_id in range(10)}
    names = {task_id: f"task_{task_id}" for task_id in range(10)}
    return build_protocol_manifest("libero_spatial", states, task_names=names)


def test_hash_partition_is_frozen_disjoint_and_exhaustive():
    partitions = partition_initial_state_ids("libero_spatial", 0)

    assert partitions["calibration"] == (24, 47, 9, 33, 14, 29, 3, 39, 34, 19)
    assert partitions["screening"] == (11, 28, 8, 21, 12, 22, 16, 38, 15, 7)
    assert partitions["final"] == (
        27,
        25,
        49,
        26,
        13,
        6,
        4,
        35,
        43,
        48,
        41,
        0,
        17,
        32,
        40,
        36,
        42,
        18,
        10,
        46,
        1,
        5,
        45,
        23,
        44,
        2,
        37,
        30,
        20,
        31,
    )
    assert {name: len(ids) for name, ids in partitions.items()} == PARTITION_SIZES
    all_ids = [state_id for ids in partitions.values() for state_id in ids]
    assert len(all_ids) == len(set(all_ids)) == OFFICIAL_INITIAL_STATE_COUNT
    assert set(all_ids) == set(range(OFFICIAL_INITIAL_STATE_COUNT))


def test_manifest_round_trip_hashes_exact_file_and_rejects_overwrite(tmp_path):
    manifest = _manifest()
    manifest_path = tmp_path / "states.json"

    written_sha = write_protocol_manifest(str(manifest_path), manifest)
    loaded, loaded_sha = load_protocol_manifest(str(manifest_path))

    assert loaded == manifest
    assert loaded_sha == written_sha
    assert loaded_sha == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_protocol_manifest(str(manifest_path), manifest)
    with pytest.raises(ValueError, match="raw initial-state file SHA-256"):
        load_protocol_manifest(str(manifest_path), require_source_file_hashes=True)


def test_raw_initial_state_file_hash_is_validated_at_runtime(tmp_path):
    source_files = {}
    for task_id in range(10):
        source_path = tmp_path / f"task_{task_id}.pt"
        source_path.write_bytes(f"serialized-task-{task_id}".encode("utf-8"))
        source_files[task_id] = str(source_path)
    manifest = build_protocol_manifest(
        "libero_spatial",
        {task_id: _states(task_id) for task_id in range(10)},
        task_initial_state_files=source_files,
    )
    manifest_path = tmp_path / "production-manifest.json"
    write_protocol_manifest(str(manifest_path), manifest)

    loaded, _ = load_protocol_manifest(str(manifest_path), require_source_file_hashes=True)
    trials = resolve_phase_trials(
        manifest=loaded,
        phase="smoke",
        task_id=0,
        initial_states=_states(0),
        base_seed=7,
        initial_state_file_path=source_files[0],
    )
    assert len(trials) == 3

    (tmp_path / "task_0.pt").write_bytes(b"mutated")
    with pytest.raises(ValueError, match="initial-state file hash mismatch"):
        resolve_phase_trials(
            manifest=loaded,
            phase="smoke",
            task_id=0,
            initial_states=_states(0),
            base_seed=7,
            initial_state_file_path=source_files[0],
        )


def test_committed_official_manifest_is_frozen_and_has_raw_file_hashes():
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "experiments/robot/libero/manifests/libero_spatial_official_50_v1.json"
    )
    manifest, manifest_sha256 = load_protocol_manifest(
        str(manifest_path), require_source_file_hashes=True
    )

    assert manifest_sha256 == "0e3c6609b719d6b0a05f79efd769dff67141b52d00b42d9e0bea904ecf493144"
    assert len(manifest["tasks"]) == 10
    assert all(entry["state_count"] == 50 for entry in manifest["tasks"].values())


def test_manifest_rejects_missing_task_and_partition_tampering():
    manifest = _manifest()
    del manifest["tasks"]["9"]
    with pytest.raises(ValueError, match="exactly task IDs 0..9"):
        validate_protocol_manifest(manifest)

    manifest = _manifest()
    manifest["tasks"]["0"]["partitions"]["calibration"][0] = 0
    with pytest.raises(ValueError, match="does not match sha256-ranked-v1"):
        validate_protocol_manifest(manifest)


def test_phase_trials_use_10_10_30_and_smoke_only_calibration_first_three():
    manifest = _manifest()
    states = _states(0)
    calibration = resolve_phase_trials(
        manifest=manifest, phase="calibration", task_id=0, initial_states=states, base_seed=7
    )
    screening = resolve_phase_trials(
        manifest=manifest, phase="screening", task_id=0, initial_states=states, base_seed=7
    )
    final = resolve_phase_trials(manifest=manifest, phase="final", task_id=0, initial_states=states, base_seed=7)
    smoke = resolve_phase_trials(manifest=manifest, phase="smoke", task_id=0, initial_states=states, base_seed=7)

    assert [len(calibration), len(screening), len(final), len(smoke)] == [10, 10, 30, 3]
    assert [trial.initial_state_id for trial in smoke] == [
        trial.initial_state_id for trial in calibration[:3]
    ]
    assert all(trial.partition == "calibration" for trial in smoke)
    assert all(trial.smoke_excluded_from_fitting for trial in smoke)
    assert not any(trial.smoke_excluded_from_fitting for trial in calibration)
    assert set(trial.initial_state_id for trial in calibration).isdisjoint(
        trial.initial_state_id for trial in screening
    )
    assert set(trial.initial_state_id for trial in calibration).isdisjoint(
        trial.initial_state_id for trial in final
    )
    assert set(trial.initial_state_id for trial in screening).isdisjoint(
        trial.initial_state_id for trial in final
    )


def test_runtime_state_hash_mismatch_fails_closed():
    manifest = _manifest()
    states = _states(0)
    states[4] = states[4].copy()
    states[4][0] += 1

    with pytest.raises(ValueError, match="initial-state hash mismatch"):
        resolve_phase_trials(
            manifest=manifest,
            phase="screening",
            task_id=0,
            initial_states=states,
            base_seed=7,
        )


def test_initial_state_hash_captures_order_dtype_shape_and_values():
    states = _states(0)
    baseline = hash_initial_states(states)
    reversed_states = list(reversed(states))
    float64_states = [state.astype(np.float64) for state in states]
    reshaped_states = [state.reshape(3, 1) for state in states]

    assert hash_initial_states(states) == baseline
    assert hash_initial_states(reversed_states) != baseline
    assert hash_initial_states(float64_states) != baseline
    assert hash_initial_states(reshaped_states) != baseline


def test_paired_seed_is_stable_and_excludes_condition_identity():
    kwargs = {
        "base_seed": 7,
        "phase": "screening",
        "task_suite_name": "libero_spatial",
        "task_id": 3,
        "initial_state_id": 21,
        "paired_trial_id": 4,
    }
    clean_arm_seed = derive_paired_episode_seed(**kwargs)
    candidate_arm_seed = derive_paired_episode_seed(**kwargs)

    assert clean_arm_seed == candidate_arm_seed
    assert 0 <= clean_arm_seed < 2**32
    assert derive_paired_episode_seed(**{**kwargs, "paired_trial_id": 5}) != clean_arm_seed
    with pytest.raises(TypeError):
        derive_paired_episode_seed(**kwargs, condition_id="candidate")


def test_nonlegacy_configuration_is_fail_closed(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")

    validate_protocol_configuration(
        phase="legacy",
        task_suite_name="libero_goal",
        num_trials_per_task=999,
        initial_states_path="custom.json",
        manifest_path="",
        reset_rng_each_episode=False,
    )
    validate_protocol_configuration(
        phase="screening",
        task_suite_name="libero_spatial",
        num_trials_per_task=10,
        initial_states_path="DEFAULT",
        manifest_path=str(manifest_path),
        reset_rng_each_episode=True,
    )

    with pytest.raises(ValueError, match="reset_rng_each_episode=True"):
        validate_protocol_configuration(
            phase="screening",
            task_suite_name="libero_spatial",
            num_trials_per_task=10,
            initial_states_path="DEFAULT",
            manifest_path=str(manifest_path),
            reset_rng_each_episode=False,
        )
    with pytest.raises(ValueError, match="requires num_trials_per_task=30"):
        validate_protocol_configuration(
            phase="final",
            task_suite_name="libero_spatial",
            num_trials_per_task=60,
            initial_states_path="DEFAULT",
            manifest_path=str(manifest_path),
            reset_rng_each_episode=True,
        )


class _FakeCuda:
    def __init__(self, events):
        self.events = events

    def is_available(self):
        return False

    def manual_seed_all(self, seed):
        self.events.append(("torch.cuda.manual_seed_all", seed))


class _FakeTorch:
    def __init__(self, events):
        self.events = events
        self.cuda = _FakeCuda(events)

    def manual_seed(self, seed):
        self.events.append(("torch.manual_seed", seed))


class _RecordingEnv:
    def __init__(self, events):
        self.events = events
        self.seed_samples = None

    def seed(self, seed):
        self.events.append(("env.seed", seed))
        self.seed_samples = (random.random(), float(np.random.random()))

    def reset(self):
        self.events.append(("env.reset", None))

    def set_init_state(self, initial_state):
        self.events.append(("env.set_init_state", initial_state))
        return "fixed-observation"

    def get_observation(self):
        self.events.append(("env.get_observation", None))
        return "reset-observation"


def test_paired_rng_is_restored_before_environment_reset_and_fixed_state():
    seed = 123456
    random.seed(seed)
    expected_python = random.random()
    np.random.seed(seed)
    expected_numpy = float(np.random.random())
    events = []
    env = _RecordingEnv(events)

    obs, environment_seed_applied = reset_episode_environment(
        env,
        initial_state="state-17",
        episode_seed=seed,
        torch_module=_FakeTorch(events),
        seed_environment=True,
    )

    assert obs == "fixed-observation"
    assert environment_seed_applied is True
    assert events == [
        ("torch.manual_seed", seed),
        ("env.seed", seed),
        ("env.reset", None),
        ("env.set_init_state", "state-17"),
    ]
    assert env.seed_samples == pytest.approx((expected_python, expected_numpy))


def test_legacy_seed_path_does_not_reseed_environment():
    events = []
    env = _RecordingEnv(events)

    _, environment_seed_applied = reset_episode_environment(
        env,
        initial_state="state-0",
        episode_seed=7,
        torch_module=_FakeTorch(events),
        seed_environment=False,
    )

    assert environment_seed_applied is False
    assert ("env.seed", 7) not in events
    assert events[-2:] == [("env.reset", None), ("env.set_init_state", "state-0")]


def test_environment_seed_requires_episode_seed():
    with pytest.raises(ValueError, match="requires an episode_seed"):
        reset_episode_environment(
            _RecordingEnv([]),
            initial_state=None,
            episode_seed=None,
            torch_module=_FakeTorch([]),
            seed_environment=True,
        )
