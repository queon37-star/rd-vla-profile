import inspect
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.robot.libero import run_libero_eval


def test_equal_returned_actions_hash_identically_and_value_change_differs():
    first = np.arange(56, dtype=np.float32).reshape(8, 7)
    equal = first.copy()
    changed = first.copy()
    changed[3, 4] = np.nextafter(changed[3, 4], np.float32(np.inf))

    first_hash = run_libero_eval._tensor_or_array_sha256(first)

    assert first_hash == run_libero_eval._tensor_or_array_sha256(equal)
    assert first_hash != run_libero_eval._tensor_or_array_sha256(changed)


def test_equal_warm_states_hash_identically_without_mutating_tensors():
    first = torch.arange(64, dtype=torch.float32).reshape(1, 8, 8).to(
        torch.bfloat16
    )
    equal = first.clone()
    first_before = first.clone()
    equal_before = equal.clone()

    first_hash = run_libero_eval._tensor_or_array_sha256(first)
    equal_hash = run_libero_eval._tensor_or_array_sha256(equal)

    assert first_hash == equal_hash
    assert torch.equal(first, first_before)
    assert torch.equal(equal, equal_before)
    assert run_libero_eval.PARITY_HASH_SCHEMA == {
        "schema_version": 1,
        "algorithm": "sha256",
        "dtype": "preserved and included in the canonical header",
        "shape": "preserved and included in the canonical header",
        "byte_order": "little-endian",
        "memory_order": "C-contiguous",
    }


def test_rng_hash_is_repeatable_and_does_not_advance_rng_state():
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()

    first_hash = run_libero_eval._rng_state_sha256()
    second_hash = run_libero_eval._rng_state_sha256()

    assert first_hash == second_hash
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)


def test_parity_hashing_occurs_after_action_latency_is_finalized():
    source = inspect.getsource(run_libero_eval.run_episode)
    parity_marker = source.index(
        "# Parity hashing is logging-only and intentionally begins only"
    )
    final_latency_assignment = source.rfind(
        "action_latency_ms = (time.perf_counter() - action_start)",
        0,
        parity_marker,
    )
    returned_hash = source.index(
        "returned_action_sha256 = (", parity_marker
    )
    next_state_hash = source.index(
        "next_warm_start_state_sha256 = (", parity_marker
    )

    assert final_latency_assignment >= 0
    assert final_latency_assignment < parity_marker < returned_hash
    assert parity_marker < next_state_hash


@pytest.mark.parametrize(
    "phase",
    ("calibration", "screening", "final_holdout"),
)
def test_frozen_protocol_phases_enable_parity_hash_logging(phase):
    assert run_libero_eval._should_log_parity_hashes(
        shadow_full_depth=False,
        evaluation_protocol_phase=phase,
    )


def test_legacy_parity_hash_logging_stays_opt_in_via_shadow_full_depth():
    assert not run_libero_eval._should_log_parity_hashes(
        shadow_full_depth=False,
        evaluation_protocol_phase="legacy",
    )
    assert run_libero_eval._should_log_parity_hashes(
        shadow_full_depth=True,
        evaluation_protocol_phase="legacy",
    )
    assert not run_libero_eval._should_log_parity_hashes(
        shadow_full_depth=False,
        evaluation_protocol_phase="smoke",
    )


def test_calibration_uses_the_shared_protocol_source_commit_path(monkeypatch):
    calls = []

    def source_commit():
        calls.append("called")
        return "exact-source-commit"

    monkeypatch.setattr(run_libero_eval, "current_source_commit", source_commit)

    for phase in ("calibration", "screening", "final_holdout"):
        assert (
            run_libero_eval._resolve_protocol_source_commit(phase)
            == "exact-source-commit"
        )
    assert run_libero_eval._resolve_protocol_source_commit("legacy") is None
    assert calls == ["called", "called", "called"]


def test_calibration_protocol_still_requires_official_ten_state_partition():
    manifest = Path(
        "experiments/robot/libero/manifests/"
        "libero_spatial_official_50_v1.json"
    )
    kwargs = {
        "phase": "calibration",
        "task_suite_name": "libero_spatial",
        "initial_states_path": "DEFAULT",
        "manifest_path": str(manifest),
        "reset_rng_each_episode": True,
    }

    run_libero_eval.validate_protocol_configuration(
        num_trials_per_task=10,
        **kwargs,
    )
    with pytest.raises(ValueError, match="requires num_trials_per_task=10"):
        run_libero_eval.validate_protocol_configuration(
            num_trials_per_task=9,
            **kwargs,
        )
