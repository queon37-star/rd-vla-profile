from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import scripts.action_latent_audit_lib as audit_lib
from scripts.action_latent_audit_lib import (
    ActionLatentAuditError,
    build_trajectory_bundle,
    build_trajectory_record,
    load_trajectory_bundle,
)
from scripts.preconvergence_trigger_lib import (
    RawPreconvergenceSequence,
    SequenceIdentity,
    save_dataset_bundle,
    sha256_file,
)


def make_sequence(
    prediction_id: int = 0,
    *,
    task_id: int = 0,
    origin: str = "ACTUAL_WARM",
    states: torch.Tensor | None = None,
    actions: torch.Tensor | None = None,
) -> RawPreconvergenceSequence:
    states = states if states is not None else torch.tensor(
        [
            [[1.0, 3.0], [3.0, 5.0]],
            [[2.0, 4.0], [4.0, 6.0]],
            [[4.0, 6.0], [6.0, 8.0]],
            [[7.0, 9.0], [9.0, 11.0]],
        ],
        dtype=torch.float16,
    )
    actions = actions if actions is not None else torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.1, 0.0], [1.11, 0.0]],
        dtype=torch.float16,
    )
    max_iter = int(states.shape[0])
    mse = [None, None] + [0.01] * (max_iter - 1)
    mse[-1] = 0.0001
    phases = [None, None] + [
        "production" if k <= 3 else "shadow_tail" for k in range(2, max_iter + 1)
    ]
    sequence = RawPreconvergenceSequence(
        identity=SequenceIdentity(task_id, 0, prediction_id),
        actual_origin=origin,
        states=states,
        actions=actions,
        action_mse=tuple(mse),
        action_mse_phase=tuple(phases),
        baseline_k=3,
        max_iter=max_iter,
    )
    sequence.validate()
    return sequence


def test_bundle_metrics_and_unavailable_nan_positions_are_exact() -> None:
    sequence = make_sequence()
    states_before = sequence.states.clone()
    actions_before = sequence.actions.clone()
    record = build_trajectory_record(sequence)

    assert record["state_mean"].dtype == torch.float32
    assert torch.equal(record["state_mean"], torch.tensor([[2, 4], [3, 5], [5, 7], [8, 10]], dtype=torch.float32))
    metrics = record["latent_metrics"]
    assert torch.isnan(metrics["delta_rms"][0])
    assert torch.isnan(metrics["second_difference_rms"][:2]).all()
    assert metrics["delta_rms"][1].item() == pytest.approx(1.0)
    assert metrics["delta_mean_abs"][1].item() == pytest.approx(1.0)
    assert metrics["delta_max_abs"][1].item() == pytest.approx(1.0)
    assert metrics["token_or_element_delta_mse_mean"][1].item() == pytest.approx(1.0)
    assert metrics["token_or_element_delta_mse_max"][1].item() == pytest.approx(1.0)
    assert metrics["full_delta_cosine_with_previous"][2].item() == pytest.approx(1.0)
    assert metrics["mean_pooled_delta_cosine_with_previous"][2].item() == pytest.approx(1.0)
    assert metrics["second_difference_rms"][2].item() == pytest.approx(1.0)
    assert torch.equal(sequence.states, states_before)
    assert torch.equal(sequence.actions, actions_before)


def test_current_delta_does_not_depend_on_future_states() -> None:
    sequence = make_sequence()
    before = build_trajectory_record(sequence)
    changed_states = sequence.states.clone()
    changed_states[3] += 1000
    changed = make_sequence(states=changed_states)
    after = build_trajectory_record(changed)
    assert torch.equal(before["state_mean"][:3], after["state_mean"][:3])
    for name in before["latent_metrics"]:
        assert torch.allclose(
            before["latent_metrics"][name][:3],
            after["latent_metrics"][name][:3],
            equal_nan=True,
        )


def test_nonfinite_source_fails_closed() -> None:
    sequence = make_sequence()
    sequence.states[1, 0, 0] = float("inf")
    with pytest.raises((ActionLatentAuditError, ValueError), match="non-finite"):
        build_trajectory_record(sequence)


def test_bundle_manifest_hashes_atomic_outputs_and_refuses_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    source.mkdir()
    save_dataset_bundle(source, {"prediction_count": 2}, [make_sequence(), make_sequence(1, task_id=1, origin="COLD")])
    output = tmp_path / "audit"
    manifest = build_trajectory_bundle(source, output)

    assert manifest["schema_version"] == 1
    assert manifest["prediction_counts_by_origin"] == {"ACTUAL_WARM": 1, "COLD": 1}
    assert manifest["task_count"] == 2
    assert manifest["absolute_input_dataset_manifest_path"] == str((source / "manifest.json").resolve())
    assert manifest["input_dataset_manifest_sha256"] == sha256_file(source / "manifest.json")
    assert manifest["output_bundle_sha256"] == sha256_file(output / "trajectory_bundle.pt")
    saved_manifest, records = load_trajectory_bundle(output)
    assert saved_manifest == json.loads((output / "manifest.json").read_text())
    assert len(records) == 2
    assert not list(output.glob("*.tmp"))
    assert not list(output.glob(".*.tmp"))
    with pytest.raises(ActionLatentAuditError, match="refusing to overwrite"):
        build_trajectory_bundle(source, output)


def test_manifest_write_failure_removes_partial_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "dataset"
    source.mkdir()
    save_dataset_bundle(source, {"prediction_count": 1}, [make_sequence()])
    output = tmp_path / "audit"

    def fail_manifest(*_args, **_kwargs):
        raise OSError("synthetic manifest failure")

    monkeypatch.setattr(audit_lib, "_atomic_write_bytes", fail_manifest)
    with pytest.raises(OSError, match="synthetic manifest failure"):
        build_trajectory_bundle(source, output)
    assert not (output / "trajectory_bundle.pt").exists()
    assert not (output / "manifest.json").exists()
    assert not list(output.glob(".*.tmp"))
