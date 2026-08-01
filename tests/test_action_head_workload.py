import types

import pytest
import torch

from prismatic.models.action_head_workload import (
    ActionHeadWorkloadError,
    build_action_head_workload,
    load_action_head_workload,
    save_action_head_workload,
)
from prismatic.models.action_heads import ActionHeadRecurrent, RecurrentConfigInternal


def _identity():
    return {
        "task_id": 2,
        "episode_id": 4,
        "paired_trial_id": 4,
        "prediction_step": 1,
        "initial_state_id": 18,
        "episode_seed": 1234,
    }


def _tensors():
    incoming = torch.arange(16, dtype=torch.float32).reshape(1, 2, 8)
    return {
        "actions_hidden_states": torch.zeros(1, 1, 513, 8),
        "proprio_input": torch.zeros(1, 8),
        "proprio_features": torch.zeros(1, 1, 8),
        "incoming_warm_start_state": incoming,
        "selected_initial_state": incoming.clone(),
    }


def test_workload_round_trip_is_cpu_finite_contiguous_and_hashed(tmp_path):
    workload = build_action_head_workload(**_tensors(), actual_origin="ACTUAL_WARM")
    path = tmp_path / "workload.pt"

    digest = save_action_head_workload(path, workload, identity=_identity())
    loaded = load_action_head_workload(
        path,
        expected_sha256=digest,
        expected_identity=_identity(),
        expected_origin="ACTUAL_WARM",
    )

    assert loaded["identity"] == _identity()
    assert all(
        tensor is None or (tensor.device.type == "cpu" and tensor.is_contiguous())
        for tensor in loaded["tensors"].values()
    )
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        save_action_head_workload(path, workload, identity=_identity())


def test_workload_rejects_noncontiguous_production_layout():
    tensors = _tensors()
    tensors["actions_hidden_states"] = tensors["actions_hidden_states"].transpose(2, 3)

    with pytest.raises(ActionHeadWorkloadError, match="must be contiguous"):
        build_action_head_workload(**tensors, actual_origin="ACTUAL_WARM")


def test_workload_rejects_origin_cache_mismatch():
    tensors = _tensors()
    tensors["selected_initial_state"] = tensors["selected_initial_state"] + 1
    with pytest.raises(ActionHeadWorkloadError, match="exactly match"):
        build_action_head_workload(**tensors, actual_origin="ACTUAL_WARM")


def test_workload_sha_validation_rejects_tampering(tmp_path):
    workload = build_action_head_workload(**_tensors(), actual_origin="ACTUAL_WARM")
    path = tmp_path / "workload.pt"
    digest = save_action_head_workload(path, workload, identity=_identity())
    path.write_bytes(path.read_bytes() + b"tamper")

    with pytest.raises(ActionHeadWorkloadError, match="SHA-256 mismatch"):
        load_action_head_workload(path, expected_sha256=digest)


def test_action_head_wrapper_captures_exact_accepted_warm_state():
    cfg = RecurrentConfigInternal(
        hidden_dim=8,
        num_heads=1,
        recurrent_vlm_layers=(0,),
        coda_vlm_layers=(),
        action_chunk_len=2,
        action_dim=2,
        mean_recurrence=2,
        backprop_depth=1,
        random_iterations=False,
    )
    head = ActionHeadRecurrent(hidden_dim=8, action_dim=2, cfg=cfg).eval()
    incoming = torch.arange(16, dtype=torch.float32).reshape(1, 2, 8)

    def fake_forward(self, h_a, h_t, p, **kwargs):
        selected = kwargs["warm_start_state"].detach().clone()
        self.last_inference_metadata = {
            "next_warm_start_state": selected.clone(),
            "warm_start": {"state_used": True},
            "_workload_selected_initial_state": selected,
        }
        return torch.zeros(1, 2, 2), 2, 0.0

    head.model.forward = types.MethodType(fake_forward, head.model)
    hidden = torch.zeros(1, 1, 513, 8)
    proprio = torch.zeros(1, 8)
    projector = torch.nn.Identity()

    result = head.predict_action(
        hidden,
        proprio=proprio,
        proprio_projector=projector,
        warm_start_state=incoming,
        enable_warm_start=True,
        warm_start_source="midpoint",
        convergence_strategy="adjacent_action_mse",
        capture_action_head_workload=True,
    )

    assert result[1:] == (2, 0.0)
    workload = head.model.last_inference_metadata["action_head_workload"]
    assert workload["actual_origin"] == "ACTUAL_WARM"
    torch.testing.assert_close(
        workload["tensors"]["incoming_warm_start_state"],
        workload["tensors"]["selected_initial_state"],
        rtol=0,
        atol=0,
    )
