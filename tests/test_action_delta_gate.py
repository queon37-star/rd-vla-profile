import hashlib
import json
from pathlib import Path

import pytest
import torch

from prismatic.models.action_delta_gate import (
    ACTION_DELTA_GATE_ARTIFACT_TYPE,
    ACTION_DELTA_GATE_CALIBRATION_METHOD,
    ACTION_DELTA_GATE_MODEL_TYPE,
    ActionDeltaGateError,
    PreparedActionDeltaGate,
    load_action_delta_gate_artifact,
    prepare_action_delta_gate,
    score_action_delta_gate,
    validate_action_delta_gate_artifact,
)


def artifact_payload():
    return {
        "schema_version": 1,
        "artifact_type": ACTION_DELTA_GATE_ARTIFACT_TYPE,
        "model_type": ACTION_DELTA_GATE_MODEL_TYPE,
        "hidden_dim": 896,
        "action_dim": 7,
        "action_chunk_len": 8,
        "held_out_task_ids": [4, 5],
        "outer_fold": 4,
        "threshold": 0.001,
        "x_mean": torch.zeros(896),
        "x_std": torch.ones(896),
        "y_mean": torch.zeros(7),
        "y_std": torch.ones(7),
        "linear_weight": torch.zeros(7, 896),
        "linear_bias": torch.zeros(7),
        "delta_quantization_dtype": "bfloat16",
        "training_seed": 1011,
        "epochs": 60,
        "batch_size": 512,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "calibration_method": ACTION_DELTA_GATE_CALIBRATION_METHOD,
        "training_row_count": 10,
        "provenance": {},
    }


def write_artifact(directory: Path):
    directory.mkdir()
    artifact = directory / "action_delta_gate.pt"
    torch.save(artifact_payload(), artifact)
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": ACTION_DELTA_GATE_ARTIFACT_TYPE,
                "artifact_file": artifact.name,
                "artifact_sha256": digest,
                "outer_fold": 4,
                "held_out_task_ids": [4, 5],
            }
        ),
        encoding="utf-8",
    )
    return artifact, digest


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda p: p.update(linear_weight=torch.zeros(7, 895)), "shape mismatch"),
        (lambda p: p["x_mean"].fill_(float("nan")), "non-finite"),
        (lambda p: p["x_std"].zero_(), "x_std must be positive"),
        (lambda p: p.update(model_type="delta_plus_anchor"), "model type mismatch"),
        (lambda p: p.update(outer_fold=3), "outer_fold must equal 4"),
        (lambda p: p.update(held_out_task_ids=[3, 4]), "held-out task identity"),
    ],
)
def test_artifact_schema_rejects_invalid_contract(mutation, message):
    payload = artifact_payload()
    mutation(payload)
    with pytest.raises(ActionDeltaGateError, match=message):
        validate_action_delta_gate_artifact(payload)


def test_loader_verifies_hash_and_weights_only_payload(tmp_path):
    artifact, digest = write_artifact(tmp_path / "gate")
    manifest, payload = load_action_delta_gate_artifact(
        tmp_path / "gate", expected_sha256=digest
    )
    assert manifest["artifact_sha256"] == digest
    assert payload["model_type"] == ACTION_DELTA_GATE_MODEL_TYPE

    with artifact.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ActionDeltaGateError, match="SHA-256 mismatch"):
        load_action_delta_gate_artifact(tmp_path / "gate", expected_sha256=digest)


def test_prepare_rejects_training_task():
    with pytest.raises(ActionDeltaGateError, match="not held out"):
        prepare_action_delta_gate(artifact_payload(), device="cpu", task_id=3)


def test_scorer_requantizes_fp32_delta_through_bfloat16():
    gate = PreparedActionDeltaGate(
        schema_version=1,
        artifact_type=ACTION_DELTA_GATE_ARTIFACT_TYPE,
        model_type=ACTION_DELTA_GATE_MODEL_TYPE,
        hidden_dim=2,
        action_dim=2,
        action_chunk_len=2,
        held_out_task_ids=(4, 5),
        outer_fold=4,
        threshold=1.0,
        x_mean=torch.zeros(2),
        x_std=torch.ones(2),
        y_mean=torch.zeros(2),
        y_std=torch.ones(2),
        linear_weight=torch.eye(2),
        linear_bias=torch.zeros(2),
        delta_quantization_dtype="bfloat16",
        training_seed=1011,
        calibration_method=ACTION_DELTA_GATE_CALIBRATION_METHOD,
    )
    anchor = torch.tensor([[[1.0, -2.0], [4.0, 8.0]]], dtype=torch.float32)
    current = torch.tensor(
        [[[1.00391, -1.99217], [4.03126, 8.06251]]], dtype=torch.float32
    )

    quantized_delta = (current.float() - anchor.float()).to(torch.bfloat16).float()
    pure_fp32_delta = current.float() - anchor.float()
    expected = quantized_delta.square().mean()
    score = score_action_delta_gate(gate, anchor, current)

    torch.testing.assert_close(score, expected, rtol=0, atol=0)
    assert not torch.equal(quantized_delta, pure_fp32_delta)

