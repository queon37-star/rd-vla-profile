import hashlib
import json
from pathlib import Path

import pytest
import torch

from prismatic.models.scalar_stopping_policy import (
    SCALAR_FEATURE_NAMES,
    SCALAR_POLICY_ARTIFACT_TYPE,
    NonFiniteScalarPolicyError,
    ScalarStoppingPolicyError,
    compute_scalar_stopping_features,
    evaluate_scalar_stopping_policy,
    load_scalar_policy_artifact,
    prepare_scalar_task_policy,
    resolve_scalar_terminal_iteration,
    score_scalar_stopping_policy,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _policy_payload():
    task_to_fold = {
        task_id: task_id // 2
        for task_id in range(10)
    }

    policies = {}

    for task_id in range(10):
        fold_id = task_to_fold[task_id]
        held = [
            candidate
            for candidate in range(10)
            if task_to_fold[candidate] == fold_id
        ]
        training = [
            candidate
            for candidate in range(10)
            if candidate not in held
        ]

        policies[str(task_id)] = {
            "task_id": task_id,
            "outer_fold": fold_id,
            "held_out_task_ids": held,
            "training_task_ids": training,
            "selected_threshold": 0.5,
            "normalizer_mean": torch.zeros(
                7,
                dtype=torch.float32,
            ),
            "normalizer_scale": torch.ones(
                7,
                dtype=torch.float32,
            ),
            "linear_weight": torch.tensor(
                [1.0, 0, 0, 0, 0, 0, 0],
                dtype=torch.float32,
            ),
            "linear_bias": torch.tensor(
                -3.0,
                dtype=torch.float32,
            ),
            "training_seed": task_id,
            "training_row_count": 1,
            "training_prediction_count_for_threshold": 1,
        }

    return {
        "schema_version": 1,
        "artifact_type": SCALAR_POLICY_ARTIFACT_TYPE,
        "policy_name": "test_policy",
        "target_reference": "K_first_authoritative",
        "model_configuration": "scalar_combo",
        "feature_names": list(
            SCALAR_FEATURE_NAMES
        ),
        "minimum_gate_iteration": 3,
        "epsilon": 1e-8,
        "supported_execution_modes": [
            "direct",
            "confirm_next",
        ],
        "task_to_fold": task_to_fold,
        "policies_by_task": policies,
        "provenance": {},
    }


def _write_artifact(directory: Path):
    directory.mkdir()
    artifact = directory / "scalar_policy.pt"
    torch.save(_policy_payload(), artifact)

    digest = _sha256(artifact)
    manifest = {
        "schema_version": 1,
        "artifact_file": artifact.name,
        "artifact_sha256": digest,
        "artifact_type": SCALAR_POLICY_ARTIFACT_TYPE,
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return artifact, digest


def test_compute_scalar_features_matches_manual_formula():
    previous_previous = torch.tensor(
        [[[0.0, 1.0], [2.0, 3.0]]],
        dtype=torch.bfloat16,
    )
    previous = torch.tensor(
        [[[1.0, 2.0], [4.0, 4.0]]],
        dtype=torch.bfloat16,
    )
    current = torch.tensor(
        [[[3.0, 3.0], [5.0, 7.0]]],
        dtype=torch.bfloat16,
    )

    result = compute_scalar_stopping_features(
        current,
        previous,
        previous_previous,
        iteration=3,
        epsilon=1e-8,
    )

    current_fp32 = current.float()
    previous_fp32 = previous.float()
    previous_previous_fp32 = (
        previous_previous.float()
    )

    current_delta = current_fp32 - previous_fp32
    previous_delta = (
        previous_fp32 - previous_previous_fp32
    )

    delta_rms = torch.sqrt(
        torch.mean(current_delta.square())
    )
    previous_delta_rms = torch.sqrt(
        torch.mean(previous_delta.square())
    )
    state_rms = torch.sqrt(
        torch.mean(current_fp32.square())
    )

    current_mean_delta = (
        current_fp32.mean(dim=(0, 1))
        - previous_fp32.mean(dim=(0, 1))
    )
    previous_mean_delta = (
        previous_fp32.mean(dim=(0, 1))
        - previous_previous_fp32.mean(
            dim=(0, 1)
        )
    )

    cosine = torch.dot(
        current_mean_delta,
        previous_mean_delta,
    ) / (
        torch.linalg.vector_norm(
            current_mean_delta
        )
        * torch.linalg.vector_norm(
            previous_mean_delta
        )
    )

    expected = torch.stack(
        (
            torch.tensor(3.0),
            delta_rms,
            previous_delta_rms,
            delta_rms / state_rms,
            delta_rms / previous_delta_rms,
            cosine,
            torch.sqrt(
                torch.mean(
                    (
                        current_delta
                        - previous_delta
                    ).square()
                )
            ),
        )
    )

    assert result.shape == (7,)
    assert result.dtype == torch.float32
    torch.testing.assert_close(
        result.cpu(),
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


def test_artifact_load_prepare_and_score(tmp_path):
    _, digest = _write_artifact(tmp_path / "policy")

    manifest, payload = load_scalar_policy_artifact(
        tmp_path / "policy",
        expected_sha256=digest,
    )

    assert manifest["artifact_sha256"] == digest

    policy = prepare_scalar_task_policy(
        payload,
        8,
        device="cpu",
    )

    features = torch.tensor(
        [3.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
        dtype=torch.float32,
    )

    score = score_scalar_stopping_policy(
        policy,
        features,
    )

    # logit = iteration_k - 3 = 0
    torch.testing.assert_close(
        score,
        torch.tensor(0.5),
    )

    score_value, triggered = (
        evaluate_scalar_stopping_policy(
            policy,
            features,
        )
    )
    assert score_value == pytest.approx(0.5)
    assert triggered is True
    assert policy.task_id == 8
    assert policy.outer_fold == 4


def test_loader_rejects_hash_mismatch(tmp_path):
    artifact, _ = _write_artifact(
        tmp_path / "policy"
    )

    with artifact.open("ab") as stream:
        stream.write(b"corruption")

    with pytest.raises(
        ScalarStoppingPolicyError,
        match="SHA-256 mismatch",
    ):
        load_scalar_policy_artifact(
            tmp_path / "policy"
        )


def test_nonfinite_state_fails_closed():
    state = torch.ones(
        1,
        2,
        3,
        dtype=torch.float32,
    )
    state[0, 0, 0] = float("nan")

    with pytest.raises(
        NonFiniteScalarPolicyError,
        match="non-finite",
    ):
        compute_scalar_stopping_features(
            state,
            torch.zeros_like(state),
            torch.zeros_like(state),
            iteration=3,
        )


@pytest.mark.parametrize(
    "gate,maximum,mode,expected",
    [
        (3, 32, "direct", 3),
        (3, 32, "confirm_next", 4),
        (32, 32, "confirm_next", 32),
        (None, 32, "direct", 32),
        (None, 32, "confirm_next", 32),
    ],
)
def test_resolve_terminal_iteration(
    gate,
    maximum,
    mode,
    expected,
):
    assert (
        resolve_scalar_terminal_iteration(
            gate,
            maximum_iteration=maximum,
            execution_mode=mode,
        )
        == expected
    )


def test_invalid_execution_mode_rejected():
    with pytest.raises(
        ScalarStoppingPolicyError,
        match="unsupported",
    ):
        resolve_scalar_terminal_iteration(
            3,
            maximum_iteration=32,
            execution_mode="unknown",
        )
