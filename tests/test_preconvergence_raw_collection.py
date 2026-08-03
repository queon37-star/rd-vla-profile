import copy
import inspect
import json
import types

import pytest
import torch

from experiments.robot.libero.raw_preconvergence_trace import (
    RAW_PRECONVERGENCE_SCHEMA_VERSION,
    RawPreconvergenceShardWriter,
    RawPreconvergenceTraceError,
    build_prediction_payload,
    load_and_validate_manifests,
)
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.models.action_heads import (
    ActionHeadRecurrent,
    RecurrentConfigInternal,
    VLARecurrent,
)
from scripts.preconvergence_trigger_lib import (
    build_training_batch,
    load_raw_manifest_sequences,
)
from scripts.validate_preconvergence_raw_shards import main as validate_raw_main


def _tiny_model():
    cfg = RecurrentConfigInternal(
        hidden_dim=4,
        num_heads=1,
        recurrent_vlm_layers=(0,),
        coda_vlm_layers=(),
        action_chunk_len=2,
        action_dim=2,
        mean_recurrence=3,
        backprop_depth=1,
        random_iterations=False,
    )
    model = VLARecurrent(cfg).eval()
    calls = {"recurrent": 0, "output": 0}

    def init_state(self, batch_size, device, dtype):
        return torch.zeros(batch_size, 2, 4, device=device, dtype=dtype)

    def run_one_iteration(self, state, *args):
        calls["recurrent"] += 1
        return state + 1

    def get_output(self, state, *args, profile=False):
        calls["output"] += 1
        return state[..., :2].clone()

    model.init_state = types.MethodType(init_state, model)
    model._run_one_iteration = types.MethodType(run_one_iteration, model)
    model._get_output = types.MethodType(get_output, model)
    model.test_calls = calls
    return model


def _run(model, *, collect):
    inputs = (
        torch.zeros(1, 1, 1, 4, dtype=torch.bfloat16),
        torch.zeros(1, 1, 1, 4, dtype=torch.bfloat16),
        torch.zeros(1, 1, 4, dtype=torch.bfloat16),
    )
    return model(
        *inputs,
        convergence_strategy="adjacent_action_mse",
        kl_thresh=2.0,
        max_iter=5,
        warm_start_state=torch.zeros(1, 2, 4, dtype=torch.bfloat16),
        enable_warm_start=True,
        warm_start_source="midpoint",
        use_cached_final_output=True,
        use_latent_precheck=False,
        latent_precheck_mode="off",
        latent_precheck_trace_level="off",
        shadow_full_depth=True,
        collect_preconvergence_raw_shadow=collect,
        preconvergence_raw_shadow_max_depth=5,
    )


def test_raw_collection_is_bitwise_noninterfering_and_uses_existing_shadow_tail():
    baseline_model = _tiny_model()
    collected_model = _tiny_model()

    torch.manual_seed(91)
    baseline = _run(baseline_model, collect=False)
    baseline_rng = torch.get_rng_state().clone()
    baseline_debug = copy.deepcopy(baseline_model.last_recurrence_debug)
    baseline_warm = baseline_model.last_inference_metadata["next_warm_start_state"].clone()

    torch.manual_seed(91)
    collected = _run(collected_model, collect=True)
    collected_rng = torch.get_rng_state().clone()
    collected_debug = copy.deepcopy(collected_model.last_recurrence_debug)
    collected_warm = collected_model.last_inference_metadata["next_warm_start_state"]

    assert torch.equal(collected[0], baseline[0])
    assert collected[1:] == baseline[1:]
    assert collected_debug == baseline_debug
    assert torch.equal(collected_warm, baseline_warm)
    assert torch.equal(collected_rng, baseline_rng)
    assert baseline_model.test_calls == collected_model.test_calls == {
        "recurrent": 5,
        "output": 5,
    }
    assert "preconvergence_raw_shadow" not in baseline_model.last_inference_metadata

    raw = collected_model.last_inference_metadata["preconvergence_raw_shadow"]
    assert raw["production_terminal_k"] == 2
    assert raw["valid_trajectory_length"] == raw["maximum_shadow_depth"] == 5
    assert raw["tensors"]["states"].shape == (5, 1, 2, 4)
    assert raw["tensors"]["actions"].shape == (5, 1, 2, 2)
    assert raw["tensors"]["states"].dtype == torch.bfloat16
    assert raw["tensors"]["actions"].dtype == torch.bfloat16
    assert raw["action_mse_source"][2:] == [
        "production_native_bf16",
        "shadow_tail_fp32",
        "shadow_tail_fp32",
        "shadow_tail_fp32",
    ]


def test_raw_collection_is_explicit_and_requires_the_clean_existing_shadow_path():
    model = _tiny_model()
    inputs = (
        torch.zeros(1, 1, 1, 4),
        torch.zeros(1, 1, 1, 4),
        torch.zeros(1, 1, 4),
    )
    with pytest.raises(ValueError, match="requires shadow_full_depth"):
        model(
            *inputs,
            convergence_strategy="adjacent_action_mse",
            max_iter=5,
            use_latent_precheck=False,
            latent_precheck_mode="off",
            collect_preconvergence_raw_shadow=True,
            preconvergence_raw_shadow_max_depth=5,
        )


@pytest.mark.parametrize(
    "callable_obj",
    [
        VLARecurrent.forward,
        ActionHeadRecurrent.predict_action,
        OpenVLAForActionPrediction._regression_or_discrete_prediction,
        OpenVLAForActionPrediction.predict_action,
    ],
)
def test_raw_collection_public_plumbing_is_disabled_by_default(callable_obj):
    parameters = inspect.signature(callable_obj).parameters
    assert parameters["collect_preconvergence_raw_shadow"].default is False
    assert parameters["preconvergence_raw_shadow_max_depth"].default == 32


def _raw_model_payload():
    states = torch.arange(5 * 1 * 2 * 4, dtype=torch.float32).reshape(5, 1, 2, 4).to(torch.bfloat16)
    actions = torch.arange(5 * 1 * 2 * 2, dtype=torch.float32).reshape(5, 1, 2, 2).to(torch.bfloat16)
    return {
        "actual_origin": "ACTUAL_WARM",
        "production_terminal_k": 4,
        "maximum_shadow_depth": 5,
        "valid_trajectory_length": 5,
        "action_mse_threshold": 0.001,
        "tensors": {"states": states, "actions": actions},
        "production_iteration_mse": [0.02, 0.005, 0.0005],
        "action_mse": [None, None, 0.02, 0.005, 0.0005, 0.0001],
        "action_mse_phase": [None, None, "production", "production", "production", "shadow_tail"],
        "action_mse_source": [None, None, "production_native_bf16", "production_native_bf16", "production_native_bf16", "shadow_tail_fp32"],
    }


def _prediction(prediction_id=0):
    return build_prediction_payload(
        _raw_model_payload(),
        task_id=0,
        task_name="tiny task",
        episode_id=0,
        timestep=10 + prediction_id,
        prediction_id=prediction_id,
        protocol_identity={"paired_trial_id": "trial-0"},
        warm_start_metadata={"state_used": True, "source": "midpoint"},
        checkpoint={"path": "/checkpoint", "sha256": "checkpoint-hash", "files": []},
        source_commit="abc123",
        run_identity={"run_id": "run-1", "seed": 7},
        returned_action_sha256="action-hash",
        rng_state_before_sha256="rng-before",
        rng_state_after_sha256="rng-after",
    )


def test_grouped_shards_manifest_validation_and_dataset_builder(tmp_path):
    writer = RawPreconvergenceShardWriter(
        tmp_path / "raw",
        shard_size=2,
        maximum_shadow_depth=5,
        source_commit="abc123",
        checkpoint={"path": "/checkpoint", "sha256": "checkpoint-hash", "files": []},
        run_identity={"run_id": "run-1", "seed": 7},
    )
    for prediction_id in range(3):
        writer.add(_prediction(prediction_id))
    manifest_path = writer.finalize()
    manifest = json.loads(manifest_path.read_text())

    assert manifest["schema_version"] == RAW_PRECONVERGENCE_SCHEMA_VERSION
    assert manifest["complete"] is True
    assert manifest["prediction_count"] == 3
    assert manifest["shard_count"] == 2
    assert manifest["origin_counts"] == {"ACTUAL_WARM": 3}
    compact, predictions = load_and_validate_manifests([manifest_path])
    assert compact["prediction_count"] == len(predictions) == 3
    assert len({tuple(item["identity"][field] for field in ("task_id", "episode_id", "prediction_id")) for item in predictions}) == 3

    metadata, sequences = load_raw_manifest_sequences([manifest_path])
    assert metadata["raw_manifest_schema_version"] == 2
    assert len(sequences) == 3
    batch = build_training_batch(sequences, origin="ACTUAL_WARM")
    assert len(batch.rows) == 3
    assert all(row.k == 3 and row.k_action == 4 and row.label == 1 for row in batch.rows)
    assert torch.isfinite(batch.auxiliary_targets).all()
    assert all(row.k < row.k_action for row in batch.rows)

    step_records = []
    for prediction_id in range(3):
        step_records.append(
            {
                "task_id": 0,
                "episode_id": 0,
                "prediction_step": prediction_id,
                "K_t": 4,
                "returned_action_sha256": "action-hash",
                "next_warm_start_state_sha256": "warm-hash",
                "iteration_mse": [0.02, 0.005, 0.0005],
                "stop_reason": "adjacent_action_mse",
                "canonical_stop_reason": "adjacent_action_mse",
                "cached_final_matches_returned": True,
                "numerical_retry_attempted": False,
                "numerical_retry_succeeded": None,
                "rng_state_before_action_sha256": "before",
                "rng_state_after_action_sha256": "after",
            }
        )
    on_steps = tmp_path / "on_steps.jsonl"
    off_steps = tmp_path / "off_steps.jsonl"
    serialized_steps = "".join(json.dumps(item) + "\n" for item in step_records)
    on_steps.write_text(serialized_steps)
    off_steps.write_text(serialized_steps)
    report_path = tmp_path / "validation.json"
    assert validate_raw_main(
        [
            "--manifest",
            str(manifest_path),
            "--step-log",
            str(on_steps),
            "--parity-step-log",
            str(off_steps),
            "--output",
            str(report_path),
        ]
    ) == 0
    report = json.loads(report_path.read_text())
    assert report["production_parity_prediction_count"] == 3
    assert report["missing_prediction_identity_count"] == 0
    assert report["duplicate_prediction_identity_count"] == 0
    assert report["post_convergence_rows_included"] == 0

    with pytest.raises(FileExistsError, match="non-empty"):
        RawPreconvergenceShardWriter(
            tmp_path / "raw",
            shard_size=2,
            maximum_shadow_depth=5,
            source_commit="abc123",
            checkpoint={},
            run_identity={},
        )


def test_partial_and_corrupt_collections_are_rejected(tmp_path):
    partial = RawPreconvergenceShardWriter(
        tmp_path / "partial",
        shard_size=2,
        maximum_shadow_depth=5,
        source_commit="abc123",
        checkpoint={},
        run_identity={},
    )
    with pytest.raises(RawPreconvergenceTraceError, match="partial"):
        load_and_validate_manifests([partial.manifest_path])

    duplicate_writer = RawPreconvergenceShardWriter(
        tmp_path / "duplicate",
        shard_size=2,
        maximum_shadow_depth=5,
        source_commit="abc123",
        checkpoint={"path": "/checkpoint", "sha256": "checkpoint-hash", "files": []},
        run_identity={"run_id": "run-1", "seed": 7},
    )
    duplicate_writer.add(_prediction())
    with pytest.raises(RawPreconvergenceTraceError, match="duplicate"):
        duplicate_writer.add(_prediction())

    writer = RawPreconvergenceShardWriter(
        tmp_path / "corrupt",
        shard_size=1,
        maximum_shadow_depth=5,
        source_commit="abc123",
        checkpoint={},
        run_identity={},
    )
    writer.add(_prediction())
    manifest_path = writer.finalize()
    shard_path = tmp_path / "corrupt" / "raw_shadow_00000.pt"
    shard_path.write_bytes(shard_path.read_bytes() + b"corrupt")
    with pytest.raises(RawPreconvergenceTraceError, match="corrupt"):
        load_and_validate_manifests([manifest_path])
