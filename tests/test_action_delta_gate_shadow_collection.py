import copy
import json
import types

import numpy as np
import pytest
import torch

import prismatic.models.action_heads as action_heads_module
from experiments.robot.libero.action_delta_gate_shadow_collection import (
    ActionDeltaGateShadowCollectionError,
    ActionDeltaGateShadowWriter,
    _CROSS_DEVICE_FP32_REDUCTION_ATOL,
    _CROSS_DEVICE_FP32_REDUCTION_RTOL,
    _validate_transition,
    build_shadow_prediction_payload,
    globally_unique_trajectory_id,
    load_action_delta_gate_shadow_collection,
)
from experiments.robot.libero.run_libero_eval import GenerateConfig, validate_config
from prismatic.models.action_delta_gate import (
    ACTION_DELTA_GATE_ARTIFACT_TYPE,
    ACTION_DELTA_GATE_CALIBRATION_METHOD,
    ACTION_DELTA_GATE_MODEL_TYPE,
    PreparedActionDeltaGate,
    load_action_delta_gate_artifact,
    prepare_action_delta_gate,
    prepare_action_delta_gate_shadow,
)
from prismatic.models.action_delta_gate_shadow import (
    ACTION_DELTA_GATE_SHADOW_FEATURE_NAMES,
    validate_action_delta_gate_shadow_configuration,
)
from prismatic.models.action_heads import RecurrentConfigInternal, VLARecurrent
from scripts.coda_anchor_feasibility.explore_false_safe_signals import (
    build_runtime_features,
)


def make_gate(*, predicted_delta=0.0, threshold=0.1):
    return PreparedActionDeltaGate(
        schema_version=1,
        artifact_type=ACTION_DELTA_GATE_ARTIFACT_TYPE,
        model_type=ACTION_DELTA_GATE_MODEL_TYPE,
        hidden_dim=4,
        action_dim=2,
        action_chunk_len=2,
        held_out_task_ids=(4, 5),
        outer_fold=4,
        threshold=threshold,
        x_mean=torch.zeros(4),
        x_std=torch.ones(4),
        y_mean=torch.full((2,), predicted_delta),
        y_std=torch.ones(2),
        linear_weight=torch.zeros(2, 4),
        linear_bias=torch.zeros(2),
        delta_quantization_dtype="bfloat16",
        training_seed=1011,
        calibration_method=ACTION_DELTA_GATE_CALIBRATION_METHOD,
    )


def make_model():
    cfg = RecurrentConfigInternal(
        hidden_dim=4,
        num_heads=1,
        recurrent_vlm_layers=(0,),
        coda_vlm_layers=(),
        action_chunk_len=2,
        action_dim=2,
        mean_recurrence=6,
        backprop_depth=1,
        random_iterations=False,
    )
    model = VLARecurrent(cfg).eval()
    model.test_iteration = 0
    model.test_coda_calls = 0

    def init_state(self, batch_size, device, dtype):
        return torch.zeros(batch_size, 2, 4, device=device, dtype=dtype)

    def run_one_iteration(self, state, *args):
        self.test_iteration += 1
        return state + 1

    def get_output(self, state, *args, profile=False):
        self.test_coda_calls += 1
        return state[..., :2].clone()

    model.init_state = types.MethodType(init_state, model)
    model._run_one_iteration = types.MethodType(run_one_iteration, model)
    model._get_output = types.MethodType(get_output, model)
    return model


def inputs():
    h_a = torch.zeros(1, 1, 1, 4, dtype=torch.bfloat16)
    return h_a, torch.zeros_like(h_a), torch.zeros(1, 1, 4, dtype=torch.bfloat16)


def kwargs(gate, *, collect):
    return {
        "convergence_strategy": "adjacent_action_mse",
        "enable_warm_start": True,
        "warm_start_source": "midpoint",
        "warm_start_min_iter": 2,
        "use_cached_final_output": True,
        "latent_precheck_mode": "off",
        "latent_precheck_trace_level": "off",
        "use_latent_precheck": False,
        "use_action_delta_gate": False,
        "action_delta_gate": gate,
        "action_delta_gate_min_terminal_iter": 5,
        "collect_action_delta_gate_shadow": collect,
        "kl_thresh": 0.001,
        "max_iter": 6,
    }


def test_shadow_predictor_never_changes_warm_only_output_k_or_exact_coda():
    baseline = make_model()
    shadow = copy.deepcopy(baseline)
    gate = make_gate(predicted_delta=0.0, threshold=1.0)
    warm = torch.zeros(1, 2, 4, dtype=torch.bfloat16)

    baseline_result = baseline(*inputs(), warm_start_state=warm, **kwargs(gate, collect=False))
    shadow_result = shadow(*inputs(), warm_start_state=warm, **kwargs(gate, collect=True))

    torch.testing.assert_close(baseline_result[0], shadow_result[0], rtol=0, atol=0)
    assert baseline_result[1:] == shadow_result[1:]
    assert baseline.test_coda_calls == shadow.test_coda_calls == 6
    assert baseline.last_recurrence_debug["iteration_mse"] == shadow.last_recurrence_debug["iteration_mse"]
    assert shadow.last_recurrence_debug["stop_reason"] == "max_iter"
    payload = shadow.last_inference_metadata["action_delta_gate_shadow"]
    assert [row["terminal_iteration"] for row in payload["transitions"]] == [5, 6]
    assert all(row["predicted_trigger"] for row in payload["transitions"])
    assert payload["production_trace"]["K_t"] == 6
    assert payload["production_trace"]["exact_coda_call_count"] == 6
    torch.testing.assert_close(
        payload["production_trace"]["returned_normalized_action"],
        payload["production_trace"]["exact_coda_outputs"][-1],
        rtol=0,
        atol=0,
    )


def test_shadow_anchor_updates_before_eligibility_and_terminal_five_uses_s4_to_s5():
    model = make_model()
    gate = make_gate(predicted_delta=0.0, threshold=1.0)
    warm = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
    model(*inputs(), warm_start_state=warm, **kwargs(gate, collect=True))

    row = model.last_inference_metadata["action_delta_gate_shadow"]["transitions"][0]
    assert row["anchor_iteration"] == 4
    assert row["terminal_iteration"] == 5
    torch.testing.assert_close(
        row["tensors"]["anchor_state"],
        torch.full((1, 2, 4), 4.0, dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        row["tensors"]["current_state"],
        torch.full((1, 2, 4), 5.0, dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        row["tensors"]["anchor_action"],
        torch.full((1, 2, 2), 4.0, dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        row["tensors"]["exact_terminal_action"],
        torch.full((1, 2, 2), 5.0, dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        row["tensors"]["previous_latent_delta_bfloat16"],
        torch.ones(1, 2, 4, dtype=torch.bfloat16),
    )
    assert row["features"]["previous_predicted_score"] is None
    assert row["features"]["previous_latent_delta_rms"] == 1.0
    assert row["features"]["latent_delta_cosine_current_previous"] == pytest.approx(
        1.0
    )
    assert row["exact_adjacent_action_mse"] == 1.0


def test_shadow_exact_safe_stop_returns_current_exact_coda_not_predicted_result():
    model = make_model()
    gate = make_gate(predicted_delta=99.0, threshold=10000.0)
    outputs = {
        1: torch.zeros(1, 2, 2),
        2: torch.ones(1, 2, 2),
        3: torch.full((1, 2, 2), 2.0),
        4: torch.full((1, 2, 2), 3.0),
        5: torch.full((1, 2, 2), 3.01),
    }

    def get_output(self, state, *args, profile=False):
        self.test_coda_calls += 1
        return outputs[self.test_iteration].clone()

    model._get_output = types.MethodType(get_output, model)
    warm = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
    result = model(*inputs(), warm_start_state=warm, **kwargs(gate, collect=True))

    assert result[1] == 5
    assert model.test_coda_calls == 5
    torch.testing.assert_close(result[0], outputs[5], rtol=0, atol=0)
    assert model.last_recurrence_debug["stop_reason"] == "adjacent_action_mse"
    assert model.last_recurrence_debug["use_action_delta_gate"] is False
    assert model.last_recurrence_debug["action_delta_gate_triggered"] is False


def test_cold_origin_never_scores_shadow_predictor(monkeypatch):
    model = make_model()
    calls = 0
    original = action_heads_module.evaluate_action_delta_gate

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(action_heads_module, "evaluate_action_delta_gate", counted)
    model(*inputs(), warm_start_state=None, **kwargs(make_gate(), collect=True))
    payload = model.last_inference_metadata["action_delta_gate_shadow"]
    assert calls == 0
    assert payload["collection_applied"] is False
    assert payload["ineligible_reason"] == "cold_origin"
    assert payload["transitions"] == []


def test_shadow_uses_one_predictor_forward_per_eligible_row(monkeypatch):
    model = make_model()
    calls = 0
    original = action_heads_module.evaluate_action_delta_gate

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(action_heads_module, "evaluate_action_delta_gate", counted)
    warm = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
    model(*inputs(), warm_start_state=warm, **kwargs(make_gate(), collect=True))
    rows = model.last_inference_metadata["action_delta_gate_shadow"]["transitions"]
    assert calls == len(rows) == 2


def test_collected_features_match_existing_offline_runtime_feature_builder():
    model = make_model()
    warm = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
    gate = make_gate(predicted_delta=0.25, threshold=1.0)
    model(*inputs(), warm_start_state=warm, **kwargs(gate, collect=True))
    rows = model.last_inference_metadata["action_delta_gate_shadow"]["transitions"]

    delta_states = torch.cat(
        [row["tensors"]["latent_delta_bfloat16"] for row in rows], dim=0
    )
    predicted = torch.cat(
        [row["tensors"]["predicted_delta_action"] for row in rows], dim=0
    )
    offline = build_runtime_features(
        delta_states,
        predicted,
        trajectory_ids=np.zeros(len(rows), dtype=np.int64),
        ks=np.asarray([row["anchor_iteration"] for row in rows]),
        gate_threshold=gate.threshold,
        x_mean=gate.x_mean,
        x_std=gate.x_std,
        prefix_steps=2,
    )
    assert set(offline) == set(ACTION_DELTA_GATE_SHADOW_FEATURE_NAMES)
    preeligible_latent_history = {
        "previous_latent_delta_rms",
        "latent_delta_rms_ratio_current_to_previous",
        "latent_delta_cosine_current_previous",
        "latent_delta_second_difference_rms",
    }
    for index, row in enumerate(rows):
        for name in ACTION_DELTA_GATE_SHADOW_FEATURE_NAMES:
            if index == 0 and name in preeligible_latent_history:
                # The deployment collector preserves S3->S4 even though the
                # predictor is intentionally not evaluated before terminal 5.
                continue
            actual = row["features"][name]
            expected = offline[name][index]
            if np.isnan(expected):
                assert actual is None
            else:
                assert actual == pytest.approx(float(expected), rel=1e-6, abs=1e-8)


def _prediction_payload(*, predicted_delta=0.0):
    model = make_model()
    warm = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
    model(
        *inputs(),
        warm_start_state=warm,
        **kwargs(make_gate(predicted_delta=predicted_delta), collect=True),
    )
    manifest_hash = "a" * 64
    return build_shadow_prediction_payload(
        model.last_inference_metadata["action_delta_gate_shadow"],
        task_id=0,
        task_name="task zero",
        episode_id=0,
        initial_state_id=7,
        paired_trial_id=0,
        episode_seed=123,
        prediction_id=0,
        environment_timestep=10,
        initial_state_manifest_sha256=manifest_hash,
        protocol_identity={"initial_state_manifest_sha256": manifest_hash},
        warm_start_metadata={"source_iteration": 3, "source_K": 6},
        returned_action=np.zeros((2, 2), dtype=np.float32),
    )


def _set_authoritative_score(row, score, *, threshold=None):
    score = float(score)
    if threshold is not None:
        row["gate_threshold"] = float(threshold)
    row["gate_score"] = score
    row["predicted_full_mse"] = score
    row["features"]["predicted_action_delta_mse"] = score
    row["residual"] = float(row["exact_adjacent_action_mse"] - score)
    row["predicted_trigger"] = bool(score <= row["gate_threshold"])
    row["false_safe"] = bool(row["predicted_trigger"] and not row["exact_safe"])


def _set_authoritative_exact_mse(row, exact_mse):
    exact_mse = float(exact_mse)
    row["exact_adjacent_action_mse"] = exact_mse
    row["residual"] = float(exact_mse - row["gate_score"])
    row["exact_safe"] = bool(exact_mse < row["recurrence_mse_threshold"])
    row["false_safe"] = bool(row["predicted_trigger"] and not row["exact_safe"])


def test_cpu_score_replay_accepts_tiny_difference_and_preserves_authoritative_decision():
    row = copy.deepcopy(
        _prediction_payload(predicted_delta=0.25)["transitions"][0]
    )
    cpu_replay = float(
        row["tensors"]["predicted_delta_action"].float().square().mean().item()
    )
    difference = max(
        _CROSS_DEVICE_FP32_REDUCTION_ATOL * 0.5,
        abs(cpu_replay) * _CROSS_DEVICE_FP32_REDUCTION_RTOL * 0.5,
    )
    authoritative_score = cpu_replay + difference
    threshold = cpu_replay + difference * 0.5
    _set_authoritative_score(row, authoritative_score, threshold=threshold)

    assert cpu_replay <= threshold < authoritative_score
    assert row["predicted_trigger"] is False
    _validate_transition(row)

    assert row["gate_score"] == authoritative_score
    assert row["predicted_trigger"] is False
    assert cpu_replay <= row["gate_threshold"]


def test_cpu_score_replay_rejects_material_difference_with_diagnostics():
    row = copy.deepcopy(
        _prediction_payload(predicted_delta=0.25)["transitions"][0]
    )
    cpu_replay = float(
        row["tensors"]["predicted_delta_action"].float().square().mean().item()
    )
    _set_authoritative_score(row, cpu_replay + 1e-3, threshold=1.0)

    with pytest.raises(
        ActionDeltaGateShadowCollectionError,
        match="predicted action-delta score CPU integrity replay mismatch",
    ) as exc_info:
        _validate_transition(row)

    message = str(exc_info.value)
    for field in (
        "authoritative_runtime_value=",
        "cpu_replay_value=",
        "absolute_difference=",
        "relative_difference=",
        "tolerance=",
        "rtol=",
        "atol=",
    ):
        assert field in message


def test_cpu_exact_mse_replay_accepts_tiny_difference():
    row = copy.deepcopy(_prediction_payload()["transitions"][0])
    cpu_replay = float(
        (
            row["tensors"]["exact_terminal_action"]
            - row["tensors"]["anchor_action"]
        )
        .square()
        .mean()
        .item()
    )
    difference = max(
        _CROSS_DEVICE_FP32_REDUCTION_ATOL * 0.5,
        abs(cpu_replay) * _CROSS_DEVICE_FP32_REDUCTION_RTOL * 0.5,
    )
    authoritative_exact_mse = cpu_replay + difference
    _set_authoritative_exact_mse(row, authoritative_exact_mse)

    _validate_transition(row)
    assert row["exact_adjacent_action_mse"] == authoritative_exact_mse


def test_cpu_exact_mse_replay_rejects_material_difference():
    row = copy.deepcopy(_prediction_payload()["transitions"][0])
    cpu_replay = float(
        (
            row["tensors"]["exact_terminal_action"]
            - row["tensors"]["anchor_action"]
        )
        .square()
        .mean()
        .item()
    )
    _set_authoritative_exact_mse(row, cpu_replay + 1e-3)

    with pytest.raises(
        ActionDeltaGateShadowCollectionError,
        match="exact adjacent action MSE CPU integrity replay mismatch",
    ):
        _validate_transition(row)


def test_writer_emits_phase_a_summary_provenance_and_global_identity(tmp_path):
    prediction = _prediction_payload()
    writer = ActionDeltaGateShadowWriter(
        tmp_path / "shadow",
        shard_size=4,
        expected_task_ids=(0,),
        expected_trajectories_per_task=1,
        source_commit="commit",
        artifact_identity={"sha256": "b" * 64, "threshold": 0.1},
        checkpoint_identity={"sha256": "c" * 64},
        initial_state_manifest_identity={"sha256": "a" * 64},
        configuration={"gate_min_terminal_iteration": 5},
        run_identity={"run_id": "run"},
    )
    writer.add_episode([prediction], success=True)
    manifest_path = writer.finalize()
    manifest = json.loads(manifest_path.read_text())

    assert manifest["complete"] is True
    assert manifest["summary"]["by_task"]["0"]["trajectories"] == 1
    assert manifest["summary"]["aggregate"]["eligible_rows"] == 2
    assert manifest["summary"]["aggregate"]["predicted_triggers"] == 2
    assert manifest["production_parity"]["shadow_values_used_for_control"] is False
    assert len(manifest["configuration_sha256"]) == 64
    expected_trajectory = globally_unique_trajectory_id(
        initial_state_manifest_sha256="a" * 64,
        task_id=0,
        initial_state_id=7,
        episode_seed=123,
    )
    assert prediction["identity"]["trajectory_id"] == expected_trajectory
    loaded_manifest, loaded_predictions = load_action_delta_gate_shadow_collection(
        manifest_path
    )
    assert loaded_manifest["dataset_identity_sha256"] == manifest[
        "dataset_identity_sha256"
    ]
    assert [item["prediction_id"] for item in loaded_predictions] == [
        prediction["prediction_id"]
    ]


def test_writer_and_runtime_validation_forbid_production_or_task4_calibration(tmp_path):
    with pytest.raises(ActionDeltaGateShadowCollectionError, match="Task 4/5"):
        ActionDeltaGateShadowWriter(
            tmp_path / "bad",
            shard_size=1,
            expected_task_ids=(4,),
            expected_trajectories_per_task=1,
            source_commit="commit",
            artifact_identity={},
            checkpoint_identity={},
            initial_state_manifest_identity={},
            configuration={},
            run_identity={},
        )


def test_shadow_artifact_preparation_is_dev_only_without_weakening_production_restriction():
    artifact = "benchmark_results/coda_anchor_feasibility/action_delta_gate_fold4/action_delta_gate.pt"
    expected_hash = "b4f9e938c72108f164b9d997a86eb4f5d9ea15b146754b9af83f9735e1ecfcf8"
    _, payload = load_action_delta_gate_artifact(
        artifact, expected_sha256=expected_hash
    )
    assert isinstance(
        prepare_action_delta_gate_shadow(payload, device="cpu", task_id=0),
        PreparedActionDeltaGate,
    )
    with pytest.raises(Exception, match="development tasks"):
        prepare_action_delta_gate_shadow(payload, device="cpu", task_id=4)
    with pytest.raises(Exception, match="not held out"):
        prepare_action_delta_gate(payload, device="cpu", task_id=0)


def _phase_a_config(**overrides):
    values = {
        "pretrained_checkpoint": "",
        "use_recurrent": True,
        "recurrence_strategy": "kl_divergence",
        "recurrence_kl_thresh": 0.001,
        "use_warm_start": True,
        "warm_start_source": "midpoint",
        "warm_start_min_iter": 2,
        "use_cached_final_output": True,
        "latent_precheck_mode": "off",
        "latent_precheck_trace_level": "off",
        "collect_action_delta_gate_shadow": True,
        "action_delta_gate_artifact_path": "benchmark_results/coda_anchor_feasibility/action_delta_gate_fold4/action_delta_gate.pt",
        "action_delta_gate_expected_sha256": "b4f9e938c72108f164b9d997a86eb4f5d9ea15b146754b9af83f9735e1ecfcf8",
        "action_delta_gate_min_terminal_iter": 5,
        "action_delta_gate_shadow_dir": "shadow-output",
        "evaluation_protocol_phase": "calibration",
        "num_trials_per_task": 10,
        "initial_state_manifest_path": "experiments/robot/libero/manifests/libero_spatial_official_50_v1.json",
        "reset_rng_each_episode": True,
    }
    values.update(overrides)
    return GenerateConfig(**values)


def test_phase_a_runner_configuration_accepts_only_development_tasks_and_terminal_five():
    validate_config(_phase_a_config(task_id=0))
    validate_config(_phase_a_config(task_id=None))
    with pytest.raises(ValueError, match="Task 4/5"):
        validate_config(_phase_a_config(task_id=4))
    with pytest.raises(ValueError, match="min_terminal_iter=5"):
        validate_config(_phase_a_config(task_id=0, action_delta_gate_min_terminal_iter=4))
    with pytest.raises(ValueError, match="recurrence_kl_thresh=0.001"):
        validate_config(_phase_a_config(task_id=0, recurrence_kl_thresh=0.002))
    with pytest.raises(Exception, match="cannot enable the production gate"):
        validate_action_delta_gate_shadow_configuration(
            enabled=True,
            production_gate_enabled=True,
            canonical_recurrence_strategy="adjacent_action_mse",
            prepared_gate=make_gate(),
            batch_size=1,
            use_warm_start=True,
            warm_start_source="midpoint",
            warm_start_min_iter=2,
            use_latent_precheck=False,
            latent_precheck_mode="off",
            latent_precheck_trace_level="off",
            shadow_full_depth=False,
            collect_preconvergence_raw_shadow=False,
            use_cached_final_output=True,
            min_terminal_iter=5,
        )
