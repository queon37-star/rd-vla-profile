import copy
import inspect
import json
import types

import pytest
import torch

import experiments.robot.libero.run_libero_eval as run_libero_eval_module
from experiments.robot.libero.run_libero_eval import (
    GenerateConfig,
    build_action_delta_deferred_backfill_log_fields,
    build_action_delta_cross_suite_log_record,
    validate_config,
)
from prismatic.models.action_delta_cross_suite_shadow import (
    ACTION_DELTA_CROSS_SUITE_ANALYSIS_TYPE,
    build_action_delta_cross_suite_transition,
)
from prismatic.models.action_delta_gate import (
    ACTION_DELTA_GATE_ARTIFACT_TYPE,
    ACTION_DELTA_GATE_CALIBRATION_METHOD,
    ACTION_DELTA_GATE_MODEL_TYPE,
    PreparedActionDeltaGate,
)
from prismatic.models.action_heads import RecurrentConfigInternal, VLARecurrent
from scripts.coda_anchor_feasibility.analyze_action_delta_cross_suite_shadow import (
    analyze_rows,
    load_rows,
)


ARTIFACT = (
    "benchmark_results/coda_anchor_feasibility/"
    "action_delta_gate_fold4/action_delta_gate.pt"
)
ARTIFACT_SHA256 = (
    "b4f9e938c72108f164b9d997a86eb4f5d9ea15b146754b9af83f9735e1ecfcf8"
)


def make_gate(*, predicted_delta=0.04, artifact_threshold=0.0007):
    return PreparedActionDeltaGate(
        schema_version=1,
        artifact_type=ACTION_DELTA_GATE_ARTIFACT_TYPE,
        model_type=ACTION_DELTA_GATE_MODEL_TYPE,
        hidden_dim=4,
        action_dim=2,
        action_chunk_len=2,
        held_out_task_ids=(4, 5),
        outer_fold=4,
        threshold=artifact_threshold,
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
    config = RecurrentConfigInternal(
        hidden_dim=4,
        num_heads=1,
        recurrent_vlm_layers=(0,),
        coda_vlm_layers=(),
        action_chunk_len=2,
        action_dim=2,
        mean_recurrence=4,
        backprop_depth=1,
        random_iterations=False,
    )
    model = VLARecurrent(config).eval()
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


def runtime_kwargs(gate, *, collect):
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
        "action_delta_gate_min_terminal_iter": 2,
        "collect_action_delta_cross_suite_shadow": collect,
        "kl_thresh": 0.001,
        "max_iter": 4,
    }


def cross_suite_config(suite, **overrides):
    values = {
        "pretrained_checkpoint": "",
        "task_suite_name": suite,
        "task_id": 0,
        "use_recurrent": True,
        "recurrence_strategy": "kl_divergence",
        "recurrence_kl_thresh": 0.001,
        "use_warm_start": True,
        "warm_start_source": "midpoint",
        "warm_start_min_iter": 2,
        "use_cached_final_output": True,
        "latent_precheck_mode": "off",
        "latent_precheck_trace_level": "off",
        "collect_action_delta_cross_suite_shadow": True,
        "action_delta_gate_artifact_path": ARTIFACT,
        "action_delta_gate_expected_sha256": ARTIFACT_SHA256,
        "action_delta_gate_min_terminal_iter": 2,
        "evaluation_protocol_phase": "legacy",
    }
    values.update(overrides)
    return GenerateConfig(**values)


@pytest.mark.parametrize("suite", ["libero_object", "libero_goal", "libero_10"])
def test_cross_suite_runner_accepts_object_goal_and_long(suite):
    validate_config(cross_suite_config(suite))


def test_cross_suite_rejects_libero_90_and_production_gate_remains_spatial_only():
    with pytest.raises(ValueError, match="supports only"):
        validate_config(cross_suite_config("libero_90"))
    with pytest.raises(ValueError, match="Spatial-only"):
        validate_config(
            cross_suite_config(
                "libero_object",
                collect_action_delta_cross_suite_shadow=False,
                use_action_delta_gate=True,
                task_id=4,
            )
        )
    with pytest.raises(ValueError, match="64-character hexadecimal SHA-256"):
        validate_config(
            cross_suite_config(
                "libero_object",
                action_delta_gate_expected_sha256="",
            )
        )


def test_cross_suite_uses_dedicated_diagnostic_artifact_preparation(monkeypatch):
    calls = []

    def prepare_cross(payload, *, device):
        calls.append((payload, device))
        return "cross-suite-prepared"

    monkeypatch.setattr(
        run_libero_eval_module,
        "prepare_action_delta_gate_cross_suite_shadow",
        prepare_cross,
    )
    result = run_libero_eval_module._prepare_action_delta_gate_for_evaluation(
        cross_suite_config("libero_object"),
        "payload",
        device=torch.device("cpu"),
        task_id=0,
    )
    assert result == "cross-suite-prepared"
    assert calls == [("payload", torch.device("cpu"))]


def test_cross_suite_shadow_does_not_change_output_k_or_exact_coda_calls():
    baseline = make_model()
    shadow = copy.deepcopy(baseline)
    gate = make_gate()
    warm = torch.zeros(1, 2, 4, dtype=torch.bfloat16)

    baseline_result = baseline(
        *inputs(), warm_start_state=warm, **runtime_kwargs(gate, collect=False)
    )
    shadow_result = shadow(
        *inputs(), warm_start_state=warm, **runtime_kwargs(gate, collect=True)
    )

    torch.testing.assert_close(baseline_result[0], shadow_result[0], rtol=0, atol=0)
    assert baseline_result[1:] == shadow_result[1:]
    assert baseline.test_coda_calls == shadow.test_coda_calls == 4
    assert baseline.last_recurrence_debug["iteration_mse"] == shadow.last_recurrence_debug[
        "iteration_mse"
    ]
    payload = shadow.last_inference_metadata["action_delta_cross_suite_shadow"]
    assert [row["terminal_iteration"] for row in payload["transitions"]] == [2, 3, 4]
    assert all(row["high_predicted_nonconvergence"] for row in payload["transitions"])
    assert all(not row["exact_safe"] for row in payload["transitions"])
    assert payload["exact_coda_call_count"] == 4
    assert payload["K_t"] == 4


def test_high_side_labels_use_frozen_q_not_artifact_low_side_threshold():
    high_safe = build_action_delta_cross_suite_transition(
        anchor_iteration=1,
        terminal_iteration=2,
        score=0.0015,
        exact_adjacent_action_mse=0.0005,
    )
    low_unsafe = build_action_delta_cross_suite_transition(
        anchor_iteration=2,
        terminal_iteration=3,
        score=0.0006,
        exact_adjacent_action_mse=0.002,
    )
    assert high_safe["high_predicted_nonconvergence"] is True
    assert high_safe["exact_safe"] is True
    assert high_safe["high_exact_safe_violation"] is True
    assert low_unsafe["high_predicted_nonconvergence"] is False
    assert low_unsafe["high_exact_safe_violation"] is False


def test_cross_suite_mode_is_mutually_exclusive_with_deferred_lazy_mode():
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_config(
            cross_suite_config(
                "libero_spatial",
                use_action_delta_deferred_backfill_filter=True,
                action_delta_deferred_runtime_policy="lazy_prefix_exact",
                evaluation_protocol_phase="calibration",
            )
        )


def test_cross_suite_provenance_record_keeps_artifact_checkpoint_and_identity():
    provenance = {
        "analysis_type": ACTION_DELTA_CROSS_SUITE_ANALYSIS_TYPE,
        "diagnostic_only": True,
        "production_efficiency_claim": False,
        "predictor_artifact_sha256": ARTIFACT_SHA256,
        "predictor_artifact_path": ARTIFACT,
        "predictor_training_suite": "libero_spatial",
        "evaluation_suite": "libero_object",
        "source_commit": "abc123",
        "checkpoint_path": "/checkpoint",
        "checkpoint_identity": {"sha256": "checkpoint-sha"},
    }
    record = build_action_delta_cross_suite_log_record(
        {
            "actual_origin": "ACTUAL_WARM",
            "transitions": [],
            "error": None,
            "exact_coda_call_count": 3,
        },
        provenance=provenance,
        task_id=2,
        task_name="task",
        episode_id=4,
        initial_state_id=7,
        episode_seed=11,
        prediction_id=9,
        environment_timestep=13,
        actual_origin="ACTUAL_WARM",
        recurrent_k=3,
        evaluation_protocol_phase="legacy",
        min_terminal_iteration=2,
    )
    assert record["predictor_artifact_sha256"] == ARTIFACT_SHA256
    assert record["checkpoint_identity"] == {"sha256": "checkpoint-sha"}
    assert record["evaluation_suite"] == "libero_object"
    assert record["task_id"] == 2
    assert record["K_t"] == 3


def test_analyzer_reports_suite_task_statistics(tmp_path):
    payload = {
        "analysis_type": ACTION_DELTA_CROSS_SUITE_ANALYSIS_TYPE,
        "diagnostic_only": True,
        "production_efficiency_claim": False,
        "predictor_training_suite": "libero_spatial",
        "predictor_artifact_sha256": ARTIFACT_SHA256,
        "high_side_threshold": 0.0015,
        "evaluation_suite": "libero_goal",
        "task_id": 1,
        "transitions": [
            build_action_delta_cross_suite_transition(
                anchor_iteration=1,
                terminal_iteration=2,
                score=0.002,
                exact_adjacent_action_mse=0.003,
            ),
            build_action_delta_cross_suite_transition(
                anchor_iteration=2,
                terminal_iteration=3,
                score=0.0015,
                exact_adjacent_action_mse=0.0005,
            ),
        ],
    }
    path = tmp_path / "predictions.jsonl"
    path.write_text(
        json.dumps({"action_delta_cross_suite_shadow": payload}) + "\n",
        encoding="utf-8",
    )
    results = analyze_rows(load_rows([path]))
    summary = results["by_suite_task"]["libero_goal/task_1"]
    assert summary["transition_count"] == 2
    assert summary["finite_score_count"] == 2
    assert summary["high_predicted_nonconvergence_count"] == 2
    assert summary["high_exact_safe_violation_count"] == 1
    assert summary["high_prediction_exact_nonconverged_precision"] == 0.5


def test_defaults_leave_both_shadow_modes_disabled():
    config = GenerateConfig()
    assert config.collect_action_delta_gate_shadow is False
    assert config.collect_action_delta_cross_suite_shadow is False


def test_legacy_none_episode_seed_is_safe_in_generic_prediction_logging():
    config = GenerateConfig(
        evaluation_protocol_phase="legacy",
        reset_rng_each_episode=False,
    )
    episode_seed = None
    assert config.evaluation_protocol_phase == "legacy"
    assert config.reset_rng_each_episode is False

    identity = {
        "task_id": 0,
        "episode_id": 0,
        "episode_seed": run_libero_eval_module._as_int(episode_seed),
        "action_prediction_index": 0,
        "environment_timestep": 10,
        "trajectory_id": None,
        "initial_state_id": None,
    }
    fields = build_action_delta_deferred_backfill_log_fields(
        {"action_delta_deferred_backfill_filter_runs": [{"run_length": 1}]},
        identity,
    )
    record = {"episode_seed": episode_seed, **fields}
    serialized = json.loads(json.dumps(record))

    assert serialized["episode_seed"] is None
    assert serialized[
        "action_delta_deferred_backfill_filter_runs"
    ][0]["prediction_identity"]["episode_seed"] is None
    run_episode_source = inspect.getsource(run_libero_eval_module.run_episode)
    assert '"episode_seed": _as_int(episode_seed)' in run_episode_source
    assert '"episode_seed": int(episode_seed)' not in run_episode_source


def test_non_null_episode_seed_remains_an_integer_in_generic_prediction_logging():
    episode_seed = 17
    identity = {
        "episode_seed": run_libero_eval_module._as_int(episode_seed),
    }
    fields = build_action_delta_deferred_backfill_log_fields(
        {"action_delta_deferred_backfill_filter_runs": [{"run_length": 1}]},
        identity,
    )
    serialized = json.loads(json.dumps(fields))

    value = serialized["action_delta_deferred_backfill_filter_runs"][0][
        "prediction_identity"
    ]["episode_seed"]
    assert value == 17
    assert isinstance(value, int)
