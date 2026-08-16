import copy
import json
import types

import pytest
import torch

import prismatic.models.action_delta_gate as action_delta_gate_module
import prismatic.models.action_heads as action_heads_module
from prismatic.models.action_delta_gate import (
    ACTION_DELTA_GATE_ARTIFACT_TYPE,
    ACTION_DELTA_GATE_CALIBRATION_METHOD,
    ACTION_DELTA_GATE_MODEL_TYPE,
    PreparedActionDeltaGate,
)
from prismatic.models.action_heads import (
    ActionHeadRecurrent,
    RecurrentConfigInternal,
    VLARecurrent,
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


def make_model(*, nonfinite_iteration=None):
    cfg = RecurrentConfigInternal(
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
    model = VLARecurrent(cfg).eval()
    model.test_coda_calls = 0
    model.test_coda_states = []
    model.test_iteration = 0

    def init_state(self, batch_size, device, dtype):
        return torch.zeros(batch_size, 2, 4, device=device, dtype=dtype)

    def run_one_iteration(self, state, *args):
        self.test_iteration += 1
        result = state + 1
        if self.test_iteration == nonfinite_iteration:
            result = result.clone()
            result[0, 0, 0] = float("nan")
        return result

    def get_output(self, state, *args, profile=False):
        self.test_coda_calls += 1
        self.test_coda_states.append(state.detach().clone())
        if profile:
            self._last_get_output_timing = {
                "get_output_ms": 0.3,
                "coda_ms": 0.2,
                "output_proj_ms": 0.1,
            }
        return state[..., :2].clone()

    model.init_state = types.MethodType(init_state, model)
    model._run_one_iteration = types.MethodType(run_one_iteration, model)
    model._get_output = types.MethodType(get_output, model)
    return model


def inputs(batch=1):
    h_a = torch.zeros(batch, 1, 1, 4, dtype=torch.bfloat16)
    h_t = torch.zeros_like(h_a)
    p = torch.zeros(batch, 1, 4, dtype=torch.bfloat16)
    return h_a, h_t, p


def gate_kwargs(gate):
    return {
        "convergence_strategy": "adjacent_action_mse",
        "enable_warm_start": True,
        "warm_start_source": "midpoint",
        "warm_start_min_iter": 2,
        "use_cached_final_output": True,
        "latent_precheck_mode": "off",
        "latent_precheck_trace_level": "off",
        "use_latent_precheck": False,
        "use_action_delta_gate": True,
        "action_delta_gate": gate,
        "action_delta_gate_max_skip": 1,
        "max_iter": 4,
    }


def install_action_outputs(model, outputs_by_iteration):
    def get_output(self, state, *args, profile=False):
        self.test_coda_calls += 1
        self.test_coda_states.append(state.detach().clone())
        if profile:
            self._last_get_output_timing = {
                "get_output_ms": 0.3,
                "coda_ms": 0.2,
                "output_proj_ms": 0.1,
            }
        return outputs_by_iteration[self.test_iteration].detach().clone()

    model._get_output = types.MethodType(get_output, model)


def test_gate_disabled_preserves_baseline_and_does_not_score():
    baseline = make_model()
    supported = copy.deepcopy(baseline)
    warm = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
    kwargs = gate_kwargs(make_gate())
    kwargs.update(
        use_action_delta_gate=False,
        action_delta_gate=None,
        kl_thresh=2.0,
    )

    first = baseline(*inputs(), warm_start_state=warm, **kwargs)
    second = supported(
        *inputs(),
        warm_start_state=warm,
        action_delta_gate_min_terminal_iter=5,
        action_delta_gate_return_mode="predicted_correction",
        **kwargs,
    )
    torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
    assert first[1:] == second[1:]
    assert baseline.test_coda_calls == supported.test_coda_calls == 2
    assert baseline.last_recurrence_debug["action_delta_gate_score_call_count"] == 0
    assert supported.last_recurrence_debug["action_delta_gate_score_call_count"] == 0
    assert supported.last_recurrence_debug[
        "action_delta_gate_first_eligible_terminal_iteration"
    ] is None


def test_explicit_anchor_return_mode_matches_default_behavior():
    default_model = make_model()
    explicit_model = make_model()
    warm = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
    kwargs = gate_kwargs(make_gate(predicted_delta=0.5, threshold=1.0))

    default_result = default_model(*inputs(), warm_start_state=warm, **kwargs)
    explicit_result = explicit_model(
        *inputs(),
        warm_start_state=warm,
        action_delta_gate_return_mode="anchor",
        **kwargs,
    )

    torch.testing.assert_close(default_result[0], explicit_result[0], rtol=0, atol=0)
    assert default_result[1:] == explicit_result[1:]
    assert default_model.test_coda_calls == explicit_model.test_coda_calls == 1
    for key in (
        "K_t",
        "stop_reason",
        "action_delta_gate_score_trace",
        "action_delta_gate_skipped_coda_count",
        "coda_call_count",
        "get_output_call_count",
        "action_delta_gate_return_mode",
        "action_delta_gate_returned_anchor",
        "action_delta_gate_returned_predicted_correction",
    ):
        assert (
            default_model.last_recurrence_debug[key]
            == explicit_model.last_recurrence_debug[key]
        )


def test_predicted_correction_return_reuses_score_prediction_and_skips_terminal_coda(
    monkeypatch,
):
    linear_call_count = 0
    original_linear = action_delta_gate_module.F.linear

    def linear_spy(*args, **kwargs):
        nonlocal linear_call_count
        linear_call_count += 1
        return original_linear(*args, **kwargs)

    monkeypatch.setattr(action_delta_gate_module.F, "linear", linear_spy)
    model = make_model()
    anchor_action = torch.ones(1, 2, 2, dtype=torch.bfloat16)
    install_action_outputs(model, {1: anchor_action})

    output, actual_iter, final_mse = model(
        *inputs(),
        warm_start_state=torch.zeros(1, 2, 4, dtype=torch.bfloat16),
        action_delta_gate_return_mode="predicted_correction",
        **gate_kwargs(make_gate(predicted_delta=0.5, threshold=1.0)),
    )

    assert linear_call_count == 1
    assert actual_iter == 2
    assert final_mse is None
    assert model.test_coda_calls == 1
    assert output.shape == anchor_action.shape
    assert output.device == anchor_action.device
    assert output.dtype == anchor_action.dtype
    torch.testing.assert_close(
        output,
        torch.full_like(anchor_action, 1.5),
        rtol=0,
        atol=0,
    )
    debug = model.last_recurrence_debug
    assert debug["action_delta_gate_return_mode"] == "predicted_correction"
    assert debug["action_delta_gate_returned_predicted_correction"] is True
    assert debug["action_delta_gate_returned_anchor"] is False
    assert debug["action_delta_gate_returned_previous_coda"] is False
    assert debug["action_delta_gate_triggered"] is True
    assert debug["action_delta_gate_skipped_coda_count"] == 1
    assert debug["stop_reason"] == "action_delta_gate"
    assert debug["K_t"] == 2
    assert debug["final_state_coda_executed"] is False
    assert debug["coda_call_count"] == 1
    assert debug["get_output_call_count"] == 1


def test_nonfinite_predicted_correction_falls_back_to_exact_coda(monkeypatch):
    def invalid_correction_result(
        gate,
        anchor_state,
        current_state,
        *,
        return_pred_delta=False,
    ):
        assert return_pred_delta is True
        pred_delta = torch.full(
            (1, gate.action_chunk_len, gate.action_dim),
            float("inf"),
            device=anchor_state.device,
        )
        return 0.0, True, pred_delta

    monkeypatch.setattr(
        action_heads_module,
        "evaluate_action_delta_gate",
        invalid_correction_result,
    )
    model = make_model()
    output, actual_iter, final_mse = model(
        *inputs(),
        warm_start_state=torch.zeros(1, 2, 4, dtype=torch.bfloat16),
        kl_thresh=2.0,
        action_delta_gate_return_mode="predicted_correction",
        **gate_kwargs(make_gate()),
    )

    assert actual_iter == 2
    assert final_mse == 1.0
    assert model.test_coda_calls == 2
    assert torch.all(output == 2)
    debug = model.last_recurrence_debug
    assert debug["action_delta_gate_triggered"] is False
    assert debug["action_delta_gate_skipped_coda_count"] == 0
    assert debug["action_delta_gate_returned_predicted_correction"] is False
    assert debug["action_delta_gate_returned_anchor"] is False
    assert "non-finite" in debug["action_delta_gate_fallback_reason"]
    assert debug["coda_call_count"] == 2
    assert debug["get_output_call_count"] == 2


def test_cold_origin_uses_normal_coda_and_captures_midpoint_candidate():
    model = make_model()
    output, actual_iter, final_mse = model(
        *inputs(),
        warm_start_state=None,
        kl_thresh=2.0,
        **gate_kwargs(make_gate()),
    )
    assert actual_iter == 2
    assert final_mse == 1.0
    assert model.test_coda_calls == 2
    assert torch.all(output == 2)
    debug = model.last_recurrence_debug
    assert debug["action_delta_gate_requested"] is True
    assert debug["action_delta_gate_applied"] is False
    assert debug["action_delta_gate_score_call_count"] == 0
    assert debug["action_delta_gate_min_terminal_iter"] == 2
    assert debug["action_delta_gate_first_eligible_terminal_iteration"] is None
    assert model.last_inference_metadata["warm_start"]["candidate_state_count"] == 2


def test_warm_above_threshold_executes_exact_coda_and_preserves_action_mse():
    model = make_model()
    gate = make_gate(predicted_delta=1.0, threshold=0.1)
    output, actual_iter, final_mse = model(
        *inputs(),
        warm_start_state=torch.zeros(1, 2, 4, dtype=torch.bfloat16),
        kl_thresh=2.0,
        **gate_kwargs(gate),
    )
    assert actual_iter == 2
    assert final_mse == 1.0
    assert model.test_coda_calls == 2
    assert torch.all(output == 2)
    debug = model.last_recurrence_debug
    assert debug["action_delta_gate_triggered"] is False
    assert debug["iteration_mse"] == [1.0]
    assert debug["final_mse"] == 1.0


def test_warm_below_threshold_skips_one_coda_and_returns_previous_exact_output():
    model = make_model()
    output, actual_iter, final_mse = model(
        *inputs(),
        warm_start_state=torch.zeros(1, 2, 4, dtype=torch.bfloat16),
        kl_thresh=0.001,
        profile_coda_cost=True,
        **gate_kwargs(make_gate(predicted_delta=0.0)),
    )
    assert actual_iter == 2
    assert final_mse is None
    assert model.test_coda_calls == 1
    assert torch.all(output == 1)

    debug = model.last_recurrence_debug
    assert debug["K_t"] == 2
    assert debug["adaptive_stop"] is True
    assert debug["stop_reason"] == "action_delta_gate"
    assert debug["canonical_stop_reason"] == "action_delta_gate"
    assert debug["action_delta_gate_anchor_iteration"] == 1
    assert debug["action_delta_gate_terminal_iteration"] == 2
    assert debug["action_delta_gate_returned_action_source_iteration"] == 1
    assert debug["action_delta_gate_skipped_coda_count"] == 1
    assert debug["action_delta_gate_returned_previous_coda"] is True
    assert debug["action_delta_gate_min_terminal_iter"] == 2
    assert debug["action_delta_gate_first_eligible_terminal_iteration"] == 2
    assert debug["final_state_coda_executed"] is False
    assert debug["returned_cached_final_output"] is False
    assert debug["coda_call_count"] == model.test_coda_calls
    assert debug["get_output_call_count"] == model.test_coda_calls
    assert debug["iteration_mse"] == []
    assert debug["conv_score_list"] == []
    assert debug["final_mse"] is None
    assert debug["action_delta_gate_score_trace"][0]["score"] == 0.0
    assert len(debug["action_delta_gate_predictor_ms_list"]) == 1
    assert debug["action_delta_gate_predictor_ms_total"] >= 0.0
    assert debug["action_delta_gate_exact_audit_enabled"] is False
    assert debug["action_delta_gate_exact_audit_performed"] is False
    assert debug["action_delta_gate_exact_audit_get_output_call_count"] == 0
    assert debug["action_delta_gate_exact_audit_get_output_ms"] == 0.0
    assert debug["action_delta_gate_exact_audit_error"] is None
    assert debug["action_delta_gate_exact_audit_anchor_action"] is None
    assert debug["action_delta_gate_exact_audit_terminal_action"] is None
    assert debug["action_delta_gate_exact_audit_predicted_delta_action"] is None
    assert debug[
        "action_delta_gate_exact_audit_predicted_corrected_action"
    ] is None
    assert debug["action_delta_gate_exact_audit_correction_full_mse"] is None
    assert model.last_inference_metadata["warm_start"]["candidate_state_count"] == 2
    json.dumps(debug, allow_nan=False)


def test_exact_audit_trigger_metrics_and_production_accounting_are_isolated():
    anchor_action = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.bfloat16
    )
    terminal_action = torch.tensor(
        [[[2.0, 4.0], [6.0, 8.0]]], dtype=torch.bfloat16
    )
    outputs = {1: anchor_action, 2: terminal_action}
    disabled = make_model()
    enabled = make_model()
    install_action_outputs(disabled, outputs)
    install_action_outputs(enabled, outputs)
    kwargs = gate_kwargs(make_gate(predicted_delta=0.5, threshold=1.0))
    warm = torch.zeros(1, 2, 4, dtype=torch.bfloat16)

    disabled_result = disabled(
        *inputs(),
        warm_start_state=warm,
        kl_thresh=0.001,
        profile_coda_cost=True,
        **kwargs,
    )
    enabled_result = enabled(
        *inputs(),
        warm_start_state=warm,
        kl_thresh=0.001,
        profile_coda_cost=True,
        action_delta_gate_exact_coda_audit=True,
        **kwargs,
    )

    torch.testing.assert_close(disabled_result[0], anchor_action, rtol=0, atol=0)
    torch.testing.assert_close(enabled_result[0], anchor_action, rtol=0, atol=0)
    assert disabled_result[1:] == enabled_result[1:] == (2, None)
    assert disabled.test_coda_calls == 1
    assert enabled.test_coda_calls == 2

    disabled_debug = disabled.last_recurrence_debug
    debug = enabled.last_recurrence_debug
    for key in (
        "coda_call_count",
        "get_output_call_count",
        "coda_ms_total",
        "get_output_ms_total",
        "iteration_mse",
        "conv_score_list",
        "final_mse",
        "stop_reason",
        "action_delta_gate_skipped_coda_count",
        "action_delta_gate_anchor_iteration",
        "action_delta_gate_terminal_iteration",
        "action_delta_gate_returned_action_source_iteration",
    ):
        assert debug[key] == disabled_debug[key]
    assert debug["coda_call_count"] == 1
    assert debug["get_output_call_count"] == 1
    assert debug["coda_ms_total"] == pytest.approx(0.2)
    assert debug["get_output_ms_total"] == pytest.approx(0.3)
    assert debug["iteration_mse"] == []
    assert debug["conv_score_list"] == []
    assert debug["final_mse"] is None
    assert debug["action_delta_gate_anchor_iteration"] == 1
    assert debug["action_delta_gate_returned_action_source_iteration"] == 1

    assert debug["action_delta_gate_exact_audit_enabled"] is True
    assert debug["action_delta_gate_exact_audit_performed"] is True
    assert debug["action_delta_gate_exact_audit_anchor_iteration"] == 1
    assert debug["action_delta_gate_exact_audit_terminal_iteration"] == 2
    assert debug["action_delta_gate_exact_audit_get_output_call_count"] == 1
    assert debug["action_delta_gate_exact_audit_get_output_ms"] >= 0.0
    assert debug["action_delta_gate_exact_audit_error"] is None
    assert debug["action_delta_gate_exact_audit_action_shape"] == [1, 2, 2]
    assert debug["action_delta_gate_exact_audit_metric_action_shape"] == [2, 2]
    assert debug[
        "action_delta_gate_exact_audit_leading_batch_dim_squeezed"
    ] is True
    assert debug["action_delta_gate_exact_audit_full_mse"] == pytest.approx(7.5)
    assert debug["action_delta_gate_exact_audit_l2"] == pytest.approx(30.0 ** 0.5)
    assert debug["action_delta_gate_exact_audit_max_abs"] == pytest.approx(4.0)
    assert debug["action_delta_gate_exact_audit_per_step_mse"] == pytest.approx(
        [2.5, 12.5]
    )
    assert debug[
        "action_delta_gate_exact_audit_per_step_max_abs"
    ] == pytest.approx([2.0, 4.0])
    assert debug["action_delta_gate_exact_audit_per_dim_mse"] == pytest.approx(
        [5.0, 10.0]
    )
    assert debug[
        "action_delta_gate_exact_audit_per_dim_max_abs"
    ] == pytest.approx([3.0, 4.0])
    assert debug["action_delta_gate_exact_audit_anchor_action"] == anchor_action.float().tolist()
    assert debug["action_delta_gate_exact_audit_terminal_action"] == terminal_action.float().tolist()
    assert debug["action_delta_gate_exact_audit_delta_action"] == [
        [[1.0, 2.0], [3.0, 4.0]]
    ]
    assert debug["action_delta_gate_exact_audit_predicted_delta_action"] == [
        [[0.5, 0.5], [0.5, 0.5]]
    ]
    assert debug[
        "action_delta_gate_exact_audit_predicted_corrected_action"
    ] == [[[1.5, 2.5], [3.5, 4.5]]]
    assert debug["action_delta_gate_exact_audit_correction_full_mse"] == pytest.approx(
        5.25
    )
    assert debug["action_delta_gate_exact_audit_correction_l2"] == pytest.approx(
        21.0 ** 0.5
    )
    assert debug["action_delta_gate_exact_audit_correction_max_abs"] == pytest.approx(
        3.5
    )
    assert debug[
        "action_delta_gate_exact_audit_correction_per_step_mse"
    ] == pytest.approx([1.25, 9.25])
    assert debug[
        "action_delta_gate_exact_audit_correction_per_step_max_abs"
    ] == pytest.approx([1.5, 3.5])
    assert debug[
        "action_delta_gate_exact_audit_correction_per_dim_mse"
    ] == pytest.approx([3.25, 7.25])
    assert debug[
        "action_delta_gate_exact_audit_correction_per_dim_max_abs"
    ] == pytest.approx([2.5, 3.5])
    assert debug["action_delta_gate_exact_audit_prefix_step_count"] == 2
    assert debug[
        "action_delta_gate_exact_audit_anchor_reuse_prefix_mse"
    ] == pytest.approx(7.5)
    assert debug[
        "action_delta_gate_exact_audit_correction_prefix_mse"
    ] == pytest.approx(5.25)
    assert debug[
        "action_delta_gate_exact_audit_correction_full_mse_ratio"
    ] == pytest.approx(0.7)
    assert debug[
        "action_delta_gate_exact_audit_correction_prefix_mse_ratio"
    ] == pytest.approx(0.7)
    json.dumps(debug, allow_nan=False)


def test_exact_audit_reuses_single_predictor_evaluation(monkeypatch):
    linear_call_count = 0
    original_linear = action_delta_gate_module.F.linear

    def linear_spy(*args, **kwargs):
        nonlocal linear_call_count
        linear_call_count += 1
        return original_linear(*args, **kwargs)

    monkeypatch.setattr(action_delta_gate_module.F, "linear", linear_spy)
    model = make_model()
    anchor_action = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.bfloat16
    )
    terminal_action = torch.tensor(
        [[[2.0, 4.0], [6.0, 8.0]]], dtype=torch.bfloat16
    )
    install_action_outputs(model, {1: anchor_action, 2: terminal_action})

    output, actual_iter, final_mse = model(
        *inputs(),
        warm_start_state=torch.zeros(1, 2, 4, dtype=torch.bfloat16),
        action_delta_gate_exact_coda_audit=True,
        **gate_kwargs(make_gate(predicted_delta=0.5, threshold=1.0)),
    )

    assert linear_call_count == 1
    assert actual_iter == 2
    assert final_mse is None
    torch.testing.assert_close(output, anchor_action, rtol=0, atol=0)
    assert model.last_recurrence_debug[
        "action_delta_gate_exact_audit_predicted_delta_action"
    ] == [[[0.5, 0.5], [0.5, 0.5]]]


def test_exact_audit_compares_correction_without_replacing_corrected_return():
    anchor_action = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.bfloat16
    )
    terminal_action = torch.tensor(
        [[[2.0, 4.0], [6.0, 8.0]]], dtype=torch.bfloat16
    )
    model = make_model()
    install_action_outputs(model, {1: anchor_action, 2: terminal_action})

    output, actual_iter, final_mse = model(
        *inputs(),
        warm_start_state=torch.zeros(1, 2, 4, dtype=torch.bfloat16),
        action_delta_gate_exact_coda_audit=True,
        action_delta_gate_return_mode="predicted_correction",
        **gate_kwargs(make_gate(predicted_delta=0.5, threshold=1.0)),
    )

    expected_return = torch.tensor(
        [[[1.5, 2.5], [3.5, 4.5]]], dtype=torch.bfloat16
    )
    assert actual_iter == 2
    assert final_mse is None
    torch.testing.assert_close(output, expected_return, rtol=0, atol=0)
    assert model.test_coda_calls == 2
    debug = model.last_recurrence_debug
    assert debug["action_delta_gate_exact_audit_performed"] is True
    assert debug[
        "action_delta_gate_exact_audit_predicted_corrected_action"
    ] == expected_return.float().tolist()
    assert debug["action_delta_gate_exact_audit_correction_full_mse"] == pytest.approx(
        5.25
    )
    assert debug["action_delta_gate_returned_predicted_correction"] is True
    assert debug["action_delta_gate_returned_anchor"] is False
    assert debug["coda_call_count"] == 1
    assert debug["get_output_call_count"] == 1
    assert debug["action_delta_gate_exact_audit_get_output_call_count"] == 1


def test_exact_audit_ratios_handle_zero_anchor_reuse_mse():
    action = torch.zeros(1, 2, 2)
    both_zero = action_heads_module._action_delta_gate_exact_audit_metrics(
        action,
        action,
        torch.zeros_like(action),
    )
    nonzero_correction = (
        action_heads_module._action_delta_gate_exact_audit_metrics(
            action,
            action,
            torch.ones_like(action),
        )
    )

    assert both_zero[
        "action_delta_gate_exact_audit_correction_full_mse_ratio"
    ] == 1.0
    assert both_zero[
        "action_delta_gate_exact_audit_correction_prefix_mse_ratio"
    ] == 1.0
    assert nonzero_correction[
        "action_delta_gate_exact_audit_correction_full_mse_ratio"
    ] is None
    assert nonzero_correction[
        "action_delta_gate_exact_audit_correction_prefix_mse_ratio"
    ] is None


def test_exact_audit_nontrigger_does_not_run_diagnostic_coda():
    model = make_model()
    output, actual_iter, final_mse = model(
        *inputs(),
        warm_start_state=torch.zeros(1, 2, 4, dtype=torch.bfloat16),
        kl_thresh=2.0,
        action_delta_gate_exact_coda_audit=True,
        **gate_kwargs(make_gate(predicted_delta=1.0, threshold=0.1)),
    )

    assert actual_iter == 2
    assert final_mse == 1.0
    assert model.test_coda_calls == 2
    assert torch.all(output == 2)
    debug = model.last_recurrence_debug
    assert debug["action_delta_gate_triggered"] is False
    assert debug["action_delta_gate_exact_audit_enabled"] is True
    assert debug["action_delta_gate_exact_audit_performed"] is False
    assert debug["action_delta_gate_exact_audit_get_output_call_count"] == 0
    assert debug["action_delta_gate_exact_audit_get_output_ms"] == 0.0
    assert debug["coda_call_count"] == model.test_coda_calls == 2
    assert debug["get_output_call_count"] == model.test_coda_calls


@pytest.mark.parametrize(
    ("terminal_action", "error_text"),
    [
        (
            torch.zeros(1, 2, 1, dtype=torch.bfloat16),
            "shape mismatch",
        ),
        (
            torch.tensor(
                [[[float("nan"), 0.0], [0.0, 0.0]]],
                dtype=torch.bfloat16,
            ),
            "non-finite",
        ),
    ],
)
def test_exact_audit_failure_records_error_without_changing_gate_return(
    terminal_action, error_text
):
    anchor_action = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.bfloat16
    )
    model = make_model()
    install_action_outputs(model, {1: anchor_action, 2: terminal_action})
    output, actual_iter, final_mse = model(
        *inputs(),
        warm_start_state=torch.zeros(1, 2, 4, dtype=torch.bfloat16),
        kl_thresh=0.001,
        action_delta_gate_exact_coda_audit=True,
        **gate_kwargs(make_gate(predicted_delta=0.0)),
    )

    assert actual_iter == 2
    assert final_mse is None
    torch.testing.assert_close(output, anchor_action, rtol=0, atol=0)
    assert model.test_coda_calls == 2
    debug = model.last_recurrence_debug
    assert debug["action_delta_gate_triggered"] is True
    assert debug["action_delta_gate_exact_audit_performed"] is True
    assert debug["action_delta_gate_exact_audit_get_output_call_count"] == 1
    assert error_text in debug["action_delta_gate_exact_audit_error"]
    assert debug["action_delta_gate_exact_audit_full_mse"] is None
    assert debug["action_delta_gate_exact_audit_anchor_action"] is None
    assert debug["action_delta_gate_exact_audit_terminal_action"] is None
    assert debug["coda_call_count"] == 1
    assert debug["get_output_call_count"] == 1
    assert debug["action_delta_gate_skipped_coda_count"] == 1
    assert debug["iteration_mse"] == []
    assert debug["final_mse"] is None
    json.dumps(debug, allow_nan=False)


def test_min_terminal_two_matches_default_gate_behavior():
    default_model = make_model()
    explicit_model = make_model()
    warm = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
    kwargs = gate_kwargs(make_gate(predicted_delta=0.0))

    default_result = default_model(
        *inputs(), warm_start_state=warm, kl_thresh=0.001, **kwargs
    )
    explicit_result = explicit_model(
        *inputs(),
        warm_start_state=warm,
        kl_thresh=0.001,
        action_delta_gate_min_terminal_iter=2,
        **kwargs,
    )

    torch.testing.assert_close(default_result[0], explicit_result[0], rtol=0, atol=0)
    assert default_result[1:] == explicit_result[1:]
    for key in (
        "K_t",
        "stop_reason",
        "action_delta_gate_score_call_count",
        "action_delta_gate_anchor_iteration",
        "action_delta_gate_terminal_iteration",
        "action_delta_gate_returned_action_source_iteration",
        "action_delta_gate_skipped_coda_count",
        "coda_call_count",
        "get_output_call_count",
    ):
        assert default_model.last_recurrence_debug[key] == explicit_model.last_recurrence_debug[key]


def test_min_terminal_five_uses_immediately_preceding_exact_anchor(monkeypatch):
    score_inputs = []

    def score_spy(gate, anchor_state, current_state):
        score_inputs.append(
            (anchor_state.detach().clone(), current_state.detach().clone())
        )
        return 0.0, True

    monkeypatch.setattr(
        action_heads_module,
        "evaluate_action_delta_gate",
        score_spy,
    )
    model = make_model()
    kwargs = gate_kwargs(make_gate())
    kwargs.update(
        max_iter=6,
        action_delta_gate_min_terminal_iter=5,
    )
    output, actual_iter, final_mse = model(
        *inputs(),
        warm_start_state=torch.zeros(1, 2, 4, dtype=torch.bfloat16),
        kl_thresh=0.001,
        **kwargs,
    )

    assert actual_iter == 5
    assert final_mse == 1.0
    assert len(score_inputs) == 1
    anchor_state, current_state = score_inputs[0]
    assert torch.all(anchor_state == 4)
    assert torch.all(current_state == 5)
    assert [int(state[0, 0, 0]) for state in model.test_coda_states] == [1, 2, 3, 4]
    assert model.test_coda_calls == 4
    assert torch.all(output == 4)

    debug = model.last_recurrence_debug
    assert debug["action_delta_gate_min_terminal_iter"] == 5
    assert debug["action_delta_gate_first_eligible_terminal_iteration"] == 5
    assert debug["action_delta_gate_score_call_count"] == 1
    assert debug["action_delta_gate_score_trace"] == [
        {
            "anchor_iteration": 4,
            "terminal_iteration": 5,
            "score": 0.0,
            "triggered": True,
        }
    ]
    assert debug["action_delta_gate_anchor_iteration"] == 4
    assert debug["action_delta_gate_terminal_iteration"] == 5
    assert debug["action_delta_gate_returned_action_source_iteration"] == 4
    assert debug["action_delta_gate_skipped_coda_count"] == 1
    assert debug["final_state_coda_executed"] is False
    assert debug["returned_cached_final_output"] is False
    assert debug["coda_call_count"] == model.test_coda_calls == 4
    assert debug["get_output_call_count"] == model.test_coda_calls
    assert debug["iteration_mse"] == [1.0, 1.0, 1.0]
    assert debug["conv_score_list"] == [1.0, 1.0, 1.0]


def test_min_terminal_five_nontrigger_executes_terminal_coda_once():
    model = make_model()
    kwargs = gate_kwargs(make_gate(predicted_delta=1.0, threshold=0.1))
    kwargs.update(
        max_iter=5,
        action_delta_gate_min_terminal_iter=5,
    )
    output, actual_iter, final_mse = model(
        *inputs(),
        warm_start_state=torch.zeros(1, 2, 4, dtype=torch.bfloat16),
        kl_thresh=0.001,
        **kwargs,
    )

    assert actual_iter == 5
    assert final_mse == 1.0
    assert model.test_coda_calls == 5
    assert torch.all(output == 5)
    debug = model.last_recurrence_debug
    assert debug["action_delta_gate_triggered"] is False
    assert debug["action_delta_gate_score_call_count"] == 1
    assert debug["action_delta_gate_score_trace"][0]["anchor_iteration"] == 4
    assert debug["action_delta_gate_score_trace"][0]["terminal_iteration"] == 5
    assert debug["coda_call_count"] == model.test_coda_calls
    assert debug["get_output_call_count"] == model.test_coda_calls


def test_delayed_gate_fails_closed_on_nonfinite_preeligibility_anchor():
    model = make_model(nonfinite_iteration=3)
    kwargs = gate_kwargs(make_gate(predicted_delta=0.0))
    kwargs.update(
        max_iter=5,
        action_delta_gate_min_terminal_iter=5,
    )
    output, actual_iter, _ = model(
        *inputs(),
        warm_start_state=torch.zeros(1, 2, 4, dtype=torch.bfloat16),
        kl_thresh=0.001,
        **kwargs,
    )

    assert actual_iter == 5
    assert torch.isnan(output).any()
    assert model.test_coda_calls == 5
    debug = model.last_recurrence_debug
    assert debug["action_delta_gate_triggered"] is False
    assert debug["action_delta_gate_score_call_count"] == 0
    assert "anchor output is non-finite" in debug["action_delta_gate_fallback_reason"]
    assert debug["coda_call_count"] == model.test_coda_calls


@pytest.mark.parametrize("minimum", [1, True, 2.5])
def test_enabled_gate_rejects_invalid_minimum_terminal_before_recurrence(minimum):
    model = make_model()
    with pytest.raises(ValueError, match="minimum terminal iteration"):
        model(
            *inputs(),
            warm_start_state=torch.zeros(1, 2, 4, dtype=torch.bfloat16),
            action_delta_gate_min_terminal_iter=minimum,
            **gate_kwargs(make_gate()),
        )
    assert model.test_iteration == 0


def test_enabled_gate_rejects_nonboolean_exact_coda_audit_before_recurrence():
    model = make_model()
    with pytest.raises(ValueError, match="exact Coda audit must be boolean"):
        model(
            *inputs(),
            warm_start_state=torch.zeros(1, 2, 4, dtype=torch.bfloat16),
            action_delta_gate_exact_coda_audit=1,
            **gate_kwargs(make_gate()),
        )
    assert model.test_iteration == 0


@pytest.mark.parametrize("return_mode", ["", "corrected", None, 1])
def test_rejects_invalid_action_delta_gate_return_mode_before_recurrence(
    return_mode,
):
    model = make_model()
    with pytest.raises(ValueError, match="return mode"):
        model(
            *inputs(),
            warm_start_state=torch.zeros(1, 2, 4, dtype=torch.bfloat16),
            action_delta_gate_return_mode=return_mode,
            **gate_kwargs(make_gate()),
        )
    assert model.test_iteration == 0


def test_nonfinite_gate_state_fails_closed_and_executes_exact_coda():
    model = make_model(nonfinite_iteration=2)
    output, actual_iter, _ = model(
        *inputs(),
        warm_start_state=torch.zeros(1, 2, 4, dtype=torch.bfloat16),
        kl_thresh=0.001,
        max_iter=2,
        **{key: value for key, value in gate_kwargs(make_gate()).items() if key != "max_iter"},
    )
    assert actual_iter == 2
    assert model.test_coda_calls == 2
    assert torch.isnan(output).any()
    debug = model.last_recurrence_debug
    assert debug["action_delta_gate_triggered"] is False
    assert "non-finite" in debug["action_delta_gate_fallback_reason"]
    assert debug["action_delta_gate_score_trace"][0]["score"] is None
    assert debug["coda_call_count"] == model.test_coda_calls


def test_enabled_gate_rejects_batch_size_greater_than_one_before_recurrence():
    model = make_model()
    with pytest.raises(ValueError, match="batch size 1"):
        model(
            *inputs(batch=2),
            warm_start_state=torch.zeros(2, 2, 4, dtype=torch.bfloat16),
            **gate_kwargs(make_gate()),
        )
    assert model.test_iteration == 0


def test_configuring_gate_does_not_change_checkpoint_keys():
    cfg = RecurrentConfigInternal(
        hidden_dim=4,
        num_heads=1,
        recurrent_vlm_layers=(0,),
        action_chunk_len=2,
        action_dim=2,
    )
    head = ActionHeadRecurrent(hidden_dim=4, action_dim=2, cfg=cfg)
    keys_before = tuple(head.state_dict())
    head.configure_action_delta_gate(make_gate())
    assert tuple(head.state_dict()) == keys_before
