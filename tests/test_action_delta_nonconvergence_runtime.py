import math
import types

import pytest
import torch

import prismatic.models.action_delta_gate as action_delta_gate_module
from prismatic.models.action_delta_gate import (
    ACTION_DELTA_GATE_ARTIFACT_TYPE,
    ACTION_DELTA_GATE_CALIBRATION_METHOD,
    ACTION_DELTA_GATE_MODEL_TYPE,
    ACTION_DELTA_NONCONVERGENCE_THRESHOLD,
    PreparedActionDeltaGate,
)
from prismatic.models.action_heads import RecurrentConfigInternal, VLARecurrent


def make_gate(predicted_delta):
    return PreparedActionDeltaGate(
        schema_version=1,
        artifact_type=ACTION_DELTA_GATE_ARTIFACT_TYPE,
        model_type=ACTION_DELTA_GATE_MODEL_TYPE,
        hidden_dim=4,
        action_dim=2,
        action_chunk_len=2,
        held_out_task_ids=(4, 5),
        outer_fold=4,
        threshold=0.0007,
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


def make_model(outputs=None, nonfinite_state_iteration=None):
    config = RecurrentConfigInternal(
        hidden_dim=4,
        num_heads=1,
        recurrent_vlm_layers=(0,),
        coda_vlm_layers=(),
        action_chunk_len=2,
        action_dim=2,
        mean_recurrence=8,
        backprop_depth=1,
        random_iterations=False,
    )
    model = VLARecurrent(config).eval()
    model.test_iteration = 0
    model.test_coda_iterations = []

    def init_state(self, batch_size, device, dtype):
        return torch.zeros(batch_size, 2, 4, device=device, dtype=dtype)

    def run_one_iteration(self, state, *args):
        self.test_iteration += 1
        result = state + 1
        if self.test_iteration == nonfinite_state_iteration:
            result = result.clone()
            result[0, 0, 0] = float("nan")
        return result

    def get_output(self, state, *args, profile=False):
        self.test_coda_iterations.append(self.test_iteration)
        if profile:
            self._last_get_output_timing = {
                "get_output_ms": 0.3,
                "coda_ms": 0.2,
                "output_proj_ms": 0.1,
            }
        if outputs is None:
            return state[..., :2].float().clone()
        return outputs[self.test_iteration].detach().clone()

    model.init_state = types.MethodType(init_state, model)
    model._run_one_iteration = types.MethodType(run_one_iteration, model)
    model._get_output = types.MethodType(get_output, model)
    return model


def inputs():
    h_a = torch.zeros(1, 1, 1, 4, dtype=torch.bfloat16)
    return h_a, torch.zeros_like(h_a), torch.zeros(1, 1, 4, dtype=torch.bfloat16)


def kwargs(gate, *, max_iter=8):
    return {
        "convergence_strategy": "adjacent_action_mse",
        "kl_thresh": 0.001,
        "enable_warm_start": True,
        "warm_start_source": "midpoint",
        "warm_start_min_iter": 2,
        "use_cached_final_output": True,
        "use_latent_precheck": False,
        "latent_precheck_mode": "off",
        "latent_precheck_trace_level": "off",
        "profile_coda_cost": True,
        "use_action_delta_gate": False,
        "action_delta_gate": gate,
        "action_delta_gate_min_terminal_iter": 5,
        "use_action_delta_nonconvergence_filter": True,
        "max_iter": max_iter,
    }


def run(model, gate, *, max_iter=8):
    return model(
        *inputs(),
        warm_start_state=torch.zeros(1, 2, 4, dtype=torch.bfloat16),
        **kwargs(gate, max_iter=max_iter),
    )


def test_mode_off_preserves_exact_path_and_never_scores(monkeypatch):
    score_calls = 0

    def forbidden_score(*args, **kwargs):
        nonlocal score_calls
        score_calls += 1
        raise AssertionError("disabled mode evaluated the predictor")

    monkeypatch.setattr(action_heads_score_module(), "evaluate_action_delta_gate", forbidden_score)
    model = make_model()
    result = model(
        *inputs(),
        warm_start_state=torch.zeros(1, 2, 4, dtype=torch.bfloat16),
        convergence_strategy="adjacent_action_mse",
        kl_thresh=0.001,
        enable_warm_start=True,
        warm_start_source="midpoint",
        warm_start_min_iter=2,
        use_cached_final_output=True,
        latent_precheck_mode="off",
        latent_precheck_trace_level="off",
        use_action_delta_nonconvergence_filter=False,
        max_iter=5,
    )
    assert result[1] == 5
    assert score_calls == 0
    assert model.test_coda_iterations == [1, 2, 3, 4, 5]
    debug = model.last_recurrence_debug
    assert debug["action_delta_nonconvergence_filter_score_call_count"] == 0
    assert debug["action_delta_nonconvergence_filter_actual_coda_skip_count"] == 0


def test_cold_origin_prediction_is_ineligible_and_executes_only_exact_coda():
    model = make_model()
    config = kwargs(make_gate(0.04), max_iter=5)

    model(*inputs(), warm_start_state=None, **config)

    assert model.test_coda_iterations == [1, 2, 3, 4, 5]
    debug = model.last_recurrence_debug
    assert debug["action_delta_nonconvergence_filter_requested"] is True
    assert debug["action_delta_nonconvergence_filter_applied"] is False
    assert debug["action_delta_nonconvergence_filter_score_call_count"] == 0


def action_heads_score_module():
    import prismatic.models.action_heads as action_heads_module

    return action_heads_module


def test_high_score_skips_then_forces_exact_and_only_exact_can_stop(monkeypatch):
    linear_calls = 0
    original_linear = action_delta_gate_module.F.linear

    def linear_spy(*args, **kwargs):
        nonlocal linear_calls
        linear_calls += 1
        return original_linear(*args, **kwargs)

    monkeypatch.setattr(action_delta_gate_module.F, "linear", linear_spy)
    outputs = {
        1: torch.zeros(1, 2, 2),
        2: torch.ones(1, 2, 2),
        3: torch.full((1, 2, 2), 2.0),
        4: torch.full((1, 2, 2), 3.0),
        6: torch.full((1, 2, 2), 3.01),
    }
    model = make_model(outputs)

    output, terminal_iteration, final_mse = run(model, make_gate(0.04))

    assert linear_calls == 1
    assert terminal_iteration == 6
    assert model.test_coda_iterations == [1, 2, 3, 4, 6]
    torch.testing.assert_close(output, outputs[6], rtol=0, atol=0)
    assert final_mse == pytest.approx(0.0001, rel=1e-4)
    debug = model.last_recurrence_debug
    assert debug["stop_reason"] == "adjacent_action_mse"
    assert debug["action_delta_nonconvergence_filter_predicted_event_count"] == 1
    assert debug["action_delta_nonconvergence_filter_actual_coda_skip_count"] == 1
    assert debug["action_delta_nonconvergence_filter_forced_next_coda_call_count"] == 1
    assert debug["action_delta_nonconvergence_filter_exact_coda_call_count"] == 5
    event = debug["action_delta_nonconvergence_filter_events"][0]
    assert event["last_exact_anchor_iteration"] == 4
    assert event["skipped_terminal_iteration"] == 5
    assert event["forced_exact_terminal_iteration"] == 6
    assert event["exact_mse_at_forced_confirmation"] == pytest.approx(0.0001, rel=1e-4)
    assert event["stopping_occurred_at_forced_confirmation"] is True
    assert event["extra_recurrent_iterations_after_skip"] == 1


def test_max_skip_one_prevents_consecutive_skips_and_updates_exact_anchor():
    model = make_model()

    _, terminal_iteration, _ = run(model, make_gate(0.04))

    assert terminal_iteration == 8
    assert model.test_coda_iterations == [1, 2, 3, 4, 6, 8]
    debug = model.last_recurrence_debug
    assert debug["action_delta_nonconvergence_filter_score_call_count"] == 2
    assert debug["action_delta_nonconvergence_filter_actual_coda_skip_count"] == 2
    assert debug["action_delta_nonconvergence_filter_forced_next_coda_call_count"] == 2
    assert debug[
        "action_delta_nonconvergence_filter_consecutive_skip_prevention_count"
    ] == 2
    events = debug["action_delta_nonconvergence_filter_events"]
    assert [(event["last_exact_anchor_iteration"], event["skipped_terminal_iteration"])
            for event in events] == [(4, 5), (6, 7)]
    assert [event["forced_exact_terminal_iteration"] for event in events] == [6, 8]


def test_low_score_executes_terminal_coda_and_predictor_cannot_stop():
    outputs = {
        1: torch.zeros(1, 2, 2),
        2: torch.ones(1, 2, 2),
        3: torch.full((1, 2, 2), 2.0),
        4: torch.full((1, 2, 2), 3.0),
        5: torch.full((1, 2, 2), 3.01),
    }
    model = make_model(outputs)

    _, terminal_iteration, final_mse = run(model, make_gate(0.01))

    assert terminal_iteration == 5
    assert model.test_coda_iterations == [1, 2, 3, 4, 5]
    assert final_mse == pytest.approx(0.0001, rel=1e-4)
    debug = model.last_recurrence_debug
    assert debug["action_delta_nonconvergence_filter_score_call_count"] == 1
    assert debug["action_delta_nonconvergence_filter_predicted_event_count"] == 0
    assert debug["action_delta_nonconvergence_filter_actual_coda_skip_count"] == 0
    assert debug["stop_reason"] == "adjacent_action_mse"


def test_high_score_at_max_iteration_fails_closed_instead_of_unconfirmed_skip():
    model = make_model()

    _, terminal_iteration, _ = run(model, make_gate(0.04), max_iter=5)

    assert terminal_iteration == 5
    assert model.test_coda_iterations == [1, 2, 3, 4, 5]
    debug = model.last_recurrence_debug
    assert debug["action_delta_nonconvergence_filter_predicted_event_count"] == 1
    assert debug["action_delta_nonconvergence_filter_actual_coda_skip_count"] == 0
    assert debug[
        "action_delta_nonconvergence_filter_max_iter_skip_prevention_count"
    ] == 1
    assert debug["action_delta_nonconvergence_filter_forced_next_coda_call_count"] == 0


def test_nonfinite_score_fails_closed_to_exact_coda():
    model = make_model(nonfinite_state_iteration=5)

    run(model, make_gate(0.04), max_iter=6)

    assert 5 in model.test_coda_iterations
    debug = model.last_recurrence_debug
    assert debug["action_delta_nonconvergence_filter_score_call_count"] == 1
    assert debug["action_delta_nonconvergence_filter_actual_coda_skip_count"] == 0
    assert "non-finite" in debug["action_delta_nonconvergence_filter_fallback_reason"]


def test_timing_and_savings_accounting_is_internally_consistent():
    model = make_model()
    run(model, make_gate(0.04))
    debug = model.last_recurrence_debug

    assert len(debug["action_delta_nonconvergence_filter_predictor_ms_list"]) == 2
    assert debug["action_delta_nonconvergence_filter_predictor_ms_total"] == pytest.approx(
        sum(debug["action_delta_nonconvergence_filter_predictor_ms_list"])
    )
    assert debug["action_delta_nonconvergence_filter_exact_coda_call_count"] == len(
        debug["get_output_ms_list"]
    )
    assert debug["action_delta_nonconvergence_filter_get_output_ms_total"] == pytest.approx(
        sum(debug["get_output_ms_list"])
    )
    assert debug["action_delta_nonconvergence_filter_coda_ms_total"] == pytest.approx(
        sum(debug["coda_ms_list"])
    )
    assert math.isfinite(
        debug["action_delta_nonconvergence_filter_estimated_net_savings_ms"]
    )
    assert debug["action_delta_nonconvergence_filter_measured_net_savings_ms"] is None
    assert debug["action_delta_nonconvergence_filter_efficiency_eligible"] is False
    assert ACTION_DELTA_NONCONVERGENCE_THRESHOLD == 0.0015
