import copy
import json
import types

import pytest
import torch

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


def test_gate_disabled_preserves_baseline_and_does_not_score():
    baseline = make_model()
    supported = copy.deepcopy(baseline)
    warm = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
    kwargs = gate_kwargs(make_gate())
    kwargs.update(use_action_delta_gate=False, action_delta_gate=None, kl_thresh=2.0)

    first = baseline(*inputs(), warm_start_state=warm, **kwargs)
    second = supported(*inputs(), warm_start_state=warm, **kwargs)
    torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
    assert first[1:] == second[1:]
    assert baseline.test_coda_calls == supported.test_coda_calls == 2
    assert baseline.last_recurrence_debug["action_delta_gate_score_call_count"] == 0


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
    assert model.last_inference_metadata["warm_start"]["candidate_state_count"] == 2
    json.dumps(debug, allow_nan=False)


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
