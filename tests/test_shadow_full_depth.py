import copy
import inspect
import types

import pytest
import torch

from configs.rdvla_precheck import validate_latent_precheck_configuration
from experiments.robot.libero.latent_metric_trace import (
    build_action_head_workload_identity,
    build_latent_metric_trace_records,
    require_prediction_id,
)
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.models.action_heads import (
    ActionHeadRecurrent,
    RecurrentConfigInternal,
    VLARecurrent,
)


def _tiny_shadow_model(*, nonfinite_recurrence_call=None):
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
        if calls["recurrent"] == nonfinite_recurrence_call:
            return torch.full_like(state, float("nan"))
        return state + 1

    def get_output(self, state, *args, profile=False):
        calls["output"] += 1
        return state[..., :2].clone()

    model.init_state = types.MethodType(init_state, model)
    model._run_one_iteration = types.MethodType(run_one_iteration, model)
    model._get_output = types.MethodType(get_output, model)
    model.test_calls = calls
    return model


def _inputs():
    return (
        torch.zeros(1, 1, 1, 4),
        torch.zeros(1, 1, 1, 4),
        torch.zeros(1, 1, 4),
    )


def _run(model, *, shadow_full_depth):
    result = model(
        *_inputs(),
        convergence_strategy="kl_divergence",
        kl_thresh=2.0,
        max_iter=5,
        warm_start_state=torch.zeros(1, 2, 4),
        enable_warm_start=True,
        warm_start_source="midpoint",
        use_cached_final_output=True,
        use_latent_precheck=False,
        latent_precheck_mode="off",
        latent_precheck_trace_level="off",
        shadow_full_depth=shadow_full_depth,
    )
    return (
        result,
        copy.deepcopy(model.last_recurrence_debug),
        copy.deepcopy(model.last_inference_metadata),
    )


def test_shadow_full_depth_preserves_production_result_and_midpoint_cache():
    baseline_model = _tiny_shadow_model()
    shadow_model = _tiny_shadow_model()

    baseline, baseline_debug, baseline_metadata = _run(
        baseline_model, shadow_full_depth=False
    )
    shadow, shadow_debug, shadow_metadata = _run(
        shadow_model, shadow_full_depth=True
    )

    torch.testing.assert_close(shadow[0], baseline[0], rtol=0, atol=0)
    assert shadow[1:] == baseline[1:] == (2, 1.0)
    assert shadow_debug["K_t"] == baseline_debug["K_t"] == 2
    assert shadow_debug["stop_reason"] == baseline_debug["stop_reason"] == "kl_divergence"
    assert shadow_debug["canonical_stop_reason"] == baseline_debug["canonical_stop_reason"]
    torch.testing.assert_close(
        shadow_metadata["next_warm_start_state"],
        baseline_metadata["next_warm_start_state"],
        rtol=0,
        atol=0,
    )
    assert shadow_metadata["warm_start"] == baseline_metadata["warm_start"]

    assert baseline_model.test_calls == {"recurrent": 2, "output": 2}
    assert shadow_model.test_calls == {"recurrent": 5, "output": 5}
    assert shadow_debug["shadow_full_depth_enabled"] is True
    assert shadow_debug["shadow_trace_complete"] is True
    assert shadow_debug["shadow_tail_start_iteration"] == 3
    assert shadow_debug["shadow_tail_iteration_count"] == 3
    assert shadow_debug["shadow_error"] is None
    assert [record["k"] for record in shadow_debug["shadow_trace"]] == [1, 2, 3, 4, 5]
    assert [record["phase"] for record in shadow_debug["shadow_trace"]] == [
        "production",
        "production",
        "shadow_tail",
        "shadow_tail",
        "shadow_tail",
    ]
    assert shadow_debug["shadow_trace"][0]["action_mse"] is None
    assert all(
        record["action_mse"] == 1.0
        for record in shadow_debug["shadow_trace"][1:]
    )
    assert all(
        set(("raw_mse", "relative_mse", "cosine_distance", "relative_l2"))
        <= set(record)
        for record in shadow_debug["shadow_trace"]
    )
    assert shadow_debug["latent_metric_trace_enabled"] is True
    assert shadow_debug["shadow_production_snapshot"] == {
        "K_t": 2,
        "terminal_iteration": 2,
        "stop_reason": "kl_divergence",
        "midpoint_source_iteration": 1,
        "cached_final_output_reused": True,
    }


def test_first_prediction_trace_has_canonical_id_and_preserves_production():
    baseline_model = _tiny_shadow_model()
    trace_model = _tiny_shadow_model()
    baseline, baseline_debug, baseline_metadata = _run(
        baseline_model, shadow_full_depth=False
    )
    traced, trace_debug, trace_metadata = _run(
        trace_model, shadow_full_depth=True
    )

    prediction_id = require_prediction_id(0)
    trace_records = build_latent_metric_trace_records(
        trace_debug,
        task_id=3,
        episode_id=0,
        prediction_id=prediction_id,
        actual_origin="ACTUAL_WARM",
    )

    assert prediction_id == 0
    assert trace_records
    assert {
        (record["task_id"], record["episode_id"], record["prediction_id"])
        for record in trace_records
    } == {(3, 0, 0)}
    assert trace_debug["canonical_recurrence_strategy"] == "adjacent_action_mse"
    assert trace_debug["shadow_full_depth_enabled"] is True
    assert trace_debug["latent_metric_trace_enabled"] is True
    torch.testing.assert_close(traced[0], baseline[0], rtol=0, atol=0)
    assert traced[1:] == baseline[1:]
    assert trace_debug["K_t"] == baseline_debug["K_t"]
    assert trace_debug["stop_reason"] == baseline_debug["stop_reason"]
    torch.testing.assert_close(
        trace_metadata["next_warm_start_state"],
        baseline_metadata["next_warm_start_state"],
        rtol=0,
        atol=0,
    )
    assert trace_metadata["warm_start"] == baseline_metadata["warm_start"]

    next_records = build_latent_metric_trace_records(
        trace_debug,
        task_id=3,
        episode_id=0,
        prediction_id=prediction_id + 1,
        actual_origin="ACTUAL_WARM",
    )
    prediction_keys = {
        (record["task_id"], record["episode_id"], record["prediction_id"])
        for record in trace_records + next_records
    }
    assert prediction_keys == {(3, 0, 0), (3, 0, 1)}


def test_prediction_id_does_not_silently_coerce_none_and_legacy_needs_no_protocol_identity():
    with pytest.raises(ValueError, match="non-null monotonically increasing integer"):
        require_prediction_id(None)

    assert (
        build_action_head_workload_identity(
            capture_requested=False,
            task_id=0,
            episode_id=0,
            paired_trial_id=None,
            prediction_id=0,
            initial_state_id=None,
            episode_seed=None,
        )
        is None
    )


def test_shadow_tail_nonfinite_is_diagnostic_only():
    baseline_model = _tiny_shadow_model()
    shadow_model = _tiny_shadow_model(nonfinite_recurrence_call=4)

    baseline, _, baseline_metadata = _run(baseline_model, shadow_full_depth=False)
    shadow, shadow_debug, shadow_metadata = _run(
        shadow_model, shadow_full_depth=True
    )

    torch.testing.assert_close(shadow[0], baseline[0], rtol=0, atol=0)
    assert shadow[1:] == baseline[1:] == (2, 1.0)
    torch.testing.assert_close(
        shadow_metadata["next_warm_start_state"],
        baseline_metadata["next_warm_start_state"],
        rtol=0,
        atol=0,
    )
    assert shadow_debug["shadow_trace_complete"] is False
    assert shadow_debug["shadow_tail_iteration_count"] == 1
    assert shadow_debug["shadow_error"] == {
        "iteration": 4,
        "stage": "recurrent_state",
        "reason": "non_finite",
    }
    assert [record["k"] for record in shadow_debug["shadow_trace"]] == [1, 2, 3]
    assert torch.isfinite(shadow[0]).all()


def _valid_shadow_configuration(**overrides):
    values = {
        "mode": "off",
        "trace_level": "off",
        "use_latent_precheck": False,
        "warm_start_source": "midpoint",
        "recurrence_strategy": "kl_divergence",
        "use_warm_start": True,
        "shadow_full_depth": True,
    }
    values.update(overrides)
    return validate_latent_precheck_configuration(**values)


def test_shadow_configuration_accepts_only_clean_midpoint_action_mse():
    assert _valid_shadow_configuration() == "off"
    assert _valid_shadow_configuration(
        recurrence_strategy="adjacent_action_mse"
    ) == "off"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"shadow_full_depth": 1}, "must be a boolean"),
        ({"mode": "legacy"}, "requires clean"),
        ({"use_warm_start": False}, "requires midpoint"),
        ({"warm_start_source": "final"}, "requires midpoint"),
        ({"recurrence_strategy": "cosine_similarity"}, "requires adjacent action-MSE"),
    ],
)
def test_shadow_configuration_rejects_unsafe_combinations(overrides, message):
    with pytest.raises(ValueError, match=message):
        _valid_shadow_configuration(**overrides)


def test_shadow_full_depth_is_inference_only():
    model = _tiny_shadow_model().train()
    with pytest.raises(ValueError, match="inference-only"):
        _run(model, shadow_full_depth=True)


@pytest.mark.parametrize(
    "callable_obj",
    [
        VLARecurrent.forward,
        ActionHeadRecurrent.predict_action,
        OpenVLAForActionPrediction._regression_or_discrete_prediction,
        OpenVLAForActionPrediction.predict_action,
    ],
)
def test_shadow_control_follows_nonfinite_policy_in_public_signatures(callable_obj):
    parameters = list(inspect.signature(callable_obj).parameters)
    assert parameters.index("shadow_full_depth") == parameters.index("nonfinite_policy") + 1
