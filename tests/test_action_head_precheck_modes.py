import contextlib
import copy
import inspect
import types
from unittest.mock import Mock, patch

import pytest
import torch

import prismatic.models.action_heads as action_heads
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.models.action_heads import ActionHeadRecurrent, RecurrentConfigInternal, VLARecurrent


@pytest.fixture
def tiny_recurrent():
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

    def init_state(self, batch_size, device, dtype):
        return torch.zeros(batch_size, 2, 4, device=device, dtype=dtype)

    def run_one_iteration(self, state, *args):
        return state + 1

    def get_output(self, state, *args, profile=False):
        return state[..., :2].clone()

    model.init_state = types.MethodType(init_state, model)
    model._run_one_iteration = types.MethodType(run_one_iteration, model)
    model._get_output = types.MethodType(get_output, model)
    return model


def _inputs():
    h_a = torch.zeros(1, 1, 1, 4)
    h_t = torch.zeros(1, 1, 1, 4)
    proprio = torch.zeros(1, 1, 4)
    return h_a, h_t, proprio


def _run_with_spies(model, *, strategy="kl_divergence", mode="legacy"):
    range_names = []
    item_calls = 0
    original_item = torch.Tensor.item

    def item_spy(tensor, *args, **kwargs):
        nonlocal item_calls
        item_calls += 1
        return original_item(tensor, *args, **kwargs)

    def range_spy(name):
        range_names.append(name)
        return contextlib.nullcontext()

    with patch.object(torch.Tensor, "item", item_spy), patch.object(
        action_heads,
        "rdvla_range",
        side_effect=range_spy,
    ):
        result = model(
            *_inputs(),
            convergence_strategy=strategy,
            kl_thresh=-1.0,
            max_iter=3,
            use_cached_final_output=True,
            use_latent_precheck=False,
            latent_precheck_mode=mode,
            latent_precheck_trace_level="off",
        )

    return result, copy.deepcopy(model.last_recurrence_debug), item_calls, range_names


def test_off_mode_removes_latent_metrics_item_sync_and_gate_trace(tiny_recurrent):
    legacy_result, legacy_debug, legacy_item_calls, legacy_ranges = _run_with_spies(
        tiny_recurrent,
        mode="legacy",
    )
    off_result, off_debug, off_item_calls, off_ranges = _run_with_spies(
        tiny_recurrent,
        mode="off",
    )

    torch.testing.assert_close(off_result[0], legacy_result[0], rtol=0, atol=0)
    assert off_result[1:] == legacy_result[1:] == (3, 1.0)

    # Two latent scalar synchronizations disappear at each of iterations 2 and 3.
    assert legacy_item_calls == 8
    assert off_item_calls == 4
    assert any("latent_precheck" in name for name in legacy_ranges)
    assert not any("latent_precheck" in name for name in off_ranges)

    assert legacy_debug["latent_mse_list"] == [1.0, 1.0]
    assert legacy_debug["latent_l2_list"]
    assert legacy_debug["latent_precheck_trace_collected"] is True
    assert legacy_debug["latent_precheck_trace_level_applied"] is None

    assert off_debug["latent_mse_list"] == []
    assert off_debug["latent_l2_list"] == []
    assert off_debug["latent_action_mse_pairs"] == []
    assert off_debug["latent_precheck_coda_call_mask"] == []
    assert off_debug["latent_precheck_decisions"] == []
    assert off_debug["latent_precheck_call_count"] is None
    assert off_debug["latent_precheck_skip_count"] is None
    assert off_debug["latent_precheck_skip_ratio"] is None
    assert off_debug["latent_precheck_trace_collected"] is False


def test_legacy_alias_and_canonical_strategy_share_internal_behavior(tiny_recurrent):
    alias_result, alias_debug, _, _ = _run_with_spies(
        tiny_recurrent,
        strategy="kl_divergence",
        mode="off",
    )
    canonical_result, canonical_debug, _, _ = _run_with_spies(
        tiny_recurrent,
        strategy="adjacent_action_mse",
        mode="off",
    )

    torch.testing.assert_close(alias_result[0], canonical_result[0], rtol=0, atol=0)
    assert alias_result[1:] == canonical_result[1:]
    assert alias_debug["strategy"] == "kl_divergence"
    assert canonical_debug["strategy"] == "adjacent_action_mse"
    assert alias_debug["canonical_recurrence_strategy"] == "adjacent_action_mse"
    assert canonical_debug["canonical_recurrence_strategy"] == "adjacent_action_mse"


def test_unimplemented_origin_aware_mode_rejects_before_recurrence(tiny_recurrent):
    recurrence = Mock(side_effect=AssertionError("recurrence must not run"))
    tiny_recurrent._run_one_iteration = recurrence
    rng_before = torch.random.get_rng_state().clone()

    with pytest.raises(NotImplementedError, match="not implemented yet"):
        tiny_recurrent(
            *_inputs(),
            convergence_strategy="kl_divergence",
            max_iter=3,
            use_latent_precheck=False,
            latent_precheck_mode="origin_aware",
            latent_precheck_trace_level="off",
        )

    recurrence.assert_not_called()
    assert torch.equal(torch.random.get_rng_state(), rng_before)


@pytest.mark.parametrize(
    "callable_obj",
    [
        VLARecurrent.forward,
        ActionHeadRecurrent.predict_action,
        OpenVLAForActionPrediction._regression_or_discrete_prediction,
        OpenVLAForActionPrediction.predict_action,
    ],
)
def test_new_precheck_controls_follow_legacy_positional_parameters(callable_obj):
    parameters = list(inspect.signature(callable_obj).parameters)
    expected_order = [
        "use_latent_precheck",
        "latent_precheck_thresh",
        "latent_precheck_min_iter",
        "latent_precheck_force_interval",
        "latent_precheck_mode",
        "latent_precheck_trace_level",
    ]

    assert [name for name in parameters if name in expected_order] == expected_order
