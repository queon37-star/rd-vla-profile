import copy
import inspect
import types
from unittest.mock import patch

import pytest
import torch

from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.models.action_heads import (
    ActionHeadRecurrent,
    RecurrentConfigInternal,
    VLARecurrent,
)
from prismatic.models.latent_metrics import compute_latent_metrics
from prismatic.models.latent_only_stopping import NonFiniteLatentOnlyInferenceError


@pytest.mark.parametrize(
    ("metric_name", "expected"),
    [
        ("raw_mse", 2.5),
        ("relative_mse", 1.0),
        ("cosine_distance", 0.0),
        ("relative_l2", 1.0),
    ],
)
def test_latent_metric_formulas(metric_name, expected):
    previous = torch.tensor([1.0, 2.0], dtype=torch.float16)
    current = torch.tensor([2.0, 4.0], dtype=torch.float16)
    metrics = compute_latent_metrics(current, previous, eps=1e-12)
    assert metrics[metric_name] == pytest.approx(expected, abs=1e-6)


def _model(*, increments=None, nonfinite_iteration=None, nonfinite_output=False):
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
    model.test_calls = {"recurrent": 0, "output": 0}
    increments = increments or [1.0] * 8

    def init_state(self, batch_size, device, dtype):
        return torch.zeros(batch_size, 2, 4, device=device, dtype=dtype)

    def run_one_iteration(self, state, *args):
        self.test_calls["recurrent"] += 1
        if self.test_calls["recurrent"] == nonfinite_iteration:
            return torch.full_like(state, float("nan"))
        return state + increments[self.test_calls["recurrent"] - 1]

    def get_output(self, state, *args, profile=False):
        self.test_calls["output"] += 1
        if nonfinite_output:
            return torch.full((1, 2, 2), float("inf"), dtype=state.dtype)
        return state[..., :2].clone()

    model.init_state = types.MethodType(init_state, model)
    model._run_one_iteration = types.MethodType(run_one_iteration, model)
    model._get_output = types.MethodType(get_output, model)
    return model


def _inputs():
    return (
        torch.zeros(1, 1, 1, 4),
        torch.zeros(1, 1, 1, 4),
        torch.zeros(1, 1, 4),
    )


def _run(model, **kwargs):
    settings = {
        "convergence_strategy": "latent_only",
        "max_iter": 4,
        "latent_only_metric": "raw_mse",
        "latent_only_cold_threshold": 0.0,
        "latent_only_warm_threshold": 0.0,
        "latent_only_min_iter": 2,
        "latent_only_eps": 1e-8,
        "use_latent_precheck": False,
        "latent_precheck_mode": "legacy",
    }
    settings.update(kwargs)
    return model(*_inputs(), **settings)


def test_threshold_stop_calls_coda_once_after_recurrence_and_preserves_midpoint():
    model = _model(increments=[1.0, 1.0, 0.1, 1.0])
    with patch(
        "prismatic.models.action_heads.run_origin_aware_adaptive",
        side_effect=AssertionError("origin-aware scheduler must not run"),
    ):
        result = _run(
            model,
            latent_only_cold_threshold=0.02,
            enable_warm_start=True,
            warm_start_source="midpoint",
        )

    assert result[1] == 3
    assert result[2] == pytest.approx(0.01, abs=1e-6)
    assert model.test_calls == {"recurrent": 3, "output": 1}
    debug = model.last_recurrence_debug
    assert debug["stop_reason"] == "latent_threshold"
    assert debug["coda_call_count"] == 1
    assert debug["latent_metric_call_count"] == 2
    assert debug["action_mse_threshold"] is None
    assert debug["latent_action_mse_pairs"] == []
    assert debug["execution_path"] == "latent_only"
    metadata = model.last_inference_metadata
    assert metadata["warm_start"]["source"] == "midpoint"
    assert metadata["warm_start"]["source_iteration"] == 1


def test_min_iter_delays_metric_and_stop():
    model = _model()
    result = _run(
        model,
        latent_only_cold_threshold=2.0,
        latent_only_min_iter=4,
    )
    assert result[1] == 4
    assert model.last_recurrence_debug["stop_reason"] == "latent_threshold"
    assert model.last_recurrence_debug["latent_metric_call_count"] == 1


def test_max_iter_fallback_still_calls_coda_exactly_once():
    model = _model()
    result = _run(model)
    assert result[1:] == (4, 1.0)
    assert model.test_calls == {"recurrent": 4, "output": 1}
    assert model.last_recurrence_debug["stop_reason"] == "max_iter"
    assert model.last_recurrence_debug["coda_call_count"] == 1


def test_actual_warm_threshold_depends_on_accepted_cache_not_enabled_flag():
    cold_model = _model()
    _run(
        cold_model,
        enable_warm_start=True,
        warm_start_source="midpoint",
        latent_only_cold_threshold=0.0,
        latent_only_warm_threshold=2.0,
    )
    assert cold_model.last_recurrence_debug["actual_origin"] == "COLD"
    assert cold_model.last_recurrence_debug["effective_threshold"] == 0.0
    assert cold_model.last_recurrence_debug["K_t"] == 4

    warm_model = _model()
    _run(
        warm_model,
        warm_start_state=torch.zeros(1, 2, 4),
        enable_warm_start=True,
        warm_start_source="midpoint",
        latent_only_cold_threshold=0.0,
        latent_only_warm_threshold=2.0,
    )
    assert warm_model.last_recurrence_debug["actual_origin"] == "ACTUAL_WARM"
    assert warm_model.last_recurrence_debug["effective_threshold"] == 2.0
    assert warm_model.last_recurrence_debug["K_t"] == 2

    rejected_model = _model()
    _run(
        rejected_model,
        warm_start_state=torch.full((1, 2, 4), float("nan")),
        enable_warm_start=True,
        warm_start_source="midpoint",
        latent_only_cold_threshold=0.0,
        latent_only_warm_threshold=2.0,
    )
    assert rejected_model.last_recurrence_debug["actual_origin"] == "COLD"
    assert rejected_model.last_inference_metadata["warm_start"]["state_used"] is False


@pytest.mark.parametrize(
    ("model", "expected_output_calls"),
    [
        (_model(nonfinite_iteration=2), 0),
        (_model(nonfinite_output=True), 1),
    ],
)
def test_nonfinite_latent_only_fails_without_returning_invalid_action(
    model, expected_output_calls
):
    with pytest.raises(NonFiniteLatentOnlyInferenceError):
        _run(model)
    assert model.test_calls["output"] == expected_output_calls


def test_legacy_action_mse_behavior_is_unchanged_after_latent_only_run():
    model = _model()
    baseline = _run(
        model,
        convergence_strategy="adjacent_action_mse",
        kl_thresh=2.0,
        latent_precheck_mode="off",
        use_cached_final_output=True,
    )
    baseline_debug = copy.deepcopy(model.last_recurrence_debug)
    assert baseline[1:] == (2, 1.0)
    assert baseline_debug["canonical_recurrence_strategy"] == "adjacent_action_mse"
    assert baseline_debug["stop_reason"] == "adjacent_action_mse"


@pytest.mark.parametrize(
    "callable_obj",
    [
        VLARecurrent.forward,
        ActionHeadRecurrent.predict_action,
        OpenVLAForActionPrediction._regression_or_discrete_prediction,
        OpenVLAForActionPrediction.predict_action,
    ],
)
def test_latent_only_options_are_present_in_every_model_wrapper(callable_obj):
    parameters = list(inspect.signature(callable_obj).parameters)
    expected = [
        "latent_only_metric",
        "latent_only_cold_threshold",
        "latent_only_warm_threshold",
        "latent_only_min_iter",
        "latent_only_eps",
    ]
    assert [name for name in parameters if name in expected] == expected
