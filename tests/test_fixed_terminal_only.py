import json
import types

import pytest
import torch

from configs.rdvla_precheck import (
    canonicalize_recurrence_strategy,
    validate_fixed_terminal_only_configuration,
)
from experiments.robot.libero.run_libero_eval import (
    GenerateConfig,
    build_decode_call_log_fields,
    resolve_fixed_k_log_value,
    validate_config,
)
from prismatic.models.action_heads import RecurrentConfigInternal, VLARecurrent
from prismatic.models.fixed_terminal_only import (
    NonFiniteFixedTerminalOnlyInferenceError,
)


def _model(*, nonfinite_initial=False, nonfinite_iteration=None, nonfinite_output=False):
    cfg = RecurrentConfigInternal(
        hidden_dim=4,
        num_heads=1,
        prelude_vlm_layers=(),
        recurrent_vlm_layers=(0,),
        coda_vlm_layers=(),
        action_chunk_len=2,
        action_dim=2,
        mean_recurrence=4,
        backprop_depth=2,
        random_iterations=False,
    )
    model = VLARecurrent(cfg).eval()
    model.test_calls = {"recurrent": 0, "output": 0}
    model.test_states = []

    def init_state(self, batch_size, device, dtype):
        value = float("nan") if nonfinite_initial else 0.0
        return torch.full(
            (batch_size, 2, 4), value, device=device, dtype=dtype
        )

    def run_one_iteration(self, state, *args):
        self.test_calls["recurrent"] += 1
        if self.test_calls["recurrent"] == nonfinite_iteration:
            state = torch.full_like(state, float("nan"))
        else:
            state = state + 1.0
        self.test_states.append(state.detach().clone())
        return state

    def get_output(self, state, *args, profile=False):
        self.test_calls["output"] += 1
        if profile:
            self._last_get_output_timing = {
                "get_output_ms": 0.3,
                "coda_ms": 0.2,
                "output_proj_ms": 0.1,
            }
        if nonfinite_output:
            return torch.full((1, 2, 2), float("inf"), dtype=state.dtype)
        return state[..., :2].clone()

    model.init_state = types.MethodType(init_state, model)
    model._run_one_iteration = types.MethodType(run_one_iteration, model)
    model._get_output = types.MethodType(get_output, model)
    return model


def _inputs():
    h_a = torch.zeros(1, 1, 1, 4)
    h_t = torch.zeros_like(h_a)
    proprio = torch.zeros(1, 1, 4)
    return h_a, h_t, proprio


def _run(model, *, k, warm_start_state=None, profile=False):
    return model(
        *_inputs(),
        num_iter=k,
        convergence_strategy="fixed_terminal_only",
        max_iter=8,
        warm_start_state=warm_start_state,
        enable_warm_start=True,
        warm_start_source="midpoint",
        profile_coda_cost=profile,
    )


def test_k3_cold_runs_recurrence_then_one_terminal_coda():
    model = _model()

    output, actual_iter, final_metric = _run(model, k=3)

    assert model.test_calls == {"recurrent": 3, "output": 1}
    assert actual_iter == 3
    assert final_metric is None
    assert torch.all(output == 3)
    debug = model.last_recurrence_debug
    assert debug["strategy"] == "fixed_terminal_only"
    assert debug["execution_path"] == "fixed_terminal_only"
    assert debug["actual_origin"] == "COLD"
    assert debug["fixed_K"] == debug["K_t"] == 3
    assert debug["coda_call_count"] == debug["get_output_call_count"] == 1
    assert debug["final_state_coda_executed"] is True
    assert debug["returned_cached_final_output"] is False


def test_k4_actual_warm_uses_provided_state_and_one_terminal_coda():
    model = _model()
    warm_state = torch.full((1, 2, 4), 10.0)

    output, actual_iter, _ = _run(model, k=4, warm_start_state=warm_state)

    assert model.test_calls == {"recurrent": 4, "output": 1}
    assert actual_iter == 4
    assert torch.all(output == 14)
    assert torch.all(model.test_states[0] == 11)
    assert model.last_recurrence_debug["actual_origin"] == "ACTUAL_WARM"
    assert model.last_recurrence_debug["warm_start_state_used"] is True


@pytest.mark.parametrize(
    ("k", "source_index", "source_iteration"),
    [(3, 0, 1), (4, 1, 2)],
)
def test_midpoint_candidate_matches_recurrent_state(
    k, source_index, source_iteration
):
    model = _model()

    _run(model, k=k)

    metadata = model.last_inference_metadata
    warm = metadata["warm_start"]
    assert warm["source_index"] == source_index
    assert warm["source_iteration"] == source_iteration
    torch.testing.assert_close(
        metadata["next_warm_start_state"],
        model.test_states[source_index],
        rtol=0,
        atol=0,
    )
    debug = model.last_recurrence_debug
    assert debug["warm_start_source_index"] == source_index
    assert debug["warm_start_source_iteration"] == source_iteration


@pytest.mark.parametrize("profile", [False, True])
def test_profile_flag_does_not_change_production_call_count(profile):
    model = _model()

    _run(model, k=3, profile=profile)

    assert model.test_calls == {"recurrent": 3, "output": 1}
    debug = model.last_recurrence_debug
    assert debug["profiling_enabled"] is profile
    assert debug["coda_call_count"] == debug["get_output_call_count"] == 1
    if profile:
        assert len(debug["get_output_ms_list"]) == 1


def test_terminal_output_matches_manual_recurrence_and_decode():
    manual = _model()
    h_a, h_t, p = _inputs()
    state = manual.init_state(1, h_a.device, h_a.dtype)
    prelude = torch.zeros(1, 2, 4)
    for _ in range(4):
        state = manual._run_one_iteration(state, prelude, h_a, h_t, p)
    expected = manual._get_output(state, h_a, h_t, p)

    strategy = _model()
    actual, actual_iter, _ = _run(strategy, k=4)

    assert actual_iter == 4
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_legacy_fixed_still_decodes_every_iteration():
    model = _model()

    output, actual_iter, _ = model(
        *_inputs(), num_iter=3, convergence_strategy="fixed", max_iter=8
    )

    assert actual_iter == 3
    assert torch.all(output == 3)
    assert model.test_calls == {"recurrent": 3, "output": 3}
    debug = model.last_recurrence_debug
    assert debug["strategy"] == "fixed"
    assert debug["fixed_K"] == debug["K_t"] == 3
    assert "coda_call_count" not in debug
    assert "execution_path" not in debug


@pytest.mark.parametrize(
    ("model", "stage", "expected_recurrent", "expected_coda"),
    [
        (_model(nonfinite_initial=True), "initial_state", 0, 0),
        (_model(nonfinite_iteration=2), "recurrent_state", 2, 0),
        (_model(nonfinite_output=True), "coda_output", 3, 1),
    ],
)
def test_nonfinite_values_fail_closed(
    model, stage, expected_recurrent, expected_coda
):
    with pytest.raises(NonFiniteFixedTerminalOnlyInferenceError) as caught:
        _run(model, k=3)

    assert caught.value.stage == stage
    assert caught.value.coda_call_count == expected_coda
    assert model.test_calls == {
        "recurrent": expected_recurrent,
        "output": expected_coda,
    }
    debug = model.last_recurrence_debug
    assert debug["coda_call_count"] == expected_coda
    assert debug["get_output_call_count"] == expected_coda


def test_runner_log_fields_are_json_safe_without_profiling():
    model = _model()
    _run(model, k=3, profile=False)
    debug = model.last_recurrence_debug
    warm = model.last_inference_metadata["warm_start"]

    record = {
        "recurrence_strategy": debug["strategy"],
        "canonical_recurrence_strategy": debug["canonical_recurrence_strategy"],
        "fixed_K": resolve_fixed_k_log_value(
            debug, "fixed_terminal_only", 3
        ),
        "K_t": debug["K_t"],
        "actual_origin": debug["actual_origin"],
        "warm_start_source_iteration": warm["source_iteration"],
        **build_decode_call_log_fields(debug),
    }

    assert record["fixed_K"] == 3
    assert record["get_output_call_count"] == 1
    assert record["warm_start_source_iteration"] == 1
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize(
    ("recurrent_num_iter", "recurrence_max_iter", "message"),
    [
        (None, 8, "positive integer"),
        (0, 8, "positive integer"),
        (-1, 8, "positive integer"),
        (9, 8, "must not exceed"),
    ],
)
def test_invalid_fixed_terminal_only_depth_is_rejected(
    recurrent_num_iter, recurrence_max_iter, message
):
    with pytest.raises(ValueError, match=message):
        validate_fixed_terminal_only_configuration(
            "fixed_terminal_only",
            recurrent_num_iter=recurrent_num_iter,
            recurrence_max_iter=recurrence_max_iter,
        )
    with pytest.raises(ValueError, match=message):
        validate_config(
            GenerateConfig(
                recurrence_strategy="fixed_terminal_only",
                recurrent_num_iter=recurrent_num_iter,
                recurrence_max_iter=recurrence_max_iter,
            )
        )


def test_fixed_terminal_only_is_a_canonical_strategy_without_alias():
    assert (
        canonicalize_recurrence_strategy("fixed_terminal_only")
        == "fixed_terminal_only"
    )
