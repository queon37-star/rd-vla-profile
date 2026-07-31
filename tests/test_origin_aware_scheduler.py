import types

import pytest
import torch

from prismatic.models.action_heads import RecurrentConfigInternal, VLARecurrent
from prismatic.models.origin_aware_scheduler import NonFiniteOriginAwareInferenceError


@pytest.fixture
def scheduler_model():
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
    model.recurrent_step = 1.0
    model.decoded_state_values = []

    def init_state(self, batch_size, device, dtype):
        return torch.zeros(batch_size, 2, 4, device=device, dtype=dtype)

    def run_one_iteration(self, state, *args):
        return state + self.recurrent_step

    def get_output(self, state, *args, profile=False):
        state_value = int(round(float(state.flatten()[0].item())))
        self.decoded_state_values.append(state_value)
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


def _run_origin_aware(
    model,
    *,
    warm_state=None,
    warm_threshold=0.05,
    max_skip_iters=1,
    confirmation_mode="next_iter",
    max_iter=6,
    action_mse_threshold=-1.0,
    trace_level="full",
):
    model.decoded_state_values.clear()
    return model(
        *_inputs(),
        convergence_strategy="kl_divergence",
        kl_thresh=action_mse_threshold,
        max_iter=max_iter,
        warm_start_state=warm_state,
        enable_warm_start=True,
        warm_start_source="midpoint",
        use_cached_final_output=False,
        use_latent_precheck=True,
        latent_precheck_mode="origin_aware",
        latent_precheck_trace_level=trace_level,
        latent_precheck_warm_thresh=warm_threshold,
        latent_precheck_min_iter=2,
        latent_precheck_max_skip_iters=max_skip_iters,
        latent_precheck_confirmation_mode=confirmation_mode,
    )


def test_next_iter_confirmation_never_compares_gapped_outputs(scheduler_model):
    result = _run_origin_aware(scheduler_model, max_iter=6)
    debug = scheduler_model.last_recurrence_debug

    assert result[1:] == (6, 1.0)
    assert scheduler_model.decoded_state_values == [1, 2, 4, 5, 6]
    assert debug["adjacent_comparison_pairs"] == [[1, 2], [4, 5], [5, 6]]
    assert [2, 4] not in debug["adjacent_comparison_pairs"]
    assert [record["reason"] for record in debug["coda_call_records"]] == [
        "forced_initial",
        "forced_second",
        "max_skip_reached",
        "confirmation",
        "max_iter",
    ]
    assert debug["coda_call_records"][2]["refresh_after_skip"] is True
    assert debug["coda_call_records"][3]["confirmation_pending"] is True
    assert debug["final_state_coda_executed"] is True
    assert debug["final_convergence_evaluable"] is True
    assert debug["returned_cached_final_output"] is True
    assert debug["use_cached_final_output_requested"] is False


def test_backfill_decodes_previous_and_current_states_as_an_adjacent_pair(scheduler_model):
    result = _run_origin_aware(
        scheduler_model,
        confirmation_mode="backfill_pair",
        max_iter=5,
    )
    debug = scheduler_model.last_recurrence_debug

    assert result[1] == 5
    assert scheduler_model.decoded_state_values == [1, 2, 3, 4, 5]
    assert debug["adjacent_comparison_pairs"] == [[1, 2], [3, 4], [4, 5]]
    assert debug["coda_call_records"][2] == {
        "iteration": 3,
        "reason": "backfill_previous",
        "is_backfill": True,
        "scheduler_state_before": "GAPPED",
        "refresh_after_skip": False,
        "confirmation_pending": False,
    }


def test_max_iter_has_priority_and_does_not_add_diagnostic_backfill(scheduler_model):
    result = _run_origin_aware(
        scheduler_model,
        confirmation_mode="backfill_pair",
        max_skip_iters=5,
        max_iter=5,
    )
    debug = scheduler_model.last_recurrence_debug

    assert result[1] == 5
    assert result[2] is None
    assert scheduler_model.decoded_state_values == [1, 2, 5]
    assert debug["stop_reason"] == "max_iter"
    assert debug["adjacent_comparison_pairs"] == [[1, 2]]
    assert debug["max_iteration_convergence_evaluable"] is False
    assert debug["final_mse"] is None
    assert debug["final_convergence_evaluable"] is False
    assert debug["coda_reason_counts"].get("backfill_previous", 0) == 0
    torch.testing.assert_close(result[0], torch.full_like(result[0], 5.0))


def test_actual_origin_selects_warm_or_cold_threshold(scheduler_model):
    scheduler_model.recurrent_step = 0.3

    cold_result = _run_origin_aware(scheduler_model, max_iter=4)
    cold_debug = scheduler_model.last_recurrence_debug.copy()
    cold_decodes = list(scheduler_model.decoded_state_values)

    warm_state = torch.zeros(1, 2, 4)
    warm_result = _run_origin_aware(
        scheduler_model,
        warm_state=warm_state,
        max_iter=4,
    )
    warm_debug = scheduler_model.last_recurrence_debug
    warm_decodes = list(scheduler_model.decoded_state_values)

    assert cold_result[1] == warm_result[1] == 4
    assert cold_debug["latent_precheck_origin"] == "COLD"
    assert cold_debug["latent_precheck_active_threshold"] == pytest.approx(0.2)
    assert cold_decodes == [0, 1, 1, 1]
    assert warm_debug["latent_precheck_origin"] == "ACTUAL_WARM"
    assert warm_debug["latent_precheck_active_threshold"] == pytest.approx(0.05)
    assert warm_decodes == [0, 1, 1]


def test_invalid_warm_dtype_falls_back_to_cold_origin(scheduler_model):
    warm_state = torch.zeros(1, 2, 4, dtype=torch.float64)

    _run_origin_aware(scheduler_model, warm_state=warm_state, max_iter=3)
    debug = scheduler_model.last_recurrence_debug
    warm_metadata = scheduler_model.last_inference_metadata["warm_start"]

    assert debug["latent_precheck_origin"] == "COLD"
    assert warm_metadata["state_used"] is False
    assert warm_metadata["reset"] is True
    assert warm_metadata["reset_reason"].startswith("warm_start_dtype_mismatch")


def test_confirmation_can_stop_only_on_the_adjacent_refreshed_pair(scheduler_model):
    def get_output(self, state, *args, profile=False):
        state_value = int(round(float(state.flatten()[0].item())))
        self.decoded_state_values.append(state_value)
        output_value = {1: 0.0, 2: 10.0, 4: 20.0, 5: 20.0}.get(
            state_value,
            float(state_value),
        )
        return torch.full((1, 2, 2), output_value, dtype=state.dtype)

    scheduler_model._get_output = types.MethodType(get_output, scheduler_model)
    result = _run_origin_aware(
        scheduler_model,
        max_iter=6,
        action_mse_threshold=0.01,
    )
    debug = scheduler_model.last_recurrence_debug

    assert result[1:] == (5, 0.0)
    assert scheduler_model.decoded_state_values == [1, 2, 4, 5]
    assert debug["adaptive_stop"] is True
    assert debug["adjacent_comparison_pairs"] == [[1, 2], [4, 5]]
    assert debug["stop_reason"] == "kl_divergence"
    assert debug["canonical_stop_reason"] == "adjacent_action_mse"
    assert debug["cached_final_matches_returned"] is True
    torch.testing.assert_close(result[0], torch.full_like(result[0], 20.0))


def test_nonfinite_recurrent_state_is_never_decoded(scheduler_model):
    def run_one_iteration(self, state, *args):
        next_state = state + 1
        if float(next_state.flatten()[0].item()) >= 3:
            next_state.fill_(float("inf"))
        return next_state

    scheduler_model._run_one_iteration = types.MethodType(
        run_one_iteration,
        scheduler_model,
    )

    with pytest.raises(NonFiniteOriginAwareInferenceError, match="iteration 3"):
        _run_origin_aware(scheduler_model, max_iter=5)

    assert scheduler_model.decoded_state_values == [1, 2]


def test_trace_off_keeps_scheduler_summary_without_iteration_trace(scheduler_model):
    _run_origin_aware(scheduler_model, max_iter=4, trace_level="off")
    debug = scheduler_model.last_recurrence_debug

    assert debug["latent_precheck_trace_collected"] is False
    assert debug["latent_precheck_decisions"] == []
    assert debug["coda_call_records"] == []
    assert debug["latent_precheck_coda_call_mask"] == []
    assert debug["latent_precheck_call_count"] == 3
