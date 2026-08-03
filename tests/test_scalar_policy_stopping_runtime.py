import time

import pytest
import torch

from prismatic.models.scalar_policy_stopping_runtime import (
    run_scalar_policy_adaptive,
)
from prismatic.models.scalar_stopping_policy import (
    NonFiniteScalarPolicyError,
    PreparedScalarTaskPolicy,
    ScalarStoppingPolicyError,
)


class DummyScalarRuntimeModel:
    def __init__(self, *, nonfinite_iteration=None):
        self.iteration = 0
        self.nonfinite_iteration = (
            nonfinite_iteration
        )
        self.coda_calls = 0
        self.last_recurrence_debug = None
        self._last_get_output_timing = None
        self.stored_candidate = None

    @staticmethod
    def _sync_time():
        return time.perf_counter()

    def _run_one_iteration(
        self,
        state,
        prelude_out,
        h_a,
        h_t,
        p,
    ):
        del prelude_out, h_a, h_t, p

        self.iteration += 1
        result = state + 1.0

        if (
            self.nonfinite_iteration
            == self.iteration
        ):
            result = result.clone()
            result.reshape(-1)[0] = float("nan")

        return result

    def _get_output(
        self,
        state,
        h_a,
        h_t,
        p,
        profile=False,
    ):
        del h_a, h_t, p

        self.coda_calls += 1

        if profile:
            self._last_get_output_timing = {
                "get_output_ms": 0.3,
                "coda_ms": 0.2,
                "output_proj_ms": 0.1,
            }

        return state.clone()

    def _store_warm_start_candidate(
        self,
        states,
        actual_iter,
        source,
    ):
        self.stored_candidate = {
            "count": len(states),
            "actual_iter": actual_iter,
            "source": source,
        }


def make_policy(
    *,
    threshold=0.5,
    bias=-3.0,
):
    return PreparedScalarTaskPolicy(
        task_id=1,
        outer_fold=1,
        threshold=threshold,
        minimum_gate_iteration=3,
        epsilon=1e-8,
        normalizer_mean=torch.zeros(
            7,
            dtype=torch.float32,
        ),
        normalizer_scale=torch.ones(
            7,
            dtype=torch.float32,
        ),
        linear_weight=torch.tensor(
            [1.0, 0, 0, 0, 0, 0, 0],
            dtype=torch.float32,
        ),
        linear_bias=torch.tensor(
            bias,
            dtype=torch.float32,
        ),
    )


def run(mode, policy, *, max_iter=8, profile=False):
    model = DummyScalarRuntimeModel()
    candidates = []

    output, actual_iter, final_score = (
        run_scalar_policy_adaptive(
            model,
            torch.zeros(
                1,
                2,
                3,
                dtype=torch.bfloat16,
            ),
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
            policy=policy,
            execution_mode=mode,
            max_iter=max_iter,
            actual_origin="ACTUAL_WARM",
            requested_recurrence_strategy=(
                "scalar_policy"
            ),
            profile_coda_cost=profile,
            capture_warm_start_candidates=True,
            warm_start_candidate_states=(
                candidates
            ),
            warm_start_source="midpoint",
            warm_start_min_iter_configured=2,
        )
    )

    return (
        model,
        output,
        actual_iter,
        final_score,
    )


def test_direct_stops_at_gate_and_calls_coda_once():
    model, output, actual_iter, score = run(
        "direct",
        make_policy(),
    )

    assert actual_iter == 3
    assert score == pytest.approx(0.5)
    assert model.coda_calls == 1
    assert torch.all(output == 3)
    assert (
        model.last_recurrence_debug[
            "scalar_policy_gate_iteration"
        ]
        == 3
    )
    assert (
        model.last_recurrence_debug[
            "stop_reason"
        ]
        == "scalar_threshold_direct"
    )
    assert model.stored_candidate == {
        "count": 3,
        "actual_iter": 3,
        "source": "midpoint",
    }


def test_confirm_next_runs_exactly_one_more_iteration():
    model, output, actual_iter, score = run(
        "confirm_next",
        make_policy(),
    )

    assert actual_iter == 4
    assert score == pytest.approx(0.5)
    assert model.coda_calls == 1
    assert torch.all(output == 4)

    debug = model.last_recurrence_debug

    assert debug[
        "scalar_policy_gate_iteration"
    ] == 3
    assert debug[
        "scalar_policy_terminal_iteration"
    ] == 4
    assert debug[
        "scalar_policy_score_call_count"
    ] == 1
    assert (
        debug["stop_reason"]
        == "scalar_threshold_confirm_next"
    )


@pytest.mark.parametrize(
    "mode",
    ["direct", "confirm_next"],
)
def test_no_trigger_runs_to_max_iter(mode):
    model, output, actual_iter, score = run(
        mode,
        make_policy(bias=-100.0),
        max_iter=6,
    )

    assert actual_iter == 6
    assert score is not None
    assert model.coda_calls == 1
    assert torch.all(output == 6)

    debug = model.last_recurrence_debug

    assert debug[
        "scalar_policy_gate_iteration"
    ] is None
    assert debug["stop_reason"] == "max_iter"
    assert debug[
        "scalar_policy_score_call_count"
    ] == 4


def test_profile_metadata_includes_scalar_cost():
    model, _, _, _ = run(
        "direct",
        make_policy(),
        profile=True,
    )

    debug = model.last_recurrence_debug

    assert debug["profiling_enabled"] is True
    assert len(
        debug["run_one_iteration_ms_list"]
    ) == 3
    assert len(
        debug["scalar_policy_ms_list"]
    ) == 1
    assert debug["get_output_ms_total"] == pytest.approx(
        0.3
    )
    assert debug["coda_ms_total"] == pytest.approx(
        0.2
    )


def test_nonfinite_recurrent_state_aborts_before_coda():
    model = DummyScalarRuntimeModel(
        nonfinite_iteration=2
    )

    with pytest.raises(
        NonFiniteScalarPolicyError,
        match="non-finite recurrent state",
    ):
        run_scalar_policy_adaptive(
            model,
            torch.zeros(1, 2, 3),
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
            policy=make_policy(),
            execution_mode="direct",
            max_iter=8,
            actual_origin="ACTUAL_WARM",
            requested_recurrence_strategy=(
                "scalar_policy"
            ),
            profile_coda_cost=False,
            capture_warm_start_candidates=False,
            warm_start_candidate_states=[],
            warm_start_source="midpoint",
            warm_start_min_iter_configured=2,
        )

    assert model.coda_calls == 0


def test_cold_origin_is_rejected_for_fallback_caller():
    with pytest.raises(
        ScalarStoppingPolicyError,
        match="ACTUAL_WARM-only",
    ):
        run_scalar_policy_adaptive(
            DummyScalarRuntimeModel(),
            torch.zeros(1, 2, 3),
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
            policy=make_policy(),
            execution_mode="direct",
            max_iter=8,
            actual_origin="COLD",
            requested_recurrence_strategy=(
                "scalar_policy"
            ),
            profile_coda_cost=False,
            capture_warm_start_candidates=False,
            warm_start_candidate_states=[],
            warm_start_source="midpoint",
            warm_start_min_iter_configured=2,
        )
