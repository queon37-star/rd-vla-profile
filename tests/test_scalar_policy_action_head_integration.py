import types

import pytest
import torch

from configs.rdvla_precheck import (
    canonicalize_recurrence_strategy,
)
from prismatic.models.action_heads import (
    RecurrentConfigInternal,
    VLARecurrent,
)
from prismatic.models.scalar_stopping_policy import (
    PreparedScalarTaskPolicy,
)


def make_policy():
    return PreparedScalarTaskPolicy(
        task_id=1,
        outer_fold=1,
        threshold=0.5,
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
            -3.0,
            dtype=torch.float32,
        ),
    )


def make_model(*, constant_state=False):
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
    model = VLARecurrent(cfg)
    model.eval()
    model.test_coda_calls = 0
    model.test_coda_states = []

    def run_one_iteration(
        self,
        state,
        prelude_out,
        h_a,
        h_t,
        p,
    ):
        del prelude_out, h_a, h_t, p

        if constant_state:
            return torch.zeros_like(state)

        return state + 1.0

    def get_output(
        self,
        state,
        h_a,
        h_t,
        p,
        profile=False,
    ):
        del h_a, h_t, p
        self.test_coda_calls += 1
        self.test_coda_states.append(
            state.detach().clone()
        )

        if profile:
            self._last_get_output_timing = {
                "get_output_ms": 0.3,
                "coda_ms": 0.2,
                "output_proj_ms": 0.1,
            }

        return state[..., :2].clone()

    model._run_one_iteration = types.MethodType(
        run_one_iteration,
        model,
    )
    model._get_output = types.MethodType(
        get_output,
        model,
    )
    return model


def inputs():
    h_a = torch.zeros(
        1,
        1,
        1,
        4,
        dtype=torch.bfloat16,
    )
    h_t = torch.zeros_like(h_a)
    p = torch.zeros(
        1,
        1,
        4,
        dtype=torch.bfloat16,
    )
    return h_a, h_t, p


def scalar_kwargs():
    return {
        "convergence_strategy": "scalar_policy",
        "enable_warm_start": True,
        "warm_start_source": "midpoint",
        "latent_precheck_mode": "off",
        "latent_precheck_trace_level": "off",
        "use_latent_precheck": False,
        "scalar_task_policy": make_policy(),
        "scalar_policy_execution_mode": "direct",
        "max_iter": 6,
    }


def test_scalar_policy_is_registered():
    assert (
        canonicalize_recurrence_strategy(
            "scalar_policy"
        )
        == "scalar_policy"
    )


def test_warm_scalar_policy_runs_direct_executor():
    model = make_model()
    h_a, h_t, p = inputs()

    output, actual_iter, final_score = model(
        h_a,
        h_t,
        p,
        warm_start_state=torch.zeros(
            1,
            2,
            4,
            dtype=torch.bfloat16,
        ),
        **scalar_kwargs(),
    )

    assert actual_iter == 3
    assert final_score == 0.5
    assert model.test_coda_calls == 1
    assert torch.all(output == 3)

    debug = model.last_recurrence_debug

    assert debug["scalar_policy_requested"] is True
    assert debug["scalar_policy_applied"] is True
    assert (
        debug["scalar_policy_cold_fallback"]
        is False
    )
    assert (
        debug["scalar_policy_gate_iteration"]
        == 3
    )
    assert debug["coda_call_count"] == 1
    assert debug["get_output_call_count"] == 1
    assert debug["final_state_coda_executed"] is True
    assert debug["returned_cached_final_output"] is False


@pytest.mark.parametrize(
    "execution_mode",
    ["direct", "confirm_next"],
)
def test_cold_scalar_request_falls_back_to_action_mse(
    execution_mode,
):
    model = make_model(constant_state=True)
    h_a, h_t, p = inputs()
    kwargs = scalar_kwargs()
    kwargs["scalar_policy_execution_mode"] = (
        execution_mode
    )

    output, actual_iter, final_mse = model(
        h_a,
        h_t,
        p,
        warm_start_state=None,
        kl_thresh=0.001,
        **kwargs,
    )

    assert actual_iter == 2
    assert final_mse == 0.0
    assert model.test_coda_calls == actual_iter == 2
    assert torch.all(output == 0)

    debug = model.last_recurrence_debug

    assert debug["scalar_policy_requested"] is True
    assert debug["scalar_policy_applied"] is False
    assert (
        debug["scalar_policy_cold_fallback"]
        is True
    )
    assert (
        debug["canonical_recurrence_strategy"]
        == "adjacent_action_mse"
    )
    assert (
        debug["stop_reason"]
        == "adjacent_action_mse_cold_fallback"
    )
    assert debug["warm_start_state_used"] is False
    assert debug["scalar_policy_execution_mode"] == execution_mode
    assert debug["coda_call_count"] == actual_iter
    assert debug["get_output_call_count"] == actual_iter
    assert debug["final_state_coda_executed"] is False
    assert debug["returned_cached_final_output"] is True
    assert debug["use_cached_final_output"] is False
    assert debug["scalar_policy_score_call_count"] == 0
    assert debug["scalar_policy_gate_iteration"] is None

    # Re-decoding the same terminal state is a reference only: the
    # production return must already equal that terminal action.
    terminal_state = model.test_coda_states[-1]
    reference_output = model._get_output(
        terminal_state,
        h_a,
        h_t,
        p,
    )
    torch.testing.assert_close(
        output,
        reference_output,
        rtol=0,
        atol=0,
    )

    warm = model.last_inference_metadata["warm_start"]
    assert warm["source"] == "midpoint"
    assert warm["source_K"] == 2

    import json

    from experiments.robot.libero.run_libero_eval import (
        build_decode_call_log_fields,
        build_scalar_policy_log_fields,
    )

    log_fields = {
        **build_decode_call_log_fields(debug),
        **build_scalar_policy_log_fields(debug),
    }
    assert log_fields["coda_call_count"] == actual_iter
    assert log_fields["get_output_call_count"] == actual_iter
    assert log_fields["final_state_coda_executed"] is False
    assert log_fields["returned_cached_final_output"] is True
    json.dumps(log_fields, allow_nan=False)


def test_action_head_resolves_bound_task_policy():
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
    from prismatic.models.action_heads import (
        ActionHeadRecurrent,
    )

    action_head = ActionHeadRecurrent(
        hidden_dim=4,
        action_dim=2,
        cfg=cfg,
    )
    policy = make_policy()

    action_head.configure_scalar_task_policy(
        policy,
        "confirm_next",
    )

    resolved_policy, resolved_mode = (
        action_head._resolve_scalar_runtime_policy(
            "scalar_policy",
            None,
            None,
        )
    )

    assert resolved_policy is policy
    assert resolved_mode == "confirm_next"

    action_head.clear_scalar_task_policy()

    resolved_policy, resolved_mode = (
        action_head._resolve_scalar_runtime_policy(
            "scalar_policy",
            None,
            None,
        )
    )

    assert resolved_policy is None
    assert resolved_mode == "direct"


def test_action_head_rejects_invalid_bound_mode():
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
    from prismatic.models.action_heads import (
        ActionHeadRecurrent,
    )

    action_head = ActionHeadRecurrent(
        hidden_dim=4,
        action_dim=2,
        cfg=cfg,
    )

    with pytest.raises(
        ValueError,
        match="execution mode",
    ):
        action_head.configure_scalar_task_policy(
            make_policy(),
            "invalid",
        )



def test_scalar_policy_log_fields_preserve_runtime_metadata():
    import json

    from experiments.robot.libero.run_libero_eval import (
        build_scalar_policy_log_fields,
    )

    model = make_model()
    h_a, h_t, p = inputs()

    model(
        h_a,
        h_t,
        p,
        warm_start_state=torch.zeros(
            1,
            2,
            4,
            dtype=torch.bfloat16,
        ),
        **scalar_kwargs(),
    )

    fields = build_scalar_policy_log_fields(
        model.last_recurrence_debug
    )

    assert fields["scalar_policy_requested"] is True
    assert fields["scalar_policy_applied"] is True
    assert fields["scalar_policy_cold_fallback"] is False
    assert fields["scalar_policy_execution_mode"] == "direct"
    assert fields["scalar_policy_task_id"] == 1
    assert fields["scalar_policy_outer_fold"] == 1
    assert fields["scalar_policy_threshold"] == 0.5
    assert fields["scalar_policy_gate_iteration"] == 3
    assert fields["scalar_policy_terminal_iteration"] == 3
    assert fields["scalar_policy_score_call_count"] == 1
    assert fields["scalar_policy_final_score"] == 0.5
    assert len(fields["scalar_policy_score_trace"]) == 1

    # The step/timing log payload must remain JSON serializable.
    json.dumps(fields, allow_nan=False)
