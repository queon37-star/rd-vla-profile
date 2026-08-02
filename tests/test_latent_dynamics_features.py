import copy
import math
import types

import pytest
import torch

from experiments.robot.libero.latent_metric_trace import (
    build_latent_metric_trace_records,
)
from prismatic.models.action_heads import RecurrentConfigInternal, VLARecurrent
from prismatic.models.latent_dynamics import (
    HISTORY_DEPENDENT_FIELDS,
    LATENT_DYNAMICS_FIELDS,
    NonFiniteLatentDynamicsError,
    WARM_ANCHOR_FIELDS,
    compute_latent_dynamics,
)
from scripts.check_latent_dynamics_trace import (
    LATENT_DYNAMICS_FIELDS as CHECKER_LATENT_DYNAMICS_FIELDS,
    LatentDynamicsContractError,
    validate_records,
)


def test_hand_computed_tiny_latent_dynamics_metrics():
    previous = torch.zeros(1, 2, 2, dtype=torch.float64)
    current = torch.tensor([[[1.0, 1.0], [3.0, 3.0]]], dtype=torch.float64)
    previous_update = torch.full_like(current, 2.0)

    values = compute_latent_dynamics(
        current,
        previous,
        previous_update=previous_update,
        warm_anchor=current.clone(),
        eps=1e-8,
    )

    expected_entropy = -(0.1 * math.log(0.1) + 0.9 * math.log(0.9)) / math.log(2)
    assert values["update_rms"] == pytest.approx(math.sqrt(5.0))
    assert values["contraction_ratio"] == pytest.approx(math.sqrt(5.0) / 2.0)
    assert values["update_turning_cosine"] == pytest.approx(2.0 / math.sqrt(5.0))
    assert values["acceleration_rms"] == pytest.approx(1.0)
    assert values["acceleration_ratio"] == pytest.approx(0.5)
    assert values["token_update_p50"] == pytest.approx(2.0)
    assert values["token_update_p90"] == pytest.approx(2.8)
    assert values["token_update_p95"] == pytest.approx(2.9)
    assert values["token_update_max"] == pytest.approx(3.0)
    assert values["token_update_cv"] == pytest.approx(0.5)
    assert values["token_update_energy_entropy"] == pytest.approx(expected_entropy)
    assert values["token_update_top10_fraction"] == pytest.approx(0.9)
    assert values["state_rms"] == pytest.approx(math.sqrt(5.0))
    assert values["state_norm_ratio"] == pytest.approx(math.sqrt(5.0) / 1e-8)
    assert values["warm_anchor_relative_l2"] == pytest.approx(0.0)
    assert values["warm_anchor_cosine_distance"] == pytest.approx(0.0, abs=2e-7)


def test_constant_and_reversed_updates_have_expected_history_metrics():
    previous = torch.zeros(1, 2, 3)
    constant = compute_latent_dynamics(
        torch.ones_like(previous),
        previous,
        previous_update=torch.ones_like(previous),
    )
    assert constant["contraction_ratio"] == pytest.approx(1.0)
    assert constant["update_turning_cosine"] == pytest.approx(1.0)
    assert constant["acceleration_rms"] == pytest.approx(0.0)
    assert constant["acceleration_ratio"] == pytest.approx(0.0)

    reversed_update = -torch.ones_like(previous)
    reversed_values = compute_latent_dynamics(
        reversed_update,
        previous,
        previous_update=torch.ones_like(previous),
    )
    assert reversed_values["update_turning_cosine"] == pytest.approx(-1.0)
    assert reversed_values["acceleration_rms"] == pytest.approx(2.0)


def test_uniform_and_concentrated_token_energy_structure():
    previous = torch.zeros(1, 10, 2)
    uniform = compute_latent_dynamics(torch.ones_like(previous), previous)
    assert uniform["token_update_energy_entropy"] == pytest.approx(1.0)
    assert uniform["token_update_top10_fraction"] == pytest.approx(0.1)

    concentrated_state = torch.zeros_like(previous)
    concentrated_state[:, 0, :] = 1.0
    concentrated = compute_latent_dynamics(concentrated_state, previous)
    assert concentrated["token_update_energy_entropy"] == pytest.approx(0.0)
    assert concentrated["token_update_top10_fraction"] == pytest.approx(1.0)


def test_zero_norm_eps_handling_is_finite_and_nonfinite_input_is_rejected():
    zeros = torch.zeros(1, 4, 3)
    values = compute_latent_dynamics(
        zeros,
        zeros,
        previous_update=zeros,
        warm_anchor=zeros,
        eps=1e-8,
    )
    assert all(value is None or math.isfinite(value) for value in values.values())
    assert values["update_rms"] == 0.0
    assert values["contraction_ratio"] == 0.0
    assert values["token_update_cv"] == 0.0
    assert values["token_update_energy_entropy"] == 0.0
    assert values["token_update_top10_fraction"] == 0.0

    nonfinite = zeros.clone()
    nonfinite[0, 0, 0] = float("nan")
    with pytest.raises(NonFiniteLatentDynamicsError, match="non-finite"):
        compute_latent_dynamics(nonfinite, zeros)


def _tiny_shadow_model():
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


def _run_shadow(model, *, shadow_full_depth, warm_state):
    result = model(
        *_inputs(),
        convergence_strategy="adjacent_action_mse",
        kl_thresh=2.0,
        max_iter=5,
        warm_start_state=warm_state,
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


@pytest.mark.parametrize(
    ("warm_state", "origin"),
    [
        (torch.zeros(1, 2, 4), "ACTUAL_WARM"),
        (None, "COLD"),
    ],
)
def test_trace_history_warm_anchor_and_behavior_invariance(warm_state, origin):
    baseline_model = _tiny_shadow_model()
    traced_model = _tiny_shadow_model()
    baseline = _run_shadow(
        baseline_model, shadow_full_depth=False, warm_state=warm_state
    )
    traced = _run_shadow(
        traced_model, shadow_full_depth=True, warm_state=warm_state
    )

    torch.testing.assert_close(traced[0][0], baseline[0][0], rtol=0, atol=0)
    assert traced[0][1:] == baseline[0][1:]
    assert traced[1]["K_t"] == baseline[1]["K_t"] == 2
    assert traced[1]["stop_reason"] == baseline[1]["stop_reason"]
    torch.testing.assert_close(
        traced[2]["next_warm_start_state"],
        baseline[2]["next_warm_start_state"],
        rtol=0,
        atol=0,
    )
    assert baseline_model.test_calls == {"recurrent": 2, "output": 2}
    assert traced_model.test_calls == {"recurrent": 5, "output": 5}

    records = build_latent_metric_trace_records(
        traced[1],
        task_id=0,
        episode_id=0,
        prediction_id=0,
        actual_origin=origin,
    )
    assert [record["iteration_index"] for record in records] == [2, 3, 4, 5]
    assert all(records[0][field] is None for field in HISTORY_DEPENDENT_FIELDS)
    assert records[1]["contraction_ratio"] == pytest.approx(1.0)
    assert records[1]["update_turning_cosine"] == pytest.approx(1.0)
    assert records[1]["acceleration_rms"] == pytest.approx(0.0)
    for record in records:
        assert set(LATENT_DYNAMICS_FIELDS) <= set(record)
        if origin == "ACTUAL_WARM":
            assert all(record[field] is not None for field in WARM_ANCHOR_FIELDS)
        else:
            assert all(record[field] is None for field in WARM_ANCHOR_FIELDS)


def _contract_record(origin="ACTUAL_WARM"):
    trace = []
    for iteration in range(2, 33):
        dynamics = {field: 1.0 for field in LATENT_DYNAMICS_FIELDS}
        if iteration == 2:
            for field in HISTORY_DEPENDENT_FIELDS:
                dynamics[field] = None
        if origin == "COLD":
            for field in WARM_ANCHOR_FIELDS:
                dynamics[field] = None
        dynamics["token_update_energy_entropy"] = 0.5
        dynamics["token_update_top10_fraction"] = 0.5
        trace.append(
            {
                "iteration_index": iteration,
                "phase": "production" if iteration <= 4 else "shadow_tail",
                "actual_origin": origin,
                "raw_mse": 1.0,
                "relative_mse": 1.0,
                "relative_l2": 1.0,
                "cosine_distance": 0.0,
                "adjacent_action_mse": 0.01,
                "action_mse_below_0_001": False,
                "baseline_stopping_iteration": 4,
                "task_id": 3,
                "episode_id": 0,
                "prediction_id": 0,
                **dynamics,
            }
        )
    return {
        "task_id": 3,
        "episode_id": 0,
        "prediction_step": 0,
        "action_prediction_index": 0,
        "paired_trial_id": 0,
        "initial_state_id": 7,
        "episode_seed": 11,
        "actual_origin": origin,
        "K_t": 4,
        "max_recurrent_iteration": 32,
        "latent_metric_trace_enabled": True,
        "latent_dynamics_trace_enabled": True,
        "latent_dynamics_warm_anchor_available": origin == "ACTUAL_WARM",
        "shadow_full_depth_enabled": True,
        "shadow_trace_complete": True,
        "shadow_error": None,
        "latent_metric_trace": trace,
    }


def test_contract_checker_accepts_complete_trace_and_rejects_contract_drift():
    assert CHECKER_LATENT_DYNAMICS_FIELDS == LATENT_DYNAMICS_FIELDS
    warm = _contract_record("ACTUAL_WARM")
    cold = _contract_record("COLD")
    cold["prediction_step"] = cold["action_prediction_index"] = 1
    for item in cold["latent_metric_trace"]:
        item["prediction_id"] = 1

    result = validate_records([warm, cold])
    assert result["passed"] is True
    assert result["prediction_count"] == 2
    assert result["transition_count"] == 62
    assert len(result["workload_identity_sha256"]) == 64
    assert validate_records(
        [warm, cold], expected_identity_sha256=result["workload_identity_sha256"]
    ) == result

    smoke_identity = copy.deepcopy(warm)
    for field in ("paired_trial_id", "initial_state_id", "episode_seed"):
        smoke_identity[field] = None
    assert validate_records([smoke_identity])["passed"] is True

    invalid = copy.deepcopy(warm)
    invalid["latent_metric_trace"][0]["token_update_energy_entropy"] = 1.1
    with pytest.raises(LatentDynamicsContractError, match="outside"):
        validate_records([invalid])

    incomplete = copy.deepcopy(warm)
    incomplete["latent_metric_trace"].pop()
    with pytest.raises(LatentDynamicsContractError, match="length must be 31"):
        validate_records([incomplete])
