"""Runtime recurrence executor for a prepared scalar stopping policy."""

from __future__ import annotations

from typing import Any

import torch

from prismatic.models.scalar_stopping_policy import (
    NonFiniteScalarPolicyError,
    PreparedScalarTaskPolicy,
    ScalarStoppingPolicyError,
    compute_scalar_stopping_features,
    evaluate_scalar_stopping_policy,
    resolve_scalar_terminal_iteration,
)
from prismatic.utils.rdvla_profiler import rdvla_range


def _finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all().item())


def run_scalar_policy_adaptive(
    model,
    state: torch.Tensor,
    prelude_out: torch.Tensor,
    h_a: torch.Tensor,
    h_t: torch.Tensor,
    p: torch.Tensor,
    *,
    policy: PreparedScalarTaskPolicy,
    execution_mode: str,
    max_iter: int,
    actual_origin: str,
    requested_recurrence_strategy: str,
    profile_coda_cost: bool,
    capture_warm_start_candidates: bool,
    warm_start_candidate_states,
    warm_start_source: str,
    warm_start_min_iter_configured: int,
):
    """Run warm recurrence with scalar stopping and one terminal Coda call."""

    if actual_origin != "ACTUAL_WARM":
        raise ScalarStoppingPolicyError(
            "scalar runtime policy is ACTUAL_WARM-only; "
            "COLD predictions must use action-MSE fallback"
        )

    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 3:
        raise ScalarStoppingPolicyError(
            "scalar runtime policy requires max_iter >= 3"
        )

    if policy.minimum_gate_iteration != 3:
        raise ScalarStoppingPolicyError(
            "scalar runtime policy schema requires minimum gate iteration 3"
        )

    previous_previous_state = None
    previous_state = None

    actual_iter = 0
    gate_iteration = None
    terminal_iteration = None
    stop_reason = "max_iter"

    score_trace: list[dict[str, Any]] = []
    score_values: list[float] = []
    recurrent_timings: list[float] = []
    scalar_timings: list[float] = []
    coda_call_count = 0

    with rdvla_range("RDVLA/action_head/recurrent_loop_total"):
        with torch.no_grad():
            for iteration in range(1, max_iter + 1):
                recurrent_start = (
                    model._sync_time()
                    if profile_coda_cost
                    else None
                )

                state = model._run_one_iteration(
                    state,
                    prelude_out,
                    h_a,
                    h_t,
                    p,
                )

                if profile_coda_cost:
                    recurrent_timings.append(
                        (
                            model._sync_time()
                            - recurrent_start
                        )
                        * 1000.0
                    )

                actual_iter = iteration

                if not _finite(state):
                    raise NonFiniteScalarPolicyError(
                        "non-finite recurrent state in scalar policy "
                        f"at iteration {iteration}"
                    )

                if capture_warm_start_candidates:
                    warm_start_candidate_states.append(
                        state.detach()
                    )

                if (
                    terminal_iteration is None
                    and iteration >= policy.minimum_gate_iteration
                ):
                    if (
                        previous_state is None
                        or previous_previous_state is None
                    ):
                        raise AssertionError(
                            "scalar policy history is unavailable at k>=3"
                        )

                    scalar_start = (
                        model._sync_time()
                        if profile_coda_cost
                        else None
                    )

                    features = compute_scalar_stopping_features(
                        state,
                        previous_state,
                        previous_previous_state,
                        iteration=iteration,
                        epsilon=policy.epsilon,
                    )
                    score, triggered = (
                        evaluate_scalar_stopping_policy(
                            policy,
                            features,
                        )
                    )

                    if profile_coda_cost:
                        scalar_timings.append(
                            (
                                model._sync_time()
                                - scalar_start
                            )
                            * 1000.0
                        )

                    score_values.append(score)
                    score_trace.append(
                        {
                            "k": int(iteration),
                            "score": float(score),
                            "threshold": float(
                                policy.threshold
                            ),
                            "triggered": bool(triggered),
                            "features": [
                                float(value)
                                for value in (
                                    features.detach()
                                    .to(
                                        device="cpu",
                                        dtype=torch.float32,
                                    )
                                    .tolist()
                                )
                            ],
                        }
                    )

                    if triggered:
                        gate_iteration = int(iteration)
                        terminal_iteration = (
                            resolve_scalar_terminal_iteration(
                                gate_iteration,
                                maximum_iteration=max_iter,
                                execution_mode=execution_mode,
                            )
                        )

                if (
                    terminal_iteration is not None
                    and iteration >= terminal_iteration
                ):
                    if execution_mode == "direct":
                        stop_reason = (
                            "scalar_threshold_direct"
                        )
                    else:
                        stop_reason = (
                            "scalar_threshold_confirm_next"
                        )
                    break

                previous_previous_state = previous_state
                previous_state = state.detach()

    with rdvla_range("RDVLA/action_head/final_get_output"):
        final_output = model._get_output(
            state,
            h_a,
            h_t,
            p,
            profile=profile_coda_cost,
        )

    coda_call_count += 1

    if not _finite(final_output):
        raise NonFiniteScalarPolicyError(
            "non-finite terminal Coda output in scalar policy"
        )

    if coda_call_count != 1:
        raise AssertionError(
            "scalar runtime policy must call Coda exactly once"
        )

    if capture_warm_start_candidates:
        model._store_warm_start_candidate(
            warm_start_candidate_states,
            actual_iter,
            warm_start_source,
        )

    final_score = (
        score_values[-1]
        if score_values
        else None
    )

    debug = {
        "strategy": "scalar_policy",
        "requested_recurrence_strategy": (
            requested_recurrence_strategy
        ),
        "canonical_recurrence_strategy": (
            "scalar_policy"
        ),
        "scalar_policy_requested": True,
        "scalar_policy_applied": True,
        "scalar_policy_cold_fallback": False,
        "execution_path": (
            f"scalar_policy_{execution_mode}"
        ),
        "scalar_policy_execution_mode": execution_mode,
        "scalar_policy_task_id": int(policy.task_id),
        "scalar_policy_outer_fold": int(
            policy.outer_fold
        ),
        "scalar_policy_threshold": float(
            policy.threshold
        ),
        "scalar_policy_minimum_gate_iteration": int(
            policy.minimum_gate_iteration
        ),
        "scalar_policy_epsilon": float(
            policy.epsilon
        ),
        "scalar_policy_gate_iteration": (
            int(gate_iteration)
            if gate_iteration is not None
            else None
        ),
        "scalar_policy_terminal_iteration": int(
            actual_iter
        ),
        "scalar_policy_score_trace": score_trace,
        "scalar_policy_score_call_count": len(
            score_trace
        ),
        "actual_origin": actual_origin,
        "fixed_K": None,
        "K_t": int(actual_iter),
        "max_iter": int(max_iter),
        "adaptive_stop": (
            gate_iteration is not None
        ),
        "stop_reason": stop_reason,
        "canonical_stop_reason": stop_reason,
        "metric_name": "scalar_combo",
        "final_conv_score": final_score,
        "final_mse": None,
        "final_convergence_evaluable": (
            final_score is not None
        ),
        "iteration_mse": [],
        "conv_score_list": score_values,
        "action_delta_list": [],
        "latent_mse_list": [],
        "latent_action_mse_pairs": [],
        "latent_action_pair_count": 0,
        "coda_call_count": coda_call_count,
        "get_output_call_count": coda_call_count,
        "final_state_coda_executed": True,
        "returned_cached_final_output": False,
        "use_cached_final_output": False,
        "use_latent_precheck": False,
        "latent_precheck_mode": (
            "bypassed_for_scalar_policy"
        ),
        "profiling_enabled": bool(
            profile_coda_cost
        ),
        "warm_start_min_iter_configured": int(
            warm_start_min_iter_configured
        ),
        "effective_min_iter": int(
            policy.minimum_gate_iteration
        ),
        "warm_start_state_used": True,
        "first_threshold_satisfied_k": (
            int(gate_iteration)
            if gate_iteration is not None
            else None
        ),
    }

    if profile_coda_cost:
        timing = model._last_get_output_timing or {}

        get_output_ms = float(
            timing.get("get_output_ms", 0.0)
        )
        coda_ms = float(
            timing.get("coda_ms", 0.0)
        )
        output_proj_ms = float(
            timing.get("output_proj_ms", 0.0)
        )

        recurrent_total = sum(
            recurrent_timings
        )
        scalar_total = sum(
            scalar_timings
        )

        debug.update(
            {
                "run_one_iteration_ms_list": (
                    recurrent_timings
                ),
                "scalar_policy_ms_list": (
                    scalar_timings
                ),
                "convergence_check_ms_list": (
                    scalar_timings
                ),
                "get_output_ms_list": [
                    get_output_ms
                ],
                "coda_ms_list": [coda_ms],
                "output_proj_ms_list": [
                    output_proj_ms
                ],
                "run_one_iteration_ms_total": (
                    recurrent_total
                ),
                "scalar_policy_ms_total": (
                    scalar_total
                ),
                "latent_metric_ms_total": (
                    scalar_total
                ),
                "get_output_ms_total": (
                    get_output_ms
                ),
                "coda_ms_total": coda_ms,
                "output_proj_ms_total": (
                    output_proj_ms
                ),
                "coda_time_ratio_total": (
                    coda_ms
                    / (
                        recurrent_total
                        + scalar_total
                        + get_output_ms
                    )
                    if (
                        recurrent_total
                        + scalar_total
                        + get_output_ms
                    )
                    else 0.0
                ),
            }
        )

    model.last_recurrence_debug = debug

    return (
        final_output,
        actual_iter,
        final_score,
    )
