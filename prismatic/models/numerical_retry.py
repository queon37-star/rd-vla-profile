"""Numerically guarded cold fallback for origin-aware RD-VLA inference."""

import math

import torch

from configs.rdvla_precheck import ORIGIN_AWARE_COLD_THRESHOLD
from prismatic.models.origin_aware_scheduler import (
    NonFiniteOriginAwareInferenceError,
    _is_finite_tensor,
)
from prismatic.utils.rdvla_profiler import rdvla_range


class NumericalInferenceAbort(RuntimeError):
    """Raised when both the scheduled attempt and its one cold retry fail."""

    def __init__(self, first_error, retry_error):
        self.first_attempt_failure = first_error.to_dict()
        self.retry_failure = retry_error.to_dict()
        super().__init__(
            "origin-aware inference and cold retry both failed: "
            f"first={first_error}; retry={retry_error}"
        )

    def to_dict(self):
        return {
            "type": type(self).__name__,
            "message": str(self),
            "first_attempt_failure": self.first_attempt_failure,
            "retry_failure": self.retry_failure,
        }


def _failure(stage, iteration, message, *, coda_attempt_count=0):
    return NonFiniteOriginAwareInferenceError(
        message,
        stage=stage,
        iteration=iteration,
        details={"coda_attempt_count": int(coda_attempt_count)},
    )


def _run_cold_full_coda(
    model,
    prelude_out,
    h_a,
    h_t,
    p,
    *,
    max_iter,
    action_mse_threshold,
    requested_recurrence_strategy,
    profile_coda_cost,
    trace_level,
):
    """Run one cold adaptive attempt with Coda decoded at every iteration."""
    batch_size = h_a.size(0)
    state = model.init_state(batch_size, h_a.device, h_a.dtype)
    if not _is_finite_tensor(state):
        raise _failure(
            "cold_retry_initial_state",
            0,
            "non-finite cold retry initial state",
        )

    collect_full_trace = trace_level == "full"
    previous_output = None
    previous_output_iter = None
    current_output = None
    current_output_iter = None
    actual_iter = 0
    final_action_mse = None
    adaptive_stop = False
    stop_reason = None
    first_converged_k_1e_4 = None
    first_converged_k_5e_4 = None
    first_threshold_satisfied_k = None

    iteration_mse = []
    action_delta_list = []
    adjacent_comparison_pairs = []
    coda_call_records = []
    run_one_iteration_ms_list = []
    get_output_ms_list = []
    coda_ms_list = []
    output_proj_ms_list = []
    convergence_check_ms_list = []

    def append_get_output_timing():
        timing = model._last_get_output_timing
        get_output_ms_list.append(timing["get_output_ms"])
        coda_ms_list.append(timing["coda_ms"])
        output_proj_ms_list.append(timing["output_proj_ms"])

    with rdvla_range("RDVLA/action_head/cold_retry_once"):
        with torch.no_grad():
            for it in range(max_iter):
                if profile_coda_cost:
                    recurrent_start = model._sync_time()
                state = model._run_one_iteration(state, prelude_out, h_a, h_t, p)
                if profile_coda_cost:
                    run_one_iteration_ms_list.append(
                        (model._sync_time() - recurrent_start) * 1000.0
                    )

                actual_iter = it + 1
                if not _is_finite_tensor(state):
                    raise _failure(
                        "cold_retry_recurrent_state",
                        actual_iter,
                        f"non-finite cold retry recurrent state at iteration {actual_iter}",
                        coda_attempt_count=actual_iter - 1,
                    )

                with rdvla_range("RDVLA/action_head/coda_stop_get_output"):
                    with rdvla_range("RDVLA/action_head/get_output_each_iter"):
                        current_output = model._get_output(
                            state,
                            h_a,
                            h_t,
                            p,
                            profile=profile_coda_cost,
                        )
                if profile_coda_cost:
                    append_get_output_timing()
                current_output_iter = actual_iter
                if not _is_finite_tensor(current_output):
                    raise _failure(
                        "cold_retry_coda_output",
                        actual_iter,
                        f"non-finite cold retry Coda output at iteration {actual_iter}",
                        coda_attempt_count=actual_iter,
                    )

                if collect_full_trace:
                    coda_call_records.append(
                        {
                            "iteration": int(actual_iter),
                            "reason": "cold_fallback",
                            "is_backfill": False,
                            "scheduler_state_before": "COLD_RETRY_FULL_CODA",
                            "refresh_after_skip": False,
                            "confirmation_pending": False,
                        }
                    )

                if previous_output is not None:
                    convergence_start = model._sync_time() if profile_coda_cost else None
                    with rdvla_range("RDVLA/action_head/stop_check/mse_compute"):
                        diff = current_output - previous_output
                        mse_tensor = torch.mean(diff ** 2)
                        l2_tensor = torch.norm(diff.float())
                    with rdvla_range("RDVLA/action_head/stop_check/item_sync"):
                        action_mse = mse_tensor.item()
                        action_l2 = l2_tensor.item()
                    if not math.isfinite(action_mse) or not math.isfinite(action_l2):
                        raise _failure(
                            "cold_retry_action_convergence_metric",
                            actual_iter,
                            "non-finite cold retry action convergence metric "
                            f"at iteration {actual_iter}",
                            coda_attempt_count=actual_iter,
                        )
                    if profile_coda_cost:
                        convergence_check_ms_list.append(
                            (model._sync_time() - convergence_start) * 1000.0
                        )

                    final_action_mse = action_mse
                    iteration_mse.append(action_mse)
                    action_delta_list.append(action_l2)
                    adjacent_comparison_pairs.append(
                        [int(previous_output_iter), int(actual_iter)]
                    )
                    if first_converged_k_1e_4 is None and action_mse < 1e-4:
                        first_converged_k_1e_4 = actual_iter
                    if first_converged_k_5e_4 is None and action_mse < 5e-4:
                        first_converged_k_5e_4 = actual_iter

                previous_output = current_output.detach()
                previous_output_iter = actual_iter

                threshold_satisfied = (
                    final_action_mse is not None
                    and final_action_mse < action_mse_threshold
                )
                if threshold_satisfied and first_threshold_satisfied_k is None:
                    first_threshold_satisfied_k = actual_iter
                if actual_iter == max_iter:
                    stop_reason = "max_iter"
                    break
                if threshold_satisfied and actual_iter >= 2:
                    adaptive_stop = True
                    stop_reason = requested_recurrence_strategy
                    break

    if current_output is None or current_output_iter != actual_iter:
        raise RuntimeError("cold retry did not decode its terminal recurrent state")

    debug = {
        "strategy": requested_recurrence_strategy,
        "requested_recurrence_strategy": requested_recurrence_strategy,
        "canonical_recurrence_strategy": "adjacent_action_mse",
        "canonical_metric_name": "adjacent_action_mse",
        "action_mse_threshold": float(action_mse_threshold),
        "threshold": float(action_mse_threshold),
        "fixed_K": None,
        "K_t": int(actual_iter),
        "max_iter": int(max_iter),
        "adaptive_stop": bool(adaptive_stop),
        "metric_name": "mse_between_adjacent_action_outputs",
        "iteration_mse": iteration_mse,
        "iteration_metric_values": iteration_mse,
        "conv_score_list": iteration_mse,
        "action_delta_list": action_delta_list,
        "adjacent_comparison_pairs": adjacent_comparison_pairs,
        "adjacent_comparison_pair_count": len(adjacent_comparison_pairs),
        "latent_mse_list": [],
        "latent_l2_list": [],
        "latent_action_mse_pairs": [],
        "latent_action_pair_count": 0,
        "use_latent_precheck": False,
        "latent_precheck_mode": "origin_aware",
        "configured_use_latent_precheck": True,
        "nonfinite_policy": "cold_retry_once",
        "latent_precheck_trace_level_requested": trace_level,
        "latent_precheck_trace_level_applied": trace_level,
        "latent_precheck_trace_collected": collect_full_trace,
        "latent_precheck_coda_call_mask": [True] * actual_iter if collect_full_trace else [],
        "latent_precheck_skipped_iters": [],
        "latent_precheck_called_iters": (
            list(range(1, actual_iter + 1)) if collect_full_trace else []
        ),
        "latent_precheck_skip_count": 0,
        "latent_precheck_call_count": int(actual_iter),
        "latent_precheck_skip_ratio": 0.0,
        "latent_precheck_decisions": [],
        "latent_metric_count": 0,
        "coda_call_records": coda_call_records,
        "coda_reason_counts": {"cold_fallback": int(actual_iter)},
        "origin_aware_scheduler_state": "COLD_RETRY_FULL_CODA",
        "first_converged_k_1e_4": first_converged_k_1e_4,
        "first_converged_k_5e_4": first_converged_k_5e_4,
        "final_mse": final_action_mse,
        "final_conv_score": final_action_mse,
        "final_convergence_evaluable": final_action_mse is not None,
        "stop_reason": stop_reason,
        "canonical_stop_reason": (
            "adjacent_action_mse" if adaptive_stop else stop_reason
        ),
        "profiling_enabled": bool(profile_coda_cost),
        "use_cached_final_output": True,
        "returned_cached_final_output": True,
        "cached_final_matches_returned": True,
        "final_state_coda_executed": True,
        "final_state_output_iteration": int(current_output_iter),
        "max_iteration_convergence_evaluable": bool(
            stop_reason == "max_iter" and final_action_mse is not None
        ),
        "warm_start_min_iter_configured": 2,
        "effective_min_iter": 2,
        "warm_start_state_used": False,
        "min_iter_gate_block_count": 0,
        "first_threshold_satisfied_k": first_threshold_satisfied_k,
        "execution_path": "cold_retry_full_coda",
    }

    if profile_coda_cost:
        run_one_iteration_ms_total = sum(run_one_iteration_ms_list)
        get_output_ms_total = sum(get_output_ms_list)
        coda_ms_total = sum(coda_ms_list)
        output_proj_ms_total = sum(output_proj_ms_list)
        profiled_recurrent_ms_total = run_one_iteration_ms_total + get_output_ms_total
        debug.update(
            {
                "run_one_iteration_ms_list": run_one_iteration_ms_list,
                "get_output_ms_list": get_output_ms_list,
                "coda_ms_list": coda_ms_list,
                "output_proj_ms_list": output_proj_ms_list,
                "convergence_check_ms_list": convergence_check_ms_list,
                "get_output_call_count": len(get_output_ms_list),
                "coda_ms_total": coda_ms_total,
                "get_output_ms_total": get_output_ms_total,
                "run_one_iteration_ms_total": run_one_iteration_ms_total,
                "output_proj_ms_total": output_proj_ms_total,
                "coda_time_ratio_total": (
                    coda_ms_total / profiled_recurrent_ms_total
                    if profiled_recurrent_ms_total
                    else 0.0
                ),
            }
        )

    return current_output, actual_iter, final_action_mse, debug


def run_cold_full_coda_retry(
    model,
    prelude_out,
    h_a,
    h_t,
    p,
    *,
    max_iter,
    action_mse_threshold,
    requested_recurrence_strategy,
    warm_threshold,
    latent_precheck_min_iter,
    max_skip_iters,
    confirmation_mode,
    profile_coda_cost,
    trace_level,
    first_error,
    first_attempt_origin,
    first_warm_metadata,
):
    """Retry once from a new cold state, invalidate the warm cache, and log it."""
    first_attempt_active_threshold = (
        float(warm_threshold)
        if first_attempt_origin == "ACTUAL_WARM"
        else ORIGIN_AWARE_COLD_THRESHOLD
    )
    warm_metadata = {
        **dict(first_warm_metadata),
        "state_used": False,
        "initial_state_origin": "random",
        "reset": True,
        "reset_reason": f"numerical_retry:{first_error.stage}",
        "source": None,
        "source_index": None,
        "source_iteration": None,
        "source_K": None,
        "candidate_state_count": None,
        "first_attempt_state_used": bool(first_warm_metadata.get("state_used", False)),
        "first_attempt_initial_state_origin": first_warm_metadata.get(
            "initial_state_origin"
        ),
    }
    model.last_inference_metadata = {
        "next_warm_start_state": None,
        "warm_start": warm_metadata,
    }

    try:
        output, actual_iter, final_mse, debug = _run_cold_full_coda(
            model,
            prelude_out,
            h_a,
            h_t,
            p,
            max_iter=max_iter,
            action_mse_threshold=action_mse_threshold,
            requested_recurrence_strategy=requested_recurrence_strategy,
            profile_coda_cost=profile_coda_cost,
            trace_level=trace_level,
        )
    except NonFiniteOriginAwareInferenceError as retry_error:
        abort = NumericalInferenceAbort(first_error, retry_error)
        model.last_recurrence_debug = {
            "strategy": requested_recurrence_strategy,
            "canonical_recurrence_strategy": "adjacent_action_mse",
            "latent_precheck_mode": "origin_aware",
            "execution_path": "numerical_abort",
            "nonfinite_policy": "cold_retry_once",
            "numerical_retry_attempted": True,
            "first_attempt_active_threshold": first_attempt_active_threshold,
            "latent_precheck_warm_thresh": float(warm_threshold),
            "latent_precheck_cold_thresh": ORIGIN_AWARE_COLD_THRESHOLD,
            "latent_precheck_max_skip_iters": int(max_skip_iters),
            "latent_precheck_confirmation_mode": confirmation_mode,
            "latent_precheck_min_iter": int(latent_precheck_min_iter),
            "numerical_retry_succeeded": False,
            "numerical_abort": abort.to_dict(),
            "first_attempt_origin": first_attempt_origin,
            "final_state_coda_executed": False,
            "final_convergence_evaluable": False,
        }
        raise abort from retry_error

    first_failure = first_error.to_dict()
    debug.update(
        {
            "numerical_retry_attempted": True,
            "numerical_retry_succeeded": True,
            "numerical_retry_count": 1,
            "first_attempt_origin": first_attempt_origin,
            "first_attempt_failure": first_failure,
            "first_attempt_coda_attempt_count": int(
                first_failure.get(
                    "coda_attempt_count",
                    first_failure.get("coda_call_count_before_failure", 0),
                )
            ),
            "retry_coda_call_count": int(actual_iter),
            "latent_precheck_warm_thresh": float(warm_threshold),
            "latent_precheck_cold_thresh": ORIGIN_AWARE_COLD_THRESHOLD,
            "latent_precheck_active_threshold": None,
            "latent_precheck_origin": "COLD",
            "latent_precheck_max_skip_iters": int(max_skip_iters),
            "latent_precheck_confirmation_mode": confirmation_mode,
            "latent_precheck_min_iter": int(latent_precheck_min_iter),
            "first_attempt_active_threshold": first_attempt_active_threshold,
            "configured_latent_precheck_mode": "origin_aware",
            "configured_use_latent_precheck": True,
        }
    )
    debug["get_output_attempt_count_intent_to_treat"] = (
        debug["first_attempt_coda_attempt_count"] + debug["retry_coda_call_count"]
    )
    model.last_recurrence_debug = debug
    return output, actual_iter, final_mse
