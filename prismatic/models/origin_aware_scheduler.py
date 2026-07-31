"""Origin-aware Coda scheduling for adaptive RD-VLA inference."""

import math
from collections import Counter

import torch

from configs.rdvla_precheck import ORIGIN_AWARE_COLD_THRESHOLD
from prismatic.utils.rdvla_profiler import rdvla_range


class NonFiniteOriginAwareInferenceError(RuntimeError):
    """Raised before a non-finite state or action can leave the action head."""

    def __init__(self, message, *, stage, iteration, details=None):
        super().__init__(message)
        self.stage = stage
        self.iteration = int(iteration)
        self.details = dict(details or {})

    def to_dict(self):
        return {
            "type": type(self).__name__,
            "message": str(self),
            "stage": self.stage,
            "iteration": self.iteration,
            **self.details,
        }



def _is_finite_tensor(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all().item())


def run_origin_aware_adaptive(
    model,
    state,
    prelude_out,
    h_a,
    h_t,
    p,
    *,
    max_iter: int,
    action_mse_threshold: float,
    effective_min_iter: int,
    warm_start_min_iter_configured: int,
    warm_threshold: float,
    latent_precheck_min_iter: int,
    max_skip_iters: int,
    confirmation_mode: str,
    trace_level: str,
    actual_origin: str,
    requested_recurrence_strategy: str,
    profile_coda_cost: bool,
    use_cached_final_output_requested: bool,
    capture_warm_start_candidates: bool,
    warm_start_candidate_states,
    warm_start_source: str,
):
    """Run the guarded scheduler while preserving adjacent-output convergence checks."""
    if max_iter < 1:
        raise ValueError("recurrence_max_iter must be >= 1")

    active_threshold = (
        float(warm_threshold)
        if actual_origin == "ACTUAL_WARM"
        else ORIGIN_AWARE_COLD_THRESHOLD
    )
    collect_full_trace = trace_level == "full"

    actual_iter = 0
    adaptive_stop = False
    stop_reason = None
    final_action_mse = None
    prev_output = None
    prev_output_iter = None
    current_output = None
    current_output_iter = None
    scheduler_state = "INITIAL"
    skip_count = 0
    total_skipped_iters = 0
    min_iter_gate_block_count = 0
    first_threshold_satisfied_k = None
    first_converged_k_1e_4 = None
    first_converged_k_5e_4 = None
    max_iteration_convergence_evaluable = False

    conv_score_list = []
    action_delta_list = []
    adjacent_comparison_pairs = []
    latent_mse_list = []
    latent_l2_list = []
    latent_action_mse_pairs = []
    coda_call_mask = []
    called_iters = []
    skipped_iters = []
    decisions = []
    coda_call_records = []
    coda_reason_counts = Counter()
    coda_call_count = 0
    latent_metric_count = 0

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

    def decode_output(decode_state, state_iter: int, reason: str, *, is_backfill: bool = False):
        nonlocal coda_call_count
        with rdvla_range("RDVLA/action_head/coda_stop_get_output"):
            with rdvla_range("RDVLA/action_head/get_output_each_iter"):
                output = model._get_output(
                    decode_state,
                    h_a,
                    h_t,
                    p,
                    profile=profile_coda_cost,
                )
        if profile_coda_cost:
            append_get_output_timing()
        with rdvla_range("RDVLA/action_head/origin_aware/finite_check/output"):
            output_is_finite = _is_finite_tensor(output)
        if not output_is_finite:
            raise NonFiniteOriginAwareInferenceError(
                f"non-finite Coda output at iteration {state_iter}",
                stage="coda_output",
                iteration=state_iter,
                details={
                    "coda_call_count_before_failure": int(coda_call_count),
                    "coda_attempt_count": int(coda_call_count + 1),
                },
            )

        coda_call_count += 1
        coda_reason_counts[reason] += 1
        if collect_full_trace:
            coda_call_records.append(
                {
                    "iteration": int(state_iter),
                    "reason": reason,
                    "is_backfill": bool(is_backfill),
                    "scheduler_state_before": scheduler_state,
                    "refresh_after_skip": scheduler_state == "GAPPED" and not is_backfill,
                    "confirmation_pending": scheduler_state == "CONFIRM_PENDING",
                }
            )
        return output

    def compare_adjacent_outputs(
        left_output,
        left_iter: int,
        right_output,
        right_iter: int,
        *,
        allow_stop: bool,
        latent_mse=None,
        latent_l2=None,
    ):
        nonlocal final_action_mse
        nonlocal first_converged_k_1e_4
        nonlocal first_converged_k_5e_4
        nonlocal first_threshold_satisfied_k
        nonlocal min_iter_gate_block_count

        if right_iter != left_iter + 1:
            raise RuntimeError(
                "origin-aware convergence requires adjacent Coda outputs: "
                f"got ({left_iter}, {right_iter})"
            )
        convergence_start = model._sync_time() if profile_coda_cost else None
        with rdvla_range("RDVLA/action_head/stop_check/mse_compute"):
            diff = right_output - left_output
            action_mse_tensor = torch.mean(diff ** 2)
            action_l2_tensor = torch.norm(diff.float())
        with rdvla_range("RDVLA/action_head/stop_check/item_sync"):
            action_mse = action_mse_tensor.item()
            action_l2 = action_l2_tensor.item()
        if not math.isfinite(action_mse) or not math.isfinite(action_l2):
            raise NonFiniteOriginAwareInferenceError(
                f"non-finite action convergence metric at iteration {right_iter}",
                stage="action_convergence_metric",
                iteration=right_iter,
                details={
                    "coda_call_count_before_failure": int(coda_call_count),
                },
            )
        if profile_coda_cost:
            convergence_check_ms_list.append(
                (model._sync_time() - convergence_start) * 1000.0
            )

        final_action_mse = action_mse
        conv_score_list.append(action_mse)
        action_delta_list.append(action_l2)
        adjacent_comparison_pairs.append([int(left_iter), int(right_iter)])
        if collect_full_trace and latent_mse is not None:
            latent_action_mse_pairs.append(
                {
                    "k": int(right_iter),
                    "latent_mse": float(latent_mse),
                    "latent_l2": float(latent_l2) if latent_l2 is not None else None,
                    "action_mse": float(action_mse),
                    "action_l2": float(action_l2),
                }
            )

        if first_converged_k_1e_4 is None and action_mse < 1e-4:
            first_converged_k_1e_4 = right_iter
        if first_converged_k_5e_4 is None and action_mse < 5e-4:
            first_converged_k_5e_4 = right_iter

        threshold_satisfied = action_mse < action_mse_threshold
        if threshold_satisfied and first_threshold_satisfied_k is None:
            first_threshold_satisfied_k = right_iter
        if threshold_satisfied and allow_stop and right_iter < effective_min_iter:
            min_iter_gate_block_count += 1
        should_stop = (
            threshold_satisfied
            and allow_stop
            and right_iter >= effective_min_iter
        )
        return action_mse, action_l2, should_stop

    def record_iteration(
        *,
        iteration: int,
        state_before: str,
        state_after: str,
        latent_mse,
        latent_l2,
        latent_trigger,
        called_current: bool,
        reason: str,
        action_mse,
    ):
        if not collect_full_trace:
            return
        coda_call_mask.append(bool(called_current))
        if called_current:
            called_iters.append(int(iteration))
        else:
            skipped_iters.append(int(iteration))
        decisions.append(
            {
                "k": int(iteration),
                "scheduler_state_before": state_before,
                "scheduler_state_after": state_after,
                "latent_mse": float(latent_mse) if latent_mse is not None else None,
                "latent_l2": float(latent_l2) if latent_l2 is not None else None,
                "latent_trigger": bool(latent_trigger) if latent_trigger is not None else None,
                "call_coda": bool(called_current),
                "reason": reason,
                "action_mse": float(action_mse) if action_mse is not None else None,
            }
        )

    with rdvla_range("RDVLA/action_head/recurrent_loop_total"):
        with torch.no_grad():
            for it in range(max_iter):
                previous_recurrent_state = state.detach()
                if profile_coda_cost:
                    recurrent_start = model._sync_time()
                state = model._run_one_iteration(state, prelude_out, h_a, h_t, p)
                if profile_coda_cost:
                    run_one_iteration_ms_list.append(
                        (model._sync_time() - recurrent_start) * 1000.0
                    )
                with rdvla_range("RDVLA/action_head/origin_aware/finite_check/state"):
                    state_is_finite = _is_finite_tensor(state)
                if not state_is_finite:
                    raise NonFiniteOriginAwareInferenceError(
                        f"non-finite recurrent state at iteration {it + 1}",
                        stage="recurrent_state",
                        iteration=it + 1,
                        details={
                            "coda_call_count_before_failure": int(coda_call_count),
                        },
                    )
                if capture_warm_start_candidates:
                    warm_start_candidate_states.append(state.detach())

                actual_iter = it + 1
                state_before = scheduler_state
                latent_mse = None
                latent_l2 = None
                latent_trigger = None
                action_mse = None
                called_current = False
                decision_reason = None

                if actual_iter == max_iter:
                    current_output = decode_output(state, actual_iter, "max_iter")
                    current_output_iter = actual_iter
                    called_current = True
                    decision_reason = "max_iter"
                    if prev_output is not None and prev_output_iter == actual_iter - 1:
                        action_mse, _, _ = compare_adjacent_outputs(
                            prev_output,
                            prev_output_iter,
                            current_output,
                            current_output_iter,
                            allow_stop=False,
                        )
                        max_iteration_convergence_evaluable = True
                    else:
                        final_action_mse = None
                    scheduler_state = "MAX_ITER"
                    stop_reason = "max_iter"
                    record_iteration(
                        iteration=actual_iter,
                        state_before=state_before,
                        state_after=scheduler_state,
                        latent_mse=None,
                        latent_l2=None,
                        latent_trigger=None,
                        called_current=called_current,
                        reason=decision_reason,
                        action_mse=action_mse,
                    )
                    break

                if actual_iter in (1, 2):
                    decision_reason = "forced_initial" if actual_iter == 1 else "forced_second"
                    current_output = decode_output(state, actual_iter, decision_reason)
                    current_output_iter = actual_iter
                    called_current = True
                    should_stop = False
                    if prev_output is not None:
                        action_mse, _, should_stop = compare_adjacent_outputs(
                            prev_output,
                            prev_output_iter,
                            current_output,
                            current_output_iter,
                            allow_stop=True,
                        )
                    prev_output = current_output.detach()
                    prev_output_iter = actual_iter
                    skip_count = 0
                    scheduler_state = "CONTIGUOUS"
                    if should_stop:
                        adaptive_stop = True
                        stop_reason = requested_recurrence_strategy

                elif scheduler_state == "CONFIRM_PENDING":
                    decision_reason = "confirmation"
                    current_output = decode_output(state, actual_iter, decision_reason)
                    current_output_iter = actual_iter
                    called_current = True
                    action_mse, _, should_stop = compare_adjacent_outputs(
                        prev_output,
                        prev_output_iter,
                        current_output,
                        current_output_iter,
                        allow_stop=True,
                    )
                    prev_output = current_output.detach()
                    prev_output_iter = actual_iter
                    skip_count = 0
                    scheduler_state = "CONTIGUOUS"
                    if should_stop:
                        adaptive_stop = True
                        stop_reason = requested_recurrence_strategy

                else:
                    with rdvla_range("RDVLA/action_head/latent_precheck_total"):
                        with rdvla_range("RDVLA/action_head/latent_precheck/mse_compute"):
                            latent_diff = state.float() - previous_recurrent_state.float()
                            latent_mse_tensor = torch.mean(latent_diff ** 2)
                            latent_l2_tensor = torch.norm(latent_diff.flatten()) if collect_full_trace else None
                        with rdvla_range("RDVLA/action_head/latent_precheck/item_sync"):
                            latent_mse = latent_mse_tensor.item()
                            latent_l2 = latent_l2_tensor.item() if latent_l2_tensor is not None else None
                    if not math.isfinite(latent_mse) or (
                        latent_l2 is not None and not math.isfinite(latent_l2)
                    ):
                        raise NonFiniteOriginAwareInferenceError(
                            f"non-finite latent gate metric at iteration {actual_iter}",
                            stage="latent_gate_metric",
                            iteration=actual_iter,
                            details={
                                "coda_call_count_before_failure": int(coda_call_count),
                            },
                        )
                    latent_metric_count += 1
                    if collect_full_trace:
                        latent_mse_list.append(latent_mse)
                        latent_l2_list.append(latent_l2)
                    latent_trigger = (
                        actual_iter >= latent_precheck_min_iter
                        and latent_mse <= active_threshold
                    )

                    if scheduler_state == "CONTIGUOUS":
                        if latent_trigger:
                            decision_reason = "latent_trigger"
                            current_output = decode_output(state, actual_iter, decision_reason)
                            current_output_iter = actual_iter
                            called_current = True
                            action_mse, _, should_stop = compare_adjacent_outputs(
                                prev_output,
                                prev_output_iter,
                                current_output,
                                current_output_iter,
                                allow_stop=True,
                                latent_mse=latent_mse,
                                latent_l2=latent_l2,
                            )
                            prev_output = current_output.detach()
                            prev_output_iter = actual_iter
                            skip_count = 0
                            if should_stop:
                                adaptive_stop = True
                                stop_reason = requested_recurrence_strategy
                        else:
                            decision_reason = "latent_skip"
                            total_skipped_iters += 1
                            skip_count = 1
                            scheduler_state = "GAPPED"

                    elif scheduler_state == "GAPPED":
                        force_reason = None
                        if latent_trigger:
                            force_reason = "latent_trigger"
                        elif skip_count >= max_skip_iters:
                            force_reason = "max_skip_reached"

                        if force_reason is None:
                            decision_reason = "latent_skip"
                            total_skipped_iters += 1
                            skip_count += 1
                        elif confirmation_mode == "backfill_pair":
                            decision_reason = force_reason
                            backfill_output = decode_output(
                                previous_recurrent_state,
                                actual_iter - 1,
                                "backfill_previous",
                                is_backfill=True,
                            )
                            current_output = decode_output(state, actual_iter, force_reason)
                            current_output_iter = actual_iter
                            called_current = True
                            action_mse, _, should_stop = compare_adjacent_outputs(
                                backfill_output,
                                actual_iter - 1,
                                current_output,
                                current_output_iter,
                                allow_stop=True,
                                latent_mse=latent_mse,
                                latent_l2=latent_l2,
                            )
                            prev_output = current_output.detach()
                            prev_output_iter = actual_iter
                            skip_count = 0
                            scheduler_state = "CONTIGUOUS"
                            if should_stop:
                                adaptive_stop = True
                                stop_reason = requested_recurrence_strategy
                        else:
                            decision_reason = force_reason
                            current_output = decode_output(state, actual_iter, force_reason)
                            current_output_iter = actual_iter
                            called_current = True
                            prev_output = current_output.detach()
                            prev_output_iter = actual_iter
                            skip_count = 0
                            scheduler_state = "CONFIRM_PENDING"
                    else:
                        raise RuntimeError(f"Unsupported scheduler state: {scheduler_state}")

                record_iteration(
                    iteration=actual_iter,
                    state_before=state_before,
                    state_after=scheduler_state,
                    latent_mse=latent_mse,
                    latent_l2=latent_l2,
                    latent_trigger=latent_trigger,
                    called_current=called_current,
                    reason=decision_reason,
                    action_mse=action_mse,
                )
                if adaptive_stop:
                    break

    if current_output is None or current_output_iter != actual_iter:
        raise RuntimeError("origin-aware scheduler did not decode the terminal recurrent state")

    final_output = current_output
    model.last_recurrence_debug = {
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
        "iteration_mse": conv_score_list,
        "iteration_metric_values": conv_score_list,
        "conv_score_list": conv_score_list,
        "action_delta_list": action_delta_list,
        "adjacent_comparison_pairs": adjacent_comparison_pairs,
        "adjacent_comparison_pair_count": len(adjacent_comparison_pairs),
        "latent_mse_list": latent_mse_list,
        "latent_l2_list": latent_l2_list,
        "latent_action_mse_pairs": latent_action_mse_pairs,
        "latent_action_pair_count": len(latent_action_mse_pairs),
        "use_latent_precheck": True,
        "latent_precheck_mode": "origin_aware",
        "configured_use_latent_precheck": True,
        "execution_path": "origin_aware",
        "nonfinite_policy": "cold_retry_once",
        "numerical_retry_attempted": False,
        "numerical_retry_succeeded": None,
        "latent_precheck_trace_level_requested": trace_level,
        "latent_precheck_trace_level_applied": trace_level,
        "latent_precheck_trace_collected": trace_level != "off",
        "latent_precheck_thresh": active_threshold,
        "latent_precheck_warm_thresh": float(warm_threshold),
        "latent_precheck_cold_thresh": ORIGIN_AWARE_COLD_THRESHOLD,
        "latent_precheck_active_threshold": active_threshold,
        "latent_precheck_origin": actual_origin,
        "latent_precheck_min_iter": int(latent_precheck_min_iter),
        "latent_precheck_force_interval": 0,
        "latent_precheck_max_skip_iters": int(max_skip_iters),
        "latent_precheck_confirmation_mode": confirmation_mode,
        "latent_precheck_coda_call_mask": coda_call_mask,
        "latent_precheck_skipped_iters": skipped_iters,
        "latent_precheck_called_iters": called_iters,
        "latent_precheck_skip_count": int(total_skipped_iters),
        "latent_precheck_call_count": int(coda_call_count),
        "latent_precheck_skip_ratio": (
            total_skipped_iters / actual_iter if actual_iter else 0.0
        ),
        "latent_precheck_decisions": decisions,
        "latent_metric_count": int(latent_metric_count),
        "coda_call_records": coda_call_records,
        "coda_reason_counts": dict(coda_reason_counts),
        "origin_aware_scheduler_state": scheduler_state,
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
        "use_cached_final_output_requested": bool(use_cached_final_output_requested),
        "returned_cached_final_output": True,
        "cached_final_matches_returned": final_output is current_output,
        "final_state_coda_executed": current_output_iter == actual_iter,
        "final_state_output_iteration": int(current_output_iter),
        "max_iteration_convergence_evaluable": bool(
            max_iteration_convergence_evaluable
        ),
        "warm_start_min_iter_configured": int(warm_start_min_iter_configured),
        "effective_min_iter": int(effective_min_iter),
        "warm_start_state_used": actual_origin == "ACTUAL_WARM",
        "min_iter_gate_block_count": int(min_iter_gate_block_count),
        "first_threshold_satisfied_k": first_threshold_satisfied_k,
    }

    if profile_coda_cost:
        run_one_iteration_ms_total = sum(run_one_iteration_ms_list)
        get_output_ms_total = sum(get_output_ms_list)
        coda_ms_total = sum(coda_ms_list)
        output_proj_ms_total = sum(output_proj_ms_list)
        profiled_recurrent_ms_total = run_one_iteration_ms_total + get_output_ms_total
        model.last_recurrence_debug.update(
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

    model._store_warm_start_candidate(
        warm_start_candidate_states,
        actual_iter,
        warm_start_source,
    )
    return final_output, actual_iter, final_action_mse
