"""Fixed-depth recurrence with exactly one terminal Coda call."""

from __future__ import annotations

from typing import Any, Dict

import torch

from prismatic.utils.rdvla_profiler import rdvla_range


class NonFiniteFixedTerminalOnlyInferenceError(RuntimeError):
    """Raised before a non-finite fixed-terminal-only result can escape."""

    def __init__(self, message: str, *, stage: str, iteration: int, coda_call_count: int):
        super().__init__(message)
        self.stage = stage
        self.iteration = int(iteration)
        self.coda_call_count = int(coda_call_count)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": type(self).__name__,
            "message": str(self),
            "stage": self.stage,
            "iteration": self.iteration,
            "coda_call_count": self.coda_call_count,
        }


def _finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all().item())


def _debug_payload(
    model,
    *,
    fixed_k: int,
    max_iter: int,
    actual_iter: int,
    actual_origin: str,
    requested_recurrence_strategy: str,
    profile_coda_cost: bool,
    coda_call_count: int,
    final_state_coda_executed: bool,
    stop_reason: str,
):
    warm = model.last_inference_metadata["warm_start"]
    return {
        "strategy": "fixed_terminal_only",
        "requested_recurrence_strategy": requested_recurrence_strategy,
        "canonical_recurrence_strategy": "fixed_terminal_only",
        "canonical_metric_name": "fixed_terminal_only",
        "execution_path": "fixed_terminal_only",
        "actual_origin": actual_origin,
        "fixed_K": int(fixed_k),
        "K_t": int(actual_iter),
        "max_iter": int(max_iter),
        "adaptive_stop": False,
        "stop_reason": stop_reason,
        "canonical_stop_reason": stop_reason,
        "metric_name": "fixed_depth",
        "threshold": None,
        "action_mse_threshold": None,
        "iteration_mse": [],
        "iteration_metric_values": [],
        "conv_score_list": [],
        "action_delta_list": [],
        "final_mse": None,
        "final_conv_score": None,
        "final_convergence_evaluable": False,
        "coda_call_count": int(coda_call_count),
        "get_output_call_count": int(coda_call_count),
        "get_output_attempt_count_intent_to_treat": int(coda_call_count),
        "final_state_coda_executed": bool(final_state_coda_executed),
        "returned_cached_final_output": False,
        "use_cached_final_output": False,
        "profiling_enabled": bool(profile_coda_cost),
        "warm_start_state_used": actual_origin == "ACTUAL_WARM",
        "warm_start_source_index": warm.get("source_index"),
        "warm_start_source_iteration": warm.get("source_iteration"),
    }


def run_fixed_terminal_only(
    model,
    state: torch.Tensor,
    prelude_out: torch.Tensor,
    h_a: torch.Tensor,
    h_t: torch.Tensor,
    p: torch.Tensor,
    *,
    fixed_k: int,
    max_iter: int,
    actual_origin: str,
    requested_recurrence_strategy: str,
    profile_coda_cost: bool,
    capture_warm_start_candidates: bool,
    warm_start_candidate_states,
    warm_start_source: str,
):
    """Run exactly K recurrent steps and decode only the terminal state."""

    recurrent_timings = []
    actual_iter = 0
    coda_call_count = 0

    if not _finite(state):
        model.last_recurrence_debug = _debug_payload(
            model,
            fixed_k=fixed_k,
            max_iter=max_iter,
            actual_iter=actual_iter,
            actual_origin=actual_origin,
            requested_recurrence_strategy=requested_recurrence_strategy,
            profile_coda_cost=profile_coda_cost,
            coda_call_count=0,
            final_state_coda_executed=False,
            stop_reason="non_finite_initial_state",
        )
        raise NonFiniteFixedTerminalOnlyInferenceError(
            "non-finite selected initial state in fixed_terminal_only",
            stage="initial_state",
            iteration=0,
            coda_call_count=0,
        )

    with rdvla_range("RDVLA/action_head/recurrent_loop_total"):
        with torch.no_grad():
            for iteration in range(1, fixed_k + 1):
                recurrent_start = model._sync_time() if profile_coda_cost else None
                state = model._run_one_iteration(state, prelude_out, h_a, h_t, p)
                if profile_coda_cost:
                    recurrent_timings.append(
                        (model._sync_time() - recurrent_start) * 1000.0
                    )
                actual_iter = iteration
                if not _finite(state):
                    model.last_recurrence_debug = _debug_payload(
                        model,
                        fixed_k=fixed_k,
                        max_iter=max_iter,
                        actual_iter=actual_iter,
                        actual_origin=actual_origin,
                        requested_recurrence_strategy=requested_recurrence_strategy,
                        profile_coda_cost=profile_coda_cost,
                        coda_call_count=0,
                        final_state_coda_executed=False,
                        stop_reason="non_finite_recurrent_state",
                    )
                    raise NonFiniteFixedTerminalOnlyInferenceError(
                        f"non-finite recurrent state at iteration {iteration}",
                        stage="recurrent_state",
                        iteration=iteration,
                        coda_call_count=0,
                    )
                if capture_warm_start_candidates:
                    warm_start_candidate_states.append(state.detach())

    with rdvla_range("RDVLA/action_head/final_get_output"):
        final_output = model._get_output(
            state, h_a, h_t, p, profile=profile_coda_cost
        )
    coda_call_count += 1

    if not _finite(final_output):
        model.last_recurrence_debug = _debug_payload(
            model,
            fixed_k=fixed_k,
            max_iter=max_iter,
            actual_iter=actual_iter,
            actual_origin=actual_origin,
            requested_recurrence_strategy=requested_recurrence_strategy,
            profile_coda_cost=profile_coda_cost,
            coda_call_count=coda_call_count,
            final_state_coda_executed=True,
            stop_reason="non_finite_coda_output",
        )
        raise NonFiniteFixedTerminalOnlyInferenceError(
            f"non-finite terminal Coda output at iteration {actual_iter}",
            stage="coda_output",
            iteration=actual_iter,
            coda_call_count=coda_call_count,
        )

    if coda_call_count != 1:
        raise AssertionError("fixed_terminal_only must call Coda exactly once")

    if capture_warm_start_candidates:
        model._store_warm_start_candidate(
            warm_start_candidate_states, actual_iter, warm_start_source
        )

    debug = _debug_payload(
        model,
        fixed_k=fixed_k,
        max_iter=max_iter,
        actual_iter=actual_iter,
        actual_origin=actual_origin,
        requested_recurrence_strategy=requested_recurrence_strategy,
        profile_coda_cost=profile_coda_cost,
        coda_call_count=coda_call_count,
        final_state_coda_executed=True,
        stop_reason="fixed_terminal_only",
    )
    if profile_coda_cost:
        timing = model._last_get_output_timing
        get_output_ms = [timing["get_output_ms"]]
        coda_ms = [timing["coda_ms"]]
        output_proj_ms = [timing["output_proj_ms"]]
        recurrent_total = sum(recurrent_timings)
        output_total = sum(get_output_ms)
        debug.update(
            {
                "run_one_iteration_ms_list": recurrent_timings,
                "convergence_check_ms_list": [],
                "get_output_ms_list": get_output_ms,
                "coda_ms_list": coda_ms,
                "output_proj_ms_list": output_proj_ms,
                "coda_ms_total": sum(coda_ms),
                "get_output_ms_total": output_total,
                "run_one_iteration_ms_total": recurrent_total,
                "output_proj_ms_total": sum(output_proj_ms),
                "coda_time_ratio_total": (
                    sum(coda_ms) / (recurrent_total + output_total)
                    if recurrent_total + output_total
                    else 0.0
                ),
            }
        )
    model.last_recurrence_debug = debug
    return final_output, actual_iter, None
