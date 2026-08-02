"""Independent latent-only recurrence stopping for RD-VLA inference."""

from __future__ import annotations

from typing import Any, Dict

import torch

from prismatic.models.latent_metrics import compute_latent_metrics
from prismatic.utils.rdvla_profiler import rdvla_range


class NonFiniteLatentOnlyInferenceError(RuntimeError):
    """Raised before a non-finite latent-only result can leave the action head."""

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


def run_latent_only_adaptive(
    model,
    state: torch.Tensor,
    prelude_out: torch.Tensor,
    h_a: torch.Tensor,
    h_t: torch.Tensor,
    p: torch.Tensor,
    *,
    max_iter: int,
    metric_name: str,
    cold_threshold: float,
    warm_threshold: float,
    min_iter: int,
    eps: float,
    actual_origin: str,
    requested_recurrence_strategy: str,
    profile_coda_cost: bool,
    capture_warm_start_candidates: bool,
    warm_start_candidate_states,
    warm_start_source: str,
    warm_start_min_iter_configured: int,
):
    """Advance recurrence using latent comparisons, then decode terminal state once."""
    if max_iter < 1:
        raise ValueError("recurrence_max_iter must be >= 1")
    effective_threshold = (
        float(warm_threshold) if actual_origin == "ACTUAL_WARM" else float(cold_threshold)
    )

    previous_state = None
    actual_iter = 0
    stop_reason = "max_iter"
    metric_trace = []
    selected_values = []
    coda_call_count = 0
    recurrent_timings = []
    metric_timings = []

    with rdvla_range("RDVLA/action_head/recurrent_loop_total"):
        with torch.no_grad():
            for iteration in range(1, max_iter + 1):
                recurrent_start = model._sync_time() if profile_coda_cost else None
                state = model._run_one_iteration(state, prelude_out, h_a, h_t, p)
                if profile_coda_cost:
                    recurrent_timings.append(
                        (model._sync_time() - recurrent_start) * 1000.0
                    )
                actual_iter = iteration
                if not _finite(state):
                    raise NonFiniteLatentOnlyInferenceError(
                        f"non-finite recurrent state at iteration {iteration}",
                        stage="recurrent_state",
                        iteration=iteration,
                        coda_call_count=0,
                    )
                if capture_warm_start_candidates:
                    warm_start_candidate_states.append(state.detach())

                if previous_state is not None and iteration >= min_iter:
                    metric_start = model._sync_time() if profile_coda_cost else None
                    try:
                        values = compute_latent_metrics(state, previous_state, eps=eps)
                    except Exception as exc:
                        raise NonFiniteLatentOnlyInferenceError(
                            f"invalid latent metric at iteration {iteration}: {exc}",
                            stage="latent_metric",
                            iteration=iteration,
                            coda_call_count=0,
                        ) from exc
                    if profile_coda_cost:
                        metric_timings.append(
                            (model._sync_time() - metric_start) * 1000.0
                        )
                    selected_value = values[metric_name]
                    selected_values.append(selected_value)
                    metric_trace.append(
                        {
                            "k": int(iteration),
                            **values,
                            "selected_metric": metric_name,
                            "selected_value": selected_value,
                            "threshold": effective_threshold,
                            "below_threshold": bool(selected_value <= effective_threshold),
                        }
                    )
                    if selected_value <= effective_threshold:
                        stop_reason = "latent_threshold"
                        break
                previous_state = state.detach()

    with rdvla_range("RDVLA/action_head/final_get_output"):
        final_output = model._get_output(
            state, h_a, h_t, p, profile=profile_coda_cost
        )
    coda_call_count += 1
    if not _finite(final_output):
        raise NonFiniteLatentOnlyInferenceError(
            f"non-finite terminal Coda output at iteration {actual_iter}",
            stage="coda_output",
            iteration=actual_iter,
            coda_call_count=coda_call_count,
        )
    if coda_call_count != 1:
        raise AssertionError("latent_only must call Coda exactly once")

    model._store_warm_start_candidate(
        warm_start_candidate_states, actual_iter, warm_start_source
    )
    final_metric = selected_values[-1] if selected_values else None
    debug = {
        "strategy": "latent_only",
        "requested_recurrence_strategy": requested_recurrence_strategy,
        "canonical_recurrence_strategy": "latent_only",
        "canonical_metric_name": metric_name,
        "action_mse_threshold": None,
        "threshold": effective_threshold,
        "configured_cold_threshold": float(cold_threshold),
        "configured_warm_threshold": float(warm_threshold),
        "effective_threshold": effective_threshold,
        "actual_origin": actual_origin,
        "fixed_K": None,
        "K_t": int(actual_iter),
        "max_iter": int(max_iter),
        "adaptive_stop": stop_reason == "latent_threshold",
        "metric_name": metric_name,
        "iteration_mse": [],
        "iteration_metric_values": selected_values,
        "conv_score_list": [],
        "action_delta_list": [],
        "latent_mse_list": [item["raw_mse"] for item in metric_trace],
        "latent_l2_list": [],
        "latent_action_mse_pairs": [],
        "latent_action_pair_count": 0,
        "latent_only_metric": metric_name,
        "latent_only_eps": float(eps),
        "latent_only_min_iter": int(min_iter),
        "latent_only_trace": metric_trace,
        "latent_metric_call_count": len(metric_trace),
        "latent_metric_count": len(metric_trace),
        "coda_call_count": coda_call_count,
        "get_output_call_count": coda_call_count,
        "final_state_coda_executed": True,
        "returned_cached_final_output": False,
        "use_cached_final_output": False,
        "use_latent_precheck": False,
        "latent_precheck_mode": "bypassed_for_latent_only",
        "origin_aware_scheduler_state": None,
        "execution_path": "latent_only",
        "final_mse": None,
        "final_conv_score": final_metric,
        "final_convergence_evaluable": final_metric is not None,
        "stop_reason": stop_reason,
        "canonical_stop_reason": stop_reason,
        "profiling_enabled": bool(profile_coda_cost),
        "warm_start_min_iter_configured": int(warm_start_min_iter_configured),
        "effective_min_iter": int(min_iter),
        "warm_start_state_used": actual_origin == "ACTUAL_WARM",
        "min_iter_gate_block_count": 0,
        "first_threshold_satisfied_k": (
            int(actual_iter) if stop_reason == "latent_threshold" else None
        ),
    }
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
                "latent_metric_ms_list": metric_timings,
                "convergence_check_ms_list": metric_timings,
                "get_output_ms_list": get_output_ms,
                "coda_ms_list": coda_ms,
                "output_proj_ms_list": output_proj_ms,
                "coda_ms_total": sum(coda_ms),
                "get_output_ms_total": output_total,
                "run_one_iteration_ms_total": recurrent_total,
                "latent_metric_ms_total": sum(metric_timings),
                "output_proj_ms_total": sum(output_proj_ms),
                "coda_time_ratio_total": (
                    sum(coda_ms) / (recurrent_total + output_total)
                    if recurrent_total + output_total
                    else 0.0
                ),
            }
        )
    model.last_recurrence_debug = debug
    return final_output, actual_iter, final_metric
