"""Diagnostic full-depth trace collection for clean midpoint warm-start runs."""

from typing import Any, Dict, Optional

import torch

from prismatic.models.latent_dynamics import (
    LATENT_DYNAMICS_FIELDS,
    NonFiniteLatentDynamicsError,
    compute_latent_dynamics,
)
from prismatic.models.latent_metrics import compute_latent_metrics
from prismatic.utils.rdvla_profiler import rdvla_range


def _is_finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all().item())


def capture_raw_shadow_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Copy a detached tensor to CPU without changing dtype, shape, or values."""
    if not torch.is_tensor(tensor):
        raise TypeError("raw shadow value must be a tensor")
    if not _is_finite(tensor):
        raise ValueError("raw shadow tensor contains non-finite values")
    return tensor.detach().to(device="cpu", copy=True)


def _metric_pair(
    previous: Optional[torch.Tensor],
    current: torch.Tensor,
) -> tuple[Optional[float], Optional[float]]:
    if previous is None or not _is_finite(previous) or not _is_finite(current):
        return None, None
    difference = current.float() - previous.float()
    return (
        float(torch.mean(difference ** 2).item()),
        float(torch.norm(difference.flatten()).item()),
    )


def build_shadow_trace_record(
    *,
    iteration: int,
    phase: str,
    previous_state: Optional[torch.Tensor],
    current_state: torch.Tensor,
    previous_output: Optional[torch.Tensor],
    current_output: torch.Tensor,
    previous_update: Optional[torch.Tensor] = None,
    warm_anchor: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> Dict[str, Any]:
    """Build a JSON-safe record without retaining GPU tensors."""
    state_finite = _is_finite(current_state)
    output_finite = _is_finite(current_output)
    latent_metrics = (
        compute_latent_metrics(current_state, previous_state, eps=eps)
        if previous_state is not None
        and _is_finite(previous_state)
        and _is_finite(current_state)
        else None
    )
    latent_dynamics_error = None
    if latent_metrics:
        try:
            latent_dynamics = compute_latent_dynamics(
                current_state,
                previous_state,
                previous_update=previous_update,
                warm_anchor=warm_anchor,
                eps=eps,
            )
        except NonFiniteLatentDynamicsError as exc:
            latent_dynamics = {name: None for name in LATENT_DYNAMICS_FIELDS}
            latent_dynamics_error = {
                "reason": "non_finite",
                "message": str(exc),
            }
    else:
        latent_dynamics = {name: None for name in LATENT_DYNAMICS_FIELDS}
    action_mse, action_l2 = _metric_pair(previous_output, current_output)
    return {
        "k": int(iteration),
        "phase": phase,
        "state_finite": state_finite,
        "output_finite": output_finite,
        "latent_mse": latent_metrics["raw_mse"] if latent_metrics else None,
        "latent_l2": (
            float(
                torch.norm(
                    current_state.float().reshape(-1)
                    - previous_state.float().reshape(-1)
                ).item()
            )
            if latent_metrics
            else None
        ),
        "raw_mse": latent_metrics["raw_mse"] if latent_metrics else None,
        "relative_mse": latent_metrics["relative_mse"] if latent_metrics else None,
        "cosine_distance": (
            latent_metrics["cosine_distance"] if latent_metrics else None
        ),
        "relative_l2": latent_metrics["relative_l2"] if latent_metrics else None,
        **latent_dynamics,
        "latent_dynamics_error": latent_dynamics_error,
        "action_mse": action_mse,
        "action_l2": action_l2,
    }


def run_shadow_tail(
    model,
    *,
    state: torch.Tensor,
    current_output: torch.Tensor,
    actual_iter: int,
    max_iter: int,
    prelude_out: torch.Tensor,
    h_a: torch.Tensor,
    h_t: torch.Tensor,
    p: torch.Tensor,
    previous_update: Optional[torch.Tensor] = None,
    warm_anchor: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
    collect_raw: bool = False,
) -> Dict[str, Any]:
    """Run a detached diagnostic tail that cannot change production state."""
    records = []
    raw_states = []
    raw_actions = []
    error = None
    tail_state = state.detach().clone()
    tail_output = current_output.detach().clone()
    tail_previous_update = (
        previous_update.detach().clone().float()
        if previous_update is not None
        else None
    )

    with torch.no_grad():
        for iteration in range(actual_iter + 1, max_iter + 1):
            previous_state = tail_state
            previous_output = tail_output
            try:
                with rdvla_range("RDVLA/action_head/shadow/recurrent_one_iteration"):
                    candidate_state = model._run_one_iteration(
                        previous_state, prelude_out, h_a, h_t, p
                    )
                if not _is_finite(candidate_state):
                    error = {
                        "iteration": int(iteration),
                        "stage": "recurrent_state",
                        "reason": "non_finite",
                    }
                    break

                with rdvla_range("RDVLA/action_head/shadow/get_output"):
                    candidate_output = model._get_output(
                        candidate_state, h_a, h_t, p, profile=False
                    )
                if not _is_finite(candidate_output):
                    error = {
                        "iteration": int(iteration),
                        "stage": "coda_output",
                        "reason": "non_finite",
                    }
                    break

                record = build_shadow_trace_record(
                    iteration=iteration,
                    phase="shadow_tail",
                    previous_state=previous_state,
                    current_state=candidate_state,
                    previous_output=previous_output,
                    current_output=candidate_output,
                    previous_update=tail_previous_update,
                    warm_anchor=warm_anchor,
                    eps=eps,
                )
                if record["latent_dynamics_error"] is not None:
                    error = {
                        "iteration": int(iteration),
                        "stage": "latent_dynamics",
                        **record["latent_dynamics_error"],
                    }
                    break
                if record["latent_mse"] is None or record["action_mse"] is None:
                    error = {
                        "iteration": int(iteration),
                        "stage": "trace_metric",
                        "reason": "non_finite",
                    }
                    break
                records.append(record)
                if collect_raw:
                    raw_states.append(capture_raw_shadow_tensor(candidate_state))
                    raw_actions.append(capture_raw_shadow_tensor(candidate_output))
                tail_previous_update = (
                    candidate_state.float() - previous_state.float()
                ).detach().clone()
                tail_state = candidate_state.detach().clone()
                tail_output = candidate_output.detach().clone()
            except Exception as exc:  # Shadow diagnostics must not replace a valid production result.
                error = {
                    "iteration": int(iteration),
                    "stage": "shadow_tail",
                    "reason": "exception",
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                }
                break

    return {
        "records": records,
        "completed": error is None and actual_iter + len(records) == max_iter,
        "error": error,
        "tail_iteration_count": len(records),
        "tail_start_iteration": actual_iter + 1 if actual_iter < max_iter else None,
        "raw_states": raw_states,
        "raw_actions": raw_actions,
    }
