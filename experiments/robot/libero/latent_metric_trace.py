"""Lightweight identity helpers for LIBERO latent metric trace logging."""

from __future__ import annotations

from numbers import Integral
from typing import Any, Dict, Mapping

from prismatic.models.latent_dynamics import LATENT_DYNAMICS_FIELDS


LATENT_ONLY_STOP_REASONS = frozenset({"max_iter", "latent_threshold"})


def build_stop_reason_fields(recurrence_debug: Mapping[str, Any]) -> Dict[str, Any]:
    """Serialize stop reasons while enforcing the latent-only logging contract."""
    stop_reason = recurrence_debug.get("stop_reason")
    canonical_stop_reason = recurrence_debug.get("canonical_stop_reason")
    if recurrence_debug.get("canonical_recurrence_strategy") == "latent_only":
        if stop_reason not in LATENT_ONLY_STOP_REASONS:
            raise ValueError(f"invalid latent_only stop_reason: {stop_reason!r}")
        if canonical_stop_reason not in LATENT_ONLY_STOP_REASONS:
            raise ValueError(
                "invalid latent_only canonical_stop_reason: "
                f"{canonical_stop_reason!r}"
            )
    return {
        "stop_reason": stop_reason,
        "canonical_stop_reason": canonical_stop_reason,
    }


def require_prediction_id(prediction_step: int) -> int:
    """Return the canonical per-episode prediction counter or fail explicitly."""
    if isinstance(prediction_step, bool) or not isinstance(prediction_step, Integral):
        raise ValueError(
            "prediction_step must be a non-null monotonically increasing integer"
        )
    prediction_id = int(prediction_step)
    if prediction_id < 0:
        raise ValueError("prediction_step must be non-negative")
    return prediction_id


def build_action_head_workload_identity(
    *,
    capture_requested: bool,
    task_id,
    episode_id,
    paired_trial_id,
    prediction_id,
    initial_state_id,
    episode_seed,
):
    """Build protocol-only workload identity without touching absent legacy fields."""
    prediction_id = require_prediction_id(prediction_id)
    if not capture_requested:
        return None
    fields = {
        "task_id": task_id,
        "episode_id": episode_id,
        "paired_trial_id": paired_trial_id,
        "initial_state_id": initial_state_id,
        "episode_seed": episode_seed,
    }
    missing = [name for name, value in fields.items() if value is None]
    if missing:
        raise ValueError(
            "action-head workload capture requires non-null protocol identity fields: "
            + ", ".join(missing)
        )
    return {
        "task_id": int(task_id),
        "episode_id": int(episode_id),
        "paired_trial_id": int(paired_trial_id),
        "prediction_step": prediction_id,
        "initial_state_id": int(initial_state_id),
        "episode_seed": int(episode_seed),
    }


def build_latent_metric_trace_records(
    recurrence_debug: Mapping[str, Any],
    *,
    task_id: int,
    episode_id: int,
    prediction_id: int,
    actual_origin: str,
) -> list[Dict[str, Any]]:
    """Attach rollout identity and action labels to scalar full-depth traces."""
    prediction_id = require_prediction_id(prediction_id)
    if not recurrence_debug.get("latent_metric_trace_enabled", False):
        return []
    baseline_k = recurrence_debug.get("K_t")
    if baseline_k is not None:
        baseline_k = int(baseline_k)
    records = []
    for item in recurrence_debug.get("shadow_trace", []):
        raw_mse = item.get("raw_mse", item.get("latent_mse"))
        action_mse = item.get("action_mse")
        if raw_mse is None or action_mse is None:
            continue
        records.append(
            {
                "iteration_index": int(item["k"]),
                "phase": item.get("phase"),
                "actual_origin": actual_origin,
                "raw_mse": float(raw_mse),
                "relative_mse": float(item["relative_mse"]),
                "cosine_distance": float(item["cosine_distance"]),
                "relative_l2": float(item["relative_l2"]),
                **{
                    name: (
                        None if item.get(name) is None else float(item[name])
                    )
                    for name in LATENT_DYNAMICS_FIELDS
                },
                "adjacent_action_mse": float(action_mse),
                "action_mse_below_0_001": bool(float(action_mse) < 0.001),
                "baseline_stopping_iteration": baseline_k,
                "task_id": int(task_id),
                "episode_id": int(episode_id),
                "prediction_id": prediction_id,
            }
        )
    return records
