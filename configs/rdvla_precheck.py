"""Pure validation helpers for RD-VLA recurrence and latent pre-check settings."""

import math
from numbers import Real
from typing import Optional

from prismatic.models.latent_metrics import LATENT_METRIC_NAMES, validate_latent_metric_eps


SUPPORTED_RECURRENCE_STRATEGIES = {
    "fixed",
    "kl_divergence",
    "adjacent_action_mse",
    "cosine_similarity",
    "latent_only",
}
SUPPORTED_LATENT_PRECHECK_MODES = {"legacy", "off", "origin_aware"}
SUPPORTED_LATENT_PRECHECK_TRACE_LEVELS = {"off", "summary", "full"}
SUPPORTED_ORIGIN_AWARE_CONFIRMATION_MODES = {"next_iter", "backfill_pair"}
SUPPORTED_NONFINITE_POLICIES = {"legacy", "cold_retry_once"}
ORIGIN_AWARE_COLD_THRESHOLD = 0.2


def _finite_non_negative(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite non-negative number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return value


def validate_latent_only_configuration(
    recurrence_strategy: Optional[str],
    *,
    metric: str,
    cold_threshold: float,
    warm_threshold: float,
    min_iter: int,
    eps: float,
    use_latent_precheck: bool = False,
    latent_precheck_mode: str = "legacy",
    shadow_full_depth: bool = False,
    use_cached_final_output: bool = False,
) -> None:
    """Validate scalar settings and keep latent-only independent of schedulers."""
    if metric not in LATENT_METRIC_NAMES:
        raise ValueError(
            f"Unsupported latent_only_metric: {metric}. "
            f"Expected one of {list(LATENT_METRIC_NAMES)}"
        )
    _finite_non_negative(cold_threshold, "latent_only_cold_threshold")
    _finite_non_negative(warm_threshold, "latent_only_warm_threshold")
    if isinstance(min_iter, bool) or not isinstance(min_iter, int) or min_iter < 2:
        raise ValueError("latent_only_min_iter must be an integer >= 2")
    validate_latent_metric_eps(eps)

    if recurrence_strategy != "latent_only":
        return
    if use_latent_precheck or latent_precheck_mode == "origin_aware":
        raise ValueError("latent_only cannot use a latent pre-check scheduler")
    if shadow_full_depth:
        raise ValueError("latent_only cannot enable shadow_full_depth")
    if use_cached_final_output:
        raise ValueError("latent_only cannot reuse a cached action output")


def canonicalize_recurrence_strategy(strategy: Optional[str]) -> Optional[str]:
    """Resolve legacy strategy names without mutating the requested value."""
    if strategy is None:
        return None
    if strategy not in SUPPORTED_RECURRENCE_STRATEGIES:
        raise ValueError(f"Unsupported recurrence strategy: {strategy}")
    if strategy == "kl_divergence":
        return "adjacent_action_mse"
    return strategy


def validate_latent_precheck_configuration(
    mode: str,
    trace_level: str,
    use_latent_precheck: bool,
    *,
    origin_aware_implemented: bool = False,
    warm_threshold: Optional[float] = None,
    max_skip_iters: int = 0,
    confirmation_mode: str = "next_iter",
    warm_start_source: Optional[str] = None,
    recurrence_strategy: Optional[str] = None,
    use_warm_start: bool = False,
    min_iter: int = 2,
    nonfinite_policy: str = "legacy",
    shadow_full_depth: bool = False,
) -> str:
    """Validate mode combinations and fail closed for unfinished schedulers."""
    if mode not in SUPPORTED_LATENT_PRECHECK_MODES:
        raise ValueError(f"Unsupported latent_precheck_mode: {mode}")
    if trace_level not in SUPPORTED_LATENT_PRECHECK_TRACE_LEVELS:
        raise ValueError(f"Unsupported latent_precheck_trace_level: {trace_level}")
    if nonfinite_policy not in SUPPORTED_NONFINITE_POLICIES:
        raise ValueError(f"Unsupported nonfinite_policy: {nonfinite_policy}")
    if not isinstance(shadow_full_depth, bool):
        raise ValueError("shadow_full_depth must be a boolean")

    if mode == "off":
        if use_latent_precheck:
            raise ValueError("latent_precheck_mode='off' requires use_latent_precheck=False")
        if trace_level != "off":
            raise ValueError("latent_precheck_mode='off' requires latent_precheck_trace_level='off'")

    if mode == "origin_aware":
        if not origin_aware_implemented:
            raise NotImplementedError("latent_precheck_mode='origin_aware' is not implemented yet")
        if not use_latent_precheck:
            raise ValueError("latent_precheck_mode='origin_aware' requires use_latent_precheck=True")
        if not use_warm_start:
            raise ValueError("latent_precheck_mode='origin_aware' requires use_warm_start=True")
        if warm_start_source != "midpoint":
            raise ValueError("latent_precheck_mode='origin_aware' requires warm_start_source='midpoint'")
        if isinstance(warm_threshold, bool) or not isinstance(warm_threshold, Real):
            raise ValueError("latent_precheck_warm_thresh must be a finite non-negative number")
        warm_threshold = float(warm_threshold)
        if not math.isfinite(warm_threshold) or warm_threshold < 0:
            raise ValueError("latent_precheck_warm_thresh must be a finite non-negative number")
        if isinstance(max_skip_iters, bool) or not isinstance(max_skip_iters, int) or max_skip_iters < 1:
            raise ValueError("latent_precheck_max_skip_iters must be an integer >= 1")
        if confirmation_mode not in SUPPORTED_ORIGIN_AWARE_CONFIRMATION_MODES:
            raise ValueError(
                "Unsupported latent_precheck_confirmation_mode: "
                f"{confirmation_mode}"
            )
        if canonicalize_recurrence_strategy(recurrence_strategy) != "adjacent_action_mse":
            raise ValueError(
                "latent_precheck_mode='origin_aware' requires "
                "recurrence_strategy='adjacent_action_mse' or legacy alias 'kl_divergence'"
            )
        if isinstance(min_iter, bool) or not isinstance(min_iter, int) or min_iter < 2:
            raise ValueError("latent_precheck_min_iter must be an integer >= 2")
        if nonfinite_policy != "cold_retry_once":
            raise ValueError(
                "latent_precheck_mode='origin_aware' requires "
                "nonfinite_policy='cold_retry_once'"
            )
    elif nonfinite_policy != "legacy":
        raise ValueError("nonfinite_policy='cold_retry_once' requires latent_precheck_mode='origin_aware'")

    if shadow_full_depth:
        if mode != "off" or use_latent_precheck or trace_level != "off":
            raise ValueError(
                "shadow_full_depth requires clean latent_precheck_mode='off'"
            )
        if not use_warm_start or warm_start_source != "midpoint":
            raise ValueError(
                "shadow_full_depth requires midpoint warm-start"
            )
        if canonicalize_recurrence_strategy(recurrence_strategy) != "adjacent_action_mse":
            raise ValueError(
                "shadow_full_depth requires adjacent action-MSE recurrence"
            )

    return mode
