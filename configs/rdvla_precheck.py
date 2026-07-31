"""Pure validation helpers for RD-VLA recurrence and latent pre-check settings."""

import math
from numbers import Real
from typing import Optional


SUPPORTED_RECURRENCE_STRATEGIES = {
    "fixed",
    "kl_divergence",
    "adjacent_action_mse",
    "cosine_similarity",
}
SUPPORTED_LATENT_PRECHECK_MODES = {"legacy", "off", "origin_aware"}
SUPPORTED_LATENT_PRECHECK_TRACE_LEVELS = {"off", "summary", "full"}
SUPPORTED_ORIGIN_AWARE_CONFIRMATION_MODES = {"next_iter", "backfill_pair"}
ORIGIN_AWARE_COLD_THRESHOLD = 0.2


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
) -> str:
    """Validate mode combinations and fail closed for unfinished schedulers."""
    if mode not in SUPPORTED_LATENT_PRECHECK_MODES:
        raise ValueError(f"Unsupported latent_precheck_mode: {mode}")
    if trace_level not in SUPPORTED_LATENT_PRECHECK_TRACE_LEVELS:
        raise ValueError(f"Unsupported latent_precheck_trace_level: {trace_level}")

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

    return mode
