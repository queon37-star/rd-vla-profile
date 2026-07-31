"""Pure validation helpers for RD-VLA recurrence and latent pre-check settings."""

from typing import Optional


SUPPORTED_RECURRENCE_STRATEGIES = {
    "fixed",
    "kl_divergence",
    "adjacent_action_mse",
    "cosine_similarity",
}
SUPPORTED_LATENT_PRECHECK_MODES = {"legacy", "off", "origin_aware"}
SUPPORTED_LATENT_PRECHECK_TRACE_LEVELS = {"off", "summary", "full"}


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

    if mode == "origin_aware" and not origin_aware_implemented:
        raise NotImplementedError("latent_precheck_mode='origin_aware' is not implemented yet")

    return mode
