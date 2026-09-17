"""LIBERO evaluation runner for the Warm-start K/2 reuse-point ablation.

This wrapper intentionally leaves the existing S1 / midpoint(K/2-1) / final
implementations untouched.  It adds one experiment-only source name,
``midpoint_k2``, whose candidate-list index is ``actual_iter // 2``.

The base runner currently accepts only {s1, midpoint, final}, so validation is
performed with the equivalent existing ``midpoint`` label and the requested
label is restored immediately afterward.  Runtime state selection is then
handled by the patched selector below.  As a result, logs and metadata retain
``warm_start_source=midpoint_k2`` while all other evaluation behavior remains
identical to the paper LIBERO runner.
"""

from experiments.robot.libero import run_libero_eval as base_runner
from prismatic.models import action_heads


WARM_START_SOURCE_K2 = "midpoint_k2"


_original_select_warm_start_candidate = action_heads.select_warm_start_candidate
_original_validate_config = base_runner.validate_config


def _select_warm_start_candidate_k2(states, actual_iter, source):
    """Add the direct K/2 reuse-point condition without changing old modes."""
    if source != WARM_START_SOURCE_K2:
        return _original_select_warm_start_candidate(states, actual_iter, source)

    if not states:
        return None, None, None

    # Existing midpoint condition:
    #   source_index = max(0, actual_iter // 2 - 1)
    # Requested direct K/2 comparison:
    #   source_index = actual_iter // 2
    source_index = max(0, actual_iter // 2)
    source_index = min(source_index, actual_iter - 1, len(states) - 1)
    return states[source_index], source_index, source_index + 1


def _validate_config_with_midpoint_k2(cfg):
    """Reuse the base validation contract while allowing midpoint_k2."""
    if cfg.warm_start_source != WARM_START_SOURCE_K2:
        return _original_validate_config(cfg)

    requested_source = cfg.warm_start_source
    cfg.warm_start_source = "midpoint"
    try:
        return _original_validate_config(cfg)
    finally:
        cfg.warm_start_source = requested_source


action_heads.select_warm_start_candidate = _select_warm_start_candidate_k2
base_runner.validate_config = _validate_config_with_midpoint_k2


if __name__ == "__main__":
    base_runner.eval_libero()
