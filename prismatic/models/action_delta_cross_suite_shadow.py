"""Diagnostic-only zero-shot cross-suite Action-Delta shadow capture.

This module deliberately contains labels and validation only. It has no
authority to skip Coda or stop recurrent inference.
"""

from __future__ import annotations

import math
from typing import Any

from prismatic.models.action_delta_gate import (
    ACTION_DELTA_NONCONVERGENCE_THRESHOLD,
    PreparedActionDeltaGate,
)


ACTION_DELTA_CROSS_SUITE_SHADOW_SCHEMA_VERSION = 1
ACTION_DELTA_CROSS_SUITE_ANALYSIS_TYPE = (
    "cross_suite_zero_shot_action_delta_transfer"
)
ACTION_DELTA_CROSS_SUITE_TRAINING_SUITE = "libero_spatial"
ACTION_DELTA_CROSS_SUITE_SUPPORTED_SUITES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
)


class ActionDeltaCrossSuiteShadowError(ValueError):
    """Raised when cross-suite shadow instrumentation violates its contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ActionDeltaCrossSuiteShadowError(message)


def validate_action_delta_cross_suite_shadow_configuration(
    *,
    enabled: bool,
    production_gate_enabled: bool,
    gate_shadow_enabled: bool,
    nonconvergence_filter_enabled: bool,
    deferred_backfill_filter_enabled: bool,
    canonical_recurrence_strategy: str | None,
    prepared_gate: Any,
    batch_size: int,
    use_warm_start: bool,
    warm_start_source: str,
    warm_start_min_iter: int,
    use_latent_precheck: bool,
    latent_precheck_mode: str,
    latent_precheck_trace_level: str,
    shadow_full_depth: bool,
    collect_preconvergence_raw_shadow: bool,
    use_cached_final_output: bool,
    min_terminal_iter: int,
    recurrence_mse_threshold: float,
) -> None:
    """Fail closed before cross-suite diagnostic instrumentation runs."""

    if not enabled:
        return
    _require(not production_gate_enabled, "cross-suite shadow cannot enable the production gate")
    _require(not gate_shadow_enabled, "cross-suite shadow cannot enable the Spatial calibration shadow")
    _require(not nonconvergence_filter_enabled, "cross-suite shadow cannot enable the max-skip filter")
    _require(not deferred_backfill_filter_enabled, "cross-suite shadow cannot enable deferred/backfill")
    _require(
        isinstance(prepared_gate, PreparedActionDeltaGate),
        "cross-suite shadow requires a prepared frozen Action-Delta predictor",
    )
    _require(batch_size == 1, "cross-suite shadow requires batch size 1")
    _require(
        canonical_recurrence_strategy == "adjacent_action_mse",
        "cross-suite shadow requires adjacent action-MSE recurrence",
    )
    _require(use_warm_start, "cross-suite shadow requires warm-start inference")
    _require(warm_start_source == "midpoint", "cross-suite shadow requires midpoint warm-start")
    _require(warm_start_min_iter == 2, "cross-suite shadow requires warm_start_min_iter=2")
    _require(not use_latent_precheck, "cross-suite shadow cannot use latent pre-check")
    _require(latent_precheck_mode == "off", "cross-suite shadow requires latent_precheck_mode='off'")
    _require(
        latent_precheck_trace_level == "off",
        "cross-suite shadow requires latent_precheck_trace_level='off'",
    )
    _require(not shadow_full_depth, "cross-suite shadow cannot enable full-depth shadow recurrence")
    _require(
        not collect_preconvergence_raw_shadow,
        "cross-suite shadow cannot collect raw post-production recurrence",
    )
    _require(use_cached_final_output, "cross-suite shadow requires exact terminal-output reuse")
    _require(
        isinstance(min_terminal_iter, int)
        and not isinstance(min_terminal_iter, bool)
        and min_terminal_iter >= 2,
        "cross-suite shadow minimum terminal iteration must be an integer >= 2",
    )
    _require(
        float(recurrence_mse_threshold) == 0.001,
        "cross-suite shadow requires recurrence_mse_threshold=0.001",
    )


def build_action_delta_cross_suite_transition(
    *,
    anchor_iteration: int,
    terminal_iteration: int,
    score: float | None,
    exact_adjacent_action_mse: float,
    scoring_error: str | None = None,
) -> dict[str, Any]:
    """Build one compact high-side diagnostic transition record."""

    _require(anchor_iteration >= 1, "anchor iteration must be >= 1")
    _require(
        terminal_iteration == anchor_iteration + 1,
        "cross-suite shadow transition must be adjacent",
    )
    _require(
        math.isfinite(exact_adjacent_action_mse),
        "exact adjacent action MSE must be finite",
    )
    finite_score = score is not None and math.isfinite(score)
    _require(
        finite_score or scoring_error is not None,
        "a non-finite cross-suite score requires a diagnostic error",
    )
    high = (
        bool(score >= ACTION_DELTA_NONCONVERGENCE_THRESHOLD)
        if finite_score
        else None
    )
    exact_safe = bool(exact_adjacent_action_mse < 0.001)
    return {
        "anchor_iteration": int(anchor_iteration),
        "terminal_iteration": int(terminal_iteration),
        "score": float(score) if finite_score else None,
        "finite_score": bool(finite_score),
        "scoring_error": scoring_error,
        "exact_adjacent_action_mse": float(exact_adjacent_action_mse),
        "high_predicted_nonconvergence": high,
        "exact_safe": exact_safe,
        "high_exact_safe_violation": (
            bool(high and exact_safe) if finite_score else None
        ),
    }
