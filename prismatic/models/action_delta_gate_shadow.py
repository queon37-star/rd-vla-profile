"""Diagnostic-only deployment shadow capture for Action-Delta Gate.

This module intentionally contains no stopping policy.  It turns the exact
pre-Coda predictor result and the subsequently computed exact Coda output into
a detached CPU record.  The caller remains responsible for running the normal
adjacent-action-MSE recurrence without consulting any value returned here.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from prismatic.models.action_delta_gate import PreparedActionDeltaGate


ACTION_DELTA_GATE_SHADOW_SCHEMA_VERSION = 1
ACTION_DELTA_GATE_SHADOW_DEVELOPMENT_TASK_IDS = (0, 1, 2, 3, 6, 7, 8, 9)
ACTION_DELTA_GATE_SHADOW_PREFIX_STEPS = 5

ACTION_DELTA_GATE_SHADOW_FEATURE_NAMES = (
    "predicted_action_delta_mse",
    "threshold_margin",
    "normalized_margin",
    "predicted_prefix5_mse",
    "predicted_full_max_abs",
    "predicted_prefix5_max_abs",
    "predicted_max_per_step_mse",
    "predicted_per_step_mse_std",
    "predicted_per_step_mse_cv",
    "predicted_max_per_dim_mse",
    "predicted_per_dim_mse_std",
    "predicted_per_dim_mse_cv",
    "predicted_max_step_to_full_mse_ratio",
    "predicted_max_dim_to_full_mse_ratio",
    "latent_delta_full_rms",
    "latent_delta_token_rms_mean",
    "latent_delta_token_rms_max",
    "latent_delta_token_rms_std",
    "latent_delta_token_rms_cv",
    "latent_delta_max_abs",
    "normalized_x_l2",
    "normalized_x_rms",
    "normalized_x_max_abs",
    "normalized_x_token_norm_mean",
    "normalized_x_token_norm_max",
    "normalized_x_token_norm_std",
    "terminal_iteration",
    "previous_predicted_score",
    "score_ratio_current_to_previous",
    "score_difference_current_minus_previous",
    "relative_score_drop",
    "previous_latent_delta_rms",
    "latent_delta_rms_ratio_current_to_previous",
    "latent_delta_cosine_current_previous",
    "latent_delta_second_difference_rms",
)


class ActionDeltaGateShadowError(ValueError):
    """Raised when a diagnostic shadow row violates its frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ActionDeltaGateShadowError(message)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else None
    return numerator / denominator


def validate_action_delta_gate_shadow_configuration(
    *,
    enabled: bool,
    production_gate_enabled: bool,
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
) -> None:
    """Fail closed before diagnostic collection can enter inference."""

    if not enabled:
        return
    _require(not production_gate_enabled, "shadow collection cannot enable the production gate")
    _require(
        isinstance(prepared_gate, PreparedActionDeltaGate),
        "shadow collection requires a prepared frozen Action-Delta Gate",
    )
    _require(batch_size == 1, "shadow collection requires batch size 1")
    _require(
        canonical_recurrence_strategy == "adjacent_action_mse",
        "shadow collection requires adjacent action-MSE recurrence",
    )
    _require(use_warm_start, "shadow collection requires warm-start inference")
    _require(warm_start_source == "midpoint", "shadow collection requires midpoint warm-start")
    _require(warm_start_min_iter == 2, "shadow collection requires warm_start_min_iter=2")
    _require(not use_latent_precheck, "shadow collection cannot use latent pre-check")
    _require(latent_precheck_mode == "off", "shadow collection requires latent_precheck_mode='off'")
    _require(
        latent_precheck_trace_level == "off",
        "shadow collection requires latent_precheck_trace_level='off'",
    )
    _require(not shadow_full_depth, "shadow collection cannot enable full-depth shadow recurrence")
    _require(
        not collect_preconvergence_raw_shadow,
        "shadow collection cannot enable raw post-production shadow recurrence",
    )
    _require(use_cached_final_output, "shadow collection requires exact terminal-output reuse")
    _require(
        isinstance(min_terminal_iter, int)
        and not isinstance(min_terminal_iter, bool)
        and min_terminal_iter >= 2,
        "shadow minimum terminal iteration must be an integer >= 2",
    )


def _validate_transition_tensors(
    gate: PreparedActionDeltaGate,
    anchor_state: torch.Tensor,
    current_state: torch.Tensor,
    anchor_output: torch.Tensor,
    current_output: torch.Tensor,
    predicted_delta: torch.Tensor,
) -> None:
    expected_state = (1, gate.action_chunk_len, gate.hidden_dim)
    expected_action = (1, gate.action_chunk_len, gate.action_dim)
    for name, tensor, shape in (
        ("anchor_state", anchor_state, expected_state),
        ("current_state", current_state, expected_state),
        ("anchor_output", anchor_output, expected_action),
        ("current_output", current_output, expected_action),
        ("predicted_delta", predicted_delta, expected_action),
    ):
        _require(torch.is_tensor(tensor), f"{name} must be a tensor")
        _require(tuple(tensor.shape) == shape, f"{name} shape mismatch: expected={shape}, actual={tuple(tensor.shape)}")
        _require(tensor.device == anchor_state.device, f"{name} must share the anchor-state device")
        _require(bool(torch.isfinite(tensor.float()).all().item()), f"{name} is non-finite")


def _predicted_and_latent_features(
    gate: PreparedActionDeltaGate,
    latent_delta_bfloat16: torch.Tensor,
    predicted_delta: torch.Tensor,
    *,
    score: float,
    terminal_iteration: int,
    previous_transition: Mapping[str, Any] | None,
    previous_latent_delta_bfloat16: torch.Tensor | None,
) -> dict[str, float | None]:
    pred = predicted_delta.float()
    pred_squared = pred.square()
    step_mse = pred_squared.mean(dim=2).squeeze(0)
    dim_mse = pred_squared.mean(dim=1).squeeze(0)
    prefix_steps = min(ACTION_DELTA_GATE_SHADOW_PREFIX_STEPS, gate.action_chunk_len)
    step_mean = float(step_mse.mean().item())
    dim_mean = float(dim_mse.mean().item())

    delta = latent_delta_bfloat16.float()
    token_rms = delta.square().mean(dim=2).sqrt().squeeze(0)
    token_rms_mean = float(token_rms.mean().item())
    token_rms_std = float(token_rms.std(unbiased=False).item())
    x = (delta - gate.x_mean) / gate.x_std
    x_token_norm = torch.linalg.vector_norm(x, dim=2).squeeze(0)
    latent_rms = float(delta.square().mean().sqrt().item())

    features: dict[str, float | None] = {
        "predicted_action_delta_mse": float(score),
        "threshold_margin": float(gate.threshold - score),
        "normalized_margin": float((gate.threshold - score) / gate.threshold),
        "predicted_prefix5_mse": float(pred_squared[:, :prefix_steps].mean().item()),
        "predicted_full_max_abs": float(pred.abs().amax().item()),
        "predicted_prefix5_max_abs": float(pred[:, :prefix_steps].abs().amax().item()),
        "predicted_max_per_step_mse": float(step_mse.amax().item()),
        "predicted_per_step_mse_std": float(step_mse.std(unbiased=False).item()),
        "predicted_per_step_mse_cv": _safe_ratio(float(step_mse.std(unbiased=False).item()), step_mean),
        "predicted_max_per_dim_mse": float(dim_mse.amax().item()),
        "predicted_per_dim_mse_std": float(dim_mse.std(unbiased=False).item()),
        "predicted_per_dim_mse_cv": _safe_ratio(float(dim_mse.std(unbiased=False).item()), dim_mean),
        "predicted_max_step_to_full_mse_ratio": _safe_ratio(float(step_mse.amax().item()), score),
        "predicted_max_dim_to_full_mse_ratio": _safe_ratio(float(dim_mse.amax().item()), score),
        "latent_delta_full_rms": latent_rms,
        "latent_delta_token_rms_mean": token_rms_mean,
        "latent_delta_token_rms_max": float(token_rms.amax().item()),
        "latent_delta_token_rms_std": token_rms_std,
        "latent_delta_token_rms_cv": _safe_ratio(token_rms_std, token_rms_mean),
        "latent_delta_max_abs": float(delta.abs().amax().item()),
        "normalized_x_l2": float(torch.linalg.vector_norm(x.flatten()).item()),
        "normalized_x_rms": float(x.square().mean().sqrt().item()),
        "normalized_x_max_abs": float(x.abs().amax().item()),
        "normalized_x_token_norm_mean": float(x_token_norm.mean().item()),
        "normalized_x_token_norm_max": float(x_token_norm.amax().item()),
        "normalized_x_token_norm_std": float(x_token_norm.std(unbiased=False).item()),
        "terminal_iteration": float(terminal_iteration),
        "previous_predicted_score": None,
        "score_ratio_current_to_previous": None,
        "score_difference_current_minus_previous": None,
        "relative_score_drop": None,
        "previous_latent_delta_rms": None,
        "latent_delta_rms_ratio_current_to_previous": None,
        "latent_delta_cosine_current_previous": None,
        "latent_delta_second_difference_rms": None,
    }

    if previous_transition is not None:
        previous_terminal = int(previous_transition["terminal_iteration"])
        if terminal_iteration == previous_terminal + 1:
            previous_score = float(previous_transition["gate_score"])
            features.update(
                {
                    "previous_predicted_score": previous_score,
                    "score_ratio_current_to_previous": _safe_ratio(score, previous_score),
                    "score_difference_current_minus_previous": score - previous_score,
                    "relative_score_drop": _safe_ratio(previous_score - score, previous_score),
                }
            )

    if previous_latent_delta_bfloat16 is not None:
        _require(
            tuple(previous_latent_delta_bfloat16.shape) == tuple(delta.shape),
            "previous latent delta shape mismatch",
        )
        _require(
            previous_latent_delta_bfloat16.dtype == torch.bfloat16,
            "previous latent delta must be BF16",
        )
        previous_delta = previous_latent_delta_bfloat16.to(
            device=delta.device, dtype=torch.float32
        )
        previous_rms = float(previous_delta.square().mean().sqrt().item())
        current_flat = delta.flatten()
        previous_flat = previous_delta.flatten()
        denominator = float(
            (torch.linalg.vector_norm(current_flat) * torch.linalg.vector_norm(previous_flat)).item()
        )
        cosine = (
            float(torch.dot(current_flat, previous_flat).item()) / denominator
            if denominator != 0.0
            else None
        )
        features.update(
            {
                "previous_latent_delta_rms": previous_rms,
                "latent_delta_rms_ratio_current_to_previous": _safe_ratio(
                    latent_rms, previous_rms
                ),
                "latent_delta_cosine_current_previous": cosine,
                "latent_delta_second_difference_rms": float(
                    (delta - previous_delta).square().mean().sqrt().item()
                ),
            }
        )

    _require(set(features) == set(ACTION_DELTA_GATE_SHADOW_FEATURE_NAMES), "shadow feature schema mismatch")
    for name, value in features.items():
        _require(value is None or math.isfinite(value), f"shadow feature {name} is non-finite")
    return features


def build_action_delta_gate_shadow_transition(
    gate: PreparedActionDeltaGate,
    *,
    anchor_iteration: int,
    terminal_iteration: int,
    anchor_state: torch.Tensor,
    current_state: torch.Tensor,
    anchor_output: torch.Tensor,
    current_output: torch.Tensor,
    predicted_delta: torch.Tensor,
    score: float,
    exact_adjacent_action_mse: float,
    recurrence_mse_threshold: float,
    previous_transition: Mapping[str, Any] | None = None,
    previous_latent_delta_bfloat16: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Build one pre-Coda predictor/exact-Coda label row on CPU."""

    _require(isinstance(gate, PreparedActionDeltaGate), "a prepared gate is required")
    _require(anchor_iteration >= 1, "anchor iteration must be >= 1")
    _require(terminal_iteration == anchor_iteration + 1, "shadow transition must be adjacent")
    _require(math.isfinite(score), "gate score must be finite")
    _require(math.isfinite(exact_adjacent_action_mse), "exact adjacent action MSE must be finite")
    _require(recurrence_mse_threshold > 0.0, "recurrence MSE threshold must be positive")
    _validate_transition_tensors(
        gate,
        anchor_state,
        current_state,
        anchor_output,
        current_output,
        predicted_delta,
    )

    # This is deliberately identical to the frozen runtime preprocessing.
    latent_delta_bfloat16 = (
        current_state.float() - anchor_state.float()
    ).to(torch.bfloat16)
    score_from_delta = float(predicted_delta.float().square().mean().item())
    _require(score_from_delta == score, "exposed predicted delta does not reproduce the gate score")
    exact_from_outputs = float(
        (current_output - anchor_output).square().mean().item()
    )
    _require(
        exact_from_outputs == exact_adjacent_action_mse,
        "exact output pair does not reproduce native adjacent action MSE",
    )
    features = _predicted_and_latent_features(
        gate,
        latent_delta_bfloat16,
        predicted_delta,
        score=score,
        terminal_iteration=terminal_iteration,
        previous_transition=previous_transition,
        previous_latent_delta_bfloat16=previous_latent_delta_bfloat16,
    )
    predicted_trigger = bool(score <= gate.threshold)
    exact_safe = bool(exact_adjacent_action_mse < recurrence_mse_threshold)
    tensors = {
        "anchor_state": anchor_state.detach().to(device="cpu", copy=True),
        "current_state": current_state.detach().to(device="cpu", copy=True),
        "latent_delta_bfloat16": latent_delta_bfloat16.detach().to(device="cpu", copy=True),
        "anchor_action": anchor_output.detach().to(device="cpu", copy=True),
        "exact_terminal_action": current_output.detach().to(device="cpu", copy=True),
        "predicted_delta_action": predicted_delta.detach().float().to(device="cpu", copy=True),
        "previous_latent_delta_bfloat16": (
            previous_latent_delta_bfloat16.detach().to(device="cpu", copy=True)
            if previous_latent_delta_bfloat16 is not None
            else torch.empty(0, dtype=torch.bfloat16)
        ),
    }
    return {
        "schema_version": ACTION_DELTA_GATE_SHADOW_SCHEMA_VERSION,
        "anchor_iteration": int(anchor_iteration),
        "terminal_iteration": int(terminal_iteration),
        "gate_score": float(score),
        "gate_threshold": float(gate.threshold),
        "predicted_trigger": predicted_trigger,
        "exact_adjacent_action_mse": float(exact_adjacent_action_mse),
        "recurrence_mse_threshold": float(recurrence_mse_threshold),
        "exact_safe": exact_safe,
        "false_safe": bool(predicted_trigger and not exact_safe),
        "residual": float(exact_adjacent_action_mse - score),
        "predicted_full_mse": float(score),
        "predicted_prefix_mse": features["predicted_prefix5_mse"],
        "predicted_max_per_step_mse": features["predicted_max_per_step_mse"],
        "predicted_max_per_dimension_mse": features["predicted_max_per_dim_mse"],
        "features": features,
        "tensors": tensors,
    }
