"""FP32 scalar diagnostics for recurrent latent-state dynamics."""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from prismatic.models.latent_metrics import validate_latent_metric_eps


LATENT_DYNAMICS_FIELDS = (
    "update_rms",
    "contraction_ratio",
    "update_turning_cosine",
    "acceleration_rms",
    "acceleration_ratio",
    "token_update_p50",
    "token_update_p90",
    "token_update_p95",
    "token_update_max",
    "token_update_cv",
    "token_update_energy_entropy",
    "token_update_top10_fraction",
    "state_rms",
    "state_norm_ratio",
    "warm_anchor_relative_l2",
    "warm_anchor_cosine_distance",
)

HISTORY_DEPENDENT_FIELDS = (
    "contraction_ratio",
    "update_turning_cosine",
    "acceleration_rms",
    "acceleration_ratio",
)

WARM_ANCHOR_FIELDS = (
    "warm_anchor_relative_l2",
    "warm_anchor_cosine_distance",
)


class NonFiniteLatentDynamicsError(RuntimeError):
    """Raised when a diagnostic input or scalar result is non-finite."""


def _require_tensor(name: str, value: torch.Tensor, shape) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a tensor")
    if tuple(value.shape) != tuple(shape):
        raise ValueError(
            f"{name} shape must match latent state: "
            f"expected={tuple(shape)}, actual={tuple(value.shape)}"
        )
    result = value.float()
    if not bool(torch.isfinite(result).all().item()):
        raise NonFiniteLatentDynamicsError(f"{name} is non-finite")
    return result


def _json_scalars(values: Dict[str, Optional[torch.Tensor]]) -> Dict[str, Optional[float]]:
    result = {
        name: None if value is None else float(value.item())
        for name, value in values.items()
    }
    if not all(value is None or math.isfinite(value) for value in result.values()):
        raise NonFiniteLatentDynamicsError("latent dynamics result is non-finite")
    return result


def compute_latent_dynamics(
    current_state: torch.Tensor,
    previous_state: torch.Tensor,
    *,
    previous_update: Optional[torch.Tensor] = None,
    warm_anchor: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> Dict[str, Optional[float]]:
    """Compute JSON-safe dynamics diagnostics without modifying input tensors."""

    eps = validate_latent_metric_eps(eps)
    if not torch.is_tensor(current_state) or not torch.is_tensor(previous_state):
        raise TypeError("latent dynamics require tensor states")
    if current_state.shape != previous_state.shape:
        raise ValueError(
            "latent dynamics states must have identical shapes: "
            f"current={tuple(current_state.shape)}, previous={tuple(previous_state.shape)}"
        )
    if current_state.ndim < 2 or current_state.shape[-1] == 0:
        raise ValueError("latent dynamics require token and non-empty feature dimensions")
    if current_state.numel() == 0:
        raise ValueError("latent dynamics states must be non-empty")

    current = _require_tensor("current_state", current_state, current_state.shape)
    previous = _require_tensor("previous_state", previous_state, current_state.shape)
    update = current - previous
    update_rms = torch.mean(update.square()).sqrt()

    token_update_rms = torch.mean(update.square(), dim=-1).sqrt().reshape(-1)
    token_count = int(token_update_rms.numel())
    token_mean = torch.mean(token_update_rms)
    token_cv = torch.std(token_update_rms, unbiased=False) / (token_mean + eps)
    token_energy = token_update_rms.square()
    total_energy = torch.sum(token_energy)
    probabilities = token_energy / (total_energy + eps)
    entropy_terms = torch.where(
        probabilities > 0,
        probabilities * torch.log(torch.clamp_min(probabilities, eps)),
        torch.zeros_like(probabilities),
    )
    if token_count > 1:
        normalized_entropy = -torch.sum(entropy_terms) / math.log(token_count)
    else:
        normalized_entropy = torch.zeros((), device=current.device, dtype=torch.float32)
    top_count = max(1, int(math.ceil(0.10 * token_count)))
    top_fraction = torch.topk(token_energy, top_count).values.sum() / (
        total_energy + eps
    )

    previous_state_rms = torch.mean(previous.square()).sqrt()
    state_rms = torch.mean(current.square()).sqrt()
    state_norm_ratio = state_rms / (previous_state_rms + eps)

    previous_update_fp32 = None
    if previous_update is not None:
        previous_update_fp32 = _require_tensor(
            "previous_update", previous_update, current_state.shape
        )
        previous_update_rms = torch.mean(previous_update_fp32.square()).sqrt()
        acceleration = update - previous_update_fp32
        contraction_ratio = update_rms / (previous_update_rms + eps)
        update_turning_cosine = F.cosine_similarity(
            update.reshape(-1),
            previous_update_fp32.reshape(-1),
            dim=0,
            eps=eps,
        )
        acceleration_rms = torch.mean(acceleration.square()).sqrt()
        acceleration_ratio = acceleration_rms / (previous_update_rms + eps)
    else:
        contraction_ratio = None
        update_turning_cosine = None
        acceleration_rms = None
        acceleration_ratio = None

    if warm_anchor is not None:
        anchor = _require_tensor("warm_anchor", warm_anchor, current_state.shape)
        anchor_difference = current - anchor
        warm_anchor_relative_l2 = torch.linalg.vector_norm(
            anchor_difference.reshape(-1)
        ) / (torch.linalg.vector_norm(anchor.reshape(-1)) + eps)
        warm_anchor_cosine_distance = 1.0 - F.cosine_similarity(
            current.reshape(-1), anchor.reshape(-1), dim=0, eps=eps
        )
    else:
        warm_anchor_relative_l2 = None
        warm_anchor_cosine_distance = None

    values = _json_scalars(
        {
            "update_rms": update_rms,
            "contraction_ratio": contraction_ratio,
            "update_turning_cosine": update_turning_cosine,
            "acceleration_rms": acceleration_rms,
            "acceleration_ratio": acceleration_ratio,
            "token_update_p50": torch.quantile(token_update_rms, 0.50),
            "token_update_p90": torch.quantile(token_update_rms, 0.90),
            "token_update_p95": torch.quantile(token_update_rms, 0.95),
            "token_update_max": torch.max(token_update_rms),
            "token_update_cv": token_cv,
            "token_update_energy_entropy": torch.clamp(
                normalized_entropy, min=0.0, max=1.0
            ),
            "token_update_top10_fraction": torch.clamp(
                top_fraction, min=0.0, max=1.0
            ),
            "state_rms": state_rms,
            "state_norm_ratio": state_norm_ratio,
            "warm_anchor_relative_l2": warm_anchor_relative_l2,
            "warm_anchor_cosine_distance": warm_anchor_cosine_distance,
        }
    )
    if tuple(values) != LATENT_DYNAMICS_FIELDS:
        raise RuntimeError("latent dynamics field order mismatch")
    return values
