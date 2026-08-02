"""Scalar convergence metrics for consecutive recurrent latent states."""

from __future__ import annotations

import math
from numbers import Real
from typing import Dict

import torch
import torch.nn.functional as F


LATENT_METRIC_NAMES = (
    "raw_mse",
    "relative_mse",
    "cosine_distance",
    "relative_l2",
)


class NonFiniteLatentMetricError(RuntimeError):
    """Raised when a latent comparison cannot produce finite scalar metrics."""


def validate_latent_metric_eps(eps: float) -> float:
    if isinstance(eps, bool) or not isinstance(eps, Real):
        raise ValueError("latent_only_eps must be a finite positive number")
    value = float(eps)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("latent_only_eps must be a finite positive number")
    return value


def compute_latent_metrics(
    current_state: torch.Tensor,
    previous_state: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Compute all supported metrics in FP32 and return JSON-safe scalars."""
    eps = validate_latent_metric_eps(eps)
    if not torch.is_tensor(current_state) or not torch.is_tensor(previous_state):
        raise TypeError("latent metrics require tensor states")
    if current_state.shape != previous_state.shape:
        raise ValueError(
            "latent metric states must have identical shapes: "
            f"current={tuple(current_state.shape)}, previous={tuple(previous_state.shape)}"
        )

    current = current_state.float().reshape(-1)
    previous = previous_state.float().reshape(-1)
    if current.numel() == 0:
        raise ValueError("latent metric states must be non-empty")
    if not bool(torch.isfinite(current).all().item()) or not bool(
        torch.isfinite(previous).all().item()
    ):
        raise NonFiniteLatentMetricError("latent metric input is non-finite")

    difference = current - previous
    raw_mse = torch.mean(difference.square())
    relative_mse = raw_mse / (torch.mean(previous.square()) + eps)
    cosine_distance = 1.0 - F.cosine_similarity(
        current, previous, dim=0, eps=eps
    )
    relative_l2 = torch.linalg.vector_norm(difference) / (
        torch.linalg.vector_norm(previous) + eps
    )
    tensors = {
        "raw_mse": raw_mse,
        "relative_mse": relative_mse,
        "cosine_distance": cosine_distance,
        "relative_l2": relative_l2,
    }
    values = {name: float(value.item()) for name, value in tensors.items()}
    if not all(math.isfinite(value) for value in values.values()):
        raise NonFiniteLatentMetricError("latent metric result is non-finite")
    return values
