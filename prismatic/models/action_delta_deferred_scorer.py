"""Development-only scorer backends for deferred/backfill diagnostics."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from prismatic.models.action_delta_gate import (
    ACTION_DELTA_NONCONVERGENCE_THRESHOLD,
    NonFiniteActionDeltaGateError,
    PreparedActionDeltaGate,
    evaluate_action_delta_gate,
)


ACTION_DELTA_DEFERRED_SCORER_BACKENDS = ("eager", "compile_default")
ACTION_DELTA_DEFERRED_COMPILED_DECISION_PARITY_COUNT = 7139
ACTION_DELTA_DEFERRED_COMPILED_DECISION_PARITY_TOTAL = 7139
ACTION_DELTA_DEFERRED_COMPILED_NUMERICAL_EQUIVALENCE = (
    "not_bitwise_equal_to_eager; development high-side decisions previously "
    "verified 7139/7139 at q=0.0015"
)


@dataclass(frozen=True)
class PreparedActionDeltaDeferredScorer:
    backend: str
    tensor_scorer: Any
    compile_setup_ms: float
    compile_fullgraph: bool
    compile_dynamic: bool
    numerical_equivalence: str
    development_decision_parity_count: int
    development_decision_parity_total: int


def action_delta_deferred_tensor_score(
    anchor_state: torch.Tensor,
    current_state: torch.Tensor,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor,
    y_std: torch.Tensor,
    y_mean: torch.Tensor,
) -> torch.Tensor:
    """Pure tensor scorer with the frozen runtime arithmetic and quantization order."""

    delta = current_state.float() - anchor_state.float()
    delta = delta.to(torch.bfloat16).float()
    x = (delta - x_mean) / x_std
    pred_norm = F.linear(x, linear_weight, linear_bias)
    pred_delta = pred_norm * y_std + y_mean
    return pred_delta.square().mean()


def _tensor_arguments(gate: PreparedActionDeltaGate) -> tuple[torch.Tensor, ...]:
    return (
        gate.x_mean,
        gate.x_std,
        gate.linear_weight,
        gate.linear_bias,
        gate.y_std,
        gate.y_mean,
    )


def prepare_action_delta_deferred_scorer(
    gate: PreparedActionDeltaGate,
    backend: str,
) -> PreparedActionDeltaDeferredScorer | None:
    """Prepare once during task setup; eager deliberately needs no wrapper."""

    if backend not in ACTION_DELTA_DEFERRED_SCORER_BACKENDS:
        raise ValueError(
            "deferred/backfill scorer backend must be one of "
            f"{ACTION_DELTA_DEFERRED_SCORER_BACKENDS}"
        )
    if not isinstance(gate, PreparedActionDeltaGate):
        raise TypeError("deferred/backfill scorer requires a prepared Action-Delta gate")
    if backend == "eager":
        return None

    setup_start = time.perf_counter()
    compiled = torch.compile(
        action_delta_deferred_tensor_score,
        fullgraph=True,
        dynamic=False,
    )
    example_shape = (1, gate.action_chunk_len, gate.hidden_dim)
    anchor = torch.zeros(
        example_shape, device=gate.x_mean.device, dtype=torch.bfloat16
    )
    current = torch.zeros_like(anchor)
    with torch.inference_mode():
        # Force lazy TorchInductor compilation here. The output is consumed
        # immediately so a CUDA-Graph-backed scalar is never retained.
        compiled(anchor, current, *_tensor_arguments(gate)).item()
    if gate.x_mean.device.type == "cuda":
        torch.cuda.synchronize(gate.x_mean.device)
    setup_ms = (time.perf_counter() - setup_start) * 1000.0
    return PreparedActionDeltaDeferredScorer(
        backend="compile_default",
        tensor_scorer=compiled,
        compile_setup_ms=float(setup_ms),
        compile_fullgraph=True,
        compile_dynamic=False,
        numerical_equivalence=(
            ACTION_DELTA_DEFERRED_COMPILED_NUMERICAL_EQUIVALENCE
        ),
        development_decision_parity_count=(
            ACTION_DELTA_DEFERRED_COMPILED_DECISION_PARITY_COUNT
        ),
        development_decision_parity_total=(
            ACTION_DELTA_DEFERRED_COMPILED_DECISION_PARITY_TOTAL
        ),
    )


def validate_action_delta_deferred_scorer_configuration(
    *,
    deferred_filter_enabled: bool,
    backend: str,
    prepared_scorer: Any,
) -> None:
    if backend not in ACTION_DELTA_DEFERRED_SCORER_BACKENDS:
        raise ValueError(
            "deferred/backfill scorer backend must be one of "
            f"{ACTION_DELTA_DEFERRED_SCORER_BACKENDS}"
        )
    if not deferred_filter_enabled:
        if backend != "eager":
            raise ValueError(
                "compile_default scorer backend is deferred/backfill-only"
            )
        return
    if backend == "eager":
        if prepared_scorer is not None:
            raise ValueError("eager deferred scorer must not bind a compiled scorer")
        return
    if not isinstance(prepared_scorer, PreparedActionDeltaDeferredScorer):
        raise ValueError("compile_default requires a prepared compiled scorer")
    if prepared_scorer.backend != backend:
        raise ValueError("prepared deferred scorer backend mismatch")


def evaluate_action_delta_deferred_scorer(
    gate: PreparedActionDeltaGate,
    anchor_state: torch.Tensor,
    current_state: torch.Tensor,
    *,
    backend: str,
    prepared_scorer: PreparedActionDeltaDeferredScorer | None,
) -> tuple[float, bool]:
    """Return score and the fixed high-side non-convergence decision."""

    if backend == "eager":
        score, _unused_low_side_decision = evaluate_action_delta_gate(
            gate, anchor_state, current_state
        )
    else:
        validate_action_delta_deferred_scorer_configuration(
            deferred_filter_enabled=True,
            backend=backend,
            prepared_scorer=prepared_scorer,
        )
        expected_shape = (1, gate.action_chunk_len, gate.hidden_dim)
        for name, state in (
            ("anchor_state", anchor_state),
            ("current_state", current_state),
        ):
            if not torch.is_tensor(state) or tuple(state.shape) != expected_shape:
                raise ValueError(
                    f"compiled deferred scorer {name} shape mismatch: "
                    f"expected={expected_shape}, actual={getattr(state, 'shape', None)}"
                )
            if state.device != gate.x_mean.device:
                raise ValueError(
                    f"compiled deferred scorer {name} device mismatch"
                )
        score_tensor = prepared_scorer.tensor_scorer(
            anchor_state,
            current_state,
            *_tensor_arguments(gate),
        )
        score = float(score_tensor.item())
        if not math.isfinite(score):
            raise NonFiniteActionDeltaGateError(
                "Action-Delta deferred scorer state or score is non-finite"
            )
    return score, bool(score >= ACTION_DELTA_NONCONVERGENCE_THRESHOLD)
