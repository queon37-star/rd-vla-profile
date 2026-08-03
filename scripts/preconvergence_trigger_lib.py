"""Offline-only feasibility helpers for a one-step-ahead Coda trigger.

The module intentionally has no dependency on the runtime action-head path.
It consumes optional post-production tensor shards plus the already frozen
authoritative action-MSE dataset, trains task-OOF probes, and replays the
predeclared CONFIRM_NEXT scheduler.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import time
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


SCHEMA_VERSION = 1
ACTION_MSE_THRESHOLD = 0.001
DEFAULT_SEED = 7
RANK_CANDIDATES = (4, 8, 16)
MODEL_VARIANTS = ("no_auxiliary", "action_delta_auxiliary")
REPLAY_CONTRACT_VERSION = 2
FORCED_CODA_PREFIX_MAX_ITERATION = 3
MINIMUM_GATE_ITERATION = 3
MODEL_APPLICABLE_MIN_K_ACTION = 4
HISTORY_UNAVAILABLE_DEFINITION = "K_action - 1 < 3"
FORCED_CODA_ITERATIONS = tuple(range(1, FORCED_CODA_PREFIX_MAX_ITERATION + 1))


def replay_contract() -> dict[str, Any]:
    return {
        "replay_contract_version": REPLAY_CONTRACT_VERSION,
        "forced_coda_prefix_max_iteration": FORCED_CODA_PREFIX_MAX_ITERATION,
        "minimum_gate_iteration": MINIMUM_GATE_ITERATION,
        "model_applicable_min_k_action": MODEL_APPLICABLE_MIN_K_ACTION,
        "history_unavailable_definition": HISTORY_UNAVAILABLE_DEFINITION,
    }


class PreconvergenceValidationError(ValueError):
    """Raised when data or replay violates the frozen feasibility contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PreconvergenceValidationError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_metadata(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "layout": str(tensor.layout),
    }


@dataclass(frozen=True)
class SequenceIdentity:
    task_id: int
    episode_id: int
    prediction_id: int

    @property
    def key(self) -> tuple[int, int, int]:
        return (self.task_id, self.episode_id, self.prediction_id)


@dataclass(frozen=True)
class RawPreconvergenceSequence:
    identity: SequenceIdentity
    actual_origin: str
    states: torch.Tensor
    actions: torch.Tensor
    action_mse: tuple[float | None, ...]
    action_mse_phase: tuple[str | None, ...]
    baseline_k: int
    max_iter: int

    def validate(self) -> None:
        _require(self.actual_origin in {"ACTUAL_WARM", "COLD"}, "invalid origin")
        _require(torch.is_tensor(self.states), "states must be a tensor")
        _require(torch.is_tensor(self.actions), "actions must be a tensor")
        _require(self.states.ndim >= 3, "states must have [iteration, token, feature] axes")
        _require(self.actions.ndim >= 2, "actions must have [iteration, action] axes")
        _require(self.states.shape[0] == self.max_iter, "state depth/max_iter mismatch")
        _require(self.actions.shape[0] == self.max_iter, "action depth/max_iter mismatch")
        _require(self.states.shape[-1] > 0, "latent feature dimension is empty")
        _require(self.actions[0].numel() > 0, "action output is empty")
        _require(len(self.action_mse) == self.max_iter + 1, "action-MSE index contract mismatch")
        _require(len(self.action_mse_phase) == self.max_iter + 1, "phase index contract mismatch")
        _require(self.action_mse[0] is None and self.action_mse[1] is None, "k=0/1 MSE must be null")
        _require(1 <= self.baseline_k <= self.max_iter, "invalid baseline K")
        _require(bool(torch.isfinite(self.states.float()).all()), "non-finite latent state")
        _require(bool(torch.isfinite(self.actions.float()).all()), "non-finite action output")
        for k in range(2, self.max_iter + 1):
            value = self.action_mse[k]
            _require(value is not None and math.isfinite(float(value)), f"k={k}: invalid action MSE")
            _require(
                self.action_mse_phase[k] in {"production", "shadow_tail"},
                f"k={k}: invalid action-MSE phase",
            )

    @property
    def latent_feature_dim(self) -> int:
        return int(self.states.shape[-1])

    @property
    def action_dim(self) -> int:
        return int(self.actions[0].numel())

    @property
    def k_action(self) -> int:
        return derive_k_action(self.action_mse, max_iter=self.max_iter)


@dataclass(frozen=True)
class ExampleRow:
    identity: SequenceIdentity
    actual_origin: str
    k: int
    k_action: int
    label: int


@dataclass(frozen=True)
class TrainingBatch:
    inputs: torch.Tensor
    labels: torch.Tensor
    auxiliary_targets: torch.Tensor
    weights: torch.Tensor
    rows: tuple[ExampleRow, ...]


@dataclass(frozen=True)
class Normalizer:
    mean: torch.Tensor
    scale: torch.Tensor

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.scale


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = DEFAULT_SEED
    steps: int = 400
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    auxiliary_weight: float = 0.1


@dataclass(frozen=True)
class FittedTrigger:
    model: "LowRankPreconvergenceTrigger"
    normalizer: Normalizer
    rank: int
    use_auxiliary: bool
    seed: int
    parameter_count: int
    inference_flops: int


def derive_k_action(
    action_mse: Sequence[float | None],
    *,
    max_iter: int | None = None,
    threshold: float = ACTION_MSE_THRESHOLD,
) -> int:
    """Return the strict first hit of the authoritative adjacent-action MSE."""

    depth = (len(action_mse) - 1) if max_iter is None else int(max_iter)
    _require(depth >= 2, "action-MSE sequence is too short")
    for k in range(2, depth + 1):
        value = action_mse[k]
        _require(value is not None and math.isfinite(float(value)), f"k={k}: missing action MSE")
        if float(value) < float(threshold):
            return k
    raise PreconvergenceValidationError(
        "authoritative sequence has no action-MSE first hit before max_iter"
    )


def preconvergence_label(k: int, k_action: int) -> int:
    _require(k < k_action, "post-convergence rows are not classification examples")
    return int(k == k_action - 1)


def pooled_trigger_input(states: torch.Tensor, k: int) -> torch.Tensor:
    """Use only S_k, S_(k-1), and S_(k-2); return a device tensor."""

    _require(torch.is_tensor(states), "states must be a tensor")
    _require(states.ndim >= 3, "states require iteration, token, and feature axes")
    _require(3 <= k <= states.shape[0], f"invalid scorer iteration: {k}")
    current = states[k - 1].float()
    previous = states[k - 2].float()
    before_previous = states[k - 3].float()
    delta_k = current - previous
    delta_previous = previous - before_previous
    feature_dim = states.shape[-1]
    pooled_current = delta_k.reshape(-1, feature_dim).mean(dim=0)
    pooled_previous = delta_previous.reshape(-1, feature_dim).mean(dim=0)
    return torch.cat((pooled_current, pooled_previous), dim=0)


def current_action_delta(actions: torch.Tensor, k: int) -> torch.Tensor:
    _require(2 <= k <= actions.shape[0], f"invalid action iteration: {k}")
    return (actions[k - 1].float() - actions[k - 2].float()).reshape(-1)


def _prediction_weights(rows: Sequence[ExampleRow]) -> torch.Tensor:
    """Give every prediction unit mass, split equally across its two classes."""

    by_prediction: dict[tuple[int, int, int], list[int]] = {}
    for index, row in enumerate(rows):
        by_prediction.setdefault(row.identity.key, []).append(index)
    weights = torch.zeros(len(rows), dtype=torch.float32)
    for indices in by_prediction.values():
        positives = [index for index in indices if rows[index].label == 1]
        negatives = [index for index in indices if rows[index].label == 0]
        _require(len(positives) == 1, "each applicable prediction must have one positive")
        weights[positives] = 0.5 if negatives else 1.0
        if negatives:
            weights[negatives] = 0.5 / len(negatives)
    return weights


def build_training_batch(
    sequences: Sequence[RawPreconvergenceSequence],
    *,
    origin: str = "ACTUAL_WARM",
) -> TrainingBatch:
    inputs: list[torch.Tensor] = []
    auxiliary: list[torch.Tensor] = []
    rows: list[ExampleRow] = []
    for sequence in sequences:
        sequence.validate()
        if sequence.actual_origin != origin:
            continue
        k_action = sequence.k_action
        # k=2 lacks delta_prev. Predictions whose target is k=2 are reported as
        # uncovered instead of receiving an invented history value.
        if k_action - 1 < 3:
            continue
        for k in range(3, k_action):
            rows.append(
                ExampleRow(
                    identity=sequence.identity,
                    actual_origin=sequence.actual_origin,
                    k=k,
                    k_action=k_action,
                    label=preconvergence_label(k, k_action),
                )
            )
            inputs.append(pooled_trigger_input(sequence.states, k))
            auxiliary.append(current_action_delta(sequence.actions, k))
    _require(bool(rows), f"no applicable {origin} training rows")
    input_tensor = torch.stack(inputs)
    auxiliary_tensor = torch.stack(auxiliary)
    labels = torch.tensor([row.label for row in rows], dtype=torch.float32)
    weights = _prediction_weights(rows)
    _require(torch.allclose(weights.sum(), torch.tensor(float(len({row.identity.key for row in rows})))), "prediction weight sum mismatch")
    return TrainingBatch(
        inputs=input_tensor,
        labels=labels,
        auxiliary_targets=auxiliary_tensor,
        weights=weights,
        rows=tuple(rows),
    )


def fit_normalizer(values: torch.Tensor) -> Normalizer:
    mean = values.mean(dim=0)
    scale = values.std(dim=0, unbiased=False)
    scale = torch.where(scale < 1e-8, torch.ones_like(scale), scale)
    return Normalizer(mean=mean, scale=scale)


class LowRankPreconvergenceTrigger(torch.nn.Module):
    """Mean-pooled two-update trigger with an optional training-only head."""

    def __init__(
        self,
        latent_feature_dim: int,
        rank: int,
        *,
        auxiliary_action_dim: int | None = None,
    ) -> None:
        super().__init__()
        _require(rank in RANK_CANDIDATES, f"unsupported rank: {rank}")
        _require(latent_feature_dim > 0, "latent feature dimension must be positive")
        self.latent_feature_dim = int(latent_feature_dim)
        self.rank = int(rank)
        self.bottleneck = torch.nn.Linear(2 * latent_feature_dim, rank)
        self.trigger_head = torch.nn.Linear(rank, 1)
        self.auxiliary_head = (
            None
            if auxiliary_action_dim is None
            else torch.nn.Linear(rank, int(auxiliary_action_dim))
        )

    def hidden(self, normalized_inputs: torch.Tensor) -> torch.Tensor:
        return self.bottleneck(normalized_inputs)

    def score_tensor(self, normalized_inputs: torch.Tensor) -> torch.Tensor:
        """Return probabilities without host transfer or scalar extraction."""

        return torch.sigmoid(self.trigger_head(self.hidden(normalized_inputs)).squeeze(-1))

    def forward(
        self, normalized_inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden = self.hidden(normalized_inputs)
        logits = self.trigger_head(hidden).squeeze(-1)
        auxiliary = None if self.auxiliary_head is None else self.auxiliary_head(hidden)
        return logits, auxiliary

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def inference_flops(self) -> int:
        # Multiply+add for the bottleneck and scalar head, plus sigmoid estimate.
        return int(2 * (2 * self.latent_feature_dim * self.rank + self.rank) + 4)


def tensor_scorer_item_call_count() -> int:
    source = inspect.getsource(LowRankPreconvergenceTrigger.score_tensor)
    source += inspect.getsource(LowRankPreconvergenceTrigger.hidden)
    return source.count(".item(")


def train_trigger(
    sequences: Sequence[RawPreconvergenceSequence],
    *,
    rank: int,
    use_auxiliary: bool,
    config: TrainingConfig = TrainingConfig(),
) -> FittedTrigger:
    batch = build_training_batch(sequences, origin="ACTUAL_WARM")
    latent_dim = batch.inputs.shape[1] // 2
    action_dim = batch.auxiliary_targets.shape[1]
    torch.manual_seed(config.seed)
    model = LowRankPreconvergenceTrigger(
        latent_dim,
        rank,
        auxiliary_action_dim=action_dim if use_auxiliary else None,
    )
    normalizer = fit_normalizer(batch.inputs)
    normalized = normalizer.transform(batch.inputs)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    denominator = batch.weights.sum()
    for _ in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        logits, auxiliary = model(normalized)
        primary_losses = F.binary_cross_entropy_with_logits(
            logits, batch.labels, reduction="none"
        )
        loss = torch.sum(primary_losses * batch.weights) / denominator
        if use_auxiliary:
            _require(auxiliary is not None, "auxiliary model has no auxiliary head")
            auxiliary_losses = F.smooth_l1_loss(
                auxiliary, batch.auxiliary_targets, reduction="none"
            ).mean(dim=1)
            loss = loss + config.auxiliary_weight * (
                torch.sum(auxiliary_losses * batch.weights) / denominator
            )
        loss.backward()
        optimizer.step()
    return FittedTrigger(
        model=model,
        normalizer=normalizer,
        rank=rank,
        use_auxiliary=use_auxiliary,
        seed=config.seed,
        parameter_count=model.parameter_count(),
        inference_flops=model.inference_flops(),
    )


def score_sequence_tensor(
    fitted: FittedTrigger, sequence: RawPreconvergenceSequence
) -> tuple[torch.Tensor, torch.Tensor]:
    iterations = torch.arange(3, sequence.max_iter + 1, dtype=torch.int64)
    inputs = torch.stack(
        [pooled_trigger_input(sequence.states, k) for k in range(3, sequence.max_iter + 1)]
    )
    with torch.inference_mode():
        scores = fitted.model.score_tensor(fitted.normalizer.transform(inputs))
    return iterations, scores


def score_sequence(
    fitted: FittedTrigger, sequence: RawPreconvergenceSequence
) -> dict[int, float]:
    iterations, scores = score_sequence_tensor(fitted, sequence)
    return dict(zip(iterations.tolist(), scores.detach().cpu().tolist()))


@dataclass(frozen=True)
class ConfirmNextReplay:
    identity: SequenceIdentity
    actual_origin: str
    max_iter: int
    k_action: int
    k_gate: int | None
    trigger_offset: int | None
    trigger_category: str
    terminal_k: int
    delta_k: int
    coda_iterations: tuple[int, ...]
    coda_call_count: int
    baseline_coda_call_count: int
    saved_coda_calls: int
    gate_evaluation_count: int
    reached_max_iter: bool


def trigger_category(k_gate: int | None, k_action: int) -> tuple[str, int | None]:
    if k_gate is None:
        return "missed", None
    offset = int(k_gate) - (int(k_action) - 1)
    if offset < 0:
        return "early", offset
    if offset == 0:
        return "ideal", offset
    return "late", offset


def replay_confirm_next(
    sequence: RawPreconvergenceSequence,
    scores_by_k: Mapping[int, float],
    threshold: float,
) -> ConfirmNextReplay:
    """Replay SEARCH -> CONFIRM_NEXT -> contiguous Coda exactly."""

    sequence.validate()
    _require(math.isfinite(float(threshold)), "decision threshold must be finite")
    k_action = sequence.k_action
    coda_iterations = list(
        range(1, min(FORCED_CODA_PREFIX_MAX_ITERATION, sequence.max_iter) + 1)
    )
    if k_action < MODEL_APPLICABLE_MIN_K_ACTION:
        category, offset = trigger_category(None, k_action)
        return ConfirmNextReplay(
            identity=sequence.identity,
            actual_origin=sequence.actual_origin,
            max_iter=sequence.max_iter,
            k_action=k_action,
            k_gate=None,
            trigger_offset=offset,
            trigger_category="forced_prefix_convergence",
            terminal_k=k_action,
            delta_k=0,
            coda_iterations=tuple(range(1, k_action + 1)),
            coda_call_count=k_action,
            baseline_coda_call_count=k_action,
            saved_coda_calls=0,
            gate_evaluation_count=0,
            reached_max_iter=k_action == sequence.max_iter,
        )

    k_gate = None
    evaluated = 0
    for k in range(MINIMUM_GATE_ITERATION, sequence.max_iter + 1):
        _require(k in scores_by_k, f"missing gate score at k={k}")
        score = float(scores_by_k[k])
        _require(math.isfinite(score), f"non-finite gate score at k={k}")
        evaluated += 1
        if score >= threshold:
            k_gate = k
            break
    category, offset = trigger_category(k_gate, k_action)
    terminal_k = sequence.max_iter
    if k_gate is None:
        if sequence.max_iter not in coda_iterations:
            coda_iterations.append(sequence.max_iter)
    else:
        if k_gate not in coda_iterations:
            coda_iterations.append(k_gate)
        for k in range(k_gate + 1, sequence.max_iter + 1):
            coda_iterations.append(k)
            if float(sequence.action_mse[k]) < ACTION_MSE_THRESHOLD:
                terminal_k = k
                break
    unique_calls = tuple(dict.fromkeys(coda_iterations))
    baseline_calls = k_action
    return ConfirmNextReplay(
        identity=sequence.identity,
        actual_origin=sequence.actual_origin,
        max_iter=sequence.max_iter,
        k_action=k_action,
        k_gate=k_gate,
        trigger_offset=offset,
        trigger_category=category,
        terminal_k=terminal_k,
        delta_k=terminal_k - k_action,
        coda_iterations=unique_calls,
        coda_call_count=len(unique_calls),
        baseline_coda_call_count=baseline_calls,
        saved_coda_calls=baseline_calls - len(unique_calls),
        gate_evaluation_count=evaluated,
        reached_max_iter=terminal_k == sequence.max_iter,
    )


def _threshold_candidates(
    scored_sequences: Sequence[tuple[RawPreconvergenceSequence, Mapping[int, float]]]
) -> list[float]:
    values = sorted(
        {
            float(score)
            for sequence, scores in scored_sequences
            if sequence.k_action >= MODEL_APPLICABLE_MIN_K_ACTION
            for score in scores.values()
        }
    )
    _require(bool(values), "threshold selection has no scores")
    return values + [float(np.nextafter(values[-1], math.inf))]


def select_training_threshold(
    scored_sequences: Sequence[tuple[RawPreconvergenceSequence, Mapping[int, float]]]
) -> dict[str, Any]:
    """Safety-first train-only selection via an exact threshold event sweep."""

    prepared = []
    score_values = set()
    history_unavailable_prediction_count = 0
    history_unavailable_scheduled_coda_calls = 0
    for sequence, scores in scored_sequences:
        # Validation is deliberately outside the candidate loop. The exhaustive
        # reference calls it from replay_confirm_next for every threshold.
        sequence.validate()
        k_action = sequence.k_action
        if k_action < MODEL_APPLICABLE_MIN_K_ACTION:
            history_unavailable_prediction_count += 1
            history_unavailable_scheduled_coda_calls += k_action
            continue
        required_iterations = range(MINIMUM_GATE_ITERATION, sequence.max_iter + 1)
        for k in required_iterations:
            _require(k in scores, f"missing gate score at k={k}")
        finite_scores = {}
        for k, value in scores.items():
            score = float(value)
            _require(math.isfinite(score), f"non-finite gate score at k={k}")
            finite_scores[k] = score
            score_values.add(score)
        prepared.append((sequence, finite_scores, k_action))

    values = sorted(score_values)
    _require(bool(values), "threshold selection has no scores")
    fail_closed = float(np.nextafter(values[-1], math.inf))
    _require(math.isfinite(fail_closed), "decision threshold must be finite")
    candidates = values + [fail_closed]
    candidate_count = len(candidates)

    late_diff = [0] * (candidate_count + 1)
    calls_diff = [0] * (candidate_count + 1)
    absolute_offset_diff = [0] * (candidate_count + 1)
    offset_diff = [0] * (candidate_count + 1)
    offset_count_diff = [0] * (candidate_count + 1)

    def add_interval(start: int, end: int, contribution: tuple[int, int, int, int, int]) -> None:
        if start >= end:
            return
        for difference, value in zip(
            (
                late_diff,
                calls_diff,
                absolute_offset_diff,
                offset_diff,
                offset_count_diff,
            ),
            contribution,
        ):
            difference[start] += value
            difference[end] -= value

    def contribution_for_gate(
        sequence: RawPreconvergenceSequence,
        k_action: int,
        k_gate: int | None,
        next_action_hit: Mapping[int, int | None],
    ) -> tuple[int, int, int, int, int]:
        if k_gate is None:
            # Model-applicable predictions have max_iter >= 4, so the forced
            # prefix plus max-iteration fallback contains four unique calls.
            return (1, 4, 0, 0, 0)
        offset = int(k_gate) - (int(k_action) - 1)
        terminal_k = next_action_hit[k_gate]
        if terminal_k is None:
            terminal_k = sequence.max_iter
        # k=3 is already in the forced prefix and must not be duplicated.
        # A later gate adds one call before the contiguous confirmation tail.
        coda_calls = (
            FORCED_CODA_PREFIX_MAX_ITERATION
            + int(k_gate > FORCED_CODA_PREFIX_MAX_ITERATION)
            + int(terminal_k)
            - int(k_gate)
        )
        return (
            int(offset > 0),
            coda_calls,
            abs(offset),
            offset,
            1,
        )

    for sequence, scores, k_action in prepared:
        next_action_hit: dict[int, int | None] = {}
        next_hit = None
        for k in range(sequence.max_iter, 2, -1):
            next_action_hit[k] = next_hit
            if float(sequence.action_mse[k]) < ACTION_MSE_THRESHOLD:
                next_hit = k

        previous_record = -math.inf
        for k in range(MINIMUM_GATE_ITERATION, sequence.max_iter + 1):
            score = scores[k]
            if score <= previous_record:
                continue
            start = bisect_right(candidates, previous_record)
            end = bisect_right(candidates, score)
            add_interval(
                start,
                end,
                contribution_for_gate(sequence, k_action, k, next_action_hit),
            )
            previous_record = score
        add_interval(
            bisect_right(candidates, previous_record),
            candidate_count,
            contribution_for_gate(sequence, k_action, None, next_action_hit),
        )

    running_late = 0
    running_calls = 0
    running_absolute_offset = 0
    running_offset = 0
    running_offset_count = 0
    selected = None
    selected_key = None
    feasible = False
    for index, threshold in enumerate(candidates):
        running_late += late_diff[index]
        running_calls += calls_diff[index]
        running_absolute_offset += absolute_offset_diff[index]
        running_offset += offset_diff[index]
        running_offset_count += offset_count_diff[index]
        item = {
            "threshold": threshold,
            "late_or_missed_count": running_late,
            "scheduled_coda_calls": (
                running_calls + history_unavailable_scheduled_coda_calls
            ),
            "model_applicable_scheduled_coda_calls": running_calls,
            "history_unavailable_scheduled_coda_calls": (
                history_unavailable_scheduled_coda_calls
            ),
            "total_scheduled_coda_calls": (
                running_calls + history_unavailable_scheduled_coda_calls
            ),
            "mean_absolute_trigger_offset": (
                float(running_absolute_offset / running_offset_count)
                if running_offset_count
                else math.inf
            ),
            "mean_trigger_lead": (
                float(-running_offset / running_offset_count)
                if running_offset_count
                else None
            ),
        }
        feasible = feasible or running_late == 0
        key = (
            item["late_or_missed_count"],
            item["scheduled_coda_calls"],
            item["mean_absolute_trigger_offset"],
            -item["threshold"],
        )
        if selected_key is None or key < selected_key:
            selected = item
            selected_key = key

    _require(selected is not None, "threshold selection produced no candidate")
    return {
        "selected_threshold": selected["threshold"],
        "selected_threshold_hex": float(selected["threshold"]).hex(),
        "selection_status": (
            "no_late_or_missed_feasible"
            if feasible
            else "no_safe_threshold_fail_closed"
        ),
        "candidate_count": candidate_count,
        "model_applicable_prediction_count": len(prepared),
        "history_unavailable_prediction_count": (
            history_unavailable_prediction_count
        ),
        "model_applicable_scheduled_coda_calls": selected[
            "model_applicable_scheduled_coda_calls"
        ],
        "history_unavailable_scheduled_coda_calls": selected[
            "history_unavailable_scheduled_coda_calls"
        ],
        "total_scheduled_coda_calls": selected["total_scheduled_coda_calls"],
        "selection_order": [
            "require zero late or missed train triggers when feasible",
            "minimize exact CONFIRM_NEXT scheduled Coda calls",
            "minimize mean absolute trigger offset",
            "maximize threshold",
        ],
        "train_metrics": selected,
    }


def _percentile(values: Sequence[float], q: float) -> float | None:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q)) if values else None


def project_latency(
    replays: Sequence[ConfirmNextReplay],
    *,
    coda_latency_ms: float,
    recurrent_iteration_latency_ms: float,
    gate_latency_ms: float,
) -> dict[str, float]:
    _require(bool(replays), "latency projection requires predictions")
    for name, value in (
        ("coda_latency_ms", coda_latency_ms),
        ("recurrent_iteration_latency_ms", recurrent_iteration_latency_ms),
        ("gate_latency_ms", gate_latency_ms),
    ):
        _require(math.isfinite(value) and value >= 0, f"invalid {name}")
    count = len(replays)
    saved = sum(replay.saved_coda_calls for replay in replays) / count
    delta_k = sum(replay.delta_k for replay in replays) / count
    gates = sum(replay.gate_evaluation_count for replay in replays) / count
    gross = saved * coda_latency_ms
    recurrent_cost = delta_k * recurrent_iteration_latency_ms
    gate_cost = gates * gate_latency_ms
    return {
        "saved_coda_calls_per_prediction": saved,
        "additional_recurrent_iterations_per_prediction": delta_k,
        "gate_evaluations_per_prediction": gates,
        "gross_coda_saving_ms": gross,
        "additional_recurrent_cost_ms": recurrent_cost,
        "gate_overhead_ms": gate_cost,
        "projected_net_saving_ms": gross - recurrent_cost - gate_cost,
        "gate_latency_assumption_ms": gate_latency_ms,
    }


def aggregate_replays(
    replays: Sequence[ConfirmNextReplay],
    *,
    latency: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    _require(bool(replays), "cannot aggregate empty replays")
    categories = {
        name: sum(replay.trigger_category == name for replay in replays)
        for name in ("early", "ideal", "late", "missed", "forced_prefix_convergence")
    }
    offsets = [
        float(replay.trigger_offset)
        for replay in replays
        if replay.trigger_offset is not None
    ]
    baseline_calls = sum(replay.baseline_coda_call_count for replay in replays)
    candidate_calls = sum(replay.coda_call_count for replay in replays)
    result: dict[str, Any] = {
        "prediction_count": len(replays),
        "trigger_category_counts": categories,
        "ideal_trigger_rate": categories["ideal"] / len(replays),
        "early_trigger_rate": categories["early"] / len(replays),
        "late_trigger_rate": categories["late"] / len(replays),
        "missed_preconvergence_trigger_rate": categories["missed"] / len(replays),
        "mean_trigger_offset": float(np.mean(offsets)) if offsets else None,
        "median_trigger_offset": float(np.median(offsets)) if offsets else None,
        "p95_trigger_offset": _percentile(offsets, 0.95),
        "mean_trigger_lead": float(np.mean([-value for value in offsets])) if offsets else None,
        "baseline_coda_calls": baseline_calls,
        "candidate_coda_calls": candidate_calls,
        "saved_coda_calls": baseline_calls - candidate_calls,
        "coda_call_reduction": (
            (baseline_calls - candidate_calls) / baseline_calls if baseline_calls else 0.0
        ),
        "mean_delta_k": float(np.mean([replay.delta_k for replay in replays])),
        "median_delta_k": float(np.median([replay.delta_k for replay in replays])),
        "p95_delta_k": _percentile([replay.delta_k for replay in replays], 0.95),
        "baseline_max_iteration_rate": float(
            np.mean([replay.k_action == replay.max_iter for replay in replays])
        ),
        "candidate_max_iteration_rate": float(
            np.mean([replay.reached_max_iter for replay in replays])
        ),
        "mean_gate_evaluations": float(
            np.mean([replay.gate_evaluation_count for replay in replays])
        ),
    }
    result["max_iteration_rate_change"] = (
        result["candidate_max_iteration_rate"] - result["baseline_max_iteration_rate"]
    )
    if latency is not None:
        result["projected_latency"] = project_latency(
            replays,
            coda_latency_ms=float(latency["coda_latency_ms"]),
            recurrent_iteration_latency_ms=float(
                latency["recurrent_iteration_latency_ms"]
            ),
            gate_latency_ms=float(latency["gate_latency_ms"]),
        )
    return result


def replay_record(replay: ConfirmNextReplay) -> dict[str, Any]:
    return {
        "task_id": replay.identity.task_id,
        "episode_id": replay.identity.episode_id,
        "prediction_id": replay.identity.prediction_id,
        "actual_origin": replay.actual_origin,
        "max_iter": replay.max_iter,
        "K_action": replay.k_action,
        "K_gate": replay.k_gate,
        "trigger_offset": replay.trigger_offset,
        "trigger_category": replay.trigger_category,
        "terminal_k": replay.terminal_k,
        "delta_k": replay.delta_k,
        "coda_iterations": list(replay.coda_iterations),
        "coda_call_count": replay.coda_call_count,
        "baseline_coda_call_count": replay.baseline_coda_call_count,
        "saved_coda_calls": replay.saved_coda_calls,
        "gate_evaluation_count": replay.gate_evaluation_count,
        "reached_max_iter": replay.reached_max_iter,
    }


def _task_macro_field(
    per_task: Mapping[str, Mapping[str, Any]], field: str
) -> float | None:
    values = [metric[field] for metric in per_task.values() if metric[field] is not None]
    return float(np.mean(values)) if values else None


def fallback_preservation(
    replays: Sequence[ConfirmNextReplay],
) -> dict[str, Any]:
    """Audit the fixed baseline behavior for history-unavailable predictions."""

    for replay in replays:
        _require(
            replay.k_action < MODEL_APPLICABLE_MIN_K_ACTION,
            "fallback audit received a model-applicable prediction",
        )
    terminal_mismatches = sum(
        replay.terminal_k != replay.k_action for replay in replays
    )
    nonzero_delta = sum(replay.delta_k != 0 for replay in replays)
    unexpected_gate_evaluations = sum(
        replay.gate_evaluation_count != 0 or replay.k_gate is not None
        for replay in replays
    )
    exact_contract = all(
        replay.trigger_category == "forced_prefix_convergence"
        and replay.terminal_k == replay.k_action
        and replay.delta_k == 0
        and replay.k_gate is None
        and replay.gate_evaluation_count == 0
        and replay.coda_iterations == tuple(range(1, replay.k_action + 1))
        and replay.coda_call_count == replay.k_action
        and replay.saved_coda_calls == 0
        and replay.reached_max_iter == (replay.k_action == replay.max_iter)
        for replay in replays
    )
    return {
        "prediction_count": len(replays),
        "terminal_k_mismatch_count": terminal_mismatches,
        "nonzero_delta_k_count": nonzero_delta,
        "unexpected_gate_evaluation_count": unexpected_gate_evaluations,
        "preserves_baseline_k": bool(exact_contract),
    }


def preconvergence_promotion_checks(
    *,
    overall_metrics: Mapping[str, Any],
    model_applicable_metrics: Mapping[str, Any],
    fallback_audit: Mapping[str, Any],
    zero_overhead_projection: Mapping[str, Any],
) -> dict[str, bool]:
    categories = model_applicable_metrics["trigger_category_counts"]
    return {
        "model_applicable_zero_late_or_missed": (
            categories["late"] + categories["missed"] == 0
        ),
        "model_applicable_mean_trigger_lead_between_0_and_1": (
            model_applicable_metrics["mean_trigger_lead"] is not None
            and 0.0 <= model_applicable_metrics["mean_trigger_lead"] <= 1.0
        ),
        "model_applicable_mean_delta_k_near_zero": (
            abs(model_applicable_metrics["mean_delta_k"]) <= 0.1
        ),
        "overall_coda_reduction_positive": (
            overall_metrics["coda_call_reduction"] > 0
        ),
        "fallback_preserves_baseline_k": bool(
            fallback_audit["preserves_baseline_k"]
        ),
        "zero_overhead_projection_improves": (
            zero_overhead_projection["projected_net_saving_ms"] > 0
        ),
    }


def leakage_audit(
    sequences: Sequence[RawPreconvergenceSequence], assignment: Mapping[str, int]
) -> dict[str, Any]:
    folds = []
    for fold_id in sorted(set(assignment.values())):
        train_tasks = {int(task) for task, fold in assignment.items() if fold != fold_id}
        held_out_tasks = {int(task) for task, fold in assignment.items() if fold == fold_id}
        _require(train_tasks.isdisjoint(held_out_tasks), f"fold {fold_id}: task leakage")
        train_keys = {s.identity.key for s in sequences if s.identity.task_id in train_tasks}
        held_out_keys = {s.identity.key for s in sequences if s.identity.task_id in held_out_tasks}
        _require(train_keys.isdisjoint(held_out_keys), f"fold {fold_id}: prediction leakage")
        folds.append(
            {
                "fold_id": fold_id,
                "training_task_ids": sorted(train_tasks),
                "held_out_task_ids": sorted(held_out_tasks),
                "task_overlap_count": 0,
                "prediction_overlap_count": 0,
            }
        )
    return {"passed": True, "folds": folds}


def serialize_fitted_trigger(fitted: FittedTrigger) -> dict[str, Any]:
    return {
        "rank": fitted.rank,
        "use_auxiliary": fitted.use_auxiliary,
        "seed": fitted.seed,
        "latent_feature_dim": fitted.model.latent_feature_dim,
        "auxiliary_action_dim": (
            None
            if fitted.model.auxiliary_head is None
            else fitted.model.auxiliary_head.out_features
        ),
        "parameter_count": fitted.parameter_count,
        "inference_flops": fitted.inference_flops,
        "normalization": {
            "mean": fitted.normalizer.mean.detach().cpu(),
            "scale": fitted.normalizer.scale.detach().cpu(),
        },
        "state_dict": {
            name: value.detach().cpu() for name, value in fitted.model.state_dict().items()
        },
    }


def deserialize_fitted_trigger(payload: Mapping[str, Any]) -> FittedTrigger:
    model = LowRankPreconvergenceTrigger(
        int(payload["latent_feature_dim"]),
        int(payload["rank"]),
        auxiliary_action_dim=payload.get("auxiliary_action_dim"),
    )
    model.load_state_dict(payload["state_dict"])
    normalization = payload["normalization"]
    return FittedTrigger(
        model=model,
        normalizer=Normalizer(
            mean=normalization["mean"].float(), scale=normalization["scale"].float()
        ),
        rank=int(payload["rank"]),
        use_auxiliary=bool(payload["use_auxiliary"]),
        seed=int(payload["seed"]),
        parameter_count=int(payload["parameter_count"]),
        inference_flops=int(payload["inference_flops"]),
    )


def load_fold_assignment(path: Path, task_ids: Iterable[int]) -> tuple[dict[str, Any], dict[str, int]]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    assignment: dict[str, int] = {}
    requested = {str(task) for task in task_ids}
    for fold in manifest["folds"]:
        fold_id = int(fold.get("fold_id", fold.get("fold")))
        values = fold.get("validation_task_ids", fold.get("task_ids", []))
        for task in values:
            key = str(task)
            _require(key not in assignment, f"duplicate task fold: {key}")
            assignment[key] = fold_id
    _require(set(assignment) == requested, "fold manifest does not cover dataset tasks")
    return manifest, assignment


def _authoritative_records(dataset_dir: Path) -> tuple[dict[str, Any], dict[tuple[int, int, int], dict[str, Any]]]:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    data_path = dataset_dir / str(manifest["dataset_file"])
    _require(sha256_file(data_path) == manifest["dataset_sha256"], "authoritative dataset hash mismatch")
    records: dict[tuple[int, int, int], dict[str, Any]] = {}
    with data_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            key = tuple(int(value) for value in record["key"])
            _require(key not in records, f"duplicate authoritative identity: {key}")
            records[key] = record
    return manifest, records


def load_raw_manifest_sequences(
    raw_manifest_path: Path | Sequence[Path],
    authoritative_dataset_dir: Path | None = None,
) -> tuple[dict[str, Any], list[RawPreconvergenceSequence]]:
    """Join optional raw shards to frozen labels by exact workload identity."""

    raw_manifest_paths = (
        [Path(path) for path in raw_manifest_path]
        if isinstance(raw_manifest_path, (list, tuple))
        else [Path(raw_manifest_path)]
    )
    _require(bool(raw_manifest_paths), "at least one raw manifest is required")
    first_manifest = json.loads(raw_manifest_paths[0].read_text(encoding="utf-8"))
    if first_manifest.get("schema_version") == 2:
        from experiments.robot.libero.raw_preconvergence_trace import (
            load_and_validate_manifests,
        )

        compact, predictions = load_and_validate_manifests(raw_manifest_paths)
        sequences = []
        for payload in predictions:
            identity = payload["identity"]
            key = (
                int(identity["task_id"]),
                int(identity["episode_id"]),
                int(identity["prediction_id"]),
            )
            raw_origin = str(payload["actual_origin"])
            actual_origin = "ACTUAL_WARM" if raw_origin == "ACTUAL_WARM" else "COLD"
            sequence = RawPreconvergenceSequence(
                identity=SequenceIdentity(*key),
                actual_origin=actual_origin,
                states=payload["tensors"]["states"],
                actions=payload["tensors"]["actions"],
                action_mse=tuple(payload["action_mse"]),
                action_mse_phase=tuple(payload["action_mse_phase"]),
                baseline_k=int(payload["production_terminal_k"]),
                max_iter=int(payload["maximum_shadow_depth"]),
            )
            sequence.validate()
            _require(sequence.k_action == sequence.baseline_k, f"{key}: collected first-hit mismatch")
            sequences.append(sequence)
        return {
            "schema_version": SCHEMA_VERSION,
            "raw_manifest_schema_version": 2,
            "raw_manifests": [str(path.resolve()) for path in raw_manifest_paths],
            "raw_manifest_sha256": [sha256_file(path) for path in raw_manifest_paths],
            "source_trace_set_sha256": compact["trace_set_sha256"],
            "authoritative_dataset_manifest": None,
            "authoritative_dataset_sha256": None,
            "authoritative_label_sources": {
                "production": "native BF16 control-flow iteration_mse embedded in raw shard",
                "shadow_tail": "FP32 diagnostic action_mse embedded in raw shard only after baseline K",
            },
            "prediction_count": len(sequences),
            "raw_states_present": True,
            "raw_actions_present_for_auxiliary_supervision": True,
        }, sequences

    _require(len(raw_manifest_paths) == 1, "legacy raw schema accepts exactly one manifest")
    _require(authoritative_dataset_dir is not None, "legacy raw schema requires authoritative_dataset_dir")
    raw_manifest_path = raw_manifest_paths[0]
    raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    _require(raw_manifest.get("schema_version") == SCHEMA_VERSION, "raw manifest schema mismatch")
    _require(
        raw_manifest.get("collection_mode") == "optional_post_production_shadow",
        "raw states must come from optional post-production shadow collection",
    )
    authoritative_manifest, authoritative = _authoritative_records(
        Path(authoritative_dataset_dir)
    )
    _require(
        raw_manifest.get("source_trace_set_sha256")
        == authoritative_manifest.get("trace_set_sha256"),
        "raw manifest and authoritative labels have different trace identities",
    )
    sequences = []
    seen = set()
    for descriptor in raw_manifest["sequences"]:
        key = (
            int(descriptor["task_id"]),
            int(descriptor["episode_id"]),
            int(descriptor["prediction_id"]),
        )
        _require(key not in seen, f"duplicate raw identity: {key}")
        seen.add(key)
        _require(key in authoritative, f"raw identity absent from authoritative labels: {key}")
        shard_path = Path(descriptor["shard_path"])
        if not shard_path.is_absolute():
            shard_path = raw_manifest_path.parent / shard_path
        _require(sha256_file(shard_path) == descriptor["sha256"], f"raw shard hash mismatch: {key}")
        payload = torch.load(shard_path, map_location="cpu", weights_only=True)
        _require(payload.get("schema_version") == SCHEMA_VERSION, f"raw shard schema mismatch: {key}")
        payload_identity = payload["identity"]
        _require(
            key
            == (
                int(payload_identity["task_id"]),
                int(payload_identity["episode_id"]),
                int(payload_identity["prediction_id"]),
            ),
            f"raw shard identity mismatch: {key}",
        )
        _require(
            str(payload.get("actual_origin")) == str(authoritative[key]["actual_origin"]),
            f"raw shard origin mismatch: {key}",
        )
        _require(
            descriptor.get("tensor_metadata")
            == {
                "states": tensor_metadata(payload["states"]),
                "actions": tensor_metadata(payload["actions"]),
            },
            f"raw shard tensor metadata mismatch: {key}",
        )
        labels = authoritative[key]
        max_iter = int(labels["max_iter"])
        mse: list[float | None] = [None] * (max_iter + 1)
        phases: list[str | None] = [None] * (max_iter + 1)
        for transition in labels["transitions"]:
            k = int(transition["k"])
            mse[k] = float(transition["action_mse"])
            phases[k] = str(transition["phase"])
            _require(
                (k <= int(labels["baseline_k"]) and phases[k] == "production")
                or (k > int(labels["baseline_k"]) and phases[k] == "shadow_tail"),
                f"{key} k={k}: authoritative BF16/shadow phase contract mismatch",
            )
        sequence = RawPreconvergenceSequence(
            identity=SequenceIdentity(*key),
            actual_origin=str(labels["actual_origin"]),
            states=payload["states"],
            actions=payload["actions"],
            action_mse=tuple(mse),
            action_mse_phase=tuple(phases),
            baseline_k=int(labels["baseline_k"]),
            max_iter=max_iter,
        )
        sequence.validate()
        _require(sequence.k_action == sequence.baseline_k, f"{key}: authoritative first-hit mismatch")
        sequences.append(sequence)
    _require(
        seen == set(authoritative),
        "raw manifest must cover every authoritative calibration prediction",
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "raw_manifest": str(raw_manifest_path.resolve()),
        "raw_manifest_sha256": sha256_file(raw_manifest_path),
        "source_trace_set_sha256": authoritative_manifest["trace_set_sha256"],
        "authoritative_dataset_manifest": str(
            (Path(authoritative_dataset_dir) / "manifest.json").resolve()
        ),
        "authoritative_dataset_sha256": authoritative_manifest["dataset_sha256"],
        "authoritative_label_sources": {
            "production": "native BF16 control-flow iteration_mse",
            "shadow_tail": "FP32 diagnostic action_mse only after baseline K",
        },
        "prediction_count": len(sequences),
        "raw_states_present": True,
        "raw_actions_present_for_auxiliary_supervision": True,
    }
    return metadata, sequences


def save_dataset_bundle(
    output_dir: Path,
    metadata: Mapping[str, Any],
    sequences: Sequence[RawPreconvergenceSequence],
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / "preconvergence_dataset.pt"
    torch.save({"schema_version": SCHEMA_VERSION, "sequences": list(sequences)}, bundle_path)
    manifest = {
        **dict(metadata),
        "dataset_file": bundle_path.name,
        "dataset_sha256": sha256_file(bundle_path),
        "classification_contract": {
            "positive": "k == K_action - 1",
            "negative": "k < K_action - 1",
            "excluded": "k >= K_action",
            "minimum_model_iteration": 3,
        },
        "replay_contract": replay_contract(),
        "history_coverage": {
            "model_applicable_prediction_count": sum(
                sequence.k_action >= MODEL_APPLICABLE_MIN_K_ACTION
                for sequence in sequences
            ),
            "history_unavailable_prediction_count": sum(
                sequence.k_action < MODEL_APPLICABLE_MIN_K_ACTION
                for sequence in sequences
            ),
            "model_applicable_min_k_action": MODEL_APPLICABLE_MIN_K_ACTION,
            "history_unavailable_definition": HISTORY_UNAVAILABLE_DEFINITION,
        },
    }
    (output_dir / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    return manifest


def load_dataset_bundle(dataset_dir: Path) -> tuple[dict[str, Any], list[RawPreconvergenceSequence]]:
    dataset_dir = Path(dataset_dir)
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    path = dataset_dir / manifest["dataset_file"]
    _require(sha256_file(path) == manifest["dataset_sha256"], "dataset bundle hash mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    sequences = list(payload["sequences"])
    for sequence in sequences:
        _require(isinstance(sequence, RawPreconvergenceSequence), "invalid sequence payload")
        sequence.validate()
    return manifest, sequences


def fit_oof_bundle(
    sequences: Sequence[RawPreconvergenceSequence],
    assignment: Mapping[str, int],
    *,
    ranks: Sequence[int] = RANK_CANDIDATES,
    variants: Sequence[str] = MODEL_VARIANTS,
    config: TrainingConfig = TrainingConfig(),
) -> dict[str, Any]:
    """Fit fold models and thresholds without scoring any held-out prediction."""

    audit = leakage_audit(sequences, assignment)
    models: dict[str, Any] = {}
    for rank in ranks:
        for variant in variants:
            _require(variant in MODEL_VARIANTS, f"unknown model variant: {variant}")
            use_auxiliary = variant == "action_delta_auxiliary"
            name = f"rank{rank}_{variant}"
            model_started = time.perf_counter()
            print(
                f"[preconvergence] model={name} stage=model start",
                flush=True,
            )
            folds = []
            for fold_id in sorted(set(assignment.values())):
                training = [
                    sequence
                    for sequence in sequences
                    if assignment[str(sequence.identity.task_id)] != fold_id
                    and sequence.actual_origin == "ACTUAL_WARM"
                ]
                train_started = time.perf_counter()
                print(
                    f"[preconvergence] model={name} fold={fold_id} "
                    f"stage=train start predictions={len(training)}",
                    flush=True,
                )
                fitted = train_trigger(
                    training,
                    rank=rank,
                    use_auxiliary=use_auxiliary,
                    config=config,
                )
                print(
                    f"[preconvergence] model={name} fold={fold_id} "
                    f"stage=train end elapsed_seconds={time.perf_counter() - train_started:.3f}",
                    flush=True,
                )
                threshold_started = time.perf_counter()
                print(
                    f"[preconvergence] model={name} fold={fold_id} "
                    "stage=threshold start",
                    flush=True,
                )
                scored_training = [
                    (
                        sequence,
                        score_sequence(fitted, sequence)
                        if sequence.k_action >= MODEL_APPLICABLE_MIN_K_ACTION
                        else {},
                    )
                    for sequence in training
                ]
                selection = select_training_threshold(scored_training)
                print(
                    f"[preconvergence] model={name} fold={fold_id} "
                    f"stage=threshold end candidates={selection['candidate_count']} "
                    f"elapsed_seconds={time.perf_counter() - threshold_started:.3f}",
                    flush=True,
                )
                folds.append(
                    {
                        "fold_id": fold_id,
                        "training_task_ids": sorted(
                            {sequence.identity.task_id for sequence in training}
                        ),
                        "held_out_task_ids": sorted(
                            int(task)
                            for task, assigned_fold in assignment.items()
                            if assigned_fold == fold_id
                        ),
                        "training_prediction_count": len(training),
                        "model_applicable_training_prediction_count": selection[
                            "model_applicable_prediction_count"
                        ],
                        "history_unavailable_training_prediction_count": selection[
                            "history_unavailable_prediction_count"
                        ],
                        "threshold_selection": selection,
                        "fitted_trigger": serialize_fitted_trigger(fitted),
                    }
                )
            models[name] = {
                "rank": rank,
                "variant": variant,
                "folds": folds,
            }
            print(
                f"[preconvergence] model={name} stage=model end "
                f"elapsed_seconds={time.perf_counter() - model_started:.3f}",
                flush=True,
            )
    return {
        "schema_version": SCHEMA_VERSION,
        **replay_contract(),
        "replay_contract": replay_contract(),
        "seed": config.seed,
        "training_config": {
            "steps": config.steps,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "auxiliary_weight": config.auxiliary_weight,
        },
        "model_fitting_scope": "outer-training ACTUAL_WARM tasks only",
        "threshold_selection_scope": (
            "outer-training ACTUAL_WARM predictions with K_action >= 4 only"
        ),
        "global_model_fitted": False,
        "global_threshold_fitted": False,
        "leakage_audit": audit,
        "models": models,
    }


def evaluate_oof_bundle(
    sequences: Sequence[RawPreconvergenceSequence],
    assignment: Mapping[str, int],
    training_bundle: Mapping[str, Any],
    *,
    latency: Mapping[str, float],
) -> dict[str, Any]:
    """Score each prediction exactly once with its held-out-task fold model."""

    _require(training_bundle.get("global_model_fitted") is False, "global model is forbidden")
    _require(
        training_bundle.get("global_threshold_fitted") is False,
        "global threshold is forbidden",
    )
    _require(
        training_bundle.get("replay_contract_version")
        == REPLAY_CONTRACT_VERSION,
        "training bundle replay contract version mismatch; contract-v1 bundles are rejected",
    )
    _require(
        training_bundle.get("replay_contract") == replay_contract(),
        "training bundle replay contract metadata mismatch",
    )
    _require(
        all(
            training_bundle.get(field) == value
            for field, value in replay_contract().items()
        ),
        "training bundle top-level replay contract metadata mismatch",
    )
    leakage_audit(sequences, assignment)
    results = {}
    for name, model_bundle in training_bundle["models"].items():
        all_replays: list[ConfirmNextReplay] = []
        fold_reports = []
        seen: set[tuple[int, int, int]] = set()
        for fold in model_bundle["folds"]:
            fold_id = int(fold["fold_id"])
            held_out = [
                sequence
                for sequence in sequences
                if assignment[str(sequence.identity.task_id)] == fold_id
            ]
            expected_tasks = sorted({sequence.identity.task_id for sequence in held_out})
            _require(
                expected_tasks == list(fold["held_out_task_ids"]),
                f"{name} fold {fold_id}: held-out task mapping changed",
            )
            _require(
                set(fold["training_task_ids"]).isdisjoint(expected_tasks),
                f"{name} fold {fold_id}: outer-held-out task entered training",
            )
            fitted = deserialize_fitted_trigger(fold["fitted_trigger"])
            selection = fold["threshold_selection"]
            threshold = float(selection["selected_threshold"])
            fold_replays = []
            for sequence in held_out:
                _require(sequence.identity.key not in seen, "duplicate OOF prediction")
                seen.add(sequence.identity.key)
                fold_replays.append(
                    replay_confirm_next(
                        sequence,
                        (
                            score_sequence(fitted, sequence)
                            if sequence.k_action >= MODEL_APPLICABLE_MIN_K_ACTION
                            else {}
                        ),
                        threshold,
                    )
                )
            all_replays.extend(fold_replays)
            fold_reports.append(
                {
                    "fold_id": fold_id,
                    "held_out_task_ids": expected_tasks,
                    "selected_threshold": threshold,
                    "selected_threshold_hex": selection["selected_threshold_hex"],
                    "selection_status": selection["selection_status"],
                    "metrics": aggregate_replays(fold_replays, latency=latency),
                }
            )
        _require(len(seen) == len(sequences), f"{name}: incomplete OOF coverage")
        primary = [replay for replay in all_replays if replay.actual_origin == "ACTUAL_WARM"]
        cold = [replay for replay in all_replays if replay.actual_origin == "COLD"]
        primary_model_applicable = [
            replay
            for replay in primary
            if replay.k_action >= MODEL_APPLICABLE_MIN_K_ACTION
        ]
        primary_fallback = [
            replay
            for replay in primary
            if replay.k_action < MODEL_APPLICABLE_MIN_K_ACTION
        ]
        _require(
            bool(primary_model_applicable),
            f"{name}: no model-applicable ACTUAL_WARM predictions",
        )
        primary_metrics = aggregate_replays(primary, latency=latency)
        primary_metrics["history_coverage"] = {
            "model_applicable_prediction_count": len(primary_model_applicable),
            "history_unavailable_prediction_count": len(primary_fallback),
            "model_applicable_min_k_action": MODEL_APPLICABLE_MIN_K_ACTION,
            "history_unavailable_definition": HISTORY_UNAVAILABLE_DEFINITION,
        }
        applicable_metrics = aggregate_replays(
            primary_model_applicable, latency=latency
        )
        fallback_audit = fallback_preservation(primary_fallback)
        fallback_metrics = (
            aggregate_replays(primary_fallback, latency=latency)
            if primary_fallback
            else {"prediction_count": 0}
        )
        fallback_metrics.update(fallback_audit)
        per_task = {
            str(task): aggregate_replays(
                [replay for replay in primary if replay.identity.task_id == task],
                latency=latency,
            )
            for task in sorted({replay.identity.task_id for replay in primary})
        }
        macro_fields = (
            "ideal_trigger_rate",
            "early_trigger_rate",
            "late_trigger_rate",
            "missed_preconvergence_trigger_rate",
            "mean_trigger_offset",
            "mean_trigger_lead",
            "coda_call_reduction",
            "mean_delta_k",
            "p95_delta_k",
            "candidate_max_iteration_rate",
        )
        primary_metrics["task_macro"] = {
            field: _task_macro_field(per_task, field)
            for field in macro_fields
        }
        primary_metrics["per_task"] = per_task
        zero_overhead = project_latency(
            primary,
            coda_latency_ms=float(latency["coda_latency_ms"]),
            recurrent_iteration_latency_ms=float(
                latency["recurrent_iteration_latency_ms"]
            ),
            gate_latency_ms=0.0,
        )
        promotion = preconvergence_promotion_checks(
            overall_metrics=primary_metrics,
            model_applicable_metrics=applicable_metrics,
            fallback_audit=fallback_audit,
            zero_overhead_projection=zero_overhead,
        )
        first_fold = model_bundle["folds"][0]["fitted_trigger"]
        results[name] = {
            "rank": model_bundle["rank"],
            "variant": model_bundle["variant"],
            "parameter_count": int(first_fold["parameter_count"]),
            "inference_flops": int(first_fold["inference_flops"]),
            "folds": fold_reports,
            "primary_actual_warm": primary_metrics,
            "primary_model_applicable": applicable_metrics,
            "primary_history_unavailable_fallback": fallback_metrics,
            "secondary_cold": aggregate_replays(cold, latency=latency) if cold else None,
            "prediction_replays": [replay_record(replay) for replay in all_replays],
            "zero_overhead_projection": zero_overhead,
            "promotion_checks": promotion,
            "passes_all_promotion_checks": all(promotion.values()),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        **replay_contract(),
        "replay_contract": replay_contract(),
        "scope": "offline preconvergence feasibility only",
        "label_definition": "y_k = 1 iff k == K_action - 1; k >= K_action excluded from training",
        "scheduler": "CONFIRM_NEXT; gate never declares convergence",
        "models": results,
        "online_integration_implemented": False,
        "gpu_microbenchmark_required": any(
            result["passes_all_promotion_checks"] for result in results.values()
        ),
    }


def run_oof_training(
    sequences: Sequence[RawPreconvergenceSequence],
    assignment: Mapping[str, int],
    *,
    ranks: Sequence[int] = RANK_CANDIDATES,
    variants: Sequence[str] = MODEL_VARIANTS,
    config: TrainingConfig = TrainingConfig(),
    latency: Mapping[str, float] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit contract-v2 folds, then evaluate them through the canonical path."""

    training_bundle = fit_oof_bundle(
        sequences,
        assignment,
        ranks=ranks,
        variants=variants,
        config=config,
    )
    evaluation = evaluate_oof_bundle(
        sequences,
        assignment,
        training_bundle,
        latency=latency
        or {
            "coda_latency_ms": 0.0,
            "recurrent_iteration_latency_ms": 0.0,
            "gate_latency_ms": 0.0,
        },
    )
    return training_bundle, evaluation
