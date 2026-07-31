"""Pure offline replay and task-level OOF selection for origin-aware Coda scheduling.

The replay is deliberately conditional on the clean warm-only observation stream
and its incoming midpoint cache.  It is suitable for candidate pruning, not for
estimating closed-loop success or latency.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from configs.rdvla_precheck import ORIGIN_AWARE_COLD_THRESHOLD


FIXED_WARM_THRESHOLDS = (0.075, 0.08, 0.10, 0.12, 0.15)
WARM_QUANTILES = (0.20, 0.40, 0.60, 0.80)
MAX_SKIP_VALUES = (1, 2, 3)
CONFIRMATION_MODES = ("next_iter", "backfill_pair")


class ShadowTraceValidationError(ValueError):
    """Raised when a shadow record cannot support exact scheduler replay."""


@dataclass(frozen=True)
class TracePoint:
    k: int
    phase: str
    latent_mse: float
    latent_l2: float
    action_mse: Optional[float]
    action_l2: Optional[float]


@dataclass(frozen=True)
class ShadowPrediction:
    task_id: str
    episode_id: int
    prediction_index: int
    actual_origin: str
    baseline_k: int
    baseline_decode_calls: int
    max_iter: int
    action_mse_threshold: float
    effective_min_iter: int
    latent_precheck_min_iter: int
    trace: Tuple[TracePoint, ...]

    @property
    def key(self) -> Tuple[str, int, int]:
        return self.task_id, self.episode_id, self.prediction_index


@dataclass(frozen=True)
class SchedulerConfig:
    warm_threshold: float
    max_skip_iters: int
    confirmation_mode: str
    cold_threshold: float = ORIGIN_AWARE_COLD_THRESHOLD

    @property
    def exact_key(self) -> Tuple[str, int, str, str]:
        return (
            float(self.warm_threshold).hex(),
            int(self.max_skip_iters),
            self.confirmation_mode,
            float(self.cold_threshold).hex(),
        )


@dataclass(frozen=True)
class ThresholdFamily:
    kind: str
    value: float

    @property
    def label(self) -> str:
        if self.kind == "fixed":
            return f"fixed:{self.value:.17g}"
        return f"quantile:Q{int(round(self.value * 100)):02d}"


@dataclass(frozen=True)
class SchedulerFamily:
    threshold: ThresholdFamily
    max_skip_iters: int
    confirmation_mode: str

    @property
    def identifier(self) -> str:
        return (
            f"{self.threshold.label}|max_skip={self.max_skip_iters}|"
            f"confirmation={self.confirmation_mode}"
        )


@dataclass(frozen=True)
class ReplayResult:
    terminal_k: int
    adaptive_stop: bool
    stop_reason: str
    decode_calls: int
    backfill_decode_calls: int
    latent_gate_calls: int
    action_comparisons: int
    finite_checks: int
    decoded_current_iterations: Tuple[int, ...]
    decoded_calls: Tuple[Tuple[int, str], ...]
    comparison_pairs: Tuple[Tuple[int, int], ...]
    confirmed_convergence_iterations: Tuple[int, ...]
    reference_first_convergence_k: Optional[int]
    reference_persistent_convergence_k: Optional[int]
    captured_reference_convergence: Optional[bool]
    recovered_persistent_convergence: Optional[bool]
    stopped_before_persistent_tail: bool
    false_convergence: bool
    final_convergence_evaluable: bool
    max_iteration_convergence_evaluable: bool


@dataclass(frozen=True)
class CostModel:
    recurrent_ms: float
    decode_ms: float
    latent_gate_ms: float
    action_compare_ms: float
    finite_check_ms: float

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not _is_finite_number(value) or float(value) < 0:
                raise ValueError(f"{name} must be a finite non-negative number")


@dataclass(frozen=True)
class SelectionConstraints:
    min_convergence_capture: float = 0.995
    max_mean_delta_k: float = 0.25
    max_p95_delta_k: float = 1.0
    max_max_iter_rate_delta: float = 0.0
    max_false_convergence_rate: float = 0.0
    max_candidate_retry_rate: float = 0.0

    def __post_init__(self):
        for name in (
            "min_convergence_capture",
            "max_false_convergence_rate",
            "max_candidate_retry_rate",
        ):
            value = getattr(self, name)
            if not _is_finite_number(value) or not 0 <= float(value) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        for name in ("max_mean_delta_k", "max_p95_delta_k"):
            if not _is_finite_number(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not _is_finite_number(self.max_max_iter_rate_delta) or not -1 <= self.max_max_iter_rate_delta <= 1:
            raise ValueError("max_max_iter_rate_delta must be between -1 and 1")


@dataclass(frozen=True)
class EvaluatedReplay:
    prediction: ShadowPrediction
    result: ReplayResult
    baseline_latency_ms: float
    candidate_latency_ms: float

    @property
    def delta_k(self) -> int:
        return self.result.terminal_k - self.prediction.baseline_k


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _finite_float(value: Any, field: str) -> float:
    if not _is_finite_number(value):
        raise ShadowTraceValidationError(f"{field} must be finite")
    return float(value)


def _positive_int(value: Any, field: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ShadowTraceValidationError(f"{field} must be an integer >= {minimum}")
    return int(value)


def parse_shadow_prediction(record: Mapping[str, Any]) -> ShadowPrediction:
    """Validate and normalize one runner step record for exact replay."""
    if record.get("shadow_full_depth_enabled") is not True:
        raise ShadowTraceValidationError("shadow_full_depth_enabled must be true")
    if record.get("shadow_trace_complete") is not True:
        raise ShadowTraceValidationError("shadow_trace_complete must be true")
    if record.get("shadow_error") is not None:
        raise ShadowTraceValidationError("shadow_error must be null")
    if record.get("numerical_retry_attempted") is True:
        raise ShadowTraceValidationError("shadow calibration cannot contain numerical retries")

    task_id = str(record.get("task_id"))
    if task_id == "None":
        raise ShadowTraceValidationError("task_id is required")
    episode_id = _positive_int(record.get("episode_id"), "episode_id", minimum=0)
    prediction_value = record.get("prediction_step", record.get("action_prediction_index"))
    prediction_index = _positive_int(prediction_value, "prediction_step", minimum=0)
    baseline_k = _positive_int(record.get("K_t"), "K_t")

    raw_trace = record.get("shadow_trace")
    if not isinstance(raw_trace, list) or not raw_trace:
        raise ShadowTraceValidationError("shadow_trace must be a non-empty list")
    max_iter = record.get("max_recurrent_iteration")
    if max_iter is None:
        max_iter = record.get("max_iter", len(raw_trace))
    max_iter = _positive_int(max_iter, "max_recurrent_iteration")
    if baseline_k > max_iter:
        raise ShadowTraceValidationError("K_t cannot exceed max_recurrent_iteration")

    action_threshold = record.get("action_mse_threshold", record.get("threshold"))
    action_threshold = _finite_float(action_threshold, "action_mse_threshold")
    if action_threshold < 0:
        raise ShadowTraceValidationError("action_mse_threshold must be non-negative")
    effective_min_iter = _positive_int(
        record.get("effective_min_iter", 2), "effective_min_iter"
    )
    latent_min_iter = _positive_int(
        record.get("latent_precheck_min_iter", 2), "latent_precheck_min_iter"
    )

    trace = []
    for expected_k, point in enumerate(raw_trace, start=1):
        if not isinstance(point, Mapping):
            raise ShadowTraceValidationError(f"shadow_trace[{expected_k}] must be an object")
        k = _positive_int(point.get("k"), f"shadow_trace[{expected_k}].k")
        if k != expected_k:
            raise ShadowTraceValidationError("shadow_trace iterations must be contiguous from 1")
        expected_phase = "production" if k <= baseline_k else "shadow_tail"
        if point.get("phase") != expected_phase:
            raise ShadowTraceValidationError(
                f"shadow_trace[{k}].phase must be {expected_phase!r}"
            )
        if point.get("state_finite") is not True or point.get("output_finite") is not True:
            raise ShadowTraceValidationError(f"shadow_trace[{k}] contains non-finite tensors")
        latent_mse = _finite_float(point.get("latent_mse"), f"shadow_trace[{k}].latent_mse")
        latent_l2 = _finite_float(point.get("latent_l2"), f"shadow_trace[{k}].latent_l2")
        if latent_mse < 0 or latent_l2 < 0:
            raise ShadowTraceValidationError(f"shadow_trace[{k}] latent metrics must be non-negative")
        if k == 1:
            if point.get("action_mse") is not None or point.get("action_l2") is not None:
                raise ShadowTraceValidationError("iteration 1 cannot have an adjacent action metric")
            action_mse = action_l2 = None
        else:
            action_mse = _finite_float(
                point.get("action_mse"), f"shadow_trace[{k}].action_mse"
            )
            action_l2 = _finite_float(
                point.get("action_l2"), f"shadow_trace[{k}].action_l2"
            )
            if action_mse < 0 or action_l2 < 0:
                raise ShadowTraceValidationError(
                    f"shadow_trace[{k}] action metrics must be non-negative"
                )
        trace.append(
            TracePoint(
                k=k,
                phase=expected_phase,
                latent_mse=latent_mse,
                latent_l2=latent_l2,
                action_mse=action_mse,
                action_l2=action_l2,
            )
        )

    if len(trace) != max_iter:
        raise ShadowTraceValidationError(
            "shadow_trace must cover every iteration through max_recurrent_iteration"
        )
    snapshot = record.get("shadow_production_snapshot")
    if not isinstance(snapshot, Mapping):
        raise ShadowTraceValidationError("shadow_production_snapshot is required")
    if snapshot.get("K_t") != baseline_k or snapshot.get("terminal_iteration") != baseline_k:
        raise ShadowTraceValidationError("shadow production snapshot does not match K_t")
    cached_final = snapshot.get("cached_final_output_reused")
    if not isinstance(cached_final, bool):
        raise ShadowTraceValidationError("cached_final_output_reused must be a boolean")

    actual_origin = "ACTUAL_WARM" if record.get("warm_start_used") is True else "COLD"
    return ShadowPrediction(
        task_id=task_id,
        episode_id=episode_id,
        prediction_index=prediction_index,
        actual_origin=actual_origin,
        baseline_k=baseline_k,
        baseline_decode_calls=baseline_k + (0 if cached_final else 1),
        max_iter=max_iter,
        action_mse_threshold=action_threshold,
        effective_min_iter=effective_min_iter,
        latent_precheck_min_iter=latent_min_iter,
        trace=tuple(trace),
    )


def parse_shadow_predictions(records: Iterable[Mapping[str, Any]]) -> List[ShadowPrediction]:
    predictions = []
    keys = set()
    for index, record in enumerate(records):
        try:
            prediction = parse_shadow_prediction(record)
        except ShadowTraceValidationError as exc:
            raise ShadowTraceValidationError(f"record {index}: {exc}") from exc
        if prediction.key in keys:
            raise ShadowTraceValidationError(f"duplicate prediction key: {prediction.key}")
        keys.add(prediction.key)
        predictions.append(prediction)
    if not predictions:
        raise ShadowTraceValidationError("no shadow predictions were provided")
    return predictions


def _first_reference_convergence(prediction: ShadowPrediction) -> Optional[int]:
    for point in prediction.trace[1:]:
        if (
            point.k >= prediction.effective_min_iter
            and point.action_mse < prediction.action_mse_threshold
        ):
            return point.k
    return None


def _persistent_reference_convergence(prediction: ShadowPrediction) -> Optional[int]:
    eligible = [
        point
        for point in prediction.trace[1:]
        if point.k >= prediction.effective_min_iter
    ]
    for index, point in enumerate(eligible):
        if all(
            later.action_mse < prediction.action_mse_threshold
            for later in eligible[index:]
        ):
            return point.k
    return None


def replay_prediction(
    prediction: ShadowPrediction,
    config: SchedulerConfig,
) -> ReplayResult:
    """Replay the online origin-aware state machine on one full-Coda trace."""
    if not _is_finite_number(config.warm_threshold) or config.warm_threshold < 0:
        raise ValueError("warm_threshold must be finite and non-negative")
    if not _is_finite_number(config.cold_threshold) or config.cold_threshold < 0:
        raise ValueError("cold_threshold must be finite and non-negative")
    if isinstance(config.max_skip_iters, bool) or config.max_skip_iters < 1:
        raise ValueError("max_skip_iters must be >= 1")
    if config.confirmation_mode not in CONFIRMATION_MODES:
        raise ValueError(f"Unsupported confirmation_mode: {config.confirmation_mode}")

    active_threshold = (
        config.warm_threshold
        if prediction.actual_origin == "ACTUAL_WARM"
        else config.cold_threshold
    )
    point_by_k = {point.k: point for point in prediction.trace}
    scheduler_state = "INITIAL"
    skip_count = 0
    previous_output_iter = None
    decode_calls = 0
    backfill_calls = 0
    latent_gate_calls = 0
    action_comparisons = 0
    decoded_current = []
    decoded_calls = []
    comparison_pairs = []
    confirmed_convergence = []
    adaptive_stop = False
    stop_reason = "max_iter"
    terminal_k = prediction.max_iter
    max_iter_evaluable = False

    def decode(iteration: int, reason: str, *, backfill: bool = False):
        nonlocal decode_calls, backfill_calls
        decode_calls += 1
        backfill_calls += int(backfill)
        decoded_calls.append((iteration, reason))
        if not backfill:
            decoded_current.append(iteration)

    def compare(left: int, right: int) -> bool:
        nonlocal action_comparisons
        if right != left + 1:
            raise RuntimeError(f"non-adjacent replay comparison: ({left}, {right})")
        action_comparisons += 1
        comparison_pairs.append((left, right))
        below = point_by_k[right].action_mse < prediction.action_mse_threshold
        if below:
            confirmed_convergence.append(right)
        return below

    for k in range(1, prediction.max_iter + 1):
        terminal_k = k
        if k == prediction.max_iter:
            decode(k, "max_iter")
            if previous_output_iter == k - 1:
                compare(previous_output_iter, k)
                max_iter_evaluable = True
            scheduler_state = "MAX_ITER"
            stop_reason = "max_iter"
            break

        if k in (1, 2):
            decode(k, "forced_initial" if k == 1 else "forced_second")
            should_stop = False
            if previous_output_iter is not None:
                should_stop = compare(previous_output_iter, k)
            previous_output_iter = k
            skip_count = 0
            scheduler_state = "CONTIGUOUS"
            if should_stop and k >= prediction.effective_min_iter:
                adaptive_stop = True

        elif scheduler_state == "CONFIRM_PENDING":
            decode(k, "confirmation")
            should_stop = compare(previous_output_iter, k)
            previous_output_iter = k
            skip_count = 0
            scheduler_state = "CONTIGUOUS"
            if should_stop and k >= prediction.effective_min_iter:
                adaptive_stop = True

        else:
            latent_gate_calls += 1
            latent_trigger = (
                k >= prediction.latent_precheck_min_iter
                and point_by_k[k].latent_mse <= active_threshold
            )
            if scheduler_state == "CONTIGUOUS":
                if latent_trigger:
                    decode(k, "latent_trigger")
                    should_stop = compare(previous_output_iter, k)
                    previous_output_iter = k
                    skip_count = 0
                    if should_stop and k >= prediction.effective_min_iter:
                        adaptive_stop = True
                else:
                    skip_count = 1
                    scheduler_state = "GAPPED"
            elif scheduler_state == "GAPPED":
                force_reason = None
                if latent_trigger:
                    force_reason = "latent_trigger"
                elif skip_count >= config.max_skip_iters:
                    force_reason = "max_skip_reached"

                if force_reason is None:
                    skip_count += 1
                elif config.confirmation_mode == "backfill_pair":
                    decode(k - 1, "backfill_previous", backfill=True)
                    decode(k, force_reason)
                    should_stop = compare(k - 1, k)
                    previous_output_iter = k
                    skip_count = 0
                    scheduler_state = "CONTIGUOUS"
                    if should_stop and k >= prediction.effective_min_iter:
                        adaptive_stop = True
                else:
                    decode(k, force_reason)
                    previous_output_iter = k
                    skip_count = 0
                    scheduler_state = "CONFIRM_PENDING"
            else:
                raise RuntimeError(f"Unsupported scheduler state: {scheduler_state}")

        if adaptive_stop:
            stop_reason = "adjacent_action_mse"
            break

    reference_k = _first_reference_convergence(prediction)
    persistent_k = _persistent_reference_convergence(prediction)
    captured = None
    if reference_k is not None:
        captured = any(k >= reference_k for k in confirmed_convergence)
    persistent_recovered = None
    if persistent_k is not None:
        persistent_recovered = any(k >= persistent_k for k in confirmed_convergence)
    stopped_before_persistent = bool(
        adaptive_stop and (persistent_k is None or terminal_k < persistent_k)
    )
    false_convergence = bool(
        adaptive_stop
        and point_by_k[terminal_k].action_mse >= prediction.action_mse_threshold
    )
    return ReplayResult(
        terminal_k=terminal_k,
        adaptive_stop=adaptive_stop,
        stop_reason=stop_reason,
        decode_calls=decode_calls,
        backfill_decode_calls=backfill_calls,
        latent_gate_calls=latent_gate_calls,
        action_comparisons=action_comparisons,
        finite_checks=terminal_k + decode_calls,
        decoded_current_iterations=tuple(decoded_current),
        decoded_calls=tuple(decoded_calls),
        comparison_pairs=tuple(comparison_pairs),
        confirmed_convergence_iterations=tuple(confirmed_convergence),
        reference_first_convergence_k=reference_k,
        reference_persistent_convergence_k=persistent_k,
        captured_reference_convergence=captured,
        recovered_persistent_convergence=persistent_recovered,
        stopped_before_persistent_tail=stopped_before_persistent,
        false_convergence=false_convergence,
        final_convergence_evaluable=bool(
            comparison_pairs and comparison_pairs[-1][1] == terminal_k
        ),
        max_iteration_convergence_evaluable=max_iter_evaluable,
    )


def predicted_latency_ms(
    *,
    recurrent_calls: int,
    decode_calls: int,
    latent_gate_calls: int,
    action_comparisons: int,
    finite_checks: int,
    costs: CostModel,
) -> float:
    return (
        recurrent_calls * costs.recurrent_ms
        + decode_calls * costs.decode_ms
        + latent_gate_calls * costs.latent_gate_ms
        + action_comparisons * costs.action_compare_ms
        + finite_checks * costs.finite_check_ms
    )


def evaluate_replay(
    prediction: ShadowPrediction,
    config: SchedulerConfig,
    costs: CostModel,
) -> EvaluatedReplay:
    result = replay_prediction(prediction, config)
    baseline_latency = predicted_latency_ms(
        recurrent_calls=prediction.baseline_k,
        decode_calls=prediction.baseline_decode_calls,
        latent_gate_calls=0,
        action_comparisons=max(0, prediction.baseline_k - 1),
        finite_checks=0,
        costs=costs,
    )
    candidate_latency = predicted_latency_ms(
        recurrent_calls=result.terminal_k,
        decode_calls=result.decode_calls,
        latent_gate_calls=result.latent_gate_calls,
        action_comparisons=result.action_comparisons,
        finite_checks=result.finite_checks,
        costs=costs,
    )
    return EvaluatedReplay(
        prediction=prediction,
        result=result,
        baseline_latency_ms=baseline_latency,
        candidate_latency_ms=candidate_latency,
    )


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * probability
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def eligible_warm_latent_values(predictions: Sequence[ShadowPrediction]) -> List[float]:
    values = []
    for prediction in predictions:
        if prediction.actual_origin != "ACTUAL_WARM":
            continue
        first_eligible = max(3, prediction.latent_precheck_min_iter)
        for point in prediction.trace:
            if first_eligible <= point.k < prediction.max_iter:
                values.append(point.latent_mse)
    return values


def make_candidate_families(
    *,
    fixed_thresholds: Sequence[float] = FIXED_WARM_THRESHOLDS,
    quantiles: Sequence[float] = WARM_QUANTILES,
    max_skip_values: Sequence[int] = MAX_SKIP_VALUES,
    confirmation_modes: Sequence[str] = CONFIRMATION_MODES,
) -> List[SchedulerFamily]:
    threshold_families = [ThresholdFamily("fixed", float(value)) for value in fixed_thresholds]
    threshold_families.extend(
        ThresholdFamily("quantile", float(value)) for value in quantiles
    )
    families = []
    for threshold in threshold_families:
        if threshold.kind == "fixed" and (
            not _is_finite_number(threshold.value) or threshold.value < 0
        ):
            raise ValueError("fixed thresholds must be finite and non-negative")
        if threshold.kind == "quantile" and not 0 <= threshold.value <= 1:
            raise ValueError("quantiles must be between 0 and 1")
        for max_skip in max_skip_values:
            if isinstance(max_skip, bool) or not isinstance(max_skip, int) or max_skip < 1:
                raise ValueError("max_skip values must be integers >= 1")
            for confirmation in confirmation_modes:
                if confirmation not in CONFIRMATION_MODES:
                    raise ValueError(f"Unsupported confirmation mode: {confirmation}")
                families.append(
                    SchedulerFamily(
                        threshold=threshold,
                        max_skip_iters=max_skip,
                        confirmation_mode=confirmation,
                    )
                )
    return families


def parse_fold_manifest(
    manifest: Mapping[str, Any],
    task_ids: Iterable[str],
) -> Dict[str, int]:
    raw_folds = manifest.get("folds")
    if not isinstance(raw_folds, list) or len(raw_folds) < 2:
        raise ValueError("fold manifest must contain at least two folds")
    assignment = {}
    seen_fold_ids = set()
    for default_id, fold in enumerate(raw_folds):
        if not isinstance(fold, Mapping):
            raise ValueError("each fold must be an object")
        fold_id = fold.get("fold_id", default_id)
        if isinstance(fold_id, bool) or not isinstance(fold_id, int):
            raise ValueError("fold_id must be an integer")
        if fold_id in seen_fold_ids:
            raise ValueError(f"duplicate fold_id: {fold_id}")
        seen_fold_ids.add(fold_id)

        validation_tasks = fold.get("validation_task_ids")
        if not isinstance(validation_tasks, list) or not validation_tasks:
            raise ValueError(f"fold {fold_id} has no validation tasks")
        for task in validation_tasks:
            task = str(task)
            if task in assignment:
                raise ValueError(f"task {task} occurs in multiple folds")
            assignment[task] = fold_id
    expected = {str(task) for task in task_ids}
    assigned = set(assignment)
    if assigned != expected:
        missing = sorted(expected - assigned)
        extra = sorted(assigned - expected)
        raise ValueError(f"fold manifest task mismatch: missing={missing}, extra={extra}")
    return assignment


def _hierarchical_mean(
    evaluations: Sequence[EvaluatedReplay],
    value_fn,
    *,
    include_fn=lambda _: True,
) -> Optional[float]:
    by_task_episode: Dict[str, Dict[int, List[float]]] = {}
    for evaluation in evaluations:
        if not include_fn(evaluation):
            continue
        prediction = evaluation.prediction
        by_task_episode.setdefault(prediction.task_id, {}).setdefault(
            prediction.episode_id, []
        ).append(float(value_fn(evaluation)))
    task_values = []
    for episodes in by_task_episode.values():
        episode_values = [statistics.mean(values) for values in episodes.values() if values]
        if episode_values:
            task_values.append(statistics.mean(episode_values))
    return statistics.mean(task_values) if task_values else None


def _hierarchical_p95_delta_k(evaluations: Sequence[EvaluatedReplay]) -> float:
    by_task_episode: Dict[str, Dict[int, List[float]]] = {}
    for evaluation in evaluations:
        prediction = evaluation.prediction
        by_task_episode.setdefault(prediction.task_id, {}).setdefault(
            prediction.episode_id, []
        ).append(float(evaluation.delta_k))
    task_values = []
    for episodes in by_task_episode.values():
        episode_p95 = [percentile(values, 0.95) for values in episodes.values() if values]
        task_values.append(statistics.mean(episode_p95))
    return statistics.mean(task_values)


def aggregate_evaluations(
    evaluations: Sequence[EvaluatedReplay],
    constraints: SelectionConstraints,
) -> Dict[str, Any]:
    if not evaluations:
        raise ValueError("cannot aggregate an empty replay set")
    mean_delta_k = _hierarchical_mean(evaluations, lambda item: item.delta_k)
    p95_delta_k = _hierarchical_p95_delta_k(evaluations)
    capture_rate = _hierarchical_mean(
        evaluations,
        lambda item: float(item.result.captured_reference_convergence),
        include_fn=lambda item: item.result.captured_reference_convergence is not None,
    )
    persistent_recovery_rate = _hierarchical_mean(
        evaluations,
        lambda item: float(item.result.recovered_persistent_convergence),
        include_fn=lambda item: item.result.recovered_persistent_convergence is not None,
    )
    baseline_max_rate = _hierarchical_mean(
        evaluations,
        lambda item: item.prediction.baseline_k == item.prediction.max_iter,
    )
    candidate_max_rate = _hierarchical_mean(
        evaluations,
        lambda item: item.result.terminal_k == item.prediction.max_iter,
    )
    false_convergence_rate = _hierarchical_mean(
        evaluations, lambda item: item.result.false_convergence
    )
    stopped_before_persistent_rate = _hierarchical_mean(
        evaluations, lambda item: item.result.stopped_before_persistent_tail
    )
    mean_candidate_calls = _hierarchical_mean(
        evaluations, lambda item: item.result.decode_calls
    )
    mean_baseline_calls = _hierarchical_mean(
        evaluations, lambda item: item.prediction.baseline_decode_calls
    )
    baseline_latency = _hierarchical_mean(
        evaluations, lambda item: item.baseline_latency_ms
    )
    candidate_latency = _hierarchical_mean(
        evaluations, lambda item: item.candidate_latency_ms
    )
    latency_delta = candidate_latency - baseline_latency
    latency_improvement = (
        -latency_delta / baseline_latency if baseline_latency > 0 else None
    )
    max_rate_delta = candidate_max_rate - baseline_max_rate
    candidate_retry_rate = 0.0
    violations = []
    if capture_rate is None:
        violations.append("no_reference_convergence_events")
    elif capture_rate < constraints.min_convergence_capture:
        violations.append("convergence_capture")
    if mean_delta_k > constraints.max_mean_delta_k:
        violations.append("mean_delta_k")
    if p95_delta_k > constraints.max_p95_delta_k:
        violations.append("p95_delta_k")
    if max_rate_delta > constraints.max_max_iter_rate_delta:
        violations.append("max_iter_rate_delta")
    if false_convergence_rate > constraints.max_false_convergence_rate:
        violations.append("false_convergence_rate")
    if candidate_retry_rate > constraints.max_candidate_retry_rate:
        violations.append("candidate_retry_rate")

    return {
        "prediction_count": len(evaluations),
        "task_count": len({item.prediction.task_id for item in evaluations}),
        "episode_count": len(
            {(item.prediction.task_id, item.prediction.episode_id) for item in evaluations}
        ),
        "convergence_capture_eligible_count": sum(
            item.result.captured_reference_convergence is not None for item in evaluations
        ),
        "convergence_capture_rate": capture_rate,
        "persistent_convergence_recovery_rate": persistent_recovery_rate,
        "false_convergence_rate": false_convergence_rate,
        "stopped_before_persistent_tail_rate_diagnostic": stopped_before_persistent_rate,
        "mean_delta_k": mean_delta_k,
        "p95_delta_k_episode_task_macro": p95_delta_k,
        "baseline_max_iter_rate": baseline_max_rate,
        "candidate_max_iter_rate": candidate_max_rate,
        "max_iter_rate_delta": max_rate_delta,
        "candidate_retry_rate": candidate_retry_rate,
        "mean_baseline_decode_calls": mean_baseline_calls,
        "mean_candidate_decode_calls": mean_candidate_calls,
        "relative_decode_call_reduction": (
            (mean_baseline_calls - mean_candidate_calls) / mean_baseline_calls
            if mean_baseline_calls > 0
            else None
        ),
        "predicted_baseline_action_head_ms": baseline_latency,
        "predicted_candidate_action_head_ms": candidate_latency,
        "predicted_action_head_delta_ms": latency_delta,
        "predicted_action_head_improvement": latency_improvement,
        "passes_safety_constraints": not violations,
        "safety_violations": violations,
    }


def _family_threshold(
    family: SchedulerFamily,
    training_predictions: Sequence[ShadowPrediction],
) -> Tuple[float, Optional[int]]:
    if family.threshold.kind == "fixed":
        return family.threshold.value, None
    values = eligible_warm_latent_values(training_predictions)
    if not values:
        raise ValueError(
            f"{family.identifier}: no eligible ACTUAL_WARM transitions in training folds"
        )
    return percentile(values, family.threshold.value), len(values)


def run_task_level_oof_selection(
    predictions: Sequence[ShadowPrediction],
    fold_manifest: Mapping[str, Any],
    costs: CostModel,
    *,
    constraints: SelectionConstraints = SelectionConstraints(),
    fixed_thresholds: Sequence[float] = FIXED_WARM_THRESHOLDS,
    quantiles: Sequence[float] = WARM_QUANTILES,
    max_skip_values: Sequence[int] = MAX_SKIP_VALUES,
    confirmation_modes: Sequence[str] = CONFIRMATION_MODES,
    top_n: int = 6,
) -> Dict[str, Any]:
    """Rank scheduler families using validation-fold-only replay metrics."""
    if not predictions:
        raise ValueError("predictions cannot be empty")
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be an integer >= 1")
    task_ids = sorted({prediction.task_id for prediction in predictions})
    assignment = parse_fold_manifest(fold_manifest, task_ids)
    fold_ids = sorted(set(assignment.values()))
    families = make_candidate_families(
        fixed_thresholds=fixed_thresholds,
        quantiles=quantiles,
        max_skip_values=max_skip_values,
        confirmation_modes=confirmation_modes,
    )
    reports = []
    for family in families:
        oof_evaluations = []
        fold_thresholds = []
        for fold_id in fold_ids:
            training = [
                prediction
                for prediction in predictions
                if assignment[prediction.task_id] != fold_id
            ]
            validation = [
                prediction
                for prediction in predictions
                if assignment[prediction.task_id] == fold_id
            ]
            threshold, quantile_sample_count = _family_threshold(family, training)
            config = SchedulerConfig(
                warm_threshold=threshold,
                max_skip_iters=family.max_skip_iters,
                confirmation_mode=family.confirmation_mode,
            )
            oof_evaluations.extend(
                evaluate_replay(prediction, config, costs) for prediction in validation
            )
            fold_thresholds.append(
                {
                    "fold_id": fold_id,
                    "warm_threshold": threshold,
                    "warm_threshold_hex": float(threshold).hex(),
                    "quantile_training_transition_count": quantile_sample_count,
                    "validation_task_ids": sorted(
                        task for task, assigned_fold in assignment.items() if assigned_fold == fold_id
                    ),
                }
            )
        metrics = aggregate_evaluations(oof_evaluations, constraints)
        reports.append(
            {
                "family_id": family.identifier,
                "threshold_family": asdict(family.threshold),
                "max_skip_iters": family.max_skip_iters,
                "confirmation_mode": family.confirmation_mode,
                "fold_thresholds": fold_thresholds,
                "oof_metrics": metrics,
                "_family": family,
            }
        )

    passing = [report for report in reports if report["oof_metrics"]["passes_safety_constraints"]]
    passing.sort(
        key=lambda report: (
            report["oof_metrics"]["predicted_candidate_action_head_ms"],
            report["oof_metrics"]["mean_candidate_decode_calls"],
            report["oof_metrics"]["mean_delta_k"],
            report["family_id"],
        )
    )
    for oof_rank, report in enumerate(passing, start=1):
        report["oof_rank"] = oof_rank

    full_values = eligible_warm_latent_values(predictions)
    selected = []
    seen_configs = set()
    for oof_rank, report in enumerate(passing, start=1):
        family = report["_family"]
        if family.threshold.kind == "fixed":
            refit_threshold = family.threshold.value
            refit_count = None
        else:
            if not full_values:
                raise ValueError("no eligible ACTUAL_WARM transitions for full-data quantile refit")
            refit_threshold = percentile(full_values, family.threshold.value)
            refit_count = len(full_values)
        config = SchedulerConfig(
            warm_threshold=refit_threshold,
            max_skip_iters=family.max_skip_iters,
            confirmation_mode=family.confirmation_mode,
        )
        if config.exact_key in seen_configs:
            continue
        seen_configs.add(config.exact_key)
        selected.append(
            {
                "oof_rank": oof_rank,
                "source_family_id": report["family_id"],
                "config": {
                    **asdict(config),
                    "warm_threshold_hex": float(refit_threshold).hex(),
                },
                "refit_quantile_transition_count": refit_count,
                "oof_metrics": report["oof_metrics"],
            }
        )
        if len(selected) == top_n:
            break

    public_reports = []
    for report in reports:
        report.setdefault("oof_rank", None)
        public_reports.append({key: value for key, value in report.items() if key != "_family"})
    return {
        "schema_version": 1,
        "scope": (
            "baseline-conditioned offline pruning only; not a closed-loop performance "
            "or deployment-latency estimate"
        ),
        "aggregation": (
            "prediction metrics are averaged within episode, then task; tasks receive equal weight. "
            "p95 delta-K is computed within episode and macro-averaged by task."
        ),
        "cost_model": asdict(costs),
        "constraints": asdict(constraints),
        "fold_assignment": assignment,
        "family_grid_size": len(families),
        "passing_family_count": len(passing),
        "selected_distinct_config_count": len(selected),
        "selected_refit_configs": selected,
        "family_reports": public_reports,
    }
