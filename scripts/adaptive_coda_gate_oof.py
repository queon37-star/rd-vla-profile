"""Leakage-safe nested task-level OOF evaluation for adaptive Coda activation."""

from __future__ import annotations

import csv
import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.analyze_latent_dynamics_features import (
    EXISTING_METRICS,
    LEAKAGE_EXCLUSIONS,
    TOKEN_DYNAMICS,
    UPDATE_DYNAMICS,
    WARM_ANCHOR,
    build_analysis_dataset,
    canonical_json,
    fit_logistic,
    fit_preprocessor,
    labels_for_rows,
    predict_logistic,
    transform_features,
)
from scripts.coda_activation_oof import (
    fixed_threshold_curve,
    fit_fixed_activation_threshold,
    replay_activation_policy as replay_fixed_raw_mse_policy,
    replay_coda_every_iteration as replay_existing_coda_every_iteration,
)
from scripts.origin_aware_replay_lib import percentile


LOGGER = logging.getLogger(__name__)
SCHEMA_VERSION = 1
ORIGIN = "ACTUAL_WARM"
MAX_ITER = 32
TRIGGER_ITERATIONS = tuple(range(2, 32))
FIXED_RAW_MSE_BETA = 0.05
QUALIFICATION_CONSTRAINTS = {
    "mean_delta_K_max": 0.1,
    "p95_delta_K_max": 1.0,
    "exact_K_preservation_min": 0.95,
    "forced_trigger_rate_required": 0.0,
}

MODEL_FEATURES = {
    "iteration_only": ("iteration_index",),
    "raw_mse_logistic": ("raw_mse",),
    "iteration_raw_mse": ("iteration_index", "raw_mse"),
    "token_update_p95": ("token_update_p95",),
    "update_dynamics": UPDATE_DYNAMICS,
    "combined": (*EXISTING_METRICS, *UPDATE_DYNAMICS, *TOKEN_DYNAMICS, *WARM_ANCHOR),
}
REFERENCE_POLICIES = ("coda_every_iteration", "fixed_raw_mse_beta_0_05")
LEARNED_POLICIES = tuple(MODEL_FEATURES)

METRIC_FIELDS = (
    "baseline_total_coda_calls",
    "scheduled_total_coda_calls",
    "coda_call_reduction",
    "mean_delta_K",
    "median_delta_K",
    "p95_delta_K",
    "max_delta_K",
    "exact_K_preservation_rate",
    "delta_K_gt_0_rate",
    "mean_trigger_delay",
    "median_trigger_delay",
    "p95_trigger_delay",
    "mean_early_trigger_distance",
    "forced_trigger_rate",
    "max_iteration_rate",
    "mean_coda_calls_per_prediction",
)


class AdaptiveCodaGateError(ValueError):
    """Raised when a nested OOF or replay invariant is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AdaptiveCodaGateError(message)


def activation_target(baseline_k: int) -> int:
    return max(2, int(baseline_k) - 1)


def difficulty_for_k(baseline_k: int) -> str:
    if baseline_k <= 4:
        return "easy"
    if baseline_k <= 7:
        return "medium"
    return "hard"


def recorded_action_mse_by_iteration(
    record: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    """Resolve authoritative production values and FP32 shadow-tail values."""

    baseline_k = int(record["K_t"])
    production_values = record.get("iteration_mse")
    _require(
        isinstance(production_values, list) and len(production_values) == baseline_k - 1,
        "iteration_mse must contain the authoritative k=2..K_t production series",
    )
    trace = {
        int(item["iteration_index"]): item for item in record["latent_metric_trace"]
    }
    values = {}
    for iteration in range(2, 33):
        if iteration <= baseline_k:
            action_mse = float(production_values[iteration - 2])
            source = "production_iteration_mse"
        else:
            action_mse = float(trace[iteration]["adjacent_action_mse"])
            source = "shadow_fp32_adjacent_action_mse"
        _require(math.isfinite(action_mse), f"k={iteration}: non-finite action MSE")
        values[iteration] = {
            "action_mse": action_mse,
            "label": action_mse < 0.001,
            "source": source,
        }
    return values


def parse_gate_predictions(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build ACTUAL_WARM full-depth replay records from validated runner rows."""

    _, analysis_rows = build_analysis_dataset(records)
    by_key_iteration = {
        (
            int(row["task_id"]),
            int(row["episode_id"]),
            int(row["prediction_id"]),
            int(row["iteration_index"]),
        ): row
        for row in analysis_rows
    }
    predictions = []
    seen = set()
    for record in records:
        if record["actual_origin"] != ORIGIN:
            continue
        task_id = str(int(record["task_id"]))
        episode_id = int(record["episode_id"])
        prediction_id = int(record["prediction_step"])
        key = (task_id, episode_id, prediction_id)
        _require(key not in seen, f"duplicate gate prediction: {key}")
        seen.add(key)
        baseline_k = int(record["K_t"])
        action_values = recorded_action_mse_by_iteration(record)
        transitions = []
        for item in record["latent_metric_trace"]:
            iteration = int(item["iteration_index"])
            analysis_row = by_key_iteration[
                (int(task_id), episode_id, prediction_id, iteration)
            ]
            action_mse = action_values[iteration]["action_mse"]
            label = action_values[iteration]["label"]
            transitions.append(
                {
                    **analysis_row,
                    "k": iteration,
                    "raw_mse": float(item["raw_mse"]),
                    "action_mse": action_mse,
                    "label": label,
                    "action_mse_source": action_values[iteration]["source"],
                }
            )
        _require(
            [item["k"] for item in transitions] == list(range(2, 33)),
            f"{key}: gate trace must cover exactly k=2..32",
        )
        predictions.append(
            {
                "key": key,
                "task_id": task_id,
                "episode_id": episode_id,
                "prediction_id": prediction_id,
                "actual_origin": ORIGIN,
                "baseline_k": baseline_k,
                "activation_target": activation_target(baseline_k),
                "difficulty": difficulty_for_k(baseline_k),
                "max_iter": MAX_ITER,
                "transitions": transitions,
            }
        )
    predictions.sort(key=lambda item: (int(item["task_id"]), item["episode_id"], item["prediction_id"]))
    _require(bool(predictions), "no ACTUAL_WARM predictions found")
    return predictions


def primary_training_rows(
    predictions: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    rows = [
        transition
        for prediction in predictions
        for transition in prediction["transitions"]
        if int(transition["k"]) <= int(prediction["baseline_k"])
    ]
    _require(rows, "primary model-fitting rows are empty")
    _require(
        all(int(row["k"]) <= int(row["baseline_k"]) for row in rows),
        "post-terminal row entered model fitting",
    )
    return rows


def full_scoring_rows(
    predictions: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    rows = [
        transition
        for prediction in predictions
        for transition in prediction["transitions"]
        if int(transition["k"]) in TRIGGER_ITERATIONS
    ]
    _require(
        len(rows) == len(predictions) * len(TRIGGER_ITERATIONS),
        "full scoring rows must contain k=2..31 for every prediction",
    )
    return rows


def split_predictions_by_outer_fold(
    predictions: Sequence[Mapping[str, Any]],
    assignment: Mapping[str, int],
) -> list[dict[str, Any]]:
    task_ids = {str(item["task_id"]) for item in predictions}
    _require(task_ids == set(assignment), "prediction/fold task IDs differ")
    folds = []
    for fold_id in sorted(set(assignment.values())):
        training = [
            item for item in predictions if assignment[str(item["task_id"])] != fold_id
        ]
        held_out = [
            item for item in predictions if assignment[str(item["task_id"])] == fold_id
        ]
        training_tasks = sorted({str(item["task_id"]) for item in training}, key=int)
        held_out_tasks = sorted({str(item["task_id"]) for item in held_out}, key=int)
        _require(training and held_out, f"outer fold {fold_id}: empty split")
        _require(
            set(training_tasks).isdisjoint(held_out_tasks),
            f"outer fold {fold_id}: task leakage",
        )
        folds.append(
            {
                "fold_id": fold_id,
                "training_task_ids": training_tasks,
                "held_out_task_ids": held_out_tasks,
                "training_predictions": training,
                "held_out_predictions": held_out,
            }
        )
    _require(len(folds) == 5, "nested evaluator requires the existing five outer folds")
    return folds


def inner_leave_one_task_out_splits(
    outer_training: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    tasks = sorted({str(item["task_id"]) for item in outer_training}, key=int)
    _require(len(tasks) >= 2, "inner cross-fitting requires at least two tasks")
    splits = []
    for omitted_task in tasks:
        training = [item for item in outer_training if str(item["task_id"]) != omitted_task]
        held_out = [item for item in outer_training if str(item["task_id"]) == omitted_task]
        training_tasks = sorted({str(item["task_id"]) for item in training}, key=int)
        _require(held_out and training, f"inner task {omitted_task}: empty split")
        _require(omitted_task not in training_tasks, f"inner task {omitted_task}: task leakage")
        splits.append(
            {
                "omitted_task_id": omitted_task,
                "training_task_ids": training_tasks,
                "training_predictions": training,
                "held_out_predictions": held_out,
            }
        )
    return splits


def fit_gate_model(
    predictions: Sequence[Mapping[str, Any]], model_name: str
) -> dict[str, Any]:
    _require(model_name in MODEL_FEATURES, f"unknown gate model: {model_name}")
    features = MODEL_FEATURES[model_name]
    _require(not (set(features) & LEAKAGE_EXCLUSIONS), "leakage field in gate features")
    rows = primary_training_rows(predictions)
    preprocessor = fit_preprocessor(rows, features)
    matrix = transform_features(rows, preprocessor)
    labels = labels_for_rows(rows)
    model = fit_logistic(matrix, labels)
    return {
        "model_name": model_name,
        "feature_names": list(features),
        "training_task_ids": sorted({str(item["task_id"]) for item in predictions}, key=int),
        "training_prediction_count": len(predictions),
        "training_transition_count": len(rows),
        "maximum_training_iteration_by_prediction": "baseline_k",
        "preprocessor": preprocessor,
        "model": model,
    }


def score_gate_predictions(
    predictions: Sequence[Mapping[str, Any]], fitted: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows = full_scoring_rows(predictions)
    matrix = transform_features(rows, fitted["preprocessor"])
    scores = predict_logistic(fitted["model"], matrix)
    _require(np.isfinite(scores).all(), "non-finite learned gate score")
    by_key: dict[tuple[Any, ...], dict[int, float]] = defaultdict(dict)
    for row, score in zip(rows, scores):
        key = (str(row["task_id"]), int(row["episode_id"]), int(row["prediction_id"]))
        by_key[key][int(row["k"])] = float(score)
    result = []
    for prediction in predictions:
        scores_by_k = by_key[tuple(prediction["key"])]
        _require(
            tuple(sorted(scores_by_k)) == TRIGGER_ITERATIONS,
            f"{prediction['key']}: learned scores must cover k=2..31",
        )
        result.append({"prediction": prediction, "scores_by_k": scores_by_k})
    return result


def replay_trigger(
    prediction: Mapping[str, Any], trigger_k: int, *, forced_trigger: bool
) -> dict[str, Any]:
    _require(trigger_k in TRIGGER_ITERATIONS, "trigger iteration must be in 2..31")
    transitions = {int(item["k"]): item for item in prediction["transitions"]}
    first_check = 2 if trigger_k == 2 else trigger_k + 1
    stopping_k = next(
        (
            iteration
            for iteration in range(first_check, int(prediction["max_iter"]) + 1)
            if bool(transitions[iteration]["label"])
        ),
        None,
    )
    terminal_k = int(stopping_k) if stopping_k is not None else int(prediction["max_iter"])
    executed_coda_iterations = [1, *range(trigger_k, terminal_k + 1)]
    scheduled_calls = 1 + terminal_k - trigger_k + 1
    _require(
        len(executed_coda_iterations) == scheduled_calls,
        f"{prediction['key']}: inconsistent scheduled Coda calls",
    )
    baseline_k = int(prediction["baseline_k"])
    target = int(prediction["activation_target"])
    return {
        "key": list(prediction["key"]),
        "task_id": str(prediction["task_id"]),
        "episode_id": int(prediction["episode_id"]),
        "prediction_id": int(prediction["prediction_id"]),
        "difficulty": prediction["difficulty"],
        "baseline_k": baseline_k,
        "activation_target": target,
        "trigger_k": int(trigger_k),
        "forced_trigger": bool(forced_trigger),
        "first_action_mse_check_k": first_check,
        "terminal_k": terminal_k,
        "stop_reason": "action_mse" if stopping_k is not None else "max_iter",
        "delta_k": terminal_k - baseline_k,
        "exact_k_preserved": terminal_k == baseline_k,
        "delta_k_gt_0": terminal_k > baseline_k,
        "trigger_delay": max(0, trigger_k - target),
        "early_trigger_distance": max(0, target - trigger_k),
        "max_iteration": terminal_k == int(prediction["max_iter"]),
        "baseline_coda_calls": baseline_k,
        "scheduled_coda_calls": scheduled_calls,
        "executed_coda_iterations": executed_coda_iterations,
        "executed_action_mse_checks": list(range(first_check, terminal_k + 1)),
    }


def replay_scored_predictions(
    scored_predictions: Sequence[Mapping[str, Any]], threshold: float
) -> list[dict[str, Any]]:
    _require(math.isfinite(float(threshold)), "gate threshold must be finite")
    replays = []
    for scored in scored_predictions:
        crossing = next(
            (
                iteration
                for iteration in TRIGGER_ITERATIONS
                if float(scored["scores_by_k"][iteration]) >= float(threshold)
            ),
            None,
        )
        forced = crossing is None
        trigger = 31 if forced else int(crossing)
        replays.append(replay_trigger(scored["prediction"], trigger, forced_trigger=forced))
    return replays


def replay_coda_every_iteration(
    predictions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    existing = replay_existing_coda_every_iteration(predictions)
    by_key = {tuple(item["key"]): item for item in predictions}
    replays = []
    for replay in existing:
        prediction = by_key[tuple(replay["key"])]
        replays.append(
            {
                **replay,
                "key": list(replay["key"]),
                "episode_id": int(prediction["episode_id"]),
                "prediction_id": int(prediction["prediction_id"]),
                "difficulty": prediction["difficulty"],
            }
        )
    return replays


def replay_fixed_reference(
    predictions: Sequence[Mapping[str, Any]], threshold_selection: Mapping[str, Any]
) -> list[dict[str, Any]]:
    existing = replay_fixed_raw_mse_policy(
        predictions, fixed_threshold_curve(threshold_selection)
    )
    by_key = {tuple(item["key"]): item for item in predictions}
    return [
        {
            **replay,
            "key": list(replay["key"]),
            "episode_id": int(by_key[tuple(replay["key"])]["episode_id"]),
            "prediction_id": int(by_key[tuple(replay["key"])]["prediction_id"]),
            "difficulty": by_key[tuple(replay["key"])]["difficulty"],
        }
        for replay in existing
    ]


def _aggregate_flat(replays: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(bool(replays), "cannot aggregate empty replays")
    baseline_calls = int(sum(int(item["baseline_coda_calls"]) for item in replays))
    scheduled_calls = int(sum(int(item["scheduled_coda_calls"]) for item in replays))
    delta = [int(item["delta_k"]) for item in replays]
    delays = [int(item["trigger_delay"]) for item in replays]
    return {
        "prediction_count": len(replays),
        "baseline_total_coda_calls": baseline_calls,
        "scheduled_total_coda_calls": scheduled_calls,
        "coda_call_reduction": (baseline_calls - scheduled_calls) / baseline_calls,
        "mean_delta_K": float(np.mean(delta)),
        "median_delta_K": percentile(delta, 0.5),
        "p95_delta_K": percentile(delta, 0.95),
        "max_delta_K": max(delta),
        "exact_K_preservation_rate": float(np.mean([item["exact_k_preserved"] for item in replays])),
        "delta_K_gt_0_rate": float(np.mean([item["delta_k_gt_0"] for item in replays])),
        "mean_trigger_delay": float(np.mean(delays)),
        "median_trigger_delay": percentile(delays, 0.5),
        "p95_trigger_delay": percentile(delays, 0.95),
        "mean_early_trigger_distance": float(
            np.mean([item["early_trigger_distance"] for item in replays])
        ),
        "forced_trigger_rate": float(np.mean([item["forced_trigger"] for item in replays])),
        "max_iteration_rate": float(np.mean([item["max_iteration"] for item in replays])),
        "mean_coda_calls_per_prediction": scheduled_calls / len(replays),
    }


def aggregate_replays(replays: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = _aggregate_flat(replays)
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for replay in replays:
        by_task[str(replay["task_id"])].append(replay)
    per_task = {
        task: _aggregate_flat(items)
        for task, items in sorted(by_task.items(), key=lambda item: int(item[0]))
    }
    result["task_macro"] = {
        field: float(np.mean([metrics[field] for metrics in per_task.values()]))
        for field in METRIC_FIELDS
        if field not in {"baseline_total_coda_calls", "scheduled_total_coda_calls"}
    }
    result["task_macro"]["task_count"] = len(per_task)
    result["per_task"] = per_task
    return result


def aggregate_by_difficulty(
    replays: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        difficulty: _aggregate_flat(
            [item for item in replays if item["difficulty"] == difficulty]
        )
        for difficulty in ("easy", "medium", "hard")
        if any(item["difficulty"] == difficulty for item in replays)
    }


class _ReplayAccumulator:
    """Incrementally maintain exact aggregate replay metrics during a score sweep."""

    def __init__(self, replays: Sequence[Mapping[str, Any]]):
        self.count = len(replays)
        self.baseline_calls = int(sum(item["baseline_coda_calls"] for item in replays))
        self.scheduled_calls = 0
        self.delta_sum = 0
        self.delay_sum = 0
        self.early_sum = 0
        self.exact_count = 0
        self.delta_positive_count = 0
        self.forced_count = 0
        self.max_iteration_count = 0
        self.delta_histogram: Counter[int] = Counter()
        self.delay_histogram: Counter[int] = Counter()
        for replay in replays:
            self.add(replay)

    def add(self, replay: Mapping[str, Any], sign: int = 1) -> None:
        delta = int(replay["delta_k"])
        delay = int(replay["trigger_delay"])
        self.scheduled_calls += sign * int(replay["scheduled_coda_calls"])
        self.delta_sum += sign * delta
        self.delay_sum += sign * delay
        self.early_sum += sign * int(replay["early_trigger_distance"])
        self.exact_count += sign * int(bool(replay["exact_k_preserved"]))
        self.delta_positive_count += sign * int(bool(replay["delta_k_gt_0"]))
        self.forced_count += sign * int(bool(replay["forced_trigger"]))
        self.max_iteration_count += sign * int(bool(replay["max_iteration"]))
        self.delta_histogram[delta] += sign
        self.delay_histogram[delay] += sign
        if self.delta_histogram[delta] == 0:
            del self.delta_histogram[delta]
        if self.delay_histogram[delay] == 0:
            del self.delay_histogram[delay]

    def replace(self, previous: Mapping[str, Any], current: Mapping[str, Any]) -> None:
        self.add(previous, -1)
        self.add(current, 1)

    @staticmethod
    def _histogram_percentile(histogram: Mapping[int, int], probability: float) -> float:
        total = sum(histogram.values())
        _require(total > 0, "empty incremental percentile histogram")
        rank = (total - 1) * probability
        lower = int(math.floor(rank))
        upper = int(math.ceil(rank))

        def value_at(index: int) -> int:
            seen = 0
            for value, count in sorted(histogram.items()):
                if seen + count > index:
                    return value
                seen += count
            raise AdaptiveCodaGateError("incremental percentile index out of range")

        lower_value = value_at(lower)
        upper_value = value_at(upper)
        weight = rank - lower
        return lower_value * (1.0 - weight) + upper_value * weight

    def metrics(self) -> dict[str, Any]:
        _require(self.count > 0 and self.baseline_calls > 0, "invalid sweep accumulator")
        return {
            "prediction_count": self.count,
            "baseline_total_coda_calls": self.baseline_calls,
            "scheduled_total_coda_calls": self.scheduled_calls,
            "coda_call_reduction": (
                self.baseline_calls - self.scheduled_calls
            )
            / self.baseline_calls,
            "mean_delta_K": self.delta_sum / self.count,
            "median_delta_K": self._histogram_percentile(self.delta_histogram, 0.5),
            "p95_delta_K": self._histogram_percentile(self.delta_histogram, 0.95),
            "max_delta_K": max(self.delta_histogram),
            "exact_K_preservation_rate": self.exact_count / self.count,
            "delta_K_gt_0_rate": self.delta_positive_count / self.count,
            "mean_trigger_delay": self.delay_sum / self.count,
            "median_trigger_delay": self._histogram_percentile(self.delay_histogram, 0.5),
            "p95_trigger_delay": self._histogram_percentile(self.delay_histogram, 0.95),
            "mean_early_trigger_distance": self.early_sum / self.count,
            "forced_trigger_rate": self.forced_count / self.count,
            "max_iteration_rate": self.max_iteration_count / self.count,
            "mean_coda_calls_per_prediction": self.scheduled_calls / self.count,
        }


def _qualify_metrics(
    metrics: Mapping[str, Any], baseline_metrics: Mapping[str, Any]
) -> dict[str, Any]:
    checks = {
        "mean_delta_K": metrics["mean_delta_K"] <= QUALIFICATION_CONSTRAINTS["mean_delta_K_max"],
        "p95_delta_K": metrics["p95_delta_K"] <= QUALIFICATION_CONSTRAINTS["p95_delta_K_max"],
        "exact_K_preservation": (
            metrics["exact_K_preservation_rate"]
            >= QUALIFICATION_CONSTRAINTS["exact_K_preservation_min"]
        ),
        "forced_trigger_rate": (
            metrics["forced_trigger_rate"]
            == QUALIFICATION_CONSTRAINTS["forced_trigger_rate_required"]
        ),
        "max_iteration_rate": (
            metrics["max_iteration_rate"] <= baseline_metrics["max_iteration_rate"]
        ),
        "finite_and_evaluable": all(
            isinstance(metrics[field], (int, float)) and math.isfinite(float(metrics[field]))
            for field in METRIC_FIELDS
        ),
    }
    return {"qualified": all(checks.values()), "checks": checks}


def _schedule_signature(replays: Sequence[Mapping[str, Any]]) -> tuple[tuple[int, bool], ...]:
    return tuple((int(item["trigger_k"]), bool(item["forced_trigger"])) for item in replays)


def exact_threshold_sweep_bruteforce(
    scored_predictions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    scores = [
        float(score)
        for item in scored_predictions
        for score in item["scores_by_k"].values()
    ]
    _require(scores and all(math.isfinite(value) for value in scores), "invalid sweep scores")
    candidates = [
        float(np.nextafter(max(scores), math.inf)),
        *sorted(set(scores), reverse=True),
        float(np.nextafter(min(scores), -math.inf)),
    ]
    by_signature = {}
    for threshold in candidates:
        replays = replay_scored_predictions(scored_predictions, threshold)
        signature = _schedule_signature(replays)
        by_signature[signature] = {
            "threshold": threshold,
            "threshold_hex": threshold.hex(),
            "metrics": _aggregate_flat(replays),
            "signature": signature,
        }
    return list(by_signature.values())


def exact_threshold_sweep(
    scored_predictions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Evaluate every distinct crossing schedule with an incremental event sweep."""

    _require(bool(scored_predictions), "threshold sweep requires scored predictions")
    events: dict[float, list[tuple[int, int]]] = defaultdict(list)
    all_scores = []
    for prediction_index, scored in enumerate(scored_predictions):
        for iteration in TRIGGER_ITERATIONS:
            score = float(scored["scores_by_k"][iteration])
            _require(math.isfinite(score), "non-finite threshold sweep score")
            events[score].append((prediction_index, iteration))
            all_scores.append(score)

    trigger = [31] * len(scored_predictions)
    forced = [True] * len(scored_predictions)
    current_replays = [
        replay_trigger(item["prediction"], 31, forced_trigger=True)
        for item in scored_predictions
    ]
    accumulator = _ReplayAccumulator(current_replays)
    high_boundary = float(np.nextafter(max(all_scores), math.inf))
    snapshots = [
        {
            "threshold": high_boundary,
            "threshold_hex": high_boundary.hex(),
            "candidate_type": "no_natural_trigger_boundary",
            "metrics": accumulator.metrics(),
            "signature": _schedule_signature(current_replays),
        }
    ]

    for score in sorted(events, reverse=True):
        updates: dict[int, int] = {}
        for prediction_index, iteration in events[score]:
            updates[prediction_index] = min(
                updates.get(prediction_index, iteration), iteration
            )
        changed = False
        for prediction_index, event_iteration in sorted(updates.items()):
            new_trigger = min(trigger[prediction_index], event_iteration)
            if forced[prediction_index] or new_trigger < trigger[prediction_index]:
                previous = current_replays[prediction_index]
                current = replay_trigger(
                    scored_predictions[prediction_index]["prediction"],
                    new_trigger,
                    forced_trigger=False,
                )
                accumulator.replace(previous, current)
                current_replays[prediction_index] = current
                trigger[prediction_index] = new_trigger
                forced[prediction_index] = False
                changed = True
        if changed:
            snapshots.append(
                {
                    "threshold": float(score),
                    "threshold_hex": float(score).hex(),
                    "candidate_type": "score_event",
                    "metrics": accumulator.metrics(),
                    "signature": _schedule_signature(current_replays),
                }
            )
        else:
            snapshots[-1]["threshold"] = float(score)
            snapshots[-1]["threshold_hex"] = float(score).hex()
            snapshots[-1]["candidate_type"] = "score_interval_lower_candidate"

    low_boundary = float(np.nextafter(min(all_scores), -math.inf))
    _require(all(value == 2 for value in trigger), "all-trigger boundary did not reach k=2")
    _require(not any(forced), "all-trigger boundary retained a forced trigger")
    snapshots[-1]["threshold"] = low_boundary
    snapshots[-1]["threshold_hex"] = low_boundary.hex()
    snapshots[-1]["candidate_type"] = "all_trigger_at_k2_boundary"
    safe_metrics = snapshots[-1]["metrics"]
    _require(safe_metrics["mean_delta_K"] == 0.0, "safe all-trigger candidate changed K")
    _require(safe_metrics["coda_call_reduction"] == 0.0, "safe all-trigger candidate changed calls")
    return snapshots


def qualify_and_select_threshold(
    sweep: Sequence[Mapping[str, Any]],
    baseline_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    evaluated = []
    for index, item in enumerate(sweep):
        qualification = _qualify_metrics(item["metrics"], baseline_metrics)
        evaluated.append(
            {
                **item,
                "candidate_index": index,
                "qualification": qualification,
            }
        )
    qualifying = [item for item in evaluated if item["qualification"]["qualified"]]
    _require(qualifying, "safe all-trigger candidate failed inner qualification")
    selected = max(
        qualifying,
        key=lambda item: (
            item["metrics"]["coda_call_reduction"],
            -item["metrics"]["mean_delta_K"],
            -item["metrics"]["p95_delta_K"],
            item["metrics"]["exact_K_preservation_rate"],
            -item["threshold"],
        ),
    )
    for item in evaluated:
        item["selected"] = item is selected
    return {
        "selection_status": "qualified_schedule_selected",
        "selection_rule": (
            "maximize Coda-call reduction; then lower mean delta K, lower p95 delta K, "
            "higher exact-K preservation, and lower q"
        ),
        "qualification_constraints": dict(QUALIFICATION_CONSTRAINTS),
        "candidate_count": len(evaluated),
        "qualifying_candidate_count": len(qualifying),
        "selected_threshold": float(selected["threshold"]),
        "selected_threshold_hex": selected["threshold_hex"],
        "selected_inner_metrics": selected["metrics"],
        "selected_qualification": selected["qualification"],
        "sweep": evaluated,
    }


def fit_nested_outer_candidate(
    outer_training: Sequence[Mapping[str, Any]], model_name: str
) -> dict[str, Any]:
    inner_scored = []
    inner_models = []
    inner_audit = []
    for split in inner_leave_one_task_out_splits(outer_training):
        fitted = fit_gate_model(split["training_predictions"], model_name)
        scored = score_gate_predictions(split["held_out_predictions"], fitted)
        inner_scored.extend(scored)
        inner_models.append(
            {
                "omitted_task_id": split["omitted_task_id"],
                "training_task_ids": split["training_task_ids"],
                "fitted": fitted,
            }
        )
        inner_audit.append(
            {
                "omitted_task_id": split["omitted_task_id"],
                "training_task_ids": split["training_task_ids"],
                "task_overlap_count": 0,
                "training_prediction_count": len(split["training_predictions"]),
                "held_out_prediction_count": len(split["held_out_predictions"]),
            }
        )
    baseline_metrics = _aggregate_flat(replay_coda_every_iteration(outer_training))
    selection = qualify_and_select_threshold(exact_threshold_sweep(inner_scored), baseline_metrics)
    outer_refit = fit_gate_model(outer_training, model_name)
    return {
        "model_name": model_name,
        "inner_cross_fitting": {
            "policy": "leave_one_outer_training_task_out",
            "leakage_audit": {"passed": True, "splits": inner_audit},
            "models": inner_models,
            "scored_prediction_count": len(inner_scored),
        },
        "threshold_selection": selection,
        "outer_training_refit": outer_refit,
    }


def _enrich_policy_replays(
    replays: Sequence[Mapping[str, Any]], *, policy: str, outer_fold: int
) -> list[dict[str, Any]]:
    return [
        {
            **replay,
            "policy": policy,
            "outer_fold": outer_fold,
        }
        for replay in replays
    ]


def _comparison_to_fixed(
    learned: Mapping[str, Any],
    fixed: Mapping[str, Any],
    coda_every_iteration: Mapping[str, Any],
) -> dict[str, Any]:
    safety_checks = {
        "mean_delta_K_within_gate": (
            learned["mean_delta_K"] <= QUALIFICATION_CONSTRAINTS["mean_delta_K_max"]
        ),
        "p95_delta_K_within_gate": (
            learned["p95_delta_K"] <= QUALIFICATION_CONSTRAINTS["p95_delta_K_max"]
        ),
        "exact_K_preservation_within_gate": (
            learned["exact_K_preservation_rate"]
            >= QUALIFICATION_CONSTRAINTS["exact_K_preservation_min"]
        ),
        "forced_trigger_rate_within_gate": (
            learned["forced_trigger_rate"]
            == QUALIFICATION_CONSTRAINTS["forced_trigger_rate_required"]
        ),
        "max_iteration_rate_no_greater_than_coda_every_iteration": (
            learned["max_iteration_rate"]
            <= coda_every_iteration["max_iteration_rate"]
        ),
    }
    reduction_improved = learned["coda_call_reduction"] > fixed["coda_call_reduction"]
    return {
        "metric_deltas_learned_minus_fixed": {
            field: learned[field] - fixed[field]
            for field in METRIC_FIELDS
            if field not in {"baseline_total_coda_calls", "scheduled_total_coda_calls"}
        },
        "comparable_safety_checks": safety_checks,
        "comparable_safety": all(safety_checks.values()),
        "coda_call_reduction_improved": reduction_improved,
        "meaningfully_better": all(safety_checks.values()) and reduction_improved,
    }


def evaluate_nested_oof(
    predictions: Sequence[Mapping[str, Any]],
    assignment: Mapping[str, int],
    *,
    prior_classification_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _require(len(predictions) == 2298 or len(predictions) < 100, "unexpected ACTUAL_WARM prediction count")
    outer_folds = split_predictions_by_outer_fold(predictions, assignment)
    all_replays: dict[str, list[dict[str, Any]]] = {
        policy: [] for policy in (*REFERENCE_POLICIES, *LEARNED_POLICIES)
    }
    outer_results = []
    model_outer_folds = []
    threshold_rows = []
    leakage_folds = []

    for outer in outer_folds:
        fold_id = int(outer["fold_id"])
        LOGGER.info(
            "outer fold=%s train_tasks=%s held_out_tasks=%s",
            fold_id,
            ",".join(outer["training_task_ids"]),
            ",".join(outer["held_out_task_ids"]),
        )
        training = outer["training_predictions"]
        held_out = outer["held_out_predictions"]
        leakage_folds.append(
            {
                "fold_id": fold_id,
                "training_task_ids": outer["training_task_ids"],
                "held_out_task_ids": outer["held_out_task_ids"],
                "task_overlap_count": 0,
                "training_prediction_count": len(training),
                "held_out_prediction_count": len(held_out),
            }
        )

        baseline_replays = _enrich_policy_replays(
            replay_coda_every_iteration(held_out),
            policy="coda_every_iteration",
            outer_fold=fold_id,
        )
        all_replays["coda_every_iteration"].extend(baseline_replays)

        fixed_selection = fit_fixed_activation_threshold(training, FIXED_RAW_MSE_BETA)
        fixed_replays = _enrich_policy_replays(
            replay_fixed_reference(held_out, fixed_selection),
            policy="fixed_raw_mse_beta_0_05",
            outer_fold=fold_id,
        )
        all_replays["fixed_raw_mse_beta_0_05"].extend(fixed_replays)
        threshold_rows.append(
            {
                "outer_fold": fold_id,
                "policy": "fixed_raw_mse_beta_0_05",
                "selection_scope": "outer_training_only",
                "candidate_index": 0,
                "candidate_type": "fixed_raw_mse_beta_0_05",
                "threshold": float(fixed_selection["threshold"]),
                "threshold_hex": fixed_selection["threshold_hex"],
                "qualified": True,
                "selected": True,
                "metrics": None,
                "qualification": None,
            }
        )

        learned_fold_results = {}
        model_fold_summary = {
            "outer_fold": fold_id,
            "outer_training_task_ids": outer["training_task_ids"],
            "outer_held_out_task_ids": outer["held_out_task_ids"],
            "fixed_raw_mse_reference": fixed_selection,
            "learned_models": {},
        }
        for model_name in LEARNED_POLICIES:
            LOGGER.info("outer fold=%s model=%s inner cross-fitting", fold_id, model_name)
            nested = fit_nested_outer_candidate(training, model_name)
            selection = nested["threshold_selection"]
            threshold = float(selection["selected_threshold"])
            held_out_scored = score_gate_predictions(
                held_out, nested["outer_training_refit"]
            )
            held_out_replays = _enrich_policy_replays(
                replay_scored_predictions(held_out_scored, threshold),
                policy=model_name,
                outer_fold=fold_id,
            )
            all_replays[model_name].extend(held_out_replays)
            learned_fold_results[model_name] = {
                "selected_inner_threshold": threshold,
                "selected_inner_threshold_hex": selection["selected_threshold_hex"],
                "inner_selection_evidence": {
                    key: value
                    for key, value in selection.items()
                    if key != "sweep"
                },
                "outer_held_out_metrics": aggregate_replays(held_out_replays),
            }
            model_fold_summary["learned_models"][model_name] = {
                **nested,
                "threshold_selection": {
                    key: value
                    for key, value in nested["threshold_selection"].items()
                    if key != "sweep"
                },
            }
            for item in selection["sweep"]:
                threshold_rows.append(
                    {
                        "outer_fold": fold_id,
                        "policy": model_name,
                        "selection_scope": "inner_leave_one_task_out_cross_fitted",
                        "candidate_index": item["candidate_index"],
                        "candidate_type": item["candidate_type"],
                        "threshold": item["threshold"],
                        "threshold_hex": item["threshold_hex"],
                        "qualified": item["qualification"]["qualified"],
                        "selected": item["selected"],
                        "metrics": item["metrics"],
                        "qualification": item["qualification"],
                    }
                )
        model_outer_folds.append(model_fold_summary)
        outer_results.append(
            {
                "outer_fold": fold_id,
                "training_task_ids": outer["training_task_ids"],
                "held_out_task_ids": outer["held_out_task_ids"],
                "coda_every_iteration_metrics": aggregate_replays(baseline_replays),
                "fixed_raw_mse_reference": {
                    "beta": FIXED_RAW_MSE_BETA,
                    "threshold_selection": fixed_selection,
                    "outer_held_out_metrics": aggregate_replays(fixed_replays),
                },
                "learned_candidates": learned_fold_results,
            }
        )

    policy_metrics = {
        policy: {
            "oof_sequence_metrics": aggregate_replays(replays),
            "difficulty_metrics": aggregate_by_difficulty(replays),
        }
        for policy, replays in all_replays.items()
    }
    fixed_metrics = policy_metrics["fixed_raw_mse_beta_0_05"]["oof_sequence_metrics"]
    baseline_metrics = policy_metrics["coda_every_iteration"]["oof_sequence_metrics"]
    learned_comparisons = {
        policy: _comparison_to_fixed(
            policy_metrics[policy]["oof_sequence_metrics"],
            fixed_metrics,
            baseline_metrics,
        )
        for policy in LEARNED_POLICIES
    }
    promotion = [
        policy for policy, comparison in learned_comparisons.items() if comparison["meaningfully_better"]
    ]
    return {
        "metric_report": {
            "schema_version": SCHEMA_VERSION,
            "status": "offline_nested_oof_diagnostic_only",
            "origin": ORIGIN,
            "prediction_count": len(predictions),
            "task_ids": sorted({str(item["task_id"]) for item in predictions}, key=int),
            "activation_target_definition": "max(2, baseline_K - 1)",
            "model_fitting_window": "ACTUAL_WARM transitions with k <= baseline_K only",
            "score_replay_window": "held-out full-depth transitions k=2..31",
            "activation_policy": {
                "mandatory_coda_iteration": 1,
                "natural_trigger": "first k in 2..31 with score_k >= fold-specific q",
                "forced_trigger_iteration": 31,
                "first_action_mse_check": "k=2 when T=2, otherwise T+1",
                "after_trigger": "execute Coda every iteration and stop on first recorded true action-MSE label",
                "action_mse_sources": {
                    "k_at_or_before_baseline_K": (
                        "authoritative recorded production iteration_mse control-flow series"
                    ),
                    "k_after_baseline_K": (
                        "recorded FP32 shadow-tail latent_metric_trace adjacent_action_mse"
                    ),
                },
            },
            "nested_selection": {
                "outer_split": "existing deterministic task-level five-fold manifest",
                "inner_split": "leave one outer-training task out",
                "threshold_candidate_policy": "all distinct score crossing schedules via exact incremental event sweep",
                "qualification_constraints": dict(QUALIFICATION_CONSTRAINTS),
                "deployment_or_global_threshold_recorded": False,
            },
            "leakage_audit": {"passed": True, "outer_folds": leakage_folds},
            "reference_policies": {
                "coda_every_iteration": policy_metrics["coda_every_iteration"],
                "fixed_raw_mse_beta_0_05": policy_metrics["fixed_raw_mse_beta_0_05"],
            },
            "learned_candidates": {
                policy: {
                    **policy_metrics[policy],
                    "comparison_to_fixed_raw_mse_reference": learned_comparisons[policy],
                }
                for policy in LEARNED_POLICIES
            },
            "outer_fold_results": outer_results,
            "promotion_assessment": {
                "criterion": (
                    "greater Coda-call reduction than fixed raw-MSE beta=0.05 while satisfying the "
                    "predeclared mean/p95 delta K, exact-K, forced-trigger, and max-iteration safety gates"
                ),
                "meaningfully_better_candidates": promotion,
                "deployment_decision_made": False,
            },
            "prior_transition_classification_context": prior_classification_context,
            "classification_scheduler_distinction": (
                "Prior AUROC/AUPRC values are transition classification diagnostics only. "
                "All metrics under reference_policies and learned_candidates are exact sequence-level scheduler replays."
            ),
            "runtime_inference_modified": False,
            "runtime_defaults_modified": False,
            "raw_traces_modified": False,
            "existing_analysis_outputs_modified": False,
            "global_or_deployment_threshold_selected": False,
        },
        "model_summary": {
            "schema_version": SCHEMA_VERSION,
            "status": "fold_specific_oof_artifacts_only",
            "model_features": {name: list(features) for name, features in MODEL_FEATURES.items()},
            "preprocessing_contract": (
                "training-task-only median imputation, explicit history availability indicators, "
                "and training-task-only mean/std scaling imported unchanged from the reviewed analysis"
            ),
            "logistic_contract": (
                "deterministic class-weighted logistic implementation imported unchanged from the reviewed analysis"
            ),
            "outer_folds": model_outer_folds,
            "global_model_fitted": False,
            "global_threshold_fitted": False,
        },
        "threshold_sweeps": threshold_rows,
        "all_replays": all_replays,
    }


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _threshold_csv_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        metrics = row["metrics"] or {}
        qualification = row["qualification"] or {"checks": {}}
        output.append(
            {
                "outer_fold": row["outer_fold"],
                "policy": row["policy"],
                "selection_scope": row["selection_scope"],
                "candidate_index": row["candidate_index"],
                "candidate_type": row["candidate_type"],
                "threshold": row["threshold"],
                "threshold_hex": row["threshold_hex"],
                "qualified": row["qualified"],
                "selected": row["selected"],
                "baseline_total_coda_calls": metrics.get("baseline_total_coda_calls"),
                "scheduled_total_coda_calls": metrics.get("scheduled_total_coda_calls"),
                "coda_call_reduction": metrics.get("coda_call_reduction"),
                "mean_delta_K": metrics.get("mean_delta_K"),
                "p95_delta_K": metrics.get("p95_delta_K"),
                "exact_K_preservation_rate": metrics.get("exact_K_preservation_rate"),
                "forced_trigger_rate": metrics.get("forced_trigger_rate"),
                "max_iteration_rate": metrics.get("max_iteration_rate"),
                **{
                    f"check_{name}": value
                    for name, value in qualification.get("checks", {}).items()
                },
            }
        )
    return output


def _replay_csv_rows(
    all_replays: Mapping[str, Sequence[Mapping[str, Any]]]
) -> list[dict[str, Any]]:
    rows = []
    for policy in (*REFERENCE_POLICIES, *LEARNED_POLICIES):
        for replay in all_replays[policy]:
            rows.append(
                {
                    "policy": policy,
                    "outer_fold": replay["outer_fold"],
                    "task_id": replay["task_id"],
                    "episode_id": replay["episode_id"],
                    "prediction_id": replay["prediction_id"],
                    "difficulty": replay["difficulty"],
                    "baseline_k": replay["baseline_k"],
                    "activation_target": replay["activation_target"],
                    "trigger_k": replay["trigger_k"],
                    "forced_trigger": replay["forced_trigger"],
                    "first_action_mse_check_k": replay["first_action_mse_check_k"],
                    "terminal_k": replay["terminal_k"],
                    "stop_reason": replay["stop_reason"],
                    "delta_k": replay["delta_k"],
                    "exact_k_preserved": replay["exact_k_preserved"],
                    "delta_k_gt_0": replay["delta_k_gt_0"],
                    "trigger_delay": replay["trigger_delay"],
                    "early_trigger_distance": replay["early_trigger_distance"],
                    "max_iteration": replay["max_iteration"],
                    "baseline_coda_calls": replay["baseline_coda_calls"],
                    "scheduled_coda_calls": replay["scheduled_coda_calls"],
                    "executed_coda_iterations_json": json.dumps(
                        replay["executed_coda_iterations"], separators=(",", ":")
                    ),
                    "executed_action_mse_checks_json": json.dumps(
                        replay["executed_action_mse_checks"], separators=(",", ":")
                    ),
                }
            )
    return rows


def _group_metric_csv_rows(
    metric_report: Mapping[str, Any], *, grouping: str
) -> list[dict[str, Any]]:
    policy_sections = {
        **metric_report["reference_policies"],
        **metric_report["learned_candidates"],
    }
    rows = []
    for policy in (*REFERENCE_POLICIES, *LEARNED_POLICIES):
        metrics = policy_sections[policy]["oof_sequence_metrics"]
        groups = metrics["per_task"] if grouping == "task" else policy_sections[policy]["difficulty_metrics"]
        for group, values in groups.items():
            rows.append(
                {
                    "policy": policy,
                    f"{grouping}_id" if grouping == "task" else "difficulty": group,
                    **{field: values[field] for field in METRIC_FIELDS},
                    "prediction_count": values["prediction_count"],
                }
            )
    return rows


def write_evaluation_outputs(
    output_dir: Path,
    evaluation: Mapping[str, Any],
    *,
    inputs: Mapping[str, Any],
    overwrite: bool = False,
) -> dict[str, str]:
    filenames = (
        "metric_report.json",
        "model_summary.json",
        "threshold_sweeps.csv",
        "oof_prediction_replays.csv",
        "task_metrics.csv",
        "difficulty_metrics.csv",
        "output_hashes.json",
    )
    existing = [output_dir / name for name in filenames if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError("refusing to overwrite OOF outputs: " + ", ".join(map(str, existing)))
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_report = dict(evaluation["metric_report"])
    metric_report["inputs"] = dict(inputs)
    metric_report["outputs"] = {name: name for name in filenames}
    (output_dir / "metric_report.json").write_text(
        canonical_json(metric_report), encoding="utf-8"
    )
    model_summary = dict(evaluation["model_summary"])
    model_summary["inputs"] = dict(inputs)
    (output_dir / "model_summary.json").write_text(
        canonical_json(model_summary), encoding="utf-8"
    )

    threshold_rows = _threshold_csv_rows(evaluation["threshold_sweeps"])
    threshold_fields = (
        "outer_fold",
        "policy",
        "selection_scope",
        "candidate_index",
        "candidate_type",
        "threshold",
        "threshold_hex",
        "qualified",
        "selected",
        "baseline_total_coda_calls",
        "scheduled_total_coda_calls",
        "coda_call_reduction",
        "mean_delta_K",
        "p95_delta_K",
        "exact_K_preservation_rate",
        "forced_trigger_rate",
        "max_iteration_rate",
        "check_mean_delta_K",
        "check_p95_delta_K",
        "check_exact_K_preservation",
        "check_forced_trigger_rate",
        "check_max_iteration_rate",
        "check_finite_and_evaluable",
    )
    _write_csv(output_dir / "threshold_sweeps.csv", threshold_rows, threshold_fields)

    replay_rows = _replay_csv_rows(evaluation["all_replays"])
    replay_fields = tuple(replay_rows[0])
    _write_csv(output_dir / "oof_prediction_replays.csv", replay_rows, replay_fields)

    task_rows = _group_metric_csv_rows(metric_report, grouping="task")
    task_fields = ("policy", "task_id", *METRIC_FIELDS, "prediction_count")
    _write_csv(output_dir / "task_metrics.csv", task_rows, task_fields)
    difficulty_rows = _group_metric_csv_rows(metric_report, grouping="difficulty")
    difficulty_fields = ("policy", "difficulty", *METRIC_FIELDS, "prediction_count")
    _write_csv(
        output_dir / "difficulty_metrics.csv", difficulty_rows, difficulty_fields
    )

    import hashlib

    hashes = {}
    for name in filenames[:-1]:
        hashes[name] = hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
    hash_report = {
        "schema_version": SCHEMA_VERSION,
        "hash_algorithm": "sha256",
        "files": hashes,
    }
    (output_dir / "output_hashes.json").write_text(
        canonical_json(hash_report), encoding="utf-8"
    )
    return hashes
