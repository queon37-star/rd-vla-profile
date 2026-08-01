"""Offline-only scalar probes for adjacent-action convergence.

This module deliberately depends on recorded full-depth shadow traces.  It has
no imports from, and makes no changes to, the online action-head path.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

from scripts.origin_aware_replay_lib import (
    ShadowPrediction,
    parse_fold_manifest,
    percentile,
)


SCHEMA_VERSION = 1
LABEL_THRESHOLD = 0.001
CAPTURE_TARGET = 0.995
ORIGIN_AWARE_DECODE_REDUCTION = 0.102
TARGET_DECODE_REDUCTION = 0.20
MODEL_NAMES = (
    "latent_mse_threshold",
    "logistic_regression",
    "class_weighted_logistic_regression",
    "tiny_mlp",
)
FEATURE_NAMES = (
    "latent_mse",
    "latent_l2",
    "iteration_index",
    "normalized_iteration_index",
    "prev1_latent_mse",
    "prev1_latent_l2",
    "prev2_latent_mse",
    "prev2_latent_l2",
    "latent_mse_slope1",
    "latent_l2_slope1",
    "latent_mse_ratio1",
    "latent_l2_ratio1",
    "latent_mse_slope2",
    "latent_l2_slope2",
    "latent_mse_ratio2",
    "latent_l2_ratio2",
    "prev2_available",
    "warm_origin",
)


class LearnedProbeValidationError(ValueError):
    """Raised when an offline probe input violates the frozen study contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LearnedProbeValidationError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_git_commit(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()


def dependency_versions() -> Dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


def feature_schema() -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "names": list(FEATURE_NAMES),
        "dtype": "float64",
        "history_missing_policy": (
            "At k=2, prev2 values equal prev1 and all two-step slopes are zero; "
            "prev2_available is zero."
        ),
        "ratio_denominator_floor": 1e-12,
        "origin_encoding": {"COLD": 0.0, "ACTUAL_WARM": 1.0},
        "cosine_features": {
            "available": False,
            "reason": "The frozen trace contains no latent cosine scalar or latent tensors.",
        },
        "label": {
            "name": "adjacent_action_converged",
            "definition": "action_mse < 0.001",
            "threshold": LABEL_THRESHOLD,
            "production_source": "iteration_mse (native control-flow metric)",
            "shadow_tail_source": "shadow_trace.action_mse (FP32 diagnostic only)",
        },
    }


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / max(float(denominator), 1e-12)


def transition_features(prediction: ShadowPrediction, k: int) -> list[float]:
    _require(2 <= k <= prediction.max_iter, f"invalid transition iteration: {k}")
    current = prediction.trace[k - 1]
    prev1 = prediction.trace[k - 2]
    has_prev2 = k >= 3
    prev2 = prediction.trace[k - 3] if has_prev2 else prev1
    return [
        current.latent_mse,
        current.latent_l2,
        float(k),
        float(k) / prediction.max_iter,
        prev1.latent_mse,
        prev1.latent_l2,
        prev2.latent_mse,
        prev2.latent_l2,
        current.latent_mse - prev1.latent_mse,
        current.latent_l2 - prev1.latent_l2,
        _ratio(current.latent_mse, prev1.latent_mse),
        _ratio(current.latent_l2, prev1.latent_l2),
        (current.latent_mse - prev2.latent_mse) / 2.0 if has_prev2 else 0.0,
        (current.latent_l2 - prev2.latent_l2) / 2.0 if has_prev2 else 0.0,
        _ratio(current.latent_mse, prev2.latent_mse) if has_prev2 else 1.0,
        _ratio(current.latent_l2, prev2.latent_l2) if has_prev2 else 1.0,
        float(has_prev2),
        float(prediction.actual_origin == "ACTUAL_WARM"),
    ]


def prediction_to_dataset_record(prediction: ShadowPrediction) -> Dict[str, Any]:
    transitions = []
    for k in range(2, prediction.max_iter + 1):
        point = prediction.trace[k - 1]
        _require(point.action_mse is not None, f"{prediction.key} k={k}: missing action MSE")
        transitions.append(
            {
                "k": k,
                "phase": point.phase,
                "action_mse": point.action_mse,
                "label": int(point.action_mse < LABEL_THRESHOLD),
                "features": transition_features(prediction, k),
            }
        )
    return {
        "key": [prediction.task_id, prediction.episode_id, prediction.prediction_index],
        "task_id": prediction.task_id,
        "episode_id": prediction.episode_id,
        "prediction_index": prediction.prediction_index,
        "actual_origin": prediction.actual_origin,
        "baseline_k": prediction.baseline_k,
        "baseline_decode_calls": prediction.baseline_decode_calls,
        "max_iter": prediction.max_iter,
        "transitions": transitions,
    }


def build_dataset_records(predictions: Sequence[ShadowPrediction]) -> list[Dict[str, Any]]:
    records = [prediction_to_dataset_record(prediction) for prediction in predictions]
    keys = [tuple(record["key"]) for record in records]
    _require(len(keys) == len(set(keys)), "duplicate prediction keys in learned-probe dataset")
    _require(records, "learned-probe dataset is empty")
    return records


def write_dataset(
    output_dir: Path,
    records: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "dataset.jsonl"
    with data_path.open("wb") as stream:
        for record in records:
            stream.write(canonical_json_bytes(record))
    manifest = dict(metadata)
    manifest.update(
        {
            "schema_version": SCHEMA_VERSION,
            "dataset_file": data_path.name,
            "dataset_sha256": sha256_file(data_path),
            "prediction_count": len(records),
            "transition_count": sum(len(record["transitions"]) for record in records),
            "episode_count": len(
                {(str(record["task_id"]), int(record["episode_id"])) for record in records}
            ),
            "task_ids": sorted({str(record["task_id"]) for record in records}, key=int),
            "origin_counts": {
                origin: sum(record["actual_origin"] == origin for record in records)
                for origin in ("ACTUAL_WARM", "COLD")
            },
            "feature_schema": feature_schema(),
        }
    )
    (output_dir / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    return manifest


def load_dataset(dataset_dir: Path) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
    manifest_path = dataset_dir / "manifest.json"
    _require(manifest_path.is_file(), f"missing dataset manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("schema_version") == SCHEMA_VERSION, "dataset schema mismatch")
    _require(manifest.get("feature_schema", {}).get("names") == list(FEATURE_NAMES), "feature schema mismatch")
    data_path = dataset_dir / str(manifest.get("dataset_file"))
    _require(data_path.is_file(), f"missing dataset file: {data_path}")
    _require(sha256_file(data_path) == manifest.get("dataset_sha256"), "dataset SHA-256 mismatch")
    records = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line]
    _require(len(records) == manifest.get("prediction_count"), "dataset prediction count mismatch")
    keys = [tuple(record["key"]) for record in records]
    _require(len(keys) == len(set(keys)), "duplicate dataset prediction key")
    return manifest, records


def load_fold_manifest(path: Path, task_ids: Iterable[str]) -> tuple[Dict[str, Any], Dict[str, int]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _require(manifest.get("schema_version") == 1, "fold manifest schema mismatch")
    assignment = parse_fold_manifest(manifest, task_ids)
    return manifest, assignment


def leakage_audit(records: Sequence[Mapping[str, Any]], assignment: Mapping[str, int]) -> Dict[str, Any]:
    audits = []
    for fold_id in sorted(set(assignment.values())):
        train = [record for record in records if assignment[str(record["task_id"])] != fold_id]
        validation = [record for record in records if assignment[str(record["task_id"])] == fold_id]
        train_predictions = {tuple(record["key"]) for record in train}
        val_predictions = {tuple(record["key"]) for record in validation}
        train_episodes = {(str(record["task_id"]), int(record["episode_id"])) for record in train}
        val_episodes = {(str(record["task_id"]), int(record["episode_id"])) for record in validation}
        _require(train_predictions.isdisjoint(val_predictions), f"fold {fold_id}: prediction leakage")
        _require(train_episodes.isdisjoint(val_episodes), f"fold {fold_id}: episode leakage")
        audits.append(
            {
                "fold_id": fold_id,
                "train_task_ids": sorted({str(record["task_id"]) for record in train}, key=int),
                "validation_task_ids": sorted({str(record["task_id"]) for record in validation}, key=int),
                "train_prediction_count": len(train),
                "validation_prediction_count": len(validation),
                "prediction_overlap_count": 0,
                "episode_overlap_count": 0,
            }
        )
    return {"passed": True, "folds": audits}


def flatten_transitions(records: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    for record in records:
        for transition in record["transitions"]:
            features.append(transition["features"])
            labels.append(transition["label"])
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    _require(x.ndim == 2 and x.shape[1] == len(FEATURE_NAMES), "invalid feature matrix")
    _require(np.isfinite(x).all(), "non-finite feature matrix")
    _require(set(np.unique(y)).issubset({0.0, 1.0}), "invalid labels")
    return x, y


def fit_normalizer(x: np.ndarray) -> Dict[str, list[float]]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return {"mean": mean.tolist(), "scale": scale.tolist()}


def normalize(x: np.ndarray, normalizer: Mapping[str, Sequence[float]]) -> np.ndarray:
    mean = np.asarray(normalizer["mean"], dtype=np.float64)
    scale = np.asarray(normalizer["scale"], dtype=np.float64)
    return (x - mean) / scale


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-value))


def fit_logistic(x: np.ndarray, y: np.ndarray, *, class_weighted: bool) -> Dict[str, Any]:
    design = np.column_stack([x, np.ones(len(x), dtype=np.float64)])
    beta = np.zeros(design.shape[1], dtype=np.float64)
    if class_weighted:
        positives = max(1, int(y.sum()))
        negatives = max(1, len(y) - positives)
        sample_weight = np.where(y == 1.0, len(y) / (2.0 * positives), len(y) / (2.0 * negatives))
    else:
        sample_weight = np.ones(len(y), dtype=np.float64)
    regularization = 1e-3
    reg_diag = np.full(design.shape[1], regularization, dtype=np.float64)
    reg_diag[-1] = 0.0
    for _ in range(60):
        probability = _sigmoid(design @ beta)
        residual_weight = sample_weight * probability * (1.0 - probability)
        gradient = design.T @ (sample_weight * (probability - y)) / sample_weight.sum()
        gradient += reg_diag * beta
        hessian = (design.T @ (design * residual_weight[:, None])) / sample_weight.sum()
        hessian += np.diag(reg_diag + 1e-9)
        step = np.linalg.solve(hessian, gradient)
        beta -= step
        if float(np.max(np.abs(step))) < 1e-8:
            break
    return {
        "kind": "logistic",
        "weights": beta[:-1].tolist(),
        "bias": float(beta[-1]),
        "class_weighted": class_weighted,
        "l2_regularization": regularization,
        "parameter_count": int(len(beta)),
        "approximate_macs": int(x.shape[1]),
        "approximate_flops": int(2 * x.shape[1] + 4),
    }


def fit_tiny_mlp(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    width: int = 16,
    steps: int = 1600,
) -> Dict[str, Any]:
    _require(1 <= width <= 16, "tiny MLP width must be in 1..16")
    rng = np.random.default_rng(seed)
    fan_in = x.shape[1]
    w1 = rng.normal(0.0, math.sqrt(2.0 / fan_in), size=(fan_in, width))
    b1 = np.zeros(width, dtype=np.float64)
    w2 = rng.normal(0.0, math.sqrt(1.0 / width), size=(width, 1))
    b2 = np.zeros(1, dtype=np.float64)
    params = [w1, b1, w2, b2]
    first = [np.zeros_like(param) for param in params]
    second = [np.zeros_like(param) for param in params]
    order = rng.permutation(len(x))
    offset = 0
    batch_size = min(1024, len(x))
    for step in range(1, steps + 1):
        if offset + batch_size > len(order):
            order = rng.permutation(len(x))
            offset = 0
        index = order[offset : offset + batch_size]
        offset += batch_size
        xb = x[index]
        yb = y[index, None]
        hidden_pre = xb @ w1 + b1
        hidden = np.maximum(hidden_pre, 0.0)
        probability = _sigmoid(hidden @ w2 + b2)
        dz2 = (probability - yb) / len(index)
        gradients = [
            xb.T @ ((dz2 @ w2.T) * (hidden_pre > 0.0)) + 1e-4 * w1,
            ((dz2 @ w2.T) * (hidden_pre > 0.0)).sum(axis=0),
            hidden.T @ dz2 + 1e-4 * w2,
            dz2.sum(axis=0),
        ]
        for item, grad, moment1, moment2 in zip(params, gradients, first, second):
            moment1 *= 0.9
            moment1 += 0.1 * grad
            moment2 *= 0.999
            moment2 += 0.001 * grad * grad
            corrected1 = moment1 / (1.0 - 0.9**step)
            corrected2 = moment2 / (1.0 - 0.999**step)
            item -= 0.003 * corrected1 / (np.sqrt(corrected2) + 1e-8)
    parameter_count = fan_in * width + width + width + 1
    return {
        "kind": "tiny_mlp",
        "hidden_width": width,
        "affine_layer_count": 2,
        "weights1": w1.tolist(),
        "bias1": b1.tolist(),
        "weights2": w2[:, 0].tolist(),
        "bias2": float(b2[0]),
        "training_steps": steps,
        "parameter_count": parameter_count,
        "approximate_macs": fan_in * width + width,
        "approximate_flops": 2 * (fan_in * width + width) + width + 4,
    }


def score_matrix(
    model_name: str,
    model: Mapping[str, Any],
    normalizer: Mapping[str, Sequence[float]] | None,
    x: np.ndarray,
) -> np.ndarray:
    if model_name == "latent_mse_threshold":
        return -x[:, FEATURE_NAMES.index("latent_mse")]
    _require(normalizer is not None, f"{model_name}: missing normalizer")
    xn = normalize(x, normalizer)
    if model["kind"] == "logistic":
        return _sigmoid(xn @ np.asarray(model["weights"]) + float(model["bias"]))
    if model["kind"] == "tiny_mlp":
        hidden = np.maximum(
            xn @ np.asarray(model["weights1"]) + np.asarray(model["bias1"]), 0.0
        )
        return _sigmoid(hidden @ np.asarray(model["weights2"]) + float(model["bias2"]))
    raise LearnedProbeValidationError(f"unsupported model kind: {model.get('kind')}")


def _attach_scores(records: Sequence[Mapping[str, Any]], scores: np.ndarray) -> list[Dict[str, Any]]:
    output = []
    offset = 0
    for record in records:
        count = len(record["transitions"])
        copied = dict(record)
        copied["scores"] = scores[offset : offset + count].tolist()
        output.append(copied)
        offset += count
    _require(offset == len(scores), "score/transition count mismatch")
    return output


def replay_scored_records(
    records: Sequence[Mapping[str, Any]], threshold: float
) -> list[Dict[str, Any]]:
    replays = []
    for record in records:
        selected = None
        for transition, score in zip(record["transitions"], record["scores"]):
            if float(score) >= threshold:
                selected = transition
                break
        if selected is None:
            terminal_k = int(record["max_iter"])
            decode_calls = terminal_k
            stopped = False
            false_convergence = False
            true_stop = False
        else:
            terminal_k = int(selected["k"])
            # The probe predicts the unavailable current adjacent-action result and
            # returns the previously cached action, so the current Coda decode is omitted.
            decode_calls = terminal_k - 1
            stopped = True
            true_stop = bool(selected["label"])
            false_convergence = not true_stop
        reference = next(
            (int(item["k"]) for item in record["transitions"] if item["label"]), None
        )
        captured = None if reference is None else bool(stopped and true_stop and terminal_k >= reference)
        replays.append(
            {
                "key": record["key"],
                "task_id": str(record["task_id"]),
                "episode_id": int(record["episode_id"]),
                "actual_origin": record["actual_origin"],
                "baseline_k": int(record["baseline_k"]),
                "baseline_decode_calls": int(record["baseline_decode_calls"]),
                "max_iter": int(record["max_iter"]),
                "terminal_k": terminal_k,
                "decode_calls": decode_calls,
                "stopped": stopped,
                "true_stop": true_stop,
                "false_convergence": false_convergence,
                "reference_convergence_k": reference,
                "captured_convergence": captured,
                "delta_k": terminal_k - int(record["baseline_k"]),
            }
        )
    return replays


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _aggregate_replays_flat(replays: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    _require(bool(replays), "cannot aggregate empty replay list")
    eligible = [item for item in replays if item["captured_convergence"] is not None]
    tp = sum(item["true_stop"] for item in replays)
    fp = sum(item["false_convergence"] for item in replays)
    fn = sum(not item["captured_convergence"] for item in eligible)
    baseline_calls = sum(item["baseline_decode_calls"] for item in replays)
    candidate_calls = sum(item["decode_calls"] for item in replays)
    baseline_max = _mean([item["baseline_k"] == item["max_iter"] for item in replays])
    candidate_max = _mean([item["terminal_k"] == item["max_iter"] for item in replays])
    return {
        "prediction_count": len(replays),
        "convergence_capture_eligible_count": len(eligible),
        "false_convergence_count": fp,
        "false_convergence_rate": fp / len(replays),
        "convergence_capture": _mean([item["captured_convergence"] for item in eligible]),
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "mean_terminal_k": _mean([item["terminal_k"] for item in replays]),
        "mean_delta_k": _mean([item["delta_k"] for item in replays]),
        "p95_delta_k": percentile([item["delta_k"] for item in replays], 0.95),
        "baseline_max_iter_rate": baseline_max,
        "candidate_max_iter_rate": candidate_max,
        "max_iter_rate_delta": candidate_max - baseline_max,
        "baseline_decode_calls": baseline_calls,
        "candidate_decode_calls": candidate_calls,
        "relative_decode_call_reduction": (baseline_calls - candidate_calls) / baseline_calls,
    }


def aggregate_scheduler_metrics(replays: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    _require(bool(replays), "cannot aggregate empty replay list")
    by_task: Dict[str, list[Mapping[str, Any]]] = {}
    for replay in replays:
        by_task.setdefault(str(replay["task_id"]), []).append(replay)
    task_metrics = {task: _aggregate_replays_flat(items) for task, items in sorted(by_task.items(), key=lambda item: int(item[0]))}
    macro_fields = (
        "false_convergence_rate",
        "convergence_capture",
        "precision",
        "recall",
        "mean_terminal_k",
        "mean_delta_k",
        "p95_delta_k",
        "baseline_max_iter_rate",
        "candidate_max_iter_rate",
        "max_iter_rate_delta",
        "relative_decode_call_reduction",
    )
    result = _aggregate_replays_flat(replays)
    for field in macro_fields:
        values = [metrics[field] for metrics in task_metrics.values() if metrics[field] is not None]
        result[f"task_macro_{field}"] = _mean(values)
    result["task_count"] = len(task_metrics)
    result["all_tasks_finite_and_evaluable"] = bool(
        len(task_metrics) == 10
        and all(
            metrics["convergence_capture_eligible_count"] > 0
            and all(value is None or _finite(value) for value in metrics.values())
            for metrics in task_metrics.values()
        )
    )
    result["task_metrics"] = task_metrics
    result["worst_task"] = {
        "convergence_capture": min(task_metrics, key=lambda task: task_metrics[task]["convergence_capture"]),
        "false_convergence_count": max(task_metrics, key=lambda task: task_metrics[task]["false_convergence_count"]),
        "p95_delta_k": max(task_metrics, key=lambda task: task_metrics[task]["p95_delta_k"]),
        "decode_call_reduction": min(task_metrics, key=lambda task: task_metrics[task]["relative_decode_call_reduction"]),
    }
    return result


def _threshold_candidates(scored_records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    scores = np.asarray([score for record in scored_records for score in record["scores"]], dtype=np.float64)
    labels = np.asarray(
        [transition["label"] for record in scored_records for transition in record["transitions"]],
        dtype=np.int8,
    )
    quantiles = np.quantile(scores, np.linspace(0.0, 1.0, 1025))
    candidates = list(quantiles)
    candidates.extend([scores.min(), np.nextafter(scores.max(), math.inf)])
    if np.any(labels == 0):
        candidates.append(np.nextafter(scores[labels == 0].max(), math.inf))
    return np.unique(np.asarray(candidates, dtype=np.float64))


def select_train_threshold(scored_records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    primary = [record for record in scored_records if record["actual_origin"] == "ACTUAL_WARM"]
    _require(bool(primary), "threshold fitting requires ACTUAL_WARM training predictions")
    candidates = _threshold_candidates(primary)
    false_counts = np.zeros(len(candidates), dtype=np.int64)
    captured_counts = np.zeros(len(candidates), dtype=np.int64)
    eligible_count = 0
    candidate_calls = np.zeros(len(candidates), dtype=np.int64)
    baseline_calls = sum(int(record["baseline_decode_calls"]) for record in primary)
    for record in primary:
        scores = np.asarray(record["scores"], dtype=np.float64)
        labels = np.asarray(
            [transition["label"] for transition in record["transitions"]], dtype=bool
        )
        crossings = scores[None, :] >= candidates[:, None]
        stopped = crossings.any(axis=1)
        first = np.argmax(crossings, axis=1)
        selected_true = stopped & labels[first]
        false_counts += stopped & ~selected_true
        reference_exists = bool(labels.any())
        if reference_exists:
            eligible_count += 1
            captured_counts += selected_true
        terminal = np.where(
            stopped,
            np.asarray([item["k"] for item in record["transitions"]])[first],
            int(record["max_iter"]),
        )
        candidate_calls += np.where(stopped, terminal - 1, terminal)
    captures = captured_counts / eligible_count
    reductions = (baseline_calls - candidate_calls) / baseline_calls
    evaluated = [
        (
            float(threshold),
            {
                "prediction_count": len(primary),
                "convergence_capture_eligible_count": eligible_count,
                "false_convergence_count": int(false_counts[index]),
                "false_convergence_rate": float(false_counts[index] / len(primary)),
                "convergence_capture": float(captures[index]),
                "baseline_decode_calls": baseline_calls,
                "candidate_decode_calls": int(candidate_calls[index]),
                "relative_decode_call_reduction": float(reductions[index]),
            },
        )
        for index, threshold in enumerate(candidates)
    ]
    feasible = [item for item in evaluated if item[1]["convergence_capture"] is not None and item[1]["convergence_capture"] >= CAPTURE_TARGET]
    if feasible:
        threshold, metrics = min(
            feasible,
            key=lambda item: (
                item[1]["false_convergence_count"],
                -item[1]["relative_decode_call_reduction"],
                item[0],
            ),
        )
        status = "capture_feasible"
    else:
        threshold, metrics = min(
            evaluated,
            key=lambda item: (
                -(item[1]["convergence_capture"] or 0.0),
                item[1]["false_convergence_count"],
                -item[1]["relative_decode_call_reduction"],
            ),
        )
        status = "capture_infeasible_fail_closed"
    return {
        "threshold": threshold,
        "threshold_hex": float(threshold).hex(),
        "candidate_count": len(evaluated),
        "selection_status": status,
        "selection_order": [
            "minimize train false convergence count subject to the capture floor",
            "require train convergence capture >= 99.5%",
            "maximize train expected Coda decode-call reduction",
        ],
        "train_primary_metrics": metrics,
    }


def fit_model(
    model_name: str, x: np.ndarray, y: np.ndarray, *, seed: int
) -> tuple[Dict[str, Any], Dict[str, Any] | None]:
    if model_name == "latent_mse_threshold":
        return {
            "kind": "latent_mse_threshold",
            "parameter_count": 1,
            "approximate_macs": 0,
            "approximate_flops": 1,
        }, None
    normalizer = fit_normalizer(x)
    normalized = normalize(x, normalizer)
    if model_name == "logistic_regression":
        return fit_logistic(normalized, y, class_weighted=False), normalizer
    if model_name == "class_weighted_logistic_regression":
        return fit_logistic(normalized, y, class_weighted=True), normalizer
    if model_name == "tiny_mlp":
        return fit_tiny_mlp(normalized, y, seed=seed), normalizer
    raise LearnedProbeValidationError(f"unsupported model: {model_name}")


def train_oof_bundle(
    records: Sequence[Mapping[str, Any]],
    assignment: Mapping[str, int],
    *,
    seed: int,
) -> Dict[str, Any]:
    audit = leakage_audit(records, assignment)
    fold_ids = sorted(set(assignment.values()))
    models: Dict[str, Any] = {}
    for model_index, model_name in enumerate(MODEL_NAMES):
        fold_models = []
        for fold_id in fold_ids:
            train_records = [record for record in records if assignment[str(record["task_id"])] != fold_id]
            validation_tasks = sorted(
                (task for task, assigned in assignment.items() if assigned == fold_id), key=int
            )
            x, y = flatten_transitions(train_records)
            model, normalizer = fit_model(model_name, x, y, seed=seed + 100 * model_index + fold_id)
            scores = score_matrix(model_name, model, normalizer, x)
            scored_train = _attach_scores(train_records, scores)
            selection = select_train_threshold(scored_train)
            fold_models.append(
                {
                    "fold_id": fold_id,
                    "validation_task_ids": validation_tasks,
                    "train_prediction_count": len(train_records),
                    "train_transition_count": len(y),
                    "train_positive_count": int(y.sum()),
                    "normalization": normalizer,
                    "model": model,
                    "threshold_selection": selection,
                }
            )
        x, y = flatten_transitions(records)
        full_model, full_normalizer = fit_model(
            model_name, x, y, seed=seed + 100 * model_index + 99
        )
        full_scores = score_matrix(model_name, full_model, full_normalizer, x)
        full_selection = select_train_threshold(_attach_scores(records, full_scores))
        models[model_name] = {
            "folds": fold_models,
            "full_data_refit_diagnostic_not_oof": {
                "normalization": full_normalizer,
                "model": full_model,
                "threshold_selection": full_selection,
            },
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "random_seed": seed,
        "model_order": list(MODEL_NAMES),
        "feature_schema": feature_schema(),
        "leakage_audit": audit,
        "models": models,
    }


def _oof_scored_records(
    records: Sequence[Mapping[str, Any]],
    assignment: Mapping[str, int],
    model_name: str,
    model_bundle: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    output = []
    seen = set()
    for fold in model_bundle["folds"]:
        fold_id = int(fold["fold_id"])
        validation = [record for record in records if assignment[str(record["task_id"])] == fold_id]
        x, _ = flatten_transitions(validation)
        scores = score_matrix(model_name, fold["model"], fold["normalization"], x)
        for record in _attach_scores(validation, scores):
            copied = dict(record)
            copied["decision_threshold"] = float(fold["threshold_selection"]["threshold"])
            key = tuple(copied["key"])
            _require(key not in seen, f"duplicate OOF prediction: {key}")
            seen.add(key)
            output.append(copied)
    _require(len(output) == len(records), "OOF coverage mismatch")
    return output


def _reliability(scored_records: Sequence[Mapping[str, Any]], model_name: str) -> Dict[str, Any]:
    if model_name == "latent_mse_threshold":
        return {"available": False, "reason": "baseline score is negative latent MSE, not a probability"}
    scores = np.asarray([score for record in scored_records if record["actual_origin"] == "ACTUAL_WARM" for score in record["scores"]])
    labels = np.asarray([transition["label"] for record in scored_records if record["actual_origin"] == "ACTUAL_WARM" for transition in record["transitions"]])
    bins = []
    ece = 0.0
    for lower_index in range(10):
        lower = lower_index / 10.0
        upper = (lower_index + 1) / 10.0
        mask = (scores >= lower) & ((scores < upper) if upper < 1.0 else (scores <= upper))
        count = int(mask.sum())
        if count:
            confidence = float(scores[mask].mean())
            frequency = float(labels[mask].mean())
            ece += count / len(scores) * abs(confidence - frequency)
        else:
            confidence = frequency = None
        bins.append({"lower": lower, "upper": upper, "count": count, "mean_probability": confidence, "observed_frequency": frequency})
    return {"available": True, "expected_calibration_error": ece, "bins": bins}


def evaluate_oof_bundle(
    records: Sequence[Mapping[str, Any]],
    assignment: Mapping[str, int],
    bundle: Mapping[str, Any],
) -> Dict[str, Any]:
    results = {}
    for model_name in bundle["model_order"]:
        model_bundle = bundle["models"][model_name]
        scored = _oof_scored_records(records, assignment, model_name, model_bundle)
        replays = []
        for record in scored:
            replays.extend(replay_scored_records([record], record["decision_threshold"]))
        primary_replays = [item for item in replays if item["actual_origin"] == "ACTUAL_WARM"]
        cold_replays = [item for item in replays if item["actual_origin"] == "COLD"]
        primary = aggregate_scheduler_metrics(primary_replays)
        cold = _aggregate_replays_flat(cold_replays)
        gate_checks = {
            "false_convergence_zero": primary["false_convergence_count"] == 0,
            "task_macro_convergence_capture_at_least_99_5pct": primary["task_macro_convergence_capture"] >= CAPTURE_TARGET,
            "mean_delta_k_at_most_0_25": primary["task_macro_mean_delta_k"] <= 0.25,
            "task_macro_p95_delta_k_at_most_1": primary["task_macro_p95_delta_k"] <= 1.0,
            "no_max_iteration_rate_increase": primary["task_macro_max_iter_rate_delta"] <= 0.0,
            "decode_reduction_at_least_20pct": primary["task_macro_relative_decode_call_reduction"] >= TARGET_DECODE_REDUCTION,
            "clearly_exceeds_origin_aware_10_2pct": primary["task_macro_relative_decode_call_reduction"] >= TARGET_DECODE_REDUCTION and primary["task_macro_relative_decode_call_reduction"] > ORIGIN_AWARE_DECODE_REDUCTION,
            "all_tasks_finite_and_evaluable": primary["all_tasks_finite_and_evaluable"],
        }
        results[model_name] = {
            "primary_actual_warm": primary,
            "supplementary_cold": cold,
            "reliability_actual_warm_transitions": _reliability(scored, model_name),
            "model_complexity": {
                "fold_parameter_counts": [fold["model"]["parameter_count"] for fold in model_bundle["folds"]],
                "parameter_count": model_bundle["folds"][0]["model"]["parameter_count"],
                "approximate_macs": model_bundle["folds"][0]["model"]["approximate_macs"],
                "approximate_flops": model_bundle["folds"][0]["model"]["approximate_flops"],
            },
            "gate_checks": gate_checks,
            "passes_all_gates": all(gate_checks.values()),
        }
    passing = [name for name, result in results.items() if result["passes_all_gates"]]
    if passing:
        selected = max(
            passing,
            key=lambda name: results[name]["primary_actual_warm"]["task_macro_relative_decode_call_reduction"],
        )
        status = "online_integration_worth_investigating"
    else:
        selected = min(
            results,
            key=lambda name: (
                results[name]["primary_actual_warm"]["false_convergence_count"],
                -results[name]["primary_actual_warm"]["task_macro_convergence_capture"],
                results[name]["primary_actual_warm"]["task_macro_mean_delta_k"],
                -results[name]["primary_actual_warm"]["task_macro_relative_decode_call_reduction"],
            ),
        )
        status = "stop_research_fail_closed"
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "offline scalar-only feasibility; baseline-conditioned, not closed-loop evidence",
        "primary_analysis_origin": "ACTUAL_WARM",
        "supplementary_origin": "COLD",
        "gate": {
            "minimum_capture": CAPTURE_TARGET,
            "maximum_mean_delta_k": 0.25,
            "maximum_task_macro_p95_delta_k": 1.0,
            "maximum_max_iter_rate_increase": 0.0,
            "false_convergence_count": 0,
            "origin_aware_reference_decode_reduction": ORIGIN_AWARE_DECODE_REDUCTION,
            "minimum_decode_reduction": TARGET_DECODE_REDUCTION,
        },
        "models": results,
        "selected_model": selected,
        "online_integration_worth_investigating": bool(passing),
        "conclusion": status,
    }


def compact_result_manifest(
    evaluation: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    training_bundle: Mapping[str, Any],
    *,
    fold_manifest_path: Path,
    training_bundle_sha256: str,
) -> Dict[str, Any]:
    compact_models = {}
    for name, result in evaluation["models"].items():
        primary = result["primary_actual_warm"]
        cold = result["supplementary_cold"]
        worst_task_ids = set(primary["worst_task"].values())
        compact_models[name] = {
            "passes_all_gates": result["passes_all_gates"],
            "prediction_count": primary["prediction_count"],
            "false_convergence_count": primary["false_convergence_count"],
            "false_convergence_rate": primary["false_convergence_rate"],
            "task_macro_convergence_capture": primary["task_macro_convergence_capture"],
            "precision": primary["precision"],
            "recall": primary["recall"],
            "task_macro_decode_call_reduction": primary["task_macro_relative_decode_call_reduction"],
            "mean_terminal_k": primary["mean_terminal_k"],
            "task_macro_mean_delta_k": primary["task_macro_mean_delta_k"],
            "task_macro_p95_delta_k": primary["task_macro_p95_delta_k"],
            "baseline_max_iter_rate": primary["baseline_max_iter_rate"],
            "candidate_max_iter_rate": primary["candidate_max_iter_rate"],
            "task_macro_max_iter_rate_delta": primary["task_macro_max_iter_rate_delta"],
            "all_tasks_finite_and_evaluable": primary["all_tasks_finite_and_evaluable"],
            "worst_task_ids": primary["worst_task"],
            "worst_task_metrics": {
                task_id: primary["task_metrics"][task_id]
                for task_id in sorted(worst_task_ids, key=int)
            },
            "supplementary_cold": {
                key: cold[key]
                for key in (
                    "prediction_count", "false_convergence_count", "convergence_capture",
                    "precision", "recall", "relative_decode_call_reduction", "mean_delta_k",
                    "p95_delta_k", "max_iter_rate_delta"
                )
            },
            "reliability_actual_warm_transitions": result["reliability_actual_warm_transitions"],
            "model_complexity": result["model_complexity"],
            "gate_checks": result["gate_checks"],
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "study": "learned-convergence-probe-feasibility",
        "source_git_commit": dataset_manifest["source_git_commit"],
        "calibration_validation_sha256": dataset_manifest["calibration_validation_sha256"],
        "trace_files": dataset_manifest["trace_files"],
        "trace_set_sha256": dataset_manifest["trace_set_sha256"],
        "fold_manifest": {"path": str(fold_manifest_path), "sha256": sha256_file(fold_manifest_path)},
        "dataset_sha256": dataset_manifest["dataset_sha256"],
        "training_bundle_sha256": training_bundle_sha256,
        "feature_schema": training_bundle["feature_schema"],
        "random_seed": training_bundle["random_seed"],
        "dependency_versions": dataset_manifest["dependency_versions"],
        "normalization_parameters_recorded_in_training_bundle": True,
        "leakage_audit": training_bundle["leakage_audit"],
        "primary_analysis_origin": evaluation["primary_analysis_origin"],
        "supplementary_origin": evaluation["supplementary_origin"],
        "gate": evaluation["gate"],
        "models": compact_models,
        "selected_model": evaluation["selected_model"],
        "online_integration_worth_investigating": evaluation["online_integration_worth_investigating"],
        "conclusion": evaluation["conclusion"],
    }
