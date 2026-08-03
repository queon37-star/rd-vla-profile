"""Offline-only action-stopping and latent-trajectory audit helpers.

This module deliberately depends only on the frozen preconvergence dataset.
It does not import or modify the runtime action head or scheduler.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from scripts.preconvergence_trigger_lib import (
    RawPreconvergenceSequence,
    load_dataset_bundle,
    sha256_file,
    tensor_metadata,
)


SCHEMA_VERSION = 1
DEFAULT_EPSILON = 1e-8
ACTION_THRESHOLDS = (0.0005, 0.001, 0.002)
TAIL_WINDOWS = (4, 8)
ORIGINS = ("ACTUAL_WARM", "COLD")

LATENT_METRIC_NAMES = (
    "state_rms",
    "delta_rms",
    "delta_mean_abs",
    "delta_max_abs",
    "relative_delta_rms",
    "full_delta_cosine_with_previous",
    "mean_pooled_delta_cosine_with_previous",
    "second_difference_rms",
    "token_or_element_delta_mse_mean",
    "token_or_element_delta_mse_max",
)

LATENT_FEATURE_DIRECTIONS = {
    "delta_rms": -1.0,
    "previous_delta_rms": -1.0,
    "delta_ratio": -1.0,
    "delta_cosine": 1.0,
    "second_difference_rms": -1.0,
    "relative_delta_rms": -1.0,
    "state_rms": -1.0,
    "iteration_k": 1.0,
}


class ActionLatentAuditError(ValueError):
    """Raised when an offline audit input violates the frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ActionLatentAuditError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def threshold_key(threshold: float) -> str:
    return format(float(threshold), ".10g")


def _git_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _temporary_path(path: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    return Path(name)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    temporary = _temporary_path(path)
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch_save(path: Path, payload: Any) -> None:
    temporary = _temporary_path(path)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _finite_tensor(tensor: torch.Tensor, context: str) -> torch.Tensor:
    result = tensor.detach().to(device="cpu", dtype=torch.float32)
    _require(bool(torch.isfinite(result).all()), f"{context}: non-finite source tensor")
    return result


def _rms(tensor: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(torch.square(tensor)))


def _cosine(left: torch.Tensor, right: torch.Tensor, epsilon: float) -> torch.Tensor:
    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    denominator = torch.linalg.vector_norm(left_flat) * torch.linalg.vector_norm(
        right_flat
    )
    denominator = torch.clamp(denominator, min=float(epsilon))
    return torch.dot(left_flat, right_flat) / denominator


def build_trajectory_record(
    sequence: RawPreconvergenceSequence,
    *,
    epsilon: float = DEFAULT_EPSILON,
) -> dict[str, Any]:
    """Build one compact record without mutating the source sequence."""

    sequence.validate()
    _require(epsilon > 0.0 and math.isfinite(epsilon), "epsilon must be finite and positive")
    states = _finite_tensor(sequence.states, f"{sequence.identity.key} states")
    actions = _finite_tensor(sequence.actions, f"{sequence.identity.key} actions")
    reduce_axes = tuple(range(1, states.ndim - 1))
    state_mean = states.mean(dim=reduce_axes) if reduce_axes else states.clone()
    state_mean = state_mean.reshape(sequence.max_iter, sequence.latent_feature_dim)
    action_flat = actions.reshape(sequence.max_iter, -1).clone()

    metrics = {
        name: torch.full((sequence.max_iter,), float("nan"), dtype=torch.float32)
        for name in LATENT_METRIC_NAMES
    }
    deltas: list[torch.Tensor | None] = [None] * sequence.max_iter
    mean_deltas: list[torch.Tensor | None] = [None] * sequence.max_iter
    for index in range(sequence.max_iter):
        current = states[index]
        state_rms = _rms(current)
        metrics["state_rms"][index] = state_rms
        if index == 0:
            continue
        delta = current - states[index - 1]
        mean_delta = state_mean[index] - state_mean[index - 1]
        deltas[index] = delta
        mean_deltas[index] = mean_delta
        delta_rms = _rms(delta)
        metrics["delta_rms"][index] = delta_rms
        metrics["delta_mean_abs"][index] = torch.mean(torch.abs(delta))
        metrics["delta_max_abs"][index] = torch.max(torch.abs(delta))
        metrics["relative_delta_rms"][index] = delta_rms / torch.clamp(
            state_rms, min=float(epsilon)
        )
        per_element_mse = torch.mean(torch.square(delta), dim=-1).reshape(-1)
        metrics["token_or_element_delta_mse_mean"][index] = torch.mean(
            per_element_mse
        )
        metrics["token_or_element_delta_mse_max"][index] = torch.max(
            per_element_mse
        )
        if index >= 2:
            previous_delta = deltas[index - 1]
            previous_mean_delta = mean_deltas[index - 1]
            assert previous_delta is not None and previous_mean_delta is not None
            metrics["full_delta_cosine_with_previous"][index] = _cosine(
                delta, previous_delta, epsilon
            )
            metrics["mean_pooled_delta_cosine_with_previous"][index] = _cosine(
                mean_delta, previous_mean_delta, epsilon
            )
            metrics["second_difference_rms"][index] = _rms(
                delta - previous_delta
            )

    for name, values in metrics.items():
        available = values[1:] if name not in {
            "full_delta_cosine_with_previous",
            "mean_pooled_delta_cosine_with_previous",
            "second_difference_rms",
        } and name != "state_rms" else values
        if name in {
            "full_delta_cosine_with_previous",
            "mean_pooled_delta_cosine_with_previous",
            "second_difference_rms",
        }:
            available = values[2:]
        _require(bool(torch.isfinite(available).all()), f"{sequence.identity.key}: invalid {name}")

    return {
        "task_id": sequence.identity.task_id,
        "episode_id": sequence.identity.episode_id,
        "prediction_id": sequence.identity.prediction_id,
        "actual_origin": sequence.actual_origin,
        "max_iter": sequence.max_iter,
        "baseline_k": sequence.baseline_k,
        "source_state_metadata": tensor_metadata(sequence.states),
        "source_action_metadata": tensor_metadata(sequence.actions),
        "action_mse_phase": list(sequence.action_mse_phase),
        "action_mse": list(sequence.action_mse),
        "state_mean": state_mean.contiguous(),
        "action_flat": action_flat.contiguous(),
        "latent_metrics": {name: value.contiguous() for name, value in metrics.items()},
    }


def _dimension_inventory(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def counts(values: Iterable[Any]) -> dict[str, int]:
        return dict(sorted(Counter(str(value) for value in values).items()))

    return {
        "state_mean_shapes": counts(tuple(record["state_mean"].shape) for record in records),
        "action_flat_shapes": counts(tuple(record["action_flat"].shape) for record in records),
        "state_mean_dtypes": counts(record["state_mean"].dtype for record in records),
        "action_flat_dtypes": counts(record["action_flat"].dtype for record in records),
        "latent_metric_dtypes": counts(
            value.dtype
            for record in records
            for value in record["latent_metrics"].values()
        ),
        "source_state_shapes": counts(
            tuple(record["source_state_metadata"]["shape"]) for record in records
        ),
        "source_action_shapes": counts(
            tuple(record["source_action_metadata"]["shape"]) for record in records
        ),
        "source_state_dtypes": counts(
            record["source_state_metadata"]["dtype"] for record in records
        ),
        "source_action_dtypes": counts(
            record["source_action_metadata"]["dtype"] for record in records
        ),
    }


def build_trajectory_bundle(
    dataset_dir: Path,
    output_dir: Path,
    *,
    epsilon: float = DEFAULT_EPSILON,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Build the versioned compact bundle and hash-anchored manifest."""

    dataset_dir = Path(dataset_dir).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / "trajectory_bundle.pt"
    manifest_path = output_dir / "manifest.json"
    _require(
        not bundle_path.exists() and not manifest_path.exists(),
        "refusing to overwrite an existing trajectory bundle or manifest",
    )
    input_manifest_path = dataset_dir / "manifest.json"
    _require(input_manifest_path.is_file(), "input dataset manifest does not exist")
    input_manifest, sequences = load_dataset_bundle(dataset_dir)
    input_data_path = dataset_dir / str(input_manifest["dataset_file"])
    input_data_hash = sha256_file(input_data_path)
    _require(
        input_data_hash == input_manifest["dataset_sha256"],
        "input dataset file hash mismatch",
    )
    records = [build_trajectory_record(sequence, epsilon=epsilon) for sequence in sequences]
    identities = [
        (record["task_id"], record["episode_id"], record["prediction_id"])
        for record in records
    ]
    _require(len(identities) == len(set(identities)), "duplicate trajectory identity")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "epsilon": float(epsilon),
        "predictions": records,
    }
    _atomic_torch_save(bundle_path, payload)
    bundle_hash = sha256_file(bundle_path)
    repo_root = Path(repo_root or Path(__file__).resolve().parents[1])
    origin_counts = Counter(record["actual_origin"] for record in records)
    task_counts = Counter(int(record["task_id"]) for record in records)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "bundle_file": bundle_path.name,
        "absolute_input_dataset_manifest_path": str(input_manifest_path.resolve()),
        "input_dataset_manifest_sha256": sha256_file(input_manifest_path),
        "input_dataset_file": str(input_data_path.resolve()),
        "input_dataset_file_sha256": input_data_hash,
        "prediction_count": len(records),
        "prediction_counts_by_origin": dict(sorted(origin_counts.items())),
        "task_count": len(task_counts),
        "prediction_counts_by_task": {
            str(key): value for key, value in sorted(task_counts.items())
        },
        "tensor_dimensions_and_dtypes": _dimension_inventory(records),
        "feature_definitions": trajectory_feature_definitions(epsilon),
        "source_git_commit": _git_commit(repo_root),
        "output_bundle_sha256": bundle_hash,
    }
    try:
        _atomic_write_bytes(manifest_path, canonical_json_bytes(manifest))
    except Exception:
        bundle_path.unlink(missing_ok=True)
        raise
    return manifest


def trajectory_feature_definitions(epsilon: float = DEFAULT_EPSILON) -> dict[str, str]:
    return {
        "iteration_indexing": "one-based in reports; tensor row i stores iteration k=i+1",
        "state_mean": "FP32 state, averaged over every axis except iteration and final feature",
        "action_flat": "FP32 stored action with all non-iteration action axes flattened",
        "state_rms": "sqrt(mean(state_k ** 2))",
        "delta_rms": "sqrt(mean((state_k - state_(k-1)) ** 2))",
        "delta_mean_abs": "mean(abs(state_k - state_(k-1)))",
        "delta_max_abs": "max(abs(state_k - state_(k-1)))",
        "relative_delta_rms": f"delta_rms / max(state_rms, {epsilon:g})",
        "full_delta_cosine_with_previous": "cosine of flattened full-state updates",
        "mean_pooled_delta_cosine_with_previous": "cosine of consecutive state_mean updates",
        "second_difference_rms": "sqrt(mean((delta_k - delta_(k-1)) ** 2))",
        "token_or_element_delta_mse_mean": "mean over per-leading-element feature-axis MSE",
        "token_or_element_delta_mse_max": "maximum per-leading-element feature-axis MSE",
        "missing_values": "NaN only at k=1 for delta metrics and before k=3 for two-delta metrics",
    }


def load_trajectory_bundle(bundle_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bundle_dir = Path(bundle_dir)
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    _require(manifest.get("schema_version") == SCHEMA_VERSION, "unsupported bundle schema")
    bundle_path = bundle_dir / str(manifest["bundle_file"])
    _require(
        sha256_file(bundle_path) == manifest["output_bundle_sha256"],
        "trajectory bundle hash mismatch",
    )
    payload = torch.load(bundle_path, map_location="cpu", weights_only=True)
    _require(payload.get("schema_version") == SCHEMA_VERSION, "bundle payload schema mismatch")
    records = list(payload["predictions"])
    _require(len(records) == int(manifest["prediction_count"]), "prediction count mismatch")
    validate_trajectory_records(records)
    return manifest, records


def validate_trajectory_records(records: Sequence[Mapping[str, Any]]) -> None:
    seen: set[tuple[int, int, int]] = set()
    for record in records:
        key = (int(record["task_id"]), int(record["episode_id"]), int(record["prediction_id"]))
        _require(key not in seen, f"duplicate trajectory identity: {key}")
        seen.add(key)
        _require(record["actual_origin"] in ORIGINS, f"{key}: invalid origin")
        max_iter = int(record["max_iter"])
        _require(record["state_mean"].shape[0] == max_iter, f"{key}: state depth mismatch")
        _require(record["action_flat"].shape[0] == max_iter, f"{key}: action depth mismatch")
        _require(record["state_mean"].dtype == torch.float32, f"{key}: state_mean dtype")
        _require(record["action_flat"].dtype == torch.float32, f"{key}: action_flat dtype")
        _require(bool(torch.isfinite(record["state_mean"]).all()), f"{key}: state_mean non-finite")
        _require(bool(torch.isfinite(record["action_flat"]).all()), f"{key}: action_flat non-finite")
        _require(len(record["action_mse"]) == max_iter + 1, f"{key}: MSE length")
        _require(len(record["action_mse_phase"]) == max_iter + 1, f"{key}: phase length")
        for name in LATENT_METRIC_NAMES:
            values = record["latent_metrics"][name]
            _require(values.dtype == torch.float32 and values.shape == (max_iter,), f"{key}: {name} contract")
            expected_nan = {0}
            if name == "state_rms":
                expected_nan = set()
            elif name in {
                "full_delta_cosine_with_previous",
                "mean_pooled_delta_cosine_with_previous",
                "second_difference_rms",
            }:
                expected_nan = {0, 1}
            actual_nan = set(torch.nonzero(torch.isnan(values), as_tuple=False).reshape(-1).tolist())
            _require(actual_nan == expected_nan, f"{key}: unexpected NaN positions for {name}")
            _require(bool(torch.isfinite(values[~torch.isnan(values)]).all()), f"{key}: non-finite {name}")


def first_hit(values: Sequence[float | None], threshold: float, *, start_k: int = 2) -> int | None:
    for k in range(start_k, len(values)):
        value = values[k]
        if value is not None and float(value) < threshold:
            return k
    return None


def consecutive_hit(values: Sequence[float | None], threshold: float, count: int) -> int | None:
    _require(count >= 1, "consecutive count must be positive")
    run = 0
    for k in range(2, len(values)):
        value = values[k]
        run = run + 1 if value is not None and float(value) < threshold else 0
        if run >= count:
            return k
    return None


def stable_suffix_hit(values: Sequence[float | None], threshold: float) -> int | None:
    candidate: int | None = None
    for k in range(len(values) - 1, 1, -1):
        value = values[k]
        if value is None or float(value) >= threshold:
            break
        candidate = k
    return candidate


def rebound_diagnostics(
    values: Sequence[float | None], threshold: float, k_first: int | None
) -> dict[str, Any]:
    if k_first is None:
        return {
            "exists": False,
            "count": 0,
            "maximum_post_hit_mse": None,
            "maximum_post_hit_mse_over_threshold": None,
            "first_rebound_iteration": None,
        }
    post = [(k, float(values[k])) for k in range(k_first + 1, len(values)) if values[k] is not None]
    rebounds = [(k, value) for k, value in post if value >= threshold]
    maximum = max((value for _, value in post), default=None)
    return {
        "exists": bool(rebounds),
        "count": len(rebounds),
        "maximum_post_hit_mse": maximum,
        "maximum_post_hit_mse_over_threshold": maximum / threshold if maximum is not None else None,
        "first_rebound_iteration": rebounds[0][0] if rebounds else None,
    }


def recompute_action_metrics(record: Mapping[str, Any]) -> dict[str, list[float | None]]:
    actions = record["action_flat"].to(dtype=torch.float32)
    max_iter = int(record["max_iter"])
    original_shape = tuple(int(value) for value in record["source_action_metadata"]["shape"][1:])
    _require(math.prod(original_shape) == actions.shape[1], "action shape/flat dimension mismatch")
    result = {
        name: [None] * (max_iter + 1)
        for name in (
            "mse",
            "rmse",
            "mean_absolute_change",
            "maximum_absolute_element_change",
            "maximum_per_action_dimension_mse",
            "maximum_per_chunk_timestep_mse",
        )
    }
    for k in range(2, max_iter + 1):
        difference = (actions[k - 1] - actions[k - 2]).reshape(original_shape)
        squared = torch.square(difference)
        mse = torch.mean(squared)
        action_matrix = difference.reshape(-1, original_shape[-1])
        per_dimension = torch.mean(torch.square(action_matrix), dim=0)
        per_timestep = torch.mean(torch.square(action_matrix), dim=-1)
        result["mse"][k] = float(mse)
        result["rmse"][k] = float(torch.sqrt(mse))
        result["mean_absolute_change"][k] = float(torch.mean(torch.abs(difference)))
        result["maximum_absolute_element_change"][k] = float(torch.max(torch.abs(difference)))
        result["maximum_per_action_dimension_mse"][k] = float(torch.max(per_dimension))
        result["maximum_per_chunk_timestep_mse"][k] = float(torch.max(per_timestep))
    return result


def tail_action_diagnostics(
    record: Mapping[str, Any], window: int, threshold: float
) -> dict[str, Any]:
    actions = record["action_flat"].to(dtype=torch.float32)
    _require(1 <= window <= len(actions), "tail window exceeds trajectory depth")
    tail = actions[-window:]
    mean_center = torch.mean(tail, dim=0)
    median_center = torch.quantile(tail, 0.5, dim=0)

    def distances(center: torch.Tensor) -> list[float | None]:
        return [None] + [float(torch.mean(torch.square(row - center))) for row in actions]

    mean_mse = distances(mean_center)
    median_mse = distances(median_center)
    return {
        "window": window,
        "offline_only_future_action_diagnostic": True,
        "mean": {
            "center_action": mean_center.tolist(),
            "first_hit_k": first_hit(mean_mse, threshold),
            "stable_suffix_k": stable_suffix_hit(mean_mse, threshold),
            "mse_to_center": mean_mse,
        },
        "median": {
            "center_action": median_center.tolist(),
            "first_hit_k": first_hit(median_mse, threshold),
            "stable_suffix_k": stable_suffix_hit(median_mse, threshold),
            "mse_to_center": median_mse,
        },
    }


def action_prediction_audit(
    record: Mapping[str, Any],
    *,
    thresholds: Sequence[float] = ACTION_THRESHOLDS,
    tail_windows: Sequence[int] = TAIL_WINDOWS,
    epsilon: float = DEFAULT_EPSILON,
) -> dict[str, Any]:
    stored = record["action_mse"]
    recomputed = recompute_action_metrics(record)
    output = {
        "task_id": int(record["task_id"]),
        "episode_id": int(record["episode_id"]),
        "prediction_id": int(record["prediction_id"]),
        "actual_origin": str(record["actual_origin"]),
        "max_iter": int(record["max_iter"]),
        "baseline_k": int(record["baseline_k"]),
        "thresholds": {},
    }
    for threshold in thresholds:
        key = threshold_key(threshold)
        stored_first = first_hit(stored, threshold)
        recomputed_first = first_hit(recomputed["mse"], threshold)
        per_transition = []
        for k in range(2, int(record["max_iter"]) + 1):
            stored_value = float(stored[k])
            recomputed_value = float(recomputed["mse"][k])
            absolute_difference = abs(stored_value - recomputed_value)
            relative_difference = absolute_difference / max(abs(stored_value), epsilon)
            global_below = recomputed_value < threshold
            maximum_dimension = float(recomputed["maximum_per_action_dimension_mse"][k])
            maximum_timestep = float(recomputed["maximum_per_chunk_timestep_mse"][k])
            per_transition.append(
                {
                    "k": k,
                    "phase": record["action_mse_phase"][k],
                    "stored_mse": stored_value,
                    "recomputed_mse": recomputed_value,
                    "absolute_difference": absolute_difference,
                    "relative_difference": relative_difference,
                    "threshold_hit_disagreement": (stored_value < threshold) != global_below,
                    "recomputed_metrics": {
                        name: values[k] for name, values in recomputed.items()
                    },
                    "dimension_masking": {
                        "eligible_global_below_threshold": global_below,
                        "max_dimension_ge_1x": global_below and maximum_dimension >= threshold,
                        "max_dimension_ge_4x": global_below and maximum_dimension >= 4 * threshold,
                        "max_dimension_ge_10x": global_below and maximum_dimension >= 10 * threshold,
                        "max_timestep_ge_1x": global_below and maximum_timestep >= threshold,
                        "max_timestep_ge_4x": global_below and maximum_timestep >= 4 * threshold,
                        "max_timestep_ge_10x": global_below and maximum_timestep >= 10 * threshold,
                    },
                }
            )
        output["thresholds"][key] = {
            "threshold": float(threshold),
            "authoritative_candidates": {
                "K_first": stored_first,
                "K_consecutive_2": consecutive_hit(stored, threshold, 2),
                "K_consecutive_3": consecutive_hit(stored, threshold, 3),
                "K_stable_suffix": stable_suffix_hit(stored, threshold),
            },
            "first_hit_rebound": rebound_diagnostics(stored, threshold, stored_first),
            "recomputed_candidates": {
                "K_first": recomputed_first,
                "K_consecutive_2": consecutive_hit(recomputed["mse"], threshold, 2),
                "K_consecutive_3": consecutive_hit(recomputed["mse"], threshold, 3),
                "K_stable_suffix": stable_suffix_hit(recomputed["mse"], threshold),
            },
            "stored_recomputed_k_disagreement": stored_first != recomputed_first,
            "tail_diagnostics": {
                str(window): tail_action_diagnostics(record, window, threshold)
                for window in tail_windows
                if window <= int(record["max_iter"])
            },
            "transitions": per_transition,
        }
    return output


def _mean(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def _distribution(values: Iterable[float | None]) -> dict[str, float | int | None]:
    finite = np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=np.float64,
    )
    return {
        "count": int(len(finite)),
        "mean": float(np.mean(finite)) if len(finite) else None,
        "p50": float(np.quantile(finite, 0.50)) if len(finite) else None,
        "p95": float(np.quantile(finite, 0.95)) if len(finite) else None,
        "maximum": float(np.max(finite)) if len(finite) else None,
    }


def _rate(count: int, denominator: int) -> float | None:
    return count / denominator if denominator else None


def aggregate_action_predictions(
    predictions: Sequence[Mapping[str, Any]], thresholds: Sequence[float] = ACTION_THRESHOLDS
) -> dict[str, Any]:
    result: dict[str, Any] = {"prediction_count": len(predictions), "thresholds": {}}
    for threshold in thresholds:
        key = threshold_key(threshold)
        threshold_rows = [prediction["thresholds"][key] for prediction in predictions]
        candidate_names = ("K_first", "K_consecutive_2", "K_consecutive_3", "K_stable_suffix")
        candidates = {}
        for name in candidate_names:
            values = [row["authoritative_candidates"][name] for row in threshold_rows]
            hit_count = sum(value is not None for value in values)
            candidates[name] = {
                "hit_prediction_count": hit_count,
                "no_hit_prediction_count": len(values) - hit_count,
                "hit_prediction_rate": _rate(hit_count, len(values)),
                "mean_hit_k": _mean(values),
            }
        rebound_count = sum(row["first_hit_rebound"]["exists"] for row in threshold_rows)
        first_hit_count = sum(row["authoritative_candidates"]["K_first"] is not None for row in threshold_rows)
        all_transitions = [transition for row in threshold_rows for transition in row["transitions"]]
        phase_reports = {}
        for phase in ("production", "shadow_tail"):
            phase_transitions = [row for row in all_transitions if row["phase"] == phase]
            disagreement = sum(row["threshold_hit_disagreement"] for row in phase_transitions)
            prediction_ks = []
            for prediction in predictions:
                transitions = [
                    row for row in prediction["thresholds"][key]["transitions"] if row["phase"] == phase
                ]
                if not transitions:
                    continue
                stored_hits = [row["k"] for row in transitions if row["stored_mse"] < threshold]
                recomputed_hits = [row["k"] for row in transitions if row["recomputed_mse"] < threshold]
                prediction_ks.append((stored_hits[0] if stored_hits else None, recomputed_hits[0] if recomputed_hits else None))
            k_disagreements = sum(left != right for left, right in prediction_ks)
            phase_reports[phase] = {
                "transition_count": len(phase_transitions),
                "absolute_difference": _distribution(
                    row["absolute_difference"] for row in phase_transitions
                ),
                "relative_difference": _distribution(
                    row["relative_difference"] for row in phase_transitions
                ),
                "threshold_hit_disagreement_count": disagreement,
                "threshold_hit_disagreement_rate": _rate(disagreement, len(phase_transitions)),
                "prediction_with_phase_count": len(prediction_ks),
                "K_disagreement_count": k_disagreements,
                "K_disagreement_rate": _rate(k_disagreements, len(prediction_ks)),
            }
        masking = {}
        for family in ("dimension", "timestep"):
            eligible = [row for row in all_transitions if row["dimension_masking"]["eligible_global_below_threshold"]]
            masking[family] = {
                f"ge_{multiple}x_count": sum(
                    row["dimension_masking"][f"max_{family}_ge_{multiple}x"] for row in eligible
                )
                for multiple in (1, 4, 10)
            }
            masking[family]["eligible_transition_count"] = len(eligible)
            for multiple in (1, 4, 10):
                masking[family][f"ge_{multiple}x_rate"] = _rate(
                    masking[family][f"ge_{multiple}x_count"], len(eligible)
                )
        tail = {}
        for window in TAIL_WINDOWS:
            window_key = str(window)
            available = [row for row in threshold_rows if window_key in row["tail_diagnostics"]]
            tail[window_key] = {}
            for center in ("mean", "median"):
                for candidate in ("first_hit_k", "stable_suffix_k"):
                    values = [row["tail_diagnostics"][window_key][center][candidate] for row in available]
                    tail[window_key][f"{center}_{candidate}"] = {
                        "hit_count": sum(value is not None for value in values),
                        "no_hit_count": sum(value is None for value in values),
                        "mean_k": _mean(values),
                    }
        result["thresholds"][key] = {
            "threshold": float(threshold),
            "candidate_coverage": candidates,
            "first_hit_rebound_prediction_count": rebound_count,
            "first_hit_rebound_rate_among_first_hits": _rate(rebound_count, first_hit_count),
            "stored_vs_recomputed_by_phase": phase_reports,
            "dimension_masking_on_recomputed_global_hits": masking,
            "tail_agreement_offline_diagnostic": tail,
        }
    return result


def build_action_stopping_audit(
    records: Sequence[Mapping[str, Any]],
    *,
    thresholds: Sequence[float] = ACTION_THRESHOLDS,
    epsilon: float = DEFAULT_EPSILON,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    predictions = [
        action_prediction_audit(record, thresholds=thresholds, epsilon=epsilon)
        for record in records
    ]
    by_origin = {
        origin: aggregate_action_predictions(
            [prediction for prediction in predictions if prediction["actual_origin"] == origin],
            thresholds,
        )
        for origin in ORIGINS
    }
    task_rows = []
    for (origin, task_id), group in sorted(
        {
            (origin, task): [
                prediction for prediction in predictions
                if prediction["actual_origin"] == origin and prediction["task_id"] == task
            ]
            for origin in ORIGINS
            for task in sorted({prediction["task_id"] for prediction in predictions})
        }.items()
    ):
        if not group:
            continue
        aggregate = aggregate_action_predictions(group, thresholds)
        for threshold in thresholds:
            key = threshold_key(threshold)
            row = aggregate["thresholds"][key]
            task_rows.append({
                "actual_origin": origin,
                "task_id": task_id,
                "threshold": threshold,
                "prediction_count": len(group),
                "K_first_hit_rate": row["candidate_coverage"]["K_first"]["hit_prediction_rate"],
                "K_consecutive_2_hit_rate": row["candidate_coverage"]["K_consecutive_2"]["hit_prediction_rate"],
                "K_consecutive_3_hit_rate": row["candidate_coverage"]["K_consecutive_3"]["hit_prediction_rate"],
                "K_stable_suffix_hit_rate": row["candidate_coverage"]["K_stable_suffix"]["hit_prediction_rate"],
                "first_hit_rebound_rate": row["first_hit_rebound_rate_among_first_hits"],
            })
    report = {
        "schema_version": SCHEMA_VERSION,
        "analysis_type": "offline_action_stopping_rule_audit",
        "primary_population": "ACTUAL_WARM",
        "thresholds": list(map(float, thresholds)),
        "definitions": action_audit_definitions(epsilon),
        "primary_actual_warm": by_origin["ACTUAL_WARM"],
        "cold_reported_separately": by_origin["COLD"],
    }
    return report, predictions, task_rows


def action_audit_definitions(epsilon: float) -> dict[str, str]:
    return {
        "K_first": "first one-based k>=2 with stored authoritative action_mse[k] < threshold",
        "K_consecutive_2": "ending k of the first two consecutive authoritative values below threshold",
        "K_consecutive_3": "ending k of the first three consecutive authoritative values below threshold",
        "K_stable_suffix": "first k whose entire authoritative suffix through max_iter is below threshold",
        "recomputed_mse": "FP32 arithmetic over stored action values; not unavailable pre-quantization FP32 production output",
        "relative_difference": f"abs(stored-recomputed) / max(abs(stored), {epsilon:g})",
        "maximum_per_action_dimension_mse": "maximum final-action-dimension MSE after reducing all preceding action axes",
        "maximum_per_chunk_timestep_mse": "maximum timestep/leading-element MSE after reducing final action dimension",
        "dimension_masking_global": "global means recomputed FP32 adjacent-action MSE",
        "tail_centers": "offline-only future-action diagnostics, not ground truth and not online inputs",
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def pearson_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    return pearson_correlation(_rankdata(x), _rankdata(y))


def exact_rank_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    _require(len(labels) == len(scores), "AUC label/score length mismatch")
    if not labels:
        return None
    y = np.asarray(labels, dtype=np.int64)
    x = np.asarray(scores, dtype=np.float64)
    _require(bool(np.isfinite(x).all()), "AUC score is non-finite")
    _require(bool(np.isin(y, [0, 1]).all()), "AUC labels must be binary")
    positive = int(np.sum(y == 1))
    negative = int(np.sum(y == 0))
    if positive == 0 or negative == 0:
        return None
    rank_sum = float(np.sum(_rankdata(x)[y == 1]))
    return (rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def _tail_mean_mse(record: Mapping[str, Any], window: int = 4) -> list[float | None]:
    actions = record["action_flat"].to(dtype=torch.float32)
    center = torch.mean(actions[-window:], dim=0)
    return [None] + [float(torch.mean(torch.square(row - center))) for row in actions]


def build_latent_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    thresholds: Sequence[float] = ACTION_THRESHOLDS,
    epsilon: float = DEFAULT_EPSILON,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    target_coverage: dict[str, dict[str, int]] = defaultdict(lambda: {"prediction_with_label": 0, "prediction_without_label": 0})
    for record in records:
        if record["actual_origin"] != "ACTUAL_WARM":
            continue
        max_iter = int(record["max_iter"])
        state_mean = record["state_mean"].to(dtype=torch.float32)
        metrics = record["latent_metrics"]
        recomputed = recompute_action_metrics(record)["mse"]
        tail_mse = _tail_mean_mse(record, 4)
        authoritative_mse = record["action_mse"]
        candidate_by_threshold: dict[str, dict[str, int | None]] = {}
        for threshold in thresholds:
            key = threshold_key(threshold)
            candidate_by_threshold[key] = {
                "K_first": first_hit(authoritative_mse, threshold),
                "K_consecutive_2": consecutive_hit(authoritative_mse, threshold, 2),
                "K_consecutive_3": consecutive_hit(authoritative_mse, threshold, 3),
                "K_stable_suffix": stable_suffix_hit(authoritative_mse, threshold),
                "K_tail_mean_4_stable_suffix": stable_suffix_hit(tail_mse, threshold),
            }
            for name, value in candidate_by_threshold[key].items():
                coverage_key = f"{name}@{key}"
                target_coverage[coverage_key][
                    "prediction_with_label" if value is not None else "prediction_without_label"
                ] += 1
        for k in range(3, max_iter + 1):
            current_delta = state_mean[k - 1] - state_mean[k - 2]
            previous_delta = state_mean[k - 2] - state_mean[k - 3]
            delta_rms = float(metrics["delta_rms"][k - 1])
            previous_delta_rms = float(metrics["delta_rms"][k - 2])
            feature_values = {
                "delta_rms": delta_rms,
                "previous_delta_rms": previous_delta_rms,
                "delta_ratio": delta_rms / max(previous_delta_rms, epsilon),
                "delta_cosine": float(_cosine(current_delta, previous_delta, epsilon)),
                "second_difference_rms": float(metrics["second_difference_rms"][k - 1]),
                "relative_delta_rms": float(metrics["relative_delta_rms"][k - 1]),
                "state_rms": float(metrics["state_rms"][k - 1]),
                "iteration_k": float(k),
            }
            _require(all(math.isfinite(value) for value in feature_values.values()), "non-finite latent feature")
            targets: dict[str, float | int | None] = {
                "log10_current_recomputed_action_mse": math.log10(max(float(recomputed[k]), epsilon)),
                "log10_next_recomputed_action_mse": (
                    math.log10(max(float(recomputed[k + 1]), epsilon)) if k < max_iter else None
                ),
                "log10_distance_to_tail_mean_4_action": math.log10(max(float(tail_mse[k]), epsilon)),
            }
            for threshold in thresholds:
                key = threshold_key(threshold)
                targets[f"current_stability@{key}"] = int(float(recomputed[k]) < threshold)
                targets[f"next_stability@{key}"] = (
                    int(float(recomputed[k + 1]) < threshold) if k < max_iter else None
                )
                for name, candidate in candidate_by_threshold[key].items():
                    target_name = f"one_step_before_{name}@{key}"
                    targets[target_name] = None if candidate is None or k >= candidate else int(k == candidate - 1)
            rows.append({
                "task_id": int(record["task_id"]),
                "episode_id": int(record["episode_id"]),
                "prediction_id": int(record["prediction_id"]),
                "k": k,
                "features": feature_values,
                "targets": targets,
            })
    return rows, dict(sorted(target_coverage.items()))


def _association(
    rows: Sequence[Mapping[str, Any]], feature: str, target: str
) -> dict[str, Any]:
    usable = [row for row in rows if row["targets"].get(target) is not None]
    feature_values = [float(row["features"][feature]) for row in usable]
    target_values = [float(row["targets"][target]) for row in usable]
    unique_predictions = {
        (row["task_id"], row["episode_id"], row["prediction_id"]) for row in usable
    }
    is_binary = bool(target_values) and set(target_values).issubset({0.0, 1.0})
    auc = None
    if is_binary:
        scores = [LATENT_FEATURE_DIRECTIONS[feature] * value for value in feature_values]
        auc = exact_rank_auc([int(value) for value in target_values], scores)
    return {
        "sample_count": len(usable),
        "prediction_count": len(unique_predictions),
        "positive_count": int(sum(target_values)) if is_binary else None,
        "no_label_row_count": len(rows) - len(usable),
        "pearson": pearson_correlation(feature_values, target_values),
        "spearman": spearman_correlation(feature_values, target_values),
        "roc_auc": auc,
    }


def build_latent_signal_audit(
    records: Sequence[Mapping[str, Any]],
    *,
    thresholds: Sequence[float] = ACTION_THRESHOLDS,
    epsilon: float = DEFAULT_EPSILON,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows, coverage = build_latent_rows(records, thresholds=thresholds, epsilon=epsilon)
    features = tuple(LATENT_FEATURE_DIRECTIONS)
    targets = sorted({target for row in rows for target in row["targets"]})
    task_ids = sorted({row["task_id"] for row in rows})
    report_rows: list[dict[str, Any]] = []
    associations: dict[str, Any] = {}
    for target in targets:
        associations[target] = {}
        for feature in features:
            micro = _association(rows, feature, target)
            per_task = {
                str(task_id): _association(
                    [row for row in rows if row["task_id"] == task_id], feature, target
                )
                for task_id in task_ids
            }
            by_prediction: dict[tuple[int, int, int], list[Mapping[str, Any]]] = defaultdict(list)
            for row in rows:
                by_prediction[
                    (int(row["task_id"]), int(row["episode_id"]), int(row["prediction_id"]))
                ].append(row)
            per_prediction = [
                _association(prediction_rows, feature, target)
                for prediction_rows in by_prediction.values()
            ]
            metric_names = ("pearson", "spearman", "roc_auc")
            task_macro = {
                name: _mean(item[name] for item in per_task.values()) for name in metric_names
            }
            task_macro["task_count"] = len(per_task)
            prediction_macro = {
                name: _mean(item[name] for item in per_prediction) for name in metric_names
            }
            prediction_macro["prediction_count"] = len(per_prediction)
            prediction_macro.update(
                {
                    f"valid_{name}_prediction_count": sum(
                        item[name] is not None for item in per_prediction
                    )
                    for name in metric_names
                }
            )
            task_min = {
                name: min(
                    (item[name] for item in per_task.values() if item[name] is not None),
                    default=None,
                )
                for name in metric_names
            }
            task_max = {
                name: max(
                    (item[name] for item in per_task.values() if item[name] is not None),
                    default=None,
                )
                for name in metric_names
            }
            associations[target][feature] = {
                "global_row_micro": micro,
                "prediction_macro": prediction_macro,
                "per_task": per_task,
                "task_macro": task_macro,
                "task_minimum": task_min,
                "task_maximum": task_max,
            }
            for scope, metrics in [("global_row_micro", micro), ("prediction_macro", prediction_macro), ("task_macro", task_macro), ("task_minimum", task_min), ("task_maximum", task_max)]:
                report_rows.append({
                    "target": target,
                    "feature": feature,
                    "score_direction": LATENT_FEATURE_DIRECTIONS[feature],
                    "scope": scope,
                    **metrics,
                })
            for task_id, metrics in per_task.items():
                report_rows.append({
                    "target": target,
                    "feature": feature,
                    "score_direction": LATENT_FEATURE_DIRECTIONS[feature],
                    "scope": "per_task",
                    "task_id": task_id,
                    **metrics,
                })
    actual_warm_predictions = {
        (record["task_id"], record["episode_id"], record["prediction_id"])
        for record in records if record["actual_origin"] == "ACTUAL_WARM"
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "analysis_type": "descriptive_in_sample_trajectory_associations",
        "not_oof": True,
        "population": "ACTUAL_WARM only, rows k>=3",
        "prediction_count": len(actual_warm_predictions),
        "row_count": len(rows),
        "thresholds": list(map(float, thresholds)),
        "feature_score_directions": {
            name: {
                "multiplier_for_positive_stability_auc": direction,
                "interpretation": (
                    "smaller values indicate stability" if direction < 0 else
                    "larger values are used as the positive stability score"
                ),
            }
            for name, direction in LATENT_FEATURE_DIRECTIONS.items()
        },
        "feature_definitions": latent_feature_definitions(epsilon),
        "target_definitions": latent_target_definitions(epsilon),
        "coverage_and_no_label_counts": coverage,
        "associations": associations,
    }
    return report, report_rows


def latent_target_definitions(epsilon: float) -> dict[str, str]:
    return {
        "log_targets": f"log10(max(FP32 recomputed stored-action diagnostic, {epsilon:g}))",
        "current_stability": "recomputed adjacent-action MSE at k is strictly below threshold",
        "next_stability": "recomputed adjacent-action MSE at k+1 is strictly below threshold",
        "one_step_before": "authoritative-action K candidates (tail candidate excepted): 1 only at k=K_candidate-1, 0 earlier, null at/after K or when unavailable",
        "tail_mean_4": "offline-only future-action diagnostic; never an online latent input or ground truth claim",
        "auc": "exact average-rank Mann-Whitney AUC with ties; null if either class is absent",
        "aggregation": "row micro, per-prediction macro, per-task, and unweighted task macro/minimum/maximum are reported",
    }


def latent_feature_definitions(epsilon: float) -> dict[str, str]:
    return {
        "current_delta_vector": "state_mean[k] - state_mean[k-1]; uses no state after k",
        "previous_delta_vector": "state_mean[k-1] - state_mean[k-2]",
        "delta_rms": "stored full-state FP32 delta RMS at k",
        "previous_delta_rms": "stored full-state FP32 delta RMS at k-1",
        "delta_ratio": f"delta_rms / max(previous_delta_rms, {epsilon:g})",
        "delta_cosine": "cosine(current mean-pooled delta, previous mean-pooled delta)",
        "second_difference_rms": "stored full-state second-difference RMS at k",
        "relative_delta_rms": "stored delta RMS relative to current state RMS",
        "state_rms": "stored full-state RMS at k",
        "iteration_k": "one-based recurrent iteration; iteration-only baseline",
    }


def write_json(path: Path, value: Any, *, refuse_overwrite: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not refuse_overwrite or not path.exists(), f"refusing to overwrite {path}")
    _atomic_write_bytes(path, canonical_json_bytes(value))


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]], *, refuse_overwrite: bool = True) -> None:
    payload = b"".join(
        (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        for row in rows
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not refuse_overwrite or not path.exists(), f"refusing to overwrite {path}")
    _atomic_write_bytes(path, payload)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], *, refuse_overwrite: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not refuse_overwrite or not path.exists(), f"refusing to overwrite {path}")
    fieldnames = sorted({key for row in rows for key in row})
    temporary = _temporary_path(path)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
