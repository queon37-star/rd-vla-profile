"""Offline-only strict-boundary latent task-OOF study.

The implementation consumes the frozen compact trajectory audit bundle and
never imports the runtime action head, scheduler, collector, or model code.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from scripts.action_latent_audit_lib import (
    exact_rank_auc,
    first_hit,
    load_trajectory_bundle,
    recompute_action_metrics,
    stable_suffix_hit,
)


SCHEMA_VERSION = 1
PRIMARY_THRESHOLD = 0.001
BOUNDARY_WIDTH = 3
DEFAULT_EPSILON = 1e-8
DEFAULT_SEED = 7
DEFAULT_STEPS = 400
DEFAULT_LEARNING_RATE = 0.003
DEFAULT_WEIGHT_DECAY = 0.0001
EXPECTED_TRAJECTORY_BUNDLE_SHA256 = (
    "9710a9c1fe4947703a103210bd715b15b24f9267ec39567edc222d18cdb23686"
)

REFERENCE_NAMES = (
    "K_first_authoritative",
    "K_stable_suffix_authoritative",
    "K_first_recomputed_fp32",
    "K_stable_suffix_recomputed_fp32",
)
PRIMARY_TARGETS = (
    "K_first_authoritative",
    "K_stable_suffix_recomputed_fp32",
)
SCALAR_FEATURE_NAMES = (
    "iteration_k",
    "delta_rms",
    "previous_delta_rms",
    "relative_delta_rms",
    "delta_ratio",
    "delta_cosine",
    "second_difference_rms",
)
MODEL_CONFIGS = {
    "iteration_only": ("scalar", (0,)),
    "delta_rms": ("scalar", (1,)),
    "relative_delta_rms": ("scalar", (3,)),
    "delta_ratio": ("scalar", (4,)),
    "scalar_combo": ("scalar", tuple(range(7))),
    "mean_pooled_low_rank4": ("mean_pooled_low_rank4", ()),
}


class BoundaryLatentOOFError(ValueError):
    """Raised when frozen offline inputs or OOF invariants are violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BoundaryLatentOOFError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def source_git_commit(repo_root: Path | None = None) -> str:
    root = Path(repo_root or Path(__file__).resolve().parents[1])
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
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


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    temporary = _temporary_path(path)
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch_save(path: Path, payload: Any) -> None:
    temporary = _temporary_path(path)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not path.exists(), f"refusing to overwrite {path}")
    atomic_write_bytes(path, canonical_json_bytes(value))


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not path.exists(), f"refusing to overwrite {path}")
    payload = b"".join(
        (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        for row in rows
    )
    atomic_write_bytes(path, payload)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not path.exists(), f"refusing to overwrite {path}")
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


def require_outputs_absent(paths: Sequence[Path]) -> None:
    existing = [str(path) for path in paths if Path(path).exists()]
    _require(not existing, "refusing to overwrite existing outputs: " + ", ".join(existing))


def load_frozen_trajectory_bundle(
    bundle_dir: Path,
    *,
    expected_sha256: str = EXPECTED_TRAJECTORY_BUNDLE_SHA256,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest, records = load_trajectory_bundle(bundle_dir)
    actual = str(manifest["output_bundle_sha256"])
    _require(actual == expected_sha256, f"trajectory bundle SHA-256 mismatch: {actual}")
    return manifest, records


def load_fold_assignment(
    fold_manifest_path: Path, task_ids: Iterable[int]
) -> tuple[dict[str, Any], dict[int, int], str]:
    path = Path(fold_manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _require(manifest.get("schema_version") == 1, "fold manifest schema mismatch")
    assignment: dict[int, int] = {}
    seen_folds: set[int] = set()
    for default_id, fold in enumerate(manifest.get("folds", [])):
        fold_id = int(fold.get("fold_id", default_id))
        _require(fold_id not in seen_folds, f"duplicate fold id: {fold_id}")
        seen_folds.add(fold_id)
        for raw_task in fold.get("validation_task_ids", []):
            task_id = int(raw_task)
            _require(task_id not in assignment, f"task appears in multiple folds: {task_id}")
            assignment[task_id] = fold_id
    expected = set(map(int, task_ids))
    _require(set(assignment) == expected, "fold manifest does not exactly cover input tasks")
    _require(len(seen_folds) >= 2, "at least two folds are required")
    return manifest, assignment, sha256_file(path)


def _distribution(values: Iterable[float | int | None]) -> dict[str, Any]:
    array = np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=np.float64,
    )
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)) if len(array) else None,
        "p50": float(np.quantile(array, 0.50)) if len(array) else None,
        "p95": float(np.quantile(array, 0.95)) if len(array) else None,
        "minimum": float(np.min(array)) if len(array) else None,
        "maximum": float(np.max(array)) if len(array) else None,
        "histogram": {
            str(int(value)): int(count)
            for value, count in sorted(Counter(map(int, array.tolist())).items())
        },
    }


def _rate(count: int, denominator: int) -> float | None:
    return count / denominator if denominator else None


def compute_boundary_references(
    record: Mapping[str, Any], threshold: float = PRIMARY_THRESHOLD
) -> dict[str, Any]:
    _require(record["actual_origin"] in {"ACTUAL_WARM", "COLD"}, "invalid origin")
    _require(math.isfinite(threshold) and threshold > 0, "threshold must be positive")
    authoritative = record["action_mse"]
    recomputed = recompute_action_metrics(record)
    references = {
        "K_first_authoritative": first_hit(authoritative, threshold),
        "K_stable_suffix_authoritative": stable_suffix_hit(authoritative, threshold),
        "K_first_recomputed_fp32": first_hit(recomputed["mse"], threshold),
        "K_stable_suffix_recomputed_fp32": stable_suffix_hit(recomputed["mse"], threshold),
    }
    k_first = references["K_first_authoritative"]
    rebound = False
    if k_first is not None:
        rebound = any(
            float(authoritative[k]) >= threshold
            for k in range(k_first + 1, int(record["max_iter"]) + 1)
        )
    masking: dict[str, Any] = {
        "evaluated_at_k": k_first,
        "eligible": k_first is not None,
    }
    if k_first is not None:
        maximum_dimension = float(recomputed["maximum_per_action_dimension_mse"][k_first])
        maximum_timestep = float(recomputed["maximum_per_chunk_timestep_mse"][k_first])
        for family, value in (
            ("dimension", maximum_dimension),
            ("chunk_timestep", maximum_timestep),
        ):
            for multiple in (1, 4, 10):
                masking[f"{family}_ge_{multiple}x"] = value >= multiple * threshold
        masking["maximum_per_action_dimension_mse"] = maximum_dimension
        masking["maximum_per_chunk_timestep_mse"] = maximum_timestep
    return {
        "task_id": int(record["task_id"]),
        "episode_id": int(record["episode_id"]),
        "prediction_id": int(record["prediction_id"]),
        "actual_origin": str(record["actual_origin"]),
        "max_iter": int(record["max_iter"]),
        "references": references,
        "authoritative_first_hit_rebound": rebound,
        "first_hit_masking": masking,
    }


def _aggregate_references(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    coverage = {}
    for name in REFERENCE_NAMES:
        values = [row["references"][name] for row in rows]
        hit = sum(value is not None for value in values)
        coverage[name] = {
            "hit_count": hit,
            "no_hit_count": len(values) - hit,
            "coverage_rate": _rate(hit, len(values)),
            "K_distribution": _distribution(values),
        }
    pairwise = {}
    for left_index, left in enumerate(REFERENCE_NAMES):
        for right in REFERENCE_NAMES[left_index + 1 :]:
            pairs = [
                (row["references"][left], row["references"][right])
                for row in rows
                if row["references"][left] is not None
                and row["references"][right] is not None
            ]
            deltas = [int(right_k) - int(left_k) for left_k, right_k in pairs]
            disagreement = sum(left_k != right_k for left_k, right_k in pairs)
            pairwise[f"{left}__vs__{right}"] = {
                "both_available_count": len(pairs),
                "either_or_both_null_count": len(rows) - len(pairs),
                "K_disagreement_count": disagreement,
                "K_disagreement_rate": _rate(disagreement, len(pairs)),
                "delta_right_minus_left": _distribution(deltas),
            }
    masking_rows = [row["first_hit_masking"] for row in rows if row["first_hit_masking"]["eligible"]]
    masking = {"eligible_prediction_count": len(masking_rows)}
    for family in ("dimension", "chunk_timestep"):
        for multiple in (1, 4, 10):
            field = f"{family}_ge_{multiple}x"
            count = sum(bool(row[field]) for row in masking_rows)
            masking[f"{field}_count"] = count
            masking[f"{field}_rate"] = _rate(count, len(masking_rows))
    return {
        "prediction_count": len(rows),
        "coverage": coverage,
        "pairwise_K_comparisons": pairwise,
        "first_hit_masking_exactly_at_K_first_authoritative": masking,
    }


def build_boundary_reference_audit(
    records: Sequence[Mapping[str, Any]],
    *,
    source_bundle_sha256: str,
    fold_manifest_sha256: str,
    git_commit: str,
    threshold: float = PRIMARY_THRESHOLD,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prediction_rows = [compute_boundary_references(record, threshold) for record in records]
    warm = [row for row in prediction_rows if row["actual_origin"] == "ACTUAL_WARM"]
    cold = [row for row in prediction_rows if row["actual_origin"] == "COLD"]
    rebound_subsets = {
        "rebound": _aggregate_references(
            [row for row in warm if row["authoritative_first_hit_rebound"]]
        ),
        "non_rebound": _aggregate_references(
            [row for row in warm if not row["authoritative_first_hit_rebound"]]
        ),
    }
    task_rows: list[dict[str, Any]] = []
    for origin in ("ACTUAL_WARM", "COLD"):
        origin_rows = [row for row in prediction_rows if row["actual_origin"] == origin]
        for task_id in sorted({row["task_id"] for row in origin_rows}):
            group = [row for row in origin_rows if row["task_id"] == task_id]
            aggregate = _aggregate_references(group)
            for reference in REFERENCE_NAMES:
                coverage = aggregate["coverage"][reference]
                task_rows.append(
                    {
                        "source_trajectory_bundle_sha256": source_bundle_sha256,
                        "boundary_dataset_sha256": None,
                        "fold_manifest_sha256": fold_manifest_sha256,
                        "source_git_commit": git_commit,
                        "model_training_configuration": "not_applicable_reference_audit",
                        "target_reference": reference,
                        "actual_origin": origin,
                        "task_id": task_id,
                        "prediction_count": len(group),
                        **coverage,
                    }
                )
    report = {
        "schema_version": SCHEMA_VERSION,
        "analysis_type": "offline_strict_boundary_reference_audit",
        "source_trajectory_bundle_sha256": source_bundle_sha256,
        "boundary_dataset_sha256": None,
        "fold_manifest_sha256": fold_manifest_sha256,
        "source_git_commit": git_commit,
        "model_training_configuration": "not_applicable_reference_audit",
        "target_references": list(REFERENCE_NAMES),
        "primary_oof_targets": list(PRIMARY_TARGETS),
        "diagnostic_only_target": "K_stable_suffix_authoritative",
        "threshold": float(threshold),
        "primary_population": "ACTUAL_WARM",
        "primary_actual_warm": _aggregate_references(warm),
        "rebound_subsets_actual_warm": rebound_subsets,
        "cold_excluded_from_oof": _aggregate_references(cold),
        "definitions": {
            "null_reference": "preserved as null; never replaced by max_iter",
            "recomputed_fp32": "FP32 arithmetic over stored action_flat values",
            "first_hit_masking": "evaluated once, exactly at K_first_authoritative",
            "pairwise_delta": "right reference K minus left reference K",
        },
    }
    return report, task_rows


def _cosine(left: torch.Tensor, right: torch.Tensor, epsilon: float) -> float:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    return float(torch.dot(left, right) / torch.clamp(denominator, min=epsilon))


def _features_at_k(
    record: Mapping[str, Any], k: int, epsilon: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _require(3 <= k <= int(record["max_iter"]), "feature iteration outside k=3..max_iter")
    state_mean = record["state_mean"].to(dtype=torch.float32)
    metrics = record["latent_metrics"]
    current_delta = (state_mean[k - 1] - state_mean[k - 2]).clone()
    previous_delta = (state_mean[k - 2] - state_mean[k - 3]).clone()
    delta_rms = float(metrics["delta_rms"][k - 1])
    previous_delta_rms = float(metrics["delta_rms"][k - 2])
    scalar = torch.tensor(
        [
            float(k),
            delta_rms,
            previous_delta_rms,
            float(metrics["relative_delta_rms"][k - 1]),
            delta_rms / max(previous_delta_rms, epsilon),
            _cosine(current_delta, previous_delta, epsilon),
            float(metrics["second_difference_rms"][k - 1]),
        ],
        dtype=torch.float32,
    )
    _require(bool(torch.isfinite(scalar).all()), "non-finite scalar boundary feature")
    _require(bool(torch.isfinite(current_delta).all()), "non-finite current delta")
    _require(bool(torch.isfinite(previous_delta).all()), "non-finite previous delta")
    return scalar, current_delta, previous_delta


def prediction_weights(rows: Sequence[Mapping[str, Any]]) -> torch.Tensor:
    grouped: dict[tuple[str, int, int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[
            (
                str(row["target_reference"]),
                int(row["task_id"]),
                int(row["episode_id"]),
                int(row["prediction_id"]),
            )
        ].append(index)
    weights = torch.zeros(len(rows), dtype=torch.float32)
    for indices in grouped.values():
        positives = [index for index in indices if int(rows[index]["label"]) == 1]
        negatives = [index for index in indices if int(rows[index]["label"]) == 0]
        _require(len(positives) == 1, "each included boundary prediction must have one positive")
        if negatives:
            weights[positives[0]] = 0.5
            weights[negatives] = 0.5 / len(negatives)
        else:
            weights[positives[0]] = 1.0
    return weights


def build_boundary_dataset_payload(
    records: Sequence[Mapping[str, Any]],
    *,
    threshold: float = PRIMARY_THRESHOLD,
    boundary_width: int = BOUNDARY_WIDTH,
    epsilon: float = DEFAULT_EPSILON,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(boundary_width == 3, "schema v1 fixes boundary width at 3")
    rows: list[dict[str, Any]] = []
    scoring_trajectories: list[dict[str, Any]] = []
    coverage: dict[str, Counter] = {target: Counter() for target in PRIMARY_TARGETS}
    for record in records:
        if record["actual_origin"] != "ACTUAL_WARM":
            continue
        reference_record = compute_boundary_references(record, threshold)
        references = reference_record["references"]
        trajectory_scalars = []
        trajectory_mean_pooled = []
        iterations = list(range(3, int(record["max_iter"]) + 1))
        for k in iterations:
            scalar, current_delta, previous_delta = _features_at_k(record, k, epsilon)
            trajectory_scalars.append(scalar)
            trajectory_mean_pooled.append(torch.cat((current_delta, previous_delta)))
        scoring_trajectories.append(
            {
                "task_id": int(record["task_id"]),
                "episode_id": int(record["episode_id"]),
                "prediction_id": int(record["prediction_id"]),
                "actual_origin": "ACTUAL_WARM",
                "max_iter": int(record["max_iter"]),
                "references": {target: references[target] for target in PRIMARY_TARGETS},
                "iterations": torch.tensor(iterations, dtype=torch.int64),
                "scalar_features": torch.stack(trajectory_scalars),
                "mean_pooled_features": torch.stack(trajectory_mean_pooled),
            }
        )
        for target in PRIMARY_TARGETS:
            reference_k = references[target]
            coverage[target]["population_prediction_count"] += 1
            if reference_k is None:
                coverage[target]["no_hit_prediction_count"] += 1
                continue
            if int(reference_k) < 4:
                coverage[target]["history_unavailable_prediction_count"] += 1
                coverage[target][f"history_unavailable_K_{int(reference_k)}_count"] += 1
                continue
            coverage[target]["included_prediction_count"] += 1
            start_k = max(3, int(reference_k) - boundary_width)
            prediction_rows = []
            for k in range(start_k, int(reference_k)):
                scalar, current_delta, previous_delta = _features_at_k(record, k, epsilon)
                prediction_rows.append(
                    {
                        "task_id": int(record["task_id"]),
                        "episode_id": int(record["episode_id"]),
                        "prediction_id": int(record["prediction_id"]),
                        "target_reference": target,
                        "k": k,
                        "K_reference": int(reference_k),
                        "label": int(k == int(reference_k) - 1),
                        "boundary_offset": k - (int(reference_k) - 1),
                        "scalar_features": scalar,
                        "current_mean_pooled_delta": current_delta,
                        "previous_mean_pooled_delta": previous_delta,
                    }
                )
            _require(all(row["k"] < int(reference_k) for row in prediction_rows), "boundary includes k>=K")
            if len(prediction_rows) == 1:
                coverage[target]["positive_only_prediction_count"] += 1
            rows.extend(prediction_rows)
    weights = prediction_weights(rows)
    for row, weight in zip(rows, weights):
        row["weight"] = float(weight)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "threshold": float(threshold),
        "boundary_width": int(boundary_width),
        "scalar_feature_names": list(SCALAR_FEATURE_NAMES),
        "target_references": list(PRIMARY_TARGETS),
        "actual_warm_primary_population_count": sum(
            record["actual_origin"] == "ACTUAL_WARM" for record in records
        ),
        "cold_excluded_from_oof_prediction_count": sum(
            record["actual_origin"] == "COLD" for record in records
        ),
        "rows": rows,
        "scoring_trajectories": scoring_trajectories,
    }
    coverage_report = {
        "targets": {
            target: dict(sorted(counter.items())) for target, counter in coverage.items()
        },
        "cold_excluded_from_oof_prediction_count": sum(
            record["actual_origin"] == "COLD" for record in records
        ),
        "actual_warm_primary_population_count": sum(
            record["actual_origin"] == "ACTUAL_WARM" for record in records
        ),
    }
    return payload, coverage_report


def boundary_feature_definitions(epsilon: float = DEFAULT_EPSILON) -> dict[str, str]:
    return {
        "iteration_k": "one-based recurrent iteration",
        "delta_rms": "full-state FP32 update RMS at k",
        "previous_delta_rms": "full-state FP32 update RMS at k-1",
        "relative_delta_rms": "delta_rms at k divided by current state RMS",
        "delta_ratio": f"delta_rms / max(previous_delta_rms, {epsilon:g})",
        "delta_cosine": "cosine of current and previous mean-pooled deltas",
        "second_difference_rms": "full-state FP32 second-difference RMS at k",
        "current_mean_pooled_delta": "state_mean[k] - state_mean[k-1]",
        "previous_mean_pooled_delta": "state_mean[k-1] - state_mean[k-2]",
        "future_exclusion": "features at k use no latent state after k",
        "boundary_rows": "max(3,K-3) through K-1; k>=K excluded",
        "label": "positive only at k=K-1",
        "prediction_weight": "unit mass: positive 0.5 and negatives total 0.5; positive-only gets 1.0",
    }


def _count_by_target_task(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output = {}
    for target in PRIMARY_TARGETS:
        target_rows = [row for row in rows if row["target_reference"] == target]
        output[target] = {
            "row_count": len(target_rows),
            "prediction_count": len(
                {(row["task_id"], row["episode_id"], row["prediction_id"]) for row in target_rows}
            ),
            "by_task": {
                str(task_id): {
                    "row_count": sum(row["task_id"] == task_id for row in target_rows),
                    "prediction_count": len(
                        {
                            (row["episode_id"], row["prediction_id"])
                            for row in target_rows
                            if row["task_id"] == task_id
                        }
                    ),
                }
                for task_id in sorted({row["task_id"] for row in target_rows})
            },
        }
    return output


def save_boundary_dataset(
    output_dir: Path,
    payload: Mapping[str, Any],
    coverage: Mapping[str, Any],
    *,
    source_bundle_path: Path,
    source_bundle_sha256: str,
    fold_manifest_sha256: str,
    git_commit: str,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "boundary_dataset.pt"
    manifest_path = output_dir / "manifest.json"
    _require(not dataset_path.exists() and not manifest_path.exists(), "refusing to overwrite boundary dataset")
    full_payload = {
        **dict(payload),
        "source_trajectory_bundle_sha256": source_bundle_sha256,
        "boundary_dataset_sha256": None,
        "fold_manifest_sha256": fold_manifest_sha256,
        "source_git_commit": git_commit,
        "model_training_configuration": "not_applicable_dataset_build",
    }
    atomic_torch_save(dataset_path, full_payload)
    dataset_hash = sha256_file(dataset_path)
    counts = _count_by_target_task(payload["rows"])
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_file": dataset_path.name,
        "source_trajectory_bundle_path": str(Path(source_bundle_path).resolve()),
        "source_trajectory_bundle_sha256": source_bundle_sha256,
        "boundary_dataset_sha256": dataset_hash,
        "fold_manifest_sha256": fold_manifest_sha256,
        "source_git_commit": git_commit,
        "model_training_configuration": "not_applicable_dataset_build",
        "target_references": list(PRIMARY_TARGETS),
        "threshold": payload["threshold"],
        "boundary_width": payload["boundary_width"],
        "counts_by_target_and_task": counts,
        "coverage": dict(coverage),
        "scalar_feature_names": list(SCALAR_FEATURE_NAMES),
        "mean_pooled_feature_dimension": (
            int(payload["rows"][0]["current_mean_pooled_delta"].numel())
            if payload["rows"] else None
        ),
        "feature_definitions": boundary_feature_definitions(),
    }
    try:
        atomic_write_bytes(manifest_path, canonical_json_bytes(manifest))
    except Exception:
        dataset_path.unlink(missing_ok=True)
        raise
    return manifest


def load_boundary_dataset(dataset_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset_dir = Path(dataset_dir)
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    _require(manifest.get("schema_version") == SCHEMA_VERSION, "boundary manifest schema mismatch")
    dataset_path = dataset_dir / str(manifest["dataset_file"])
    _require(sha256_file(dataset_path) == manifest["boundary_dataset_sha256"], "boundary dataset hash mismatch")
    payload = torch.load(dataset_path, map_location="cpu", weights_only=True)
    _require(payload.get("schema_version") == SCHEMA_VERSION, "boundary payload schema mismatch")
    _require(payload["source_trajectory_bundle_sha256"] == manifest["source_trajectory_bundle_sha256"], "source bundle identity mismatch")
    payload["boundary_dataset_sha256"] = manifest["boundary_dataset_sha256"]
    validate_boundary_payload(payload)
    return manifest, payload


def validate_boundary_payload(payload: Mapping[str, Any]) -> None:
    rows = payload["rows"]
    grouped: dict[tuple[str, int, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        _require(row["target_reference"] in PRIMARY_TARGETS, "invalid target reference")
        _require(3 <= int(row["k"]) < int(row["K_reference"]), "invalid boundary iteration")
        _require(int(row["label"]) == int(row["k"] == row["K_reference"] - 1), "invalid boundary label")
        _require(row["scalar_features"].shape == (len(SCALAR_FEATURE_NAMES),), "scalar shape mismatch")
        for field in ("scalar_features", "current_mean_pooled_delta", "previous_mean_pooled_delta"):
            _require(bool(torch.isfinite(row[field]).all()), f"non-finite {field}")
        grouped[(row["target_reference"], row["task_id"], row["episode_id"], row["prediction_id"])].append(row)
    for group in grouped.values():
        _require(abs(sum(float(row["weight"]) for row in group) - 1.0) < 1e-6, "prediction weights do not sum to one")
    for trajectory in payload["scoring_trajectories"]:
        _require(trajectory["actual_origin"] == "ACTUAL_WARM", "COLD trajectory in OOF dataset")
        _require(bool(torch.isfinite(trajectory["scalar_features"]).all()), "non-finite scoring scalar")
        _require(bool(torch.isfinite(trajectory["mean_pooled_features"]).all()), "non-finite scoring latent")


def leakage_audit(
    payload: Mapping[str, Any], assignment: Mapping[int, int]
) -> dict[str, Any]:
    identities = {
        (int(row["task_id"]), int(row["episode_id"]), int(row["prediction_id"]))
        for row in payload["rows"]
    }
    folds = []
    for fold_id in sorted(set(assignment.values())):
        held_tasks = {task for task, assigned in assignment.items() if assigned == fold_id}
        train_tasks = set(assignment) - held_tasks
        train_predictions = {identity for identity in identities if identity[0] in train_tasks}
        held_predictions = {identity for identity in identities if identity[0] in held_tasks}
        folds.append(
            {
                "fold_id": fold_id,
                "training_task_ids": sorted(train_tasks),
                "held_out_task_ids": sorted(held_tasks),
                "task_overlap_count": len(train_tasks & held_tasks),
                "prediction_overlap_count": len(train_predictions & held_predictions),
                "training_prediction_count": len(train_predictions),
                "held_out_prediction_count": len(held_predictions),
            }
        )
    passed = all(
        fold["task_overlap_count"] == 0 and fold["prediction_overlap_count"] == 0
        for fold in folds
    )
    _require(passed, "task-level OOF leakage detected")
    return {"passed": True, "folds": folds}


def _model_input_from_row(row: Mapping[str, Any], model_name: str) -> torch.Tensor:
    kind, indices = MODEL_CONFIGS[model_name]
    if kind == "scalar":
        return row["scalar_features"][list(indices)].to(dtype=torch.float32)
    return torch.cat(
        (row["current_mean_pooled_delta"], row["previous_mean_pooled_delta"])
    ).to(dtype=torch.float32)


def _model_input_from_trajectory(
    trajectory: Mapping[str, Any], model_name: str
) -> torch.Tensor:
    kind, indices = MODEL_CONFIGS[model_name]
    if kind == "scalar":
        return trajectory["scalar_features"][:, list(indices)].to(dtype=torch.float32)
    return trajectory["mean_pooled_features"].to(dtype=torch.float32)


def fit_normalizer(inputs: torch.Tensor) -> dict[str, torch.Tensor]:
    _require(inputs.ndim == 2 and len(inputs) > 0, "normalizer needs non-empty matrix")
    _require(bool(torch.isfinite(inputs).all()), "normalizer input is non-finite")
    mean = inputs.mean(dim=0)
    scale = inputs.std(dim=0, unbiased=False)
    scale = torch.where(scale < 1e-8, torch.ones_like(scale), scale)
    return {"mean": mean, "scale": scale}


def normalize(inputs: torch.Tensor, normalizer: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return (inputs - normalizer["mean"]) / normalizer["scale"]


class LinearLogistic(torch.nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.linear(values).squeeze(-1)


class MeanPooledLowRank4(torch.nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.bottleneck = torch.nn.Linear(input_dim, 4)
        self.head = torch.nn.Linear(4, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.head(self.bottleneck(values)).squeeze(-1)


def _make_model(model_name: str, input_dim: int) -> torch.nn.Module:
    _require(model_name in MODEL_CONFIGS, f"unknown model configuration: {model_name}")
    return (
        MeanPooledLowRank4(input_dim)
        if MODEL_CONFIGS[model_name][0] == "mean_pooled_low_rank4"
        else LinearLogistic(input_dim)
    )


def fit_model(
    rows: Sequence[Mapping[str, Any]],
    model_name: str,
    *,
    seed: int,
    steps: int = DEFAULT_STEPS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
) -> dict[str, Any]:
    _require(bool(rows), "model fitting has no boundary rows")
    inputs = torch.stack([_model_input_from_row(row, model_name) for row in rows])
    labels = torch.tensor([float(row["label"]) for row in rows], dtype=torch.float32)
    weights = torch.tensor([float(row["weight"]) for row in rows], dtype=torch.float32)
    normalizer = fit_normalizer(inputs)
    normalized = normalize(inputs, normalizer)
    torch.manual_seed(seed)
    model = _make_model(model_name, inputs.shape[1])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(normalized)
        losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        loss = torch.sum(losses * weights) / torch.sum(weights)
        loss.backward()
        optimizer.step()
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(normalized))
    _require(bool(torch.isfinite(probabilities).all()), "model produced non-finite scores")
    return {
        "model_name": model_name,
        "input_dim": int(inputs.shape[1]),
        "normalizer": {key: value.detach().clone() for key, value in normalizer.items()},
        "state_dict": {key: value.detach().clone() for key, value in model.state_dict().items()},
        "training_row_count": len(rows),
        "training_task_ids": sorted({int(row["task_id"]) for row in rows}),
        "seed": int(seed),
    }


def restore_model(fitted: Mapping[str, Any]) -> torch.nn.Module:
    model = _make_model(str(fitted["model_name"]), int(fitted["input_dim"]))
    model.load_state_dict(fitted["state_dict"])
    model.eval()
    return model


def score_rows(
    fitted: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> list[float]:
    if not rows:
        return []
    inputs = torch.stack(
        [_model_input_from_row(row, str(fitted["model_name"])) for row in rows]
    )
    model = restore_model(fitted)
    with torch.inference_mode():
        scores = torch.sigmoid(model(normalize(inputs, fitted["normalizer"])))
    _require(bool(torch.isfinite(scores).all()), "non-finite row score")
    return [float(value) for value in scores]


def score_trajectory(
    fitted: Mapping[str, Any], trajectory: Mapping[str, Any]
) -> dict[int, float]:
    return _score_trajectory_with_model(fitted, trajectory, restore_model(fitted))


def _score_trajectory_with_model(
    fitted: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    model: torch.nn.Module,
) -> dict[int, float]:
    inputs = _model_input_from_trajectory(trajectory, str(fitted["model_name"]))
    with torch.inference_mode():
        scores = torch.sigmoid(model(normalize(inputs, fitted["normalizer"])))
    _require(bool(torch.isfinite(scores).all()), "non-finite trajectory score")
    return {
        int(k): float(score)
        for k, score in zip(trajectory["iterations"].tolist(), scores.tolist())
    }


def gate_timing(
    scores_by_k: Mapping[int, float],
    threshold: float,
    K_reference: int,
    max_iter: int,
) -> dict[str, Any]:
    _require(K_reference >= 4, "gate timing reference lacks required history")
    k_gate = None
    for k in range(3, max_iter + 1):
        _require(k in scores_by_k, f"missing score at k={k}")
        score = float(scores_by_k[k])
        _require(math.isfinite(score), f"non-finite score at k={k}")
        if score >= threshold:
            k_gate = k
            break
    if k_gate is None or k_gate >= max_iter:
        return {
            "k_gate": k_gate,
            "category": "missed",
            "trigger_offset": None,
            "absolute_offset": None,
        }
    offset = k_gate - (K_reference - 1)
    return {
        "k_gate": k_gate,
        "category": "early" if offset < 0 else "ideal" if offset == 0 else "late",
        "trigger_offset": offset,
        "absolute_offset": abs(offset),
    }


def aggregate_gate_timings(timings: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(item["category"]) for item in timings)
    offsets = [item["trigger_offset"] for item in timings if item["trigger_offset"] is not None]
    absolute = [abs(int(value)) for value in offsets]
    return {
        "prediction_count": len(timings),
        **{
            f"{category}_count": int(counts[category])
            for category in ("early", "ideal", "late", "missed")
        },
        **{
            f"{category}_rate": _rate(int(counts[category]), len(timings))
            for category in ("early", "ideal", "late", "missed")
        },
        "late_plus_missed_count": int(counts["late"] + counts["missed"]),
        "total_late_offset": int(
            sum(int(value) for value in offsets if int(value) > 0)
        ),
        "total_absolute_early_offset": int(
            sum(-int(value) for value in offsets if int(value) < 0)
        ),
        "trigger_offset_distribution": _distribution(offsets),
        "mean_absolute_offset": float(np.mean(absolute)) if absolute else None,
        "p95_absolute_offset": float(np.quantile(absolute, 0.95)) if absolute else None,
        "mean_signed_offset": float(np.mean(offsets)) if offsets else None,
    }


def select_gate_threshold(
    scored_predictions: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Exact prefix-record event sweep for the frozen safety-first order."""

    _require(bool(scored_predictions), "threshold selection has no predictions")
    prepared = []
    score_values: set[float] = set()
    for prediction in scored_predictions:
        K_reference = int(prediction["K_reference"])
        max_iter = int(prediction["max_iter"])
        _require(K_reference >= 4, "threshold selection includes history-unavailable prediction")
        scores = {}
        for k in range(3, max_iter + 1):
            _require(k in prediction["scores_by_k"], f"missing train score at k={k}")
            score = float(prediction["scores_by_k"][k])
            _require(math.isfinite(score), f"non-finite train score at k={k}")
            scores[k] = score
            score_values.add(score)
        prepared.append((K_reference, max_iter, scores))
    values = sorted(score_values)
    _require(bool(values), "threshold candidate set is empty")
    fail_closed = float(np.nextafter(values[-1], math.inf))
    _require(math.isfinite(fail_closed), "fail-closed threshold is non-finite")
    candidates = values + [fail_closed]
    field_names = (
        "late_plus_missed_count",
        "missed_count",
        "total_late_offset",
        "early_count",
        "total_absolute_early_offset",
        "ideal_count",
    )
    differences = {name: np.zeros(len(candidates) + 1, dtype=np.int64) for name in field_names}

    def contribution(k_gate: int | None, K_reference: int, max_iter: int) -> dict[str, int]:
        timing = gate_timing(
            {k: 1.0 if k == k_gate else 0.0 for k in range(3, max_iter + 1)},
            0.5,
            K_reference,
            max_iter,
        ) if k_gate is not None else {
            "category": "missed",
            "trigger_offset": None,
        }
        category = timing["category"]
        offset = timing["trigger_offset"]
        return {
            "late_plus_missed_count": int(category in {"late", "missed"}),
            "missed_count": int(category == "missed"),
            "total_late_offset": int(offset) if offset is not None and offset > 0 else 0,
            "early_count": int(category == "early"),
            "total_absolute_early_offset": -int(offset) if offset is not None and offset < 0 else 0,
            "ideal_count": int(category == "ideal"),
        }

    def add_interval(start: int, end: int, values_to_add: Mapping[str, int]) -> None:
        if start >= end:
            return
        for name in field_names:
            differences[name][start] += int(values_to_add[name])
            differences[name][end] -= int(values_to_add[name])

    for K_reference, max_iter, scores in prepared:
        previous_record = -math.inf
        for k in range(3, max_iter + 1):
            score = scores[k]
            if score <= previous_record:
                continue
            add_interval(
                bisect_right(candidates, previous_record),
                bisect_right(candidates, score),
                contribution(k, K_reference, max_iter),
            )
            previous_record = score
        add_interval(
            bisect_right(candidates, previous_record),
            len(candidates),
            contribution(None, K_reference, max_iter),
        )
    running = {name: 0 for name in field_names}
    selected = None
    selected_key = None
    for index, threshold in enumerate(candidates):
        for name in field_names:
            running[name] += int(differences[name][index])
        metrics = {name: int(value) for name, value in running.items()}
        key = (
            metrics["late_plus_missed_count"],
            metrics["missed_count"],
            metrics["total_late_offset"],
            metrics["early_count"],
            metrics["total_absolute_early_offset"],
            -metrics["ideal_count"],
            -threshold,
        )
        if selected_key is None or key < selected_key:
            selected_key = key
            selected = {"threshold": threshold, **metrics}
    assert selected is not None
    return {
        "selected_threshold": float(selected["threshold"]),
        "selected_threshold_hex": float(selected["threshold"]).hex(),
        "candidate_count": len(candidates),
        "prediction_count": len(prepared),
        "candidate_definition": "sorted unique training scores plus nextafter(max_score,+inf)",
        "selection_order": [
            "minimize late + missed count",
            "minimize missed count",
            "minimize total late offset",
            "minimize early count",
            "minimize total absolute early offset",
            "maximize ideal count",
            "maximize threshold",
        ],
        "train_metrics": selected,
    }


def _scored_trajectories_for_target(
    fitted: Mapping[str, Any],
    trajectories: Sequence[Mapping[str, Any]],
    target: str,
) -> list[dict[str, Any]]:
    output = []
    model = restore_model(fitted)
    for trajectory in trajectories:
        reference = trajectory["references"][target]
        if reference is None or int(reference) < 4:
            continue
        output.append(
            {
                "task_id": int(trajectory["task_id"]),
                "episode_id": int(trajectory["episode_id"]),
                "prediction_id": int(trajectory["prediction_id"]),
                "K_reference": int(reference),
                "max_iter": int(trajectory["max_iter"]),
                "scores_by_k": _score_trajectory_with_model(fitted, trajectory, model),
            }
        )
    return output


def training_configuration(
    *, seed: int, steps: int, learning_rate: float, weight_decay: float
) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "optimization_steps": int(steps),
        "optimizer": "Adam",
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "scalar_head": "linear logistic",
        "mean_pooled_head": "rank-4 linear bottleneck plus scalar logistic head",
        "normalization_scope": "outer-training boundary rows only",
        "threshold_scope": "outer-training predictions only",
    }


def fit_boundary_oof_bundle(
    payload: Mapping[str, Any],
    assignment: Mapping[int, int],
    *,
    source_bundle_sha256: str,
    boundary_dataset_sha256: str,
    fold_manifest_sha256: str,
    git_commit: str,
    seed: int = DEFAULT_SEED,
    steps: int = DEFAULT_STEPS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
) -> dict[str, Any]:
    validate_boundary_payload(payload)
    leakage = leakage_audit(payload, assignment)
    configuration = training_configuration(
        seed=seed,
        steps=steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    folds = []
    fold_ids = sorted(set(assignment.values()))
    for fold_id in fold_ids:
        held_tasks = {task for task, value in assignment.items() if value == fold_id}
        train_rows_all = [row for row in payload["rows"] if int(row["task_id"]) not in held_tasks]
        train_trajectories_all = [
            row for row in payload["scoring_trajectories"] if int(row["task_id"]) not in held_tasks
        ]
        fold_models = {}
        for target_index, target in enumerate(PRIMARY_TARGETS):
            train_rows = [row for row in train_rows_all if row["target_reference"] == target]
            for model_index, model_name in enumerate(MODEL_CONFIGS):
                deterministic_seed = seed + fold_id * 1000 + target_index * 100 + model_index
                started = time.perf_counter()
                print(
                    f"[boundary-oof] fold={fold_id} target={target} model={model_name} train start",
                    flush=True,
                )
                fitted = fit_model(
                    train_rows,
                    model_name,
                    seed=deterministic_seed,
                    steps=steps,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                )
                train_trajectories = _scored_trajectories_for_target(
                    fitted, train_trajectories_all, target
                )
                selection = select_gate_threshold(train_trajectories)
                fold_models[f"{target}::{model_name}"] = {
                    "target_reference": target,
                    "model_configuration": model_name,
                    "fitted": fitted,
                    "threshold_selection": selection,
                    "training_prediction_count_for_threshold": len(train_trajectories),
                    "training_task_ids": sorted(set(assignment) - held_tasks),
                    "held_out_task_ids": sorted(held_tasks),
                }
                print(
                    f"[boundary-oof] fold={fold_id} target={target} model={model_name} done "
                    f"threshold={selection['selected_threshold']:.9g} elapsed={time.perf_counter()-started:.2f}s",
                    flush=True,
                )
        folds.append(
            {
                "fold_id": fold_id,
                "training_task_ids": sorted(set(assignment) - held_tasks),
                "held_out_task_ids": sorted(held_tasks),
                "models": fold_models,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "source_trajectory_bundle_sha256": source_bundle_sha256,
        "boundary_dataset_sha256": boundary_dataset_sha256,
        "fold_manifest_sha256": fold_manifest_sha256,
        "source_git_commit": git_commit,
        "model_training_configuration": configuration,
        "target_references": list(PRIMARY_TARGETS),
        "model_configurations": list(MODEL_CONFIGS),
        "leakage_audit": leakage,
        "folds": folds,
    }


def save_training_bundle(output_dir: Path, bundle: Mapping[str, Any]) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / "boundary_oof_bundle.pt"
    summary_path = output_dir / "training_summary.json"
    _require(not bundle_path.exists() and not summary_path.exists(), "refusing to overwrite training output")
    atomic_torch_save(bundle_path, dict(bundle))
    bundle_hash = sha256_file(bundle_path)
    summary = {
        key: value
        for key, value in bundle.items()
        if key != "folds"
    }
    summary["training_bundle_file"] = bundle_path.name
    summary["training_bundle_sha256"] = bundle_hash
    summary["folds"] = [
        {
            "fold_id": fold["fold_id"],
            "training_task_ids": fold["training_task_ids"],
            "held_out_task_ids": fold["held_out_task_ids"],
            "models": {
                name: {
                    "target_reference": model["target_reference"],
                    "model_configuration": model["model_configuration"],
                    "training_task_ids": model["training_task_ids"],
                    "held_out_task_ids": model["held_out_task_ids"],
                    "training_prediction_count_for_threshold": model[
                        "training_prediction_count_for_threshold"
                    ],
                    "normalizer_mean": model["fitted"]["normalizer"]["mean"].tolist(),
                    "normalizer_scale": model["fitted"]["normalizer"]["scale"].tolist(),
                    "threshold_selection": model["threshold_selection"],
                }
                for name, model in fold["models"].items()
            },
        }
        for fold in bundle["folds"]
    ]
    try:
        atomic_write_bytes(summary_path, canonical_json_bytes(summary))
    except Exception:
        bundle_path.unlink(missing_ok=True)
        raise
    return summary


def load_training_bundle(training_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    training_dir = Path(training_dir)
    summary = json.loads((training_dir / "training_summary.json").read_text(encoding="utf-8"))
    path = training_dir / str(summary["training_bundle_file"])
    _require(sha256_file(path) == summary["training_bundle_sha256"], "training bundle hash mismatch")
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    _require(bundle.get("schema_version") == SCHEMA_VERSION, "training schema mismatch")
    return summary, bundle


def exact_average_precision(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    _require(len(labels) == len(scores), "AP label/score length mismatch")
    if not labels:
        return None
    y = np.asarray(labels, dtype=np.int64)
    x = np.asarray(scores, dtype=np.float64)
    _require(bool(np.isin(y, [0, 1]).all()), "AP labels must be binary")
    _require(bool(np.isfinite(x).all()), "AP score is non-finite")
    positives = int(np.sum(y == 1))
    if positives == 0:
        return None
    order = np.argsort(-x, kind="mergesort")
    y = y[order]
    x = x[order]
    true_positive = 0
    observed = 0
    average_precision = 0.0
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and x[end] == x[start]:
            end += 1
        group_positive = int(np.sum(y[start:end]))
        true_positive += group_positive
        observed += end - start
        average_precision += (group_positive / positives) * (true_positive / observed)
        start = end
    return float(average_precision)


def binary_metrics(labels: Sequence[int], scores: Sequence[float]) -> dict[str, Any]:
    _require(len(labels) == len(scores), "metric label/score length mismatch")
    positives = sum(int(label) == 1 for label in labels)
    negatives = len(labels) - positives
    return {
        "row_count": len(labels),
        "positive_count": positives,
        "negative_count": negatives,
        "roc_auc": exact_rank_auc(labels, scores),
        "average_precision": exact_average_precision(labels, scores),
        "brier_score": (
            float(np.mean((np.asarray(scores) - np.asarray(labels)) ** 2))
            if labels else None
        ),
    }


def aggregate_row_metrics(scored_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    micro = binary_metrics(
        [int(row["label"]) for row in scored_rows],
        [float(row["score"]) for row in scored_rows],
    )
    per_task = {}
    for task_id in sorted({int(row["task_id"]) for row in scored_rows}):
        group = [row for row in scored_rows if int(row["task_id"]) == task_id]
        per_task[str(task_id)] = binary_metrics(
            [int(row["label"]) for row in group],
            [float(row["score"]) for row in group],
        )
    metric_names = ("roc_auc", "average_precision", "brier_score")
    task_macro = {
        name: _mean_non_null(item[name] for item in per_task.values())
        for name in metric_names
    }
    task_macro["task_count"] = len(per_task)
    task_minimum = {
        name: min((item[name] for item in per_task.values() if item[name] is not None), default=None)
        for name in metric_names
    }
    task_maximum = {
        name: max((item[name] for item in per_task.values() if item[name] is not None), default=None)
        for name in metric_names
    }
    return {
        "row_micro": micro,
        "per_task": per_task,
        "task_macro": task_macro,
        "task_minimum": task_minimum,
        "task_maximum": task_maximum,
    }


def _mean_non_null(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return float(np.mean(finite)) if finite else None


def iteration_matched_metrics(
    scored_rows: Sequence[Mapping[str, Any]], model_name: str
) -> dict[str, Any]:
    by_iteration = {}
    valid = []
    for k in sorted({int(row["k"]) for row in scored_rows}):
        group = [row for row in scored_rows if int(row["k"]) == k]
        labels = [int(row["label"]) for row in group]
        scores = [float(row["score"]) for row in group]
        both_classes = len(set(labels)) == 2
        auc = None if model_name == "iteration_only" or not both_classes else exact_rank_auc(labels, scores)
        by_iteration[str(k)] = {
            "row_count": len(group),
            "positive_count": sum(labels),
            "negative_count": len(labels) - sum(labels),
            "roc_auc": auc,
            "valid": auc is not None,
            "null_reason": (
                "constant_iteration_only_score_within_k"
                if model_name == "iteration_only" and both_classes
                else "single_class"
                if not both_classes
                else None
            ),
        }
        if auc is not None:
            valid.append(auc)
    return {
        "by_iteration": by_iteration,
        "unweighted_valid_k_macro_roc_auc": _mean_non_null(valid),
        "valid_k_count": len(valid),
    }


def evaluate_boundary_oof(
    payload: Mapping[str, Any],
    training_bundle: Mapping[str, Any],
    assignment: Mapping[int, int],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    validate_boundary_payload(payload)
    _require(training_bundle["boundary_dataset_sha256"] == payload["boundary_dataset_sha256"], "training/dataset identity mismatch")
    if "source_trajectory_bundle_sha256" in payload:
        _require(
            training_bundle["source_trajectory_bundle_sha256"]
            == payload["source_trajectory_bundle_sha256"],
            "training/source trajectory identity mismatch",
        )
    expected_folds = {
        fold_id: sorted(task for task, value in assignment.items() if value == fold_id)
        for fold_id in sorted(set(assignment.values()))
    }
    actual_folds = {
        int(fold["fold_id"]): sorted(map(int, fold["held_out_task_ids"]))
        for fold in training_bundle["folds"]
    }
    _require(actual_folds == expected_folds, "training bundle outer-fold assignment mismatch")
    prediction_outputs: list[dict[str, Any]] = []
    report_models: dict[str, Any] = {}
    by_task_rows: list[dict[str, Any]] = []
    for target in PRIMARY_TARGETS:
        for model_name in MODEL_CONFIGS:
            scored_boundary_rows = []
            fold_reports = []
            timing_outputs = []
            for fold in training_bundle["folds"]:
                fold_id = int(fold["fold_id"])
                fitted_entry = fold["models"][f"{target}::{model_name}"]
                fitted = fitted_entry["fitted"]
                restored_model = restore_model(fitted)
                held_tasks = set(map(int, fold["held_out_task_ids"]))
                held_rows = [
                    row for row in payload["rows"]
                    if row["target_reference"] == target and int(row["task_id"]) in held_tasks
                ]
                scores = score_rows(fitted, held_rows)
                fold_scored_rows = [
                    {**{key: row[key] for key in ("task_id", "episode_id", "prediction_id", "k", "label")}, "score": score}
                    for row, score in zip(held_rows, scores)
                ]
                scored_boundary_rows.extend(fold_scored_rows)
                threshold = float(fitted_entry["threshold_selection"]["selected_threshold"])
                held_trajectories = [
                    trajectory for trajectory in payload["scoring_trajectories"]
                    if int(trajectory["task_id"]) in held_tasks
                    and trajectory["references"][target] is not None
                    and int(trajectory["references"][target]) >= 4
                ]
                fold_timings = []
                for trajectory in held_trajectories:
                    scores_by_k = _score_trajectory_with_model(
                        fitted, trajectory, restored_model
                    )
                    timing = gate_timing(
                        scores_by_k,
                        threshold,
                        int(trajectory["references"][target]),
                        int(trajectory["max_iter"]),
                    )
                    output = {
                        "task_id": int(trajectory["task_id"]),
                        "episode_id": int(trajectory["episode_id"]),
                        "prediction_id": int(trajectory["prediction_id"]),
                        "target_reference": target,
                        "model_configuration": model_name,
                        "outer_fold": fold_id,
                        "K_reference": int(trajectory["references"][target]),
                        "decision_threshold": threshold,
                        "decision_threshold_hex": threshold.hex(),
                        **timing,
                    }
                    fold_timings.append(output)
                    timing_outputs.append(output)
                    prediction_outputs.append(output)
                held_metrics = aggregate_gate_timings(fold_timings)
                train_metrics = fitted_entry["threshold_selection"]["train_metrics"]
                fold_reports.append(
                    {
                        "fold_id": fold_id,
                        "training_task_ids": fold["training_task_ids"],
                        "held_out_task_ids": fold["held_out_task_ids"],
                        "selected_threshold": threshold,
                        "selected_threshold_hex": threshold.hex(),
                        "training_gate_metrics": train_metrics,
                        "held_out_gate_metrics": held_metrics,
                        "train_vs_held_out_degradation": {
                            "late_plus_missed_rate_delta": (
                                held_metrics["late_plus_missed_count"]
                                / held_metrics["prediction_count"]
                                - int(train_metrics["late_plus_missed_count"])
                                / fitted_entry["training_prediction_count_for_threshold"]
                            ),
                            "missed_rate_delta": held_metrics["missed_rate"] - (
                                int(train_metrics["missed_count"]) / fitted_entry["training_prediction_count_for_threshold"]
                            ),
                        },
                    }
                )
            row_metrics = aggregate_row_metrics(scored_boundary_rows)
            matched = iteration_matched_metrics(scored_boundary_rows, model_name)
            gate_metrics = aggregate_gate_timings(timing_outputs)
            per_task_gate = {}
            for task_id in sorted({row["task_id"] for row in timing_outputs}):
                task_group = [row for row in timing_outputs if row["task_id"] == task_id]
                metrics = aggregate_gate_timings(task_group)
                per_task_gate[str(task_id)] = metrics
                by_task_rows.append(
                    {
                        "source_trajectory_bundle_sha256": training_bundle["source_trajectory_bundle_sha256"],
                        "boundary_dataset_sha256": training_bundle["boundary_dataset_sha256"],
                        "fold_manifest_sha256": training_bundle["fold_manifest_sha256"],
                        "source_git_commit": training_bundle["source_git_commit"],
                        "model_training_configuration": json.dumps(training_bundle["model_training_configuration"], sort_keys=True),
                        "target_reference": target,
                        "model_configuration": model_name,
                        "task_id": task_id,
                        **metrics,
                    }
                )
            report_models[f"{target}::{model_name}"] = {
                "target_reference": target,
                "model_configuration": model_name,
                "task_level_oof": True,
                "threshold_free_metrics": row_metrics,
                "iteration_matched_metrics": matched,
                "thresholded_gate_timing": {
                    "overall": gate_metrics,
                    "per_task": per_task_gate,
                    "folds": fold_reports,
                },
            }
    for output in prediction_outputs:
        output.update(
            {
                "source_trajectory_bundle_sha256": training_bundle["source_trajectory_bundle_sha256"],
                "boundary_dataset_sha256": training_bundle["boundary_dataset_sha256"],
                "fold_manifest_sha256": training_bundle["fold_manifest_sha256"],
                "source_git_commit": training_bundle["source_git_commit"],
                "model_training_configuration": training_bundle["model_training_configuration"],
            }
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "analysis_type": "strict_boundary_task_level_oof_gate_timing_study",
        "source_trajectory_bundle_sha256": training_bundle["source_trajectory_bundle_sha256"],
        "boundary_dataset_sha256": training_bundle["boundary_dataset_sha256"],
        "fold_manifest_sha256": training_bundle["fold_manifest_sha256"],
        "source_git_commit": training_bundle["source_git_commit"],
        "model_training_configuration": training_bundle["model_training_configuration"],
        "target_references": list(PRIMARY_TARGETS),
        "primary_population": "ACTUAL_WARM",
        "cold_population": {
            "status": "excluded from OOF population",
            "prediction_count": int(payload.get("cold_excluded_from_oof_prediction_count", 0)),
        },
        "interpretation_constraints": [
            "all reported model metrics are task-level outer-fold OOF",
            "this is a gate-timing study, not an online-success or safe-stopping claim",
            "K_first and stable-suffix targets are trained and reported separately",
            "no Coda reduction, latency, terminal-K, or promotion result is computed",
        ],
        "models": report_models,
    }
    return report, prediction_outputs, by_task_rows
