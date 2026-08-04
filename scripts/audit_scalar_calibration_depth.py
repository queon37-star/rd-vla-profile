#!/usr/bin/env python3
"""Audit authoritative recurrence-depth coverage and k=3 scalar hard negatives."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prismatic.models.scalar_stopping_policy import (  # noqa: E402
    compute_scalar_stopping_features,
    load_scalar_policy_artifact,
    prepare_scalar_task_policy,
    score_scalar_stopping_policy,
)
from scripts.origin_aware_gpu_microbenchmark_lib import (  # noqa: E402
    GPUMicrobenchmarkValidationError,
    load_json_object,
    sha256_file,
)
from scripts.preconvergence_trigger_lib import (  # noqa: E402
    RawPreconvergenceSequence,
    load_raw_manifest_sequences,
)


DEFAULT_ROOT = (
    REPO_ROOT
    / "benchmark_results/preconvergence_trigger/raw_shadow_calibration_seed7"
)
DEFAULT_PROTOCOL = (
    REPO_ROOT
    / "experiments/robot/libero/manifests/scalar_calibration_depth_audit_v1.json"
)
DEFAULT_SCALAR_POLICY = (
    REPO_ROOT
    / "benchmark_results/preconvergence_trigger/seed7/scalar_runtime_policy_kfirst_v1"
)


class CalibrationDepthAuditError(GPUMicrobenchmarkValidationError):
    """Raised when the frozen calibration audit contract is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CalibrationDepthAuditError(message)


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_jsonl(paths: Sequence[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        _require(path.is_file(), f"missing step log: {path}")
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CalibrationDepthAuditError(
                    f"invalid JSONL record at {path}:{line_number}: {exc}"
                ) from exc
            _require(
                isinstance(value, dict),
                f"step-log record must be an object at {path}:{line_number}",
            )
            records.append(value)
    return records


def _record_key(record: Mapping[str, Any]) -> tuple[int, int, int]:
    prediction_id = record.get(
        "prediction_step", record.get("action_prediction_index")
    )
    _require(prediction_id is not None, "step record has no prediction identity")
    return (
        int(record["task_id"]),
        int(record["episode_id"]),
        int(prediction_id),
    )


def _step_index(
    paths: Sequence[Path],
) -> tuple[dict[tuple[int, int, int], dict[str, Any]], dict[tuple[int, int], bool]]:
    predictions: dict[tuple[int, int, int], dict[str, Any]] = {}
    episode_success: dict[tuple[int, int], bool] = {}
    for record in _load_jsonl(paths):
        key = _record_key(record)
        _require(key not in predictions, f"duplicate step-log identity: {key}")
        success = record.get("success")
        _require(
            isinstance(success, bool),
            f"step-log success is not finalized for identity {key}",
        )
        predictions[key] = record
        episode_key = key[:2]
        if episode_key in episode_success:
            _require(
                episode_success[episode_key] == success,
                f"inconsistent success labels in episode {episode_key}",
            )
        else:
            episode_success[episode_key] = success
    return predictions, episode_success


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    _require(protocol.get("schema_version") == 1, "unsupported audit protocol")
    _require(
        protocol.get("primary_origin") == "ACTUAL_WARM",
        "primary audit origin must be ACTUAL_WARM",
    )
    for field in (
        "expected_prediction_count",
        "expected_actual_warm_count",
        "expected_cold_count",
        "hard_depth_threshold",
        "very_hard_depth_threshold",
        "minimum_scalar_gate_iteration",
    ):
        value = protocol.get(field)
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value > 0,
            f"protocol field {field!r} must be a positive integer",
        )
    _require(
        int(protocol["minimum_scalar_gate_iteration"]) == 3,
        "this audit requires scalar scoring at k=3",
    )
    expected_tasks = protocol.get("expected_task_ids")
    _require(
        expected_tasks == list(range(10)),
        "formal audit must cover LIBERO Spatial task IDs 0..9",
    )
    _require(
        protocol.get("latency_scope")
        == "post-VLM action-policy path; VLM backbone excluded",
        "latency reporting scope changed unexpectedly",
    )


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _exclusive_bucket(k_action: int) -> str:
    if k_action <= 3:
        return "K<=3"
    if k_action == 4:
        return "K=4"
    if k_action == 5:
        return "K=5"
    if k_action <= 7:
        return "K=6-7"
    return "K>=8"


def _depth_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    hard_depth: int,
    very_hard_depth: int,
) -> dict[str, Any]:
    values = [int(row["K_action"]) for row in rows]
    exact = Counter(values)
    exclusive = Counter(_exclusive_bucket(value) for value in values)
    count = len(values)
    return {
        "prediction_count": count,
        "mean_K_action": float(statistics.fmean(values)) if values else None,
        "median_K_action": float(statistics.median(values)) if values else None,
        "p95_K_action": _percentile(values, 0.95),
        "minimum_K_action": min(values) if values else None,
        "maximum_K_action": max(values) if values else None,
        "exact_K_histogram": {
            str(key): int(value) for key, value in sorted(exact.items())
        },
        "requested_exact_counts": {
            "K=3": int(exact.get(3, 0)),
            "K=4": int(exact.get(4, 0)),
            "K=5": int(exact.get(5, 0)),
        },
        "threshold_counts": {
            f"K>={hard_depth}": int(sum(value >= hard_depth for value in values)),
            f"K>={very_hard_depth}": int(
                sum(value >= very_hard_depth for value in values)
            ),
        },
        "threshold_rates": {
            f"K>={hard_depth}": (
                float(sum(value >= hard_depth for value in values) / count)
                if count
                else None
            ),
            f"K>={very_hard_depth}": (
                float(sum(value >= very_hard_depth for value in values) / count)
                if count
                else None
            ),
        },
        "exclusive_bucket_counts": {
            name: int(exclusive.get(name, 0))
            for name in ("K<=3", "K=4", "K=5", "K=6-7", "K>=8")
        },
    }


def _per_task_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    hard_depth: int,
    very_hard_depth: int,
) -> dict[str, Any]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["task_id"])].append(row)
    total_hard = sum(int(row["K_action"]) >= hard_depth for row in rows)
    total_very_hard = sum(
        int(row["K_action"]) >= very_hard_depth for row in rows
    )
    result = {}
    for task_id, task_rows in sorted(grouped.items()):
        summary = _depth_summary(
            task_rows,
            hard_depth=hard_depth,
            very_hard_depth=very_hard_depth,
        )
        hard_count = summary["threshold_counts"][f"K>={hard_depth}"]
        very_hard_count = summary["threshold_counts"][f"K>={very_hard_depth}"]
        summary["share_of_all_hard_predictions"] = (
            hard_count / total_hard if total_hard else 0.0
        )
        summary["share_of_all_very_hard_predictions"] = (
            very_hard_count / total_very_hard if total_very_hard else 0.0
        )
        result[str(task_id)] = summary
    return result


def _episode_concentration(
    rows: Sequence[Mapping[str, Any]],
    *,
    hard_depth: int,
    very_hard_depth: int,
) -> dict[str, Any]:
    grouped: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["task_id"]), int(row["episode_id"]))].append(row)
    episodes = []
    for (task_id, episode_id), episode_rows in grouped.items():
        values = [int(row["K_action"]) for row in episode_rows]
        episodes.append(
            {
                "task_id": task_id,
                "episode_id": episode_id,
                "success": bool(episode_rows[0]["success"]),
                "prediction_count": len(values),
                "mean_K_action": float(statistics.fmean(values)),
                "maximum_K_action": max(values),
                f"K>={hard_depth}_count": int(
                    sum(value >= hard_depth for value in values)
                ),
                f"K>={very_hard_depth}_count": int(
                    sum(value >= very_hard_depth for value in values)
                ),
            }
        )
    episodes.sort(
        key=lambda row: (
            -int(row[f"K>={hard_depth}_count"]),
            -int(row[f"K>={very_hard_depth}_count"]),
            -float(row["mean_K_action"]),
            int(row["task_id"]),
            int(row["episode_id"]),
        )
    )
    return {
        "episode_count": len(episodes),
        "episodes_with_hard_prediction_count": int(
            sum(row[f"K>={hard_depth}_count"] > 0 for row in episodes)
        ),
        "episodes_with_very_hard_prediction_count": int(
            sum(row[f"K>={very_hard_depth}_count"] > 0 for row in episodes)
        ),
        "top_hard_episodes": episodes[:20],
        "all_episodes": sorted(
            episodes, key=lambda row: (row["task_id"], row["episode_id"])
        ),
    }


def _outcome_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    hard_depth: int,
    very_hard_depth: int,
) -> dict[str, Any]:
    result = {}
    for success, name in ((True, "success"), (False, "failure")):
        selected = [row for row in rows if bool(row["success"]) is success]
        episodes = {(int(row["task_id"]), int(row["episode_id"])) for row in selected}
        result[name] = {
            "episode_count": len(episodes),
            **_depth_summary(
                selected,
                hard_depth=hard_depth,
                very_hard_depth=very_hard_depth,
            ),
        }
    return result


def _score_k3(
    sequence: RawPreconvergenceSequence,
    policy,
) -> tuple[float, float, bool]:
    features = compute_scalar_stopping_features(
        sequence.states[2],
        sequence.states[1],
        sequence.states[0],
        iteration=3,
        epsilon=policy.epsilon,
    )
    score = float(score_scalar_stopping_policy(policy, features).item())
    _require(math.isfinite(score), "non-finite scalar score at k=3")
    margin = score - float(policy.threshold)
    return score, margin, score >= float(policy.threshold)


def _hard_negative_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    hard_depth: int,
    very_hard_depth: int,
) -> dict[str, Any]:
    warm = [row for row in rows if row["actual_origin"] == "ACTUAL_WARM"]
    hard = [row for row in warm if int(row["K_action"]) >= hard_depth]
    very_hard = [
        row for row in warm if int(row["K_action"]) >= very_hard_depth
    ]
    hard_negatives = [row for row in hard if row["k3_triggered"]]
    very_hard_negatives = [row for row in very_hard if row["k3_triggered"]]
    by_task = Counter(int(row["task_id"]) for row in hard_negatives)
    top = sorted(
        hard_negatives,
        key=lambda row: (
            -float(row["k3_score_margin"]),
            -float(row["k3_scalar_score"]),
            -int(row["K_action"]),
            int(row["task_id"]),
            int(row["episode_id"]),
            int(row["prediction_id"]),
        ),
    )
    return {
        "definition": (
            f"ACTUAL_WARM, K_action>={hard_depth}, and task-OOF scalar "
            "score at k=3 >= task threshold"
        ),
        "hard_prediction_count": len(hard),
        "hard_negative_count": len(hard_negatives),
        "hard_negative_rate_within_hard": (
            len(hard_negatives) / len(hard) if hard else None
        ),
        "very_hard_prediction_count": len(very_hard),
        "very_hard_negative_count": len(very_hard_negatives),
        "very_hard_negative_rate_within_very_hard": (
            len(very_hard_negatives) / len(very_hard) if very_hard else None
        ),
        "hard_negative_task_counts": {
            str(task_id): int(by_task.get(task_id, 0)) for task_id in range(10)
        },
        "top_hard_negatives": [
            {
                "task_id": int(row["task_id"]),
                "episode_id": int(row["episode_id"]),
                "prediction_id": int(row["prediction_id"]),
                "success": bool(row["success"]),
                "K_action": int(row["K_action"]),
                "k3_scalar_score": float(row["k3_scalar_score"]),
                "task_threshold": float(row["task_threshold"]),
                "k3_score_margin": float(row["k3_score_margin"]),
            }
            for row in top[:50]
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-manifest",
        type=Path,
        action="append",
        default=None,
        help="Schema-2 raw manifest; may be supplied more than once.",
    )
    parser.add_argument(
        "--step-log",
        type=Path,
        action="append",
        default=None,
        help="Finalized rollout step JSONL; may be supplied more than once.",
    )
    parser.add_argument("--scalar-policy", type=Path, default=DEFAULT_SCALAR_POLICY)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite audit report: {args.output}")

    raw_manifests = args.raw_manifest or [DEFAULT_ROOT / "manifest.json"]
    step_logs = args.step_log or [DEFAULT_ROOT / "steps.jsonl"]
    protocol = load_json_object(args.protocol)
    _validate_protocol(protocol)

    dataset_metadata, sequences = load_raw_manifest_sequences(raw_manifests)
    step_records, episode_success = _step_index(step_logs)
    scalar_manifest, scalar_payload = load_scalar_policy_artifact(args.scalar_policy)

    expected_tasks = set(int(value) for value in protocol["expected_task_ids"])
    observed_tasks = {sequence.identity.task_id for sequence in sequences}
    _require(observed_tasks == expected_tasks, "raw calibration task coverage mismatch")

    origin_counts = Counter(sequence.actual_origin for sequence in sequences)
    _require(
        len(sequences) == int(protocol["expected_prediction_count"]),
        "raw calibration prediction count mismatch",
    )
    _require(
        origin_counts["ACTUAL_WARM"]
        == int(protocol["expected_actual_warm_count"]),
        "actual-warm prediction count mismatch",
    )
    _require(
        origin_counts["COLD"] == int(protocol["expected_cold_count"]),
        "cold prediction count mismatch",
    )

    policy_cache = {
        task_id: prepare_scalar_task_policy(
            scalar_payload, task_id, device=torch.device("cpu")
        )
        for task_id in sorted(expected_tasks)
    }
    rows: list[dict[str, Any]] = []
    missing_step_keys = []
    for sequence in sequences:
        key = sequence.identity.key
        if key not in step_records:
            missing_step_keys.append(key)
            continue
        step = step_records[key]
        _require(
            str(step.get("actual_origin"))
            in {sequence.actual_origin, "COLD_PRIMARY", "COLD_RETRY"},
            f"origin mismatch for prediction {key}",
        )
        success = episode_success[key[:2]]
        row = {
            "task_id": sequence.identity.task_id,
            "episode_id": sequence.identity.episode_id,
            "prediction_id": sequence.identity.prediction_id,
            "actual_origin": sequence.actual_origin,
            "success": success,
            "K_action": int(sequence.k_action),
            "production_terminal_k": int(sequence.baseline_k),
            "timestep": int(step.get("timestep", -1)),
        }
        _require(
            row["K_action"] == row["production_terminal_k"],
            f"authoritative K mismatch for prediction {key}",
        )
        if sequence.actual_origin == "ACTUAL_WARM":
            policy = policy_cache[sequence.identity.task_id]
            score, margin, triggered = _score_k3(sequence, policy)
            row.update(
                {
                    "k3_scalar_score": score,
                    "task_threshold": float(policy.threshold),
                    "k3_score_margin": margin,
                    "k3_triggered": bool(triggered),
                }
            )
        else:
            row.update(
                {
                    "k3_scalar_score": None,
                    "task_threshold": None,
                    "k3_score_margin": None,
                    "k3_triggered": False,
                }
            )
        rows.append(row)

    _require(not missing_step_keys, f"missing step records: {len(missing_step_keys)}")
    _require(len(rows) == len(sequences), "audit row count mismatch")

    hard_depth = int(protocol["hard_depth_threshold"])
    very_hard_depth = int(protocol["very_hard_depth_threshold"])
    warm_rows = [row for row in rows if row["actual_origin"] == "ACTUAL_WARM"]
    cold_rows = [row for row in rows if row["actual_origin"] == "COLD"]

    report = {
        "schema_version": 1,
        "formal_run": True,
        "code_git_commit": _git_commit(),
        "latency_reporting_scope": protocol["latency_scope"],
        "protocol": protocol,
        "inputs": {
            "raw_manifests": [
                {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for path in raw_manifests
            ],
            "step_logs": [
                {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for path in step_logs
            ],
            "scalar_policy": {
                "path": str(args.scalar_policy.resolve()),
                "manifest": scalar_manifest,
            },
            "protocol_manifest": {
                "path": str(args.protocol.resolve()),
                "sha256": sha256_file(args.protocol),
            },
            "dataset_metadata": dataset_metadata,
        },
        "validation": {
            "prediction_count": len(rows),
            "actual_warm_count": len(warm_rows),
            "cold_count": len(cold_rows),
            "task_ids": sorted(observed_tasks),
            "step_log_join_missing_count": 0,
            "authoritative_K_mismatch_count": 0,
            "scalar_score_nonfinite_count": 0,
        },
        "actual_warm": {
            "overall_depth": _depth_summary(
                warm_rows,
                hard_depth=hard_depth,
                very_hard_depth=very_hard_depth,
            ),
            "per_task_depth": _per_task_summary(
                warm_rows,
                hard_depth=hard_depth,
                very_hard_depth=very_hard_depth,
            ),
            "episode_concentration": _episode_concentration(
                warm_rows,
                hard_depth=hard_depth,
                very_hard_depth=very_hard_depth,
            ),
            "success_failure_depth": _outcome_summary(
                warm_rows,
                hard_depth=hard_depth,
                very_hard_depth=very_hard_depth,
            ),
            "k3_scalar_hard_negatives": _hard_negative_summary(
                warm_rows,
                hard_depth=hard_depth,
                very_hard_depth=very_hard_depth,
            ),
        },
        "cold_supplementary": {
            "overall_depth": _depth_summary(
                cold_rows,
                hard_depth=hard_depth,
                very_hard_depth=very_hard_depth,
            ),
            "per_task_depth": _per_task_summary(
                cold_rows,
                hard_depth=hard_depth,
                very_hard_depth=very_hard_depth,
            ),
        },
        "prediction_rows": sorted(
            rows,
            key=lambda row: (
                row["task_id"], row["episode_id"], row["prediction_id"]
            ),
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    warm_depth = report["actual_warm"]["overall_depth"]
    hard_negative = report["actual_warm"]["k3_scalar_hard_negatives"]
    print(f"Formal run: {report['formal_run']}")
    print(f"Predictions: {len(rows)} (warm={len(warm_rows)}, cold={len(cold_rows)})")
    print(
        f"Warm K>=6: {warm_depth['threshold_counts']['K>=6']} "
        f"({100.0 * warm_depth['threshold_rates']['K>=6']:.3f}%)"
    )
    print(
        f"Warm K>=8: {warm_depth['threshold_counts']['K>=8']} "
        f"({100.0 * warm_depth['threshold_rates']['K>=8']:.3f}%)"
    )
    print(
        "k=3 scalar hard negatives (K>=6): "
        f"{hard_negative['hard_negative_count']}/"
        f"{hard_negative['hard_prediction_count']}"
    )
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
