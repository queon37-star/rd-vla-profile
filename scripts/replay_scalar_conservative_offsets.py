#!/usr/bin/env python3
"""Replay conservative scalar threshold offsets over full recurrent trajectories."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    load_json_object,
    sha256_file,
)
from scripts.preconvergence_trigger_lib import (  # noqa: E402
    RawPreconvergenceSequence,
    load_raw_manifest_sequences,
)


DEFAULT_RAW_ROOT = (
    REPO_ROOT
    / "benchmark_results/preconvergence_trigger/raw_shadow_calibration_seed7"
)
DEFAULT_SCALAR_POLICY = (
    REPO_ROOT
    / "benchmark_results/preconvergence_trigger/seed7/scalar_runtime_policy_kfirst_v1"
)
DEFAULT_DEPTH_AUDIT = (
    REPO_ROOT
    / "benchmark_results/preconvergence_trigger/seed7/calibration_depth_audit/report.json"
)
DEFAULT_K3_AUDIT = (
    REPO_ROOT
    / "benchmark_results/preconvergence_trigger/seed7/k3_separability_audit/report.json"
)
DEFAULT_PROTOCOL = (
    REPO_ROOT
    / "experiments/robot/libero/manifests/scalar_conservative_offset_replay_v1.json"
)


class ConservativeOffsetReplayError(ValueError):
    """Raised when an input or replay violates the frozen descriptive contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ConservativeOffsetReplayError(message)


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def numeric_summary(values: Sequence[float]) -> dict[str, Any]:
    values = [float(value) for value in values]
    require(all(math.isfinite(value) for value in values), "non-finite numeric value")
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p05": percentile(values, 0.05),
        "p95": percentile(values, 0.95),
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    require(protocol.get("schema_version") == 1, "unsupported replay protocol")
    require(protocol.get("primary_origin") == "ACTUAL_WARM", "primary origin mismatch")
    require(protocol.get("execution_mode") == "confirm_next", "execution mode mismatch")
    require(protocol.get("minimum_gate_iteration") == 3, "minimum gate iteration mismatch")
    require(protocol.get("promotion_allowed") is False, "descriptive replay cannot promote")
    require(
        protocol.get("latency_scope")
        == "post-VLM action-policy path; VLM backbone excluded",
        "latency scope mismatch",
    )
    require(protocol.get("expected_task_ids") == list(range(10)), "task coverage mismatch")
    for field in (
        "expected_prediction_count",
        "expected_actual_warm_count",
        "expected_cold_count",
    ):
        value = protocol.get(field)
        require(
            isinstance(value, int) and not isinstance(value, bool) and value > 0,
            f"invalid protocol field: {field}",
        )
    for field in (
        "planning_recurrent_iteration_ms",
        "planning_repeated_output_path_ms",
        "planning_gate_evaluation_ms",
    ):
        value = float(protocol.get(field, float("nan")))
        require(math.isfinite(value) and value >= 0.0, f"invalid cost anchor: {field}")


def identity_key(row: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        int(row["task_id"]),
        int(row["episode_id"]),
        int(row["prediction_id"]),
    )


def load_depth_index(path: Path) -> tuple[dict[str, Any], dict[tuple[int, int, int], dict[str, Any]]]:
    report = load_json_object(path)
    require(report.get("formal_run") is True, "depth audit is not formal")
    rows = report.get("prediction_rows")
    require(isinstance(rows, list), "depth audit has no prediction rows")
    index: dict[tuple[int, int, int], dict[str, Any]] = {}
    for row in rows:
        require(isinstance(row, dict), "depth audit row must be an object")
        key = identity_key(row)
        require(key not in index, f"duplicate depth-audit identity: {key}")
        index[key] = row
    return report, index


def candidate_offsets(k3_report: Mapping[str, Any]) -> list[dict[str, Any]]:
    require(k3_report.get("formal_run") is True, "k=3 separability audit is not formal")
    candidates = [
        {
            "name": "deployed_offset_0",
            "source": "deployed task-specific threshold",
            "margin_offset": 0.0,
            "k3_severe_fpr_limit": None,
        }
    ]
    selected = k3_report.get("uniform_margin_offset_sweep", {}).get(
        "selected_descriptive_operating_points"
    )
    require(isinstance(selected, Mapping), "k=3 audit has no descriptive offsets")
    expected_limits = (0.01, 0.05, 0.10, 0.20)
    for limit in expected_limits:
        key = f"severe_FPR<={limit:.2f}"
        point = selected.get(key)
        require(isinstance(point, Mapping), f"missing k=3 operating point: {key}")
        offset = float(point.get("margin_offset", float("nan")))
        observed_fpr = float(point.get("severe_false_trigger_rate", float("nan")))
        require(math.isfinite(offset), f"non-finite margin offset: {key}")
        require(
            math.isfinite(observed_fpr) and observed_fpr <= limit + 1e-12,
            f"invalid severe-FPR point: {key}",
        )
        candidates.append(
            {
                "name": f"k3_severe_fpr_cap_{int(round(limit * 100)):02d}pct",
                "source": key,
                "margin_offset": offset,
                "k3_severe_fpr_limit": limit,
                "k3_observed_severe_fpr": observed_fpr,
                "k3_observed_safe_recall": float(point["safe_trigger_recall"]),
            }
        )
    offsets = [candidate["margin_offset"] for candidate in candidates]
    require(len(set(offsets)) == len(offsets), "candidate offsets are not unique")
    return candidates


def score_full_trajectory(
    sequence: RawPreconvergenceSequence,
    policy,
) -> dict[int, float]:
    scores: dict[int, float] = {}
    for iteration in range(policy.minimum_gate_iteration, sequence.max_iter + 1):
        features = compute_scalar_stopping_features(
            sequence.states[iteration - 1],
            sequence.states[iteration - 2],
            sequence.states[iteration - 3],
            iteration=iteration,
            epsilon=policy.epsilon,
        )
        score = float(score_scalar_stopping_policy(policy, features).item())
        require(math.isfinite(score), f"non-finite scalar score at k={iteration}")
        scores[iteration] = score
    return scores


def replay_one(
    sequence: RawPreconvergenceSequence,
    *,
    scores: Mapping[int, float],
    base_threshold: float,
    offset: float,
    success: bool,
    costs: Mapping[str, float],
) -> dict[str, Any]:
    effective_threshold = float(base_threshold) + float(offset)
    require(math.isfinite(effective_threshold), "non-finite effective threshold")
    gate_iteration = next(
        (
            iteration
            for iteration in range(3, sequence.max_iter + 1)
            if float(scores[iteration]) >= effective_threshold
        ),
        None,
    )
    terminal_k = (
        sequence.max_iter
        if gate_iteration is None
        else min(sequence.max_iter, int(gate_iteration) + 1)
    )
    baseline_k = int(sequence.k_action)
    delta_k = int(terminal_k - baseline_k)
    baseline_coda_calls = baseline_k
    candidate_coda_calls = 1
    saved_coda_calls = baseline_coda_calls - candidate_coda_calls
    gate_evaluations = (
        sequence.max_iter - 2 if gate_iteration is None else int(gate_iteration) - 2
    )
    recurrent_ms = float(costs["recurrent"])
    output_ms = float(costs["output"])
    gate_ms = float(costs["gate"])
    projected_net_saving_ms = (
        saved_coda_calls * output_ms
        - delta_k * recurrent_ms
        - gate_evaluations * gate_ms
    )
    baseline_variable_ms = baseline_k * (recurrent_ms + output_ms)
    candidate_variable_ms = (
        terminal_k * recurrent_ms
        + candidate_coda_calls * output_ms
        + gate_evaluations * gate_ms
    )
    return {
        "task_id": sequence.identity.task_id,
        "episode_id": sequence.identity.episode_id,
        "prediction_id": sequence.identity.prediction_id,
        "success": bool(success),
        "K_action": baseline_k,
        "gate_iteration": gate_iteration,
        "terminal_k": int(terminal_k),
        "delta_k": delta_k,
        "effective_threshold": effective_threshold,
        "base_task_threshold": float(base_threshold),
        "margin_offset": float(offset),
        "gate_evaluation_count": int(gate_evaluations),
        "baseline_coda_calls": baseline_coda_calls,
        "candidate_coda_calls": candidate_coda_calls,
        "saved_coda_calls": saved_coda_calls,
        "terminal_matches_action_mse_k": terminal_k == baseline_k,
        "terminal_earlier_than_action_mse": terminal_k < baseline_k,
        "terminal_later_than_action_mse": terminal_k > baseline_k,
        "projected_net_saving_ms": projected_net_saving_ms,
        "planning_baseline_variable_ms": baseline_variable_ms,
        "planning_candidate_variable_ms": candidate_variable_ms,
    }


def aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(bool(records), "cannot aggregate an empty replay")
    count = len(records)
    deltas = [int(record["delta_k"]) for record in records]
    baseline_k = [int(record["K_action"]) for record in records]
    terminal_k = [int(record["terminal_k"]) for record in records]
    early = [record for record in records if int(record["delta_k"]) < 0]
    exact = [record for record in records if int(record["delta_k"]) == 0]
    late = [record for record in records if int(record["delta_k"]) > 0]
    severe = [record for record in records if int(record["K_action"]) >= 6]
    very_severe = [record for record in records if int(record["K_action"]) >= 8]
    baseline_coda = sum(int(record["baseline_coda_calls"]) for record in records)
    candidate_coda = sum(int(record["candidate_coda_calls"]) for record in records)
    baseline_variable = sum(float(record["planning_baseline_variable_ms"]) for record in records)
    candidate_variable = sum(float(record["planning_candidate_variable_ms"]) for record in records)
    return {
        "prediction_count": count,
        "baseline_K": numeric_summary(baseline_k),
        "candidate_terminal_K": numeric_summary(terminal_k),
        "delta_K": numeric_summary(deltas),
        "early_terminal_count": len(early),
        "early_terminal_rate": len(early) / count,
        "exact_terminal_count": len(exact),
        "exact_terminal_rate": len(exact) / count,
        "late_terminal_count": len(late),
        "late_terminal_rate": len(late) / count,
        "delta_K_le_minus_2_count": sum(value <= -2 for value in deltas),
        "delta_K_le_minus_4_count": sum(value <= -4 for value in deltas),
        "delta_K_ge_plus_2_count": sum(value >= 2 for value in deltas),
        "delta_K_ge_plus_4_count": sum(value >= 4 for value in deltas),
        "severe_K_ge_6_count": len(severe),
        "severe_early_terminal_count": sum(int(record["delta_k"]) < 0 for record in severe),
        "severe_early_terminal_rate": (
            sum(int(record["delta_k"]) < 0 for record in severe) / len(severe)
            if severe
            else None
        ),
        "very_severe_K_ge_8_count": len(very_severe),
        "very_severe_early_terminal_count": sum(
            int(record["delta_k"]) < 0 for record in very_severe
        ),
        "very_severe_early_terminal_rate": (
            sum(int(record["delta_k"]) < 0 for record in very_severe)
            / len(very_severe)
            if very_severe
            else None
        ),
        "no_gate_before_max_count": sum(record["gate_iteration"] is None for record in records),
        "no_gate_before_max_rate": sum(record["gate_iteration"] is None for record in records) / count,
        "mean_gate_evaluations": statistics.fmean(
            float(record["gate_evaluation_count"]) for record in records
        ),
        "baseline_coda_calls": baseline_coda,
        "candidate_coda_calls": candidate_coda,
        "saved_coda_calls": baseline_coda - candidate_coda,
        "coda_call_reduction": (baseline_coda - candidate_coda) / baseline_coda,
        "planning_cost": {
            "mean_projected_net_saving_ms": statistics.fmean(
                float(record["projected_net_saving_ms"]) for record in records
            ),
            "mean_baseline_variable_ms": baseline_variable / count,
            "mean_candidate_variable_ms": candidate_variable / count,
            "relative_variable_latency_reduction": (
                (baseline_variable - candidate_variable) / baseline_variable
                if baseline_variable
                else None
            ),
        },
    }


def aggregate_per_task(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[int(record["task_id"])].append(record)
    return {str(task_id): aggregate(items) for task_id, items in sorted(grouped.items())}


def aggregate_by_success(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for value, name in ((True, "success"), (False, "failure")):
        selected = [record for record in records if bool(record["success"]) is value]
        result[name] = {
            "episode_count": len(
                {(int(record["task_id"]), int(record["episode_id"])) for record in selected}
            ),
            "metrics": aggregate(selected) if selected else None,
        }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-manifest", type=Path, default=DEFAULT_RAW_ROOT / "manifest.json")
    parser.add_argument("--scalar-policy", type=Path, default=DEFAULT_SCALAR_POLICY)
    parser.add_argument("--depth-audit", type=Path, default=DEFAULT_DEPTH_AUDIT)
    parser.add_argument("--k3-audit", type=Path, default=DEFAULT_K3_AUDIT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite replay report: {args.output}")

    protocol = load_json_object(args.protocol)
    validate_protocol(protocol)
    depth_report, depth_index = load_depth_index(args.depth_audit)
    k3_report = load_json_object(args.k3_audit)
    require(
        depth_report.get("latency_reporting_scope") == protocol["latency_scope"],
        "depth-audit latency scope mismatch",
    )
    require(
        k3_report.get("latency_reporting_scope") == protocol["latency_scope"],
        "k=3 audit latency scope mismatch",
    )
    candidates = candidate_offsets(k3_report)

    dataset_metadata, sequences = load_raw_manifest_sequences(args.raw_manifest)
    origin_counts = Counter(sequence.actual_origin for sequence in sequences)
    require(
        len(sequences) == int(protocol["expected_prediction_count"]),
        "raw prediction count mismatch",
    )
    require(
        origin_counts["ACTUAL_WARM"] == int(protocol["expected_actual_warm_count"]),
        "raw actual-warm count mismatch",
    )
    require(
        origin_counts["COLD"] == int(protocol["expected_cold_count"]),
        "raw cold count mismatch",
    )
    observed_tasks = sorted({sequence.identity.task_id for sequence in sequences})
    require(observed_tasks == protocol["expected_task_ids"], "raw task coverage mismatch")

    scalar_manifest, scalar_payload = load_scalar_policy_artifact(args.scalar_policy)
    policies = {
        task_id: prepare_scalar_task_policy(
            scalar_payload, task_id, device=torch.device("cpu")
        )
        for task_id in observed_tasks
    }
    warm_sequences = [
        sequence for sequence in sequences if sequence.actual_origin == "ACTUAL_WARM"
    ]
    costs = {
        "recurrent": float(protocol["planning_recurrent_iteration_ms"]),
        "output": float(protocol["planning_repeated_output_path_ms"]),
        "gate": float(protocol["planning_gate_evaluation_ms"]),
    }

    scored: list[tuple[RawPreconvergenceSequence, dict[int, float], bool]] = []
    missing_depth = []
    for index, sequence in enumerate(warm_sequences, start=1):
        key = sequence.identity.key
        row = depth_index.get(key)
        if row is None:
            missing_depth.append(key)
            continue
        require(row["actual_origin"] == "ACTUAL_WARM", f"depth origin mismatch: {key}")
        require(int(row["K_action"]) == int(sequence.k_action), f"depth K mismatch: {key}")
        scores = score_full_trajectory(sequence, policies[sequence.identity.task_id])
        scored.append((sequence, scores, bool(row["success"])))
        if index % 200 == 0 or index == len(warm_sequences):
            print(f"Scored trajectories: {index}/{len(warm_sequences)}")
    require(not missing_depth, f"missing depth-audit rows: {len(missing_depth)}")
    require(len(scored) == len(warm_sequences), "scored trajectory count mismatch")

    candidate_reports = []
    for candidate in candidates:
        records = []
        offset = float(candidate["margin_offset"])
        for sequence, scores, success in scored:
            policy = policies[sequence.identity.task_id]
            records.append(
                replay_one(
                    sequence,
                    scores=scores,
                    base_threshold=policy.threshold,
                    offset=offset,
                    success=success,
                    costs=costs,
                )
            )
        candidate_reports.append(
            {
                "candidate": candidate,
                "overall": aggregate(records),
                "per_task": aggregate_per_task(records),
                "success_failure": aggregate_by_success(records),
                "prediction_records": records,
            }
        )

    report = {
        "schema_version": 1,
        "formal_input_validation": True,
        "descriptive_replay": True,
        "promotion_eligible": False,
        "promotion_block_reason": (
            "offsets were selected after inspecting the same calibration population; "
            "a fresh development or nested train-only selection protocol is required"
        ),
        "code_git_commit": git_commit(),
        "latency_reporting_scope": protocol["latency_scope"],
        "protocol": protocol,
        "inputs": {
            "raw_manifest": {
                "path": str(args.raw_manifest.resolve()),
                "sha256": sha256_file(args.raw_manifest),
            },
            "scalar_policy": {
                "path": str(args.scalar_policy.resolve()),
                "manifest": scalar_manifest,
            },
            "depth_audit": {
                "path": str(args.depth_audit.resolve()),
                "sha256": sha256_file(args.depth_audit),
            },
            "k3_audit": {
                "path": str(args.k3_audit.resolve()),
                "sha256": sha256_file(args.k3_audit),
            },
            "protocol_manifest": {
                "path": str(args.protocol.resolve()),
                "sha256": sha256_file(args.protocol),
            },
            "dataset_metadata": dataset_metadata,
        },
        "validation": {
            "prediction_count": len(sequences),
            "actual_warm_count": len(warm_sequences),
            "cold_count": origin_counts["COLD"],
            "task_ids": observed_tasks,
            "missing_depth_row_count": 0,
            "nonfinite_score_count": 0,
            "candidate_count": len(candidates),
        },
        "planning_cost_contract": {
            "status": "mechanism_only_delta_projection",
            "recurrent_iteration_ms": costs["recurrent"],
            "repeated_output_path_ms": costs["output"],
            "gate_evaluation_ms_assumption": costs["gate"],
            "excluded": [
                "VLM backbone",
                "fixed post-VLM overhead that cancels in the delta model",
                "unmeasured scalar control and synchronization overhead",
            ],
        },
        "candidates": candidate_reports,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(f"Formal input validation: {report['formal_input_validation']}")
    print(f"Descriptive replay only: {report['descriptive_replay']}")
    print(f"Actual-warm predictions: {len(warm_sequences)}")
    for item in candidate_reports:
        name = item["candidate"]["name"]
        overall = item["overall"]
        print(
            f"{name}: early={100.0 * overall['early_terminal_rate']:.3f}%, "
            f"exact={100.0 * overall['exact_terminal_rate']:.3f}%, "
            f"mean_delta_K={overall['delta_K']['mean']:.4f}, "
            f"p95_delta_K={overall['delta_K']['p95']:.4f}, "
            f"projected_net={overall['planning_cost']['mean_projected_net_saving_ms']:.4f} ms"
        )
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
