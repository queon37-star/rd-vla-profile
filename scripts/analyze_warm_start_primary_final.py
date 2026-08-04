#!/usr/bin/env python3
"""Analyze the frozen 47-state/task warm-start primary evaluation.

The analyzer combines calibration, screening, and final phase validation
reports, excludes the three predeclared preflight-observed states per task, and
applies the frozen pooled paired-trinomial profile-likelihood decision rule.
Secondary efficiency outcomes remain descriptive.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from validate_warm_start_interval_methods import one_sided_profile_p_value


TASK_COUNT = 10
PAIRS_PER_TASK = 47
TOTAL_PAIRS = TASK_COUNT * PAIRS_PER_TASK
MARGIN = 0.05
CUTOFF = 0.045
PHASES = ("calibration", "screening", "final")
EFFICIENCY_FIELDS = (
    "num_predictions",
    "recurrent_calls",
    "get_output_calls",
    "inference_time_ms",
    "mean_prediction_latency_ms",
    "mean_K",
)


class FinalAnalysisError(ValueError):
    """Raised when final acquisition artifacts violate the frozen contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalAnalysisError(message)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def percentile(values: list[float], probability: float) -> float:
    require(values, "percentile requires data")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def numeric_summary(values: Iterable[float]) -> dict[str, Any]:
    numbers = [float(value) for value in values if math.isfinite(float(value))]
    require(numbers, "numeric summary requires finite values")
    return {
        "count": len(numbers),
        "mean": sum(numbers) / len(numbers),
        "median": percentile(numbers, 0.50),
        "p05": percentile(numbers, 0.05),
        "p95": percentile(numbers, 0.95),
        "min": min(numbers),
        "max": max(numbers),
    }


def calibrated_profile_lower_bound(
    minus: int, zero: int, plus: int, *, cutoff: float
) -> float:
    total = minus + zero + plus
    require(total > 0, "profile lower bound requires pairs")
    estimate = (plus - minus) / total
    low = -1.0
    high = float(estimate)
    if one_sided_profile_p_value(minus, zero, plus, low) >= cutoff:
        return low
    for _ in range(120):
        midpoint = 0.5 * (low + high)
        p_value = one_sided_profile_p_value(minus, zero, plus, midpoint)
        if p_value < cutoff:
            low = midpoint
        else:
            high = midpoint
    return 0.5 * (low + high)


def category(record: dict[str, Any]) -> str:
    cold = bool(record["cold_success"])
    warm = bool(record["warm_success"])
    if cold and warm:
        return "both_success"
    if cold:
        return "cold_only_success"
    if warm:
        return "warm_only_success"
    return "both_failure"


def validate_freeze(payload: dict[str, Any]) -> None:
    require(
        payload.get("status") == "primary_analysis_frozen_runtime_not_authorized",
        "wrong analysis-freeze status",
    )
    primary = payload.get("primary_estimand")
    noninferiority = payload.get("noninferiority")
    require(isinstance(primary, dict) and isinstance(noninferiority, dict), "freeze is incomplete")
    require(primary.get("task_count") == TASK_COUNT, "task count mismatch")
    require(primary.get("pairs_per_task") == PAIRS_PER_TASK, "pairs/task mismatch")
    require(primary.get("total_primary_pairs") == TOTAL_PAIRS, "total-pair mismatch")
    require(
        abs(float(noninferiority.get("margin_absolute_success_probability", -1.0)) - MARGIN)
        <= 1e-12,
        "margin mismatch",
    )
    require(
        abs(float(noninferiority.get("profile_p_value_cutoff", -1.0)) - CUTOFF)
        <= 1e-12,
        "profile cutoff mismatch",
    )


def run_self_test() -> None:
    # Monotonicity and decision consistency around the frozen boundary.
    minus, zero, plus = 15, 440, 15
    p_boundary = one_sided_profile_p_value(minus, zero, plus, -MARGIN)
    lower = calibrated_profile_lower_bound(minus, zero, plus, cutoff=CUTOFF)
    require(0.0 <= p_boundary <= 1.0, "self-test p-value outside [0,1]")
    require(-1.0 <= lower <= 1.0, "self-test lower bound outside [-1,1]")
    require(
        (p_boundary < CUTOFF) == (lower > -MARGIN),
        "self-test p-value/lower-bound decisions disagree",
    )

    favorable = one_sided_profile_p_value(5, 450, 15, -MARGIN)
    unfavorable = one_sided_profile_p_value(25, 440, 5, -MARGIN)
    require(favorable < unfavorable, "self-test directional ordering failed")
    print("Warm-start final analyzer self-test: PASS")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--acquisition-root", type=Path)
    parser.add_argument("--analysis-freeze", type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        require(
            args.acquisition_root is None
            and args.analysis_freeze is None
            and args.authorization is None
            and args.output is None,
            "--self-test cannot be combined with artifact arguments",
        )
        run_self_test()
        return 0

    require(args.acquisition_root is not None, "--acquisition-root is required")
    require(args.analysis_freeze is not None, "--analysis-freeze is required")
    require(args.authorization is not None, "--authorization is required")
    require(args.output is not None, "--output is required")
    require(not args.output.exists(), f"refusing to overwrite output: {args.output}")

    freeze = load_json(args.analysis_freeze)
    authorization = load_json(args.authorization)
    validate_freeze(freeze)
    require(authorization.get("authorized") is True, "runtime authorization is not valid")
    require(
        authorization.get("analysis_freeze", {}).get("sha256")
        == authorization.get("frozen_inputs", {}).get("analysis_freeze_sha256"),
        "authorization analysis-freeze identity is inconsistent",
    )

    acquisition_report = load_json(args.acquisition_root / "acquisition_report.json")
    require(acquisition_report.get("completed") is True, "acquisition is incomplete")
    require(acquisition_report.get("authorization_sha256") == authorization.get("self_sha256"), "authorization digest mismatch")

    records_by_task_state: dict[tuple[int, int], dict[str, Any]] = {}
    phase_reports = {}
    for phase in PHASES:
        path = args.acquisition_root / phase / "validation_report.json"
        report = load_json(path)
        phase_reports[phase] = report
        require(report.get("schema_version") == 2, f"{phase}: validator schema v2 required")
        require(report.get("contract_passed") is True, f"{phase}: runtime contract failed")
        require(report.get("tasks") == list(range(TASK_COUNT)), f"{phase}: task coverage mismatch")
        require(report.get("warm_start_resets") == [], f"{phase}: warm-state reset observed")
        per_task = report.get("per_task")
        require(isinstance(per_task, dict), f"{phase}: missing per-task results")
        for task_id in range(TASK_COUNT):
            task_payload = per_task.get(str(task_id))
            require(isinstance(task_payload, dict), f"{phase}: missing task {task_id}")
            paired = task_payload.get("paired_episodes")
            require(isinstance(paired, list), f"{phase}/task{task_id}: missing paired episodes")
            for record in paired:
                state_id = int(record["initial_state_id"])
                key = (task_id, state_id)
                require(key not in records_by_task_state, f"duplicate task/state pair: {key}")
                records_by_task_state[key] = {**record, "phase": phase}

    state_allocation = freeze.get("state_allocation")
    require(isinstance(state_allocation, dict), "analysis freeze lacks state allocation")
    included = state_allocation.get("included_initial_state_ids_by_task")
    excluded = state_allocation.get("excluded_initial_state_ids_by_task")
    require(isinstance(included, dict) and isinstance(excluded, dict), "state allocation is incomplete")

    primary_records: list[dict[str, Any]] = []
    excluded_records: list[dict[str, Any]] = []
    for task_id in range(TASK_COUNT):
        included_ids = [int(value) for value in included[str(task_id)]]
        excluded_ids = [int(value) for value in excluded[str(task_id)]]
        require(len(included_ids) == PAIRS_PER_TASK, f"task {task_id}: included-state count mismatch")
        require(len(excluded_ids) == 3, f"task {task_id}: excluded-state count mismatch")
        require(not set(included_ids).intersection(excluded_ids), f"task {task_id}: included/excluded overlap")
        require(set(included_ids).union(excluded_ids) == set(range(50)), f"task {task_id}: state coverage mismatch")
        for state_id in included_ids:
            key = (task_id, state_id)
            require(key in records_by_task_state, f"missing primary pair {key}")
            primary_records.append(records_by_task_state[key])
        for state_id in excluded_ids:
            key = (task_id, state_id)
            require(key in records_by_task_state, f"missing excluded acquisition pair {key}")
            excluded_records.append(records_by_task_state[key])

    require(len(primary_records) == TOTAL_PAIRS, "primary pair count mismatch")
    require(len(excluded_records) == 30, "excluded acquisition pair count mismatch")

    counts = {
        "both_success": 0,
        "cold_only_success": 0,
        "warm_only_success": 0,
        "both_failure": 0,
    }
    per_task_counts: dict[str, dict[str, int]] = {}
    for task_id in range(TASK_COUNT):
        task_counts = dict.fromkeys(counts, 0)
        for record in primary_records:
            if int(record["task_id"]) != task_id:
                continue
            label = category(record)
            counts[label] += 1
            task_counts[label] += 1
        require(sum(task_counts.values()) == PAIRS_PER_TASK, f"task {task_id}: primary pair total mismatch")
        per_task_counts[str(task_id)] = task_counts

    minus = counts["cold_only_success"]
    plus = counts["warm_only_success"]
    zero = counts["both_success"] + counts["both_failure"]
    estimate = (plus - minus) / TOTAL_PAIRS
    p_value = one_sided_profile_p_value(minus, zero, plus, -MARGIN)
    lower_bound = calibrated_profile_lower_bound(minus, zero, plus, cutoff=CUTOFF)
    passed = p_value < CUTOFF
    require(passed == (lower_bound > -MARGIN), "profile decision and calibrated lower bound disagree")

    efficiency = {
        field: numeric_summary(
            record["difference_warm_minus_cold"][field]
            for record in primary_records
        )
        for field in EFFICIENCY_FIELDS
    }
    per_task = {}
    for task_id in range(TASK_COUNT):
        task_records = [
            record for record in primary_records if int(record["task_id"]) == task_id
        ]
        task_counts = per_task_counts[str(task_id)]
        per_task[str(task_id)] = {
            "paired_counts": task_counts,
            "success_difference": (
                task_counts["warm_only_success"]
                - task_counts["cold_only_success"]
            )
            / PAIRS_PER_TASK,
            "paired_efficiency_warm_minus_cold": {
                field: numeric_summary(
                    record["difference_warm_minus_cold"][field]
                    for record in task_records
                )
                for field in EFFICIENCY_FIELDS
            },
        }

    payload = {
        "schema_version": 1,
        "role": "warm_start_primary_final_analysis",
        "study": "midpoint_warm_start_primary_evaluation",
        "primary_analysis": {
            "task_count": TASK_COUNT,
            "pairs_per_task": PAIRS_PER_TASK,
            "total_pairs": TOTAL_PAIRS,
            "paired_counts": counts,
            "p01": plus / TOTAL_PAIRS,
            "p10": minus / TOTAL_PAIRS,
            "paired_success_difference_warm_minus_cold": estimate,
            "noninferiority_margin": MARGIN,
            "method": "pooled paired-trinomial profile likelihood",
            "one_sided_profile_p_value": p_value,
            "simulation_calibrated_cutoff": CUTOFF,
            "simulation_calibrated_one_sided_lower_bound": lower_bound,
            "noninferiority_passed": passed,
        },
        "secondary_descriptive_efficiency": {
            "latency_scope": "synchronized online policy-query latency around get_action",
            "paired_warm_minus_cold": efficiency,
            "causal_decomposition_claim_allowed": False,
        },
        "per_task_descriptive": per_task,
        "excluded_preflight_state_acquisition_pairs": {
            "count": len(excluded_records),
            "used_in_primary_analysis": False,
        },
        "artifact_paths": {
            "acquisition_root": str(args.acquisition_root.resolve()),
            "analysis_freeze": str(args.analysis_freeze.resolve()),
            "authorization": str(args.authorization.resolve()),
            "phase_validation_reports": {
                phase: str((args.acquisition_root / phase / "validation_report.json").resolve())
                for phase in PHASES
            },
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote: {args.output}")
    print(
        "Primary paired success: "
        f"warm-cold={estimate:+.4f}, cold_only={minus}, warm_only={plus}, "
        f"profile_p={p_value:.6g}, calibrated_lower={lower_bound:+.4f}"
    )
    print(f"Non-inferiority at margin -0.05 and cutoff 0.045: {passed}")
    print(
        "Descriptive paired efficiency mean, warm-cold: "
        f"K={efficiency['mean_K']['mean']:+.4f}, "
        f"recurrent_calls/episode={efficiency['recurrent_calls']['mean']:+.4f}, "
        f"inference_time/episode={efficiency['inference_time_ms']['mean']:+.2f} ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
