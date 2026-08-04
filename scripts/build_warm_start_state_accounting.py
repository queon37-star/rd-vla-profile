#!/usr/bin/env python3
"""Build warm-start state accounting after the all-task paired preflight.

The script does not authorize a final run. It records which official LIBERO
Spatial initial states have already been observed under the warm-start policy
and identifies the remaining warm-start-outcome-unseen states.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


EXPECTED_TASK_IDS = tuple(range(10))
EXPECTED_OBSERVED_PER_TASK = 3
EXPECTED_REMAINING_PER_TASK = 47


class StateAccountingError(ValueError):
    """Raised when preflight provenance or state allocation is inconsistent."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise StateAccountingError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--official-manifest",
        type=Path,
        default=Path(
            "experiments/robot/libero/manifests/"
            "libero_spatial_official_50_v1.json"
        ),
    )
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require(not args.output.exists(), f"refusing to overwrite existing output: {args.output}")

    manifest = load_json(args.official_manifest)
    report = load_json(args.validation_report)

    require(manifest.get("task_suite_name") == "libero_spatial", "official manifest must be LIBERO Spatial")
    require(manifest.get("expected_state_count") == 50, "official manifest must expose 50 states/task")
    require(report.get("schema_version") == 2, "validation report schema v2 is required")
    require(report.get("contract_passed") is True, "extended preflight contract did not pass")
    require(
        report.get("role") == "extended_planning_preflight_not_evidence",
        "validation report is not the all-task extended preflight",
    )
    require(report.get("tasks") == list(EXPECTED_TASK_IDS), "extended preflight must cover tasks 0..9")
    require(
        int(report.get("episodes_per_task", -1)) == EXPECTED_OBSERVED_PER_TASK,
        "extended preflight must contain 3 pairs/task",
    )

    manifest_tasks = manifest.get("tasks")
    observed_by_task = report.get("observed_initial_state_ids_by_task")
    require(isinstance(manifest_tasks, dict), "official manifest has no tasks mapping")
    require(isinstance(observed_by_task, dict), "validation report has no observed state mapping")

    task_rows: dict[str, Any] = {}
    all_observed_pairs = 0
    all_remaining_pairs = 0
    for task_id in EXPECTED_TASK_IDS:
        task_key = str(task_id)
        entry = manifest_tasks.get(task_key)
        require(isinstance(entry, dict), f"official manifest is missing task {task_id}")
        partitions = entry.get("partitions")
        require(isinstance(partitions, dict), f"task {task_id} has no partitions")
        calibration = [int(value) for value in partitions.get("calibration", [])]
        screening = [int(value) for value in partitions.get("screening", [])]
        final = [int(value) for value in partitions.get("final", [])]
        require(
            len(calibration) == 10 and len(screening) == 10 and len(final) == 30,
            f"task {task_id} has invalid partition sizes",
        )
        canonical_order = calibration + screening + final
        require(
            len(canonical_order) == 50 and len(set(canonical_order)) == 50,
            f"task {task_id} does not contain 50 unique states",
        )

        observed = observed_by_task.get(task_key)
        require(isinstance(observed, list), f"task {task_id} has no observed state list")
        observed = sorted(int(value) for value in observed)
        expected_smoke = sorted(calibration[:EXPECTED_OBSERVED_PER_TASK])
        require(
            observed == expected_smoke,
            f"task {task_id} observed states do not match the first three calibration states: "
            f"observed={observed}, expected={expected_smoke}",
        )
        observed_set = set(observed)
        remaining = [state_id for state_id in canonical_order if state_id not in observed_set]
        require(
            len(remaining) == EXPECTED_REMAINING_PER_TASK,
            f"task {task_id} must have 47 warm-start-outcome-unseen states",
        )
        require(not observed_set.intersection(remaining), f"task {task_id} observed/remaining overlap")
        require(set(observed).union(remaining) == set(canonical_order), f"task {task_id} accounting is incomplete")

        task_rows[task_key] = {
            "task_name": entry.get("task_name"),
            "observed_in_warm_start_preflight": observed,
            "warm_start_outcome_unseen": remaining,
            "warm_start_outcome_unseen_by_original_partition": {
                "calibration_remainder": [
                    state_id for state_id in calibration if state_id not in observed_set
                ],
                "screening": screening,
                "final": final,
            },
            "observed_count": len(observed),
            "remaining_count": len(remaining),
        }
        all_observed_pairs += len(observed)
        all_remaining_pairs += len(remaining)

    payload = {
        "schema_version": 1,
        "study": "midpoint_warm_start_primary_evaluation",
        "role": "planning_state_accounting_not_final_authorization",
        "terminology": {
            "allowed": "warm-start-outcome-unseen states",
            "forbidden": "untouched states",
            "reason": (
                "the official states were used in earlier scalar studies; only their "
                "warm-start paired outcomes remain unseen"
            ),
        },
        "official_manifest": {
            "path": str(args.official_manifest.resolve()),
            "sha256": sha256_file(args.official_manifest),
        },
        "extended_preflight_validation": {
            "path": str(args.validation_report.resolve()),
            "sha256": sha256_file(args.validation_report),
            "contract_passed": True,
            "success_result_used_for_method_selection": False,
        },
        "state_accounting": {
            "official_states_per_task": 50,
            "observed_warm_start_preflight_states_per_task": EXPECTED_OBSERVED_PER_TASK,
            "maximum_warm_start_outcome_unseen_states_per_task": EXPECTED_REMAINING_PER_TASK,
            "total_observed_task_state_pairs": all_observed_pairs,
            "total_remaining_task_state_pairs": all_remaining_pairs,
            "preflight_states_allowed_in_primary_confirmatory_analysis": False,
            "repeated_state_as_independent_pair_allowed": False,
        },
        "tasks": task_rows,
        "final_run_authorized": False,
        "remaining_authorization_requirements": [
            "primary interval procedure validated by simulation",
            "non-inferiority margin justified and frozen",
            "target power frozen",
            "sample size shown feasible with at most 47 pairs/task",
            "warm-start-specific paired seed namespace frozen",
            "runtime source commit and checkpoint digest frozen",
            "final launcher and validator reviewed",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote: {args.output}")
    print(
        "Warm-start state accounting: "
        f"observed={EXPECTED_OBSERVED_PER_TASK}/task, "
        f"remaining={EXPECTED_REMAINING_PER_TASK}/task, "
        f"total_remaining={all_remaining_pairs}"
    )
    print("Final run authorized: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
