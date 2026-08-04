#!/usr/bin/env python3
"""Audit repeated warm-start preflight traces on overlapping task/state/seed pairs.

The audit compares the original task-0/task-5 preflight with the all-task
extended preflight. Latency is intentionally ignored. Action tensors were not
hashed in these runs, so a pass establishes repeatability of success, prediction
count, K/stop traces, decode-call accounting, and warm-start provenance—not
bitwise action equality.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


ARMS = ("cold_initialized_adaptive", "midpoint_warm_start_adaptive")
PAIR_FIELDS = ("paired_trial_id", "initial_state_id", "episode_seed")
EXACT_STEP_FIELDS = (
    "recurrent_iteration_count",
    "canonical_stop_reason",
    "stop_reason",
    "get_output_call_count",
    "coda_call_count",
    "returned_cached_final_output",
    "warm_start_enabled",
    "warm_start_used",
    "warm_start_state_provided",
    "warm_start_source",
    "warm_start_source_iteration",
    "warm_start_source_K",
    "warm_start_candidate_state_count",
    "warm_start_reset",
    "warm_start_reset_reason",
    "actual_origin",
    "initial_state_origin",
    "executed_actions_from_prediction",
)
FLOAT_STEP_FIELDS = ("final_mse", "final_conv_score")


class RepeatabilityError(ValueError):
    """Raised when overlapping preflight traces are not reproducible."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RepeatabilityError(message)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"missing JSONL file: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            require(isinstance(value, dict), f"{path}:{line_number} is not an object")
            rows.append(value)
    require(rows, f"empty JSONL file: {path}")
    return rows


def parse_tasks(text: str) -> tuple[int, ...]:
    tasks = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    require(tasks, "at least one task is required")
    require(len(tasks) == len(set(tasks)), "task IDs must be unique")
    return tasks


def pair_key(record: dict[str, Any], *, context: str) -> tuple[int, int, int]:
    values = []
    for field in PAIR_FIELDS:
        value = record.get(field)
        require(
            isinstance(value, int) and not isinstance(value, bool),
            f"{context}: missing integer {field}",
        )
        values.append(int(value))
    return values[0], values[1], values[2]


def run_entry(plan: dict[str, Any], *, task_id: int, arm: str) -> dict[str, Any]:
    matches = [
        item
        for item in plan.get("runs", [])
        if int(item.get("task_id", -1)) == task_id and item.get("arm") == arm
    ]
    require(len(matches) == 1, f"task {task_id}/{arm}: expected one run entry")
    return matches[0]


def episode_map(result_path: Path, *, context: str) -> dict[tuple[int, int, int], bool]:
    result = load_json(result_path)
    tasks = result.get("tasks")
    require(isinstance(tasks, dict) and len(tasks) == 1, f"{context}: expected one task")
    records = next(iter(tasks.values()))
    require(isinstance(records, list) and records, f"{context}: missing episodes")
    mapped = {}
    for record in records:
        require(isinstance(record, dict), f"{context}: episode is not an object")
        key = pair_key(record, context=context)
        require(key not in mapped, f"{context}: duplicate pair {key}")
        require(isinstance(record.get("success"), bool), f"{context}: invalid success")
        mapped[key] = bool(record["success"])
    return mapped


def step_map(
    step_path: Path, *, context: str
) -> dict[tuple[int, int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(step_path):
        key = pair_key(row, context=context)
        prediction = row.get("prediction_step")
        require(isinstance(prediction, int), f"{context}/{key}: invalid prediction_step")
        grouped[key].append(row)
    for key, rows in grouped.items():
        rows.sort(key=lambda item: int(item["prediction_step"]))
        observed = [int(item["prediction_step"]) for item in rows]
        require(observed == list(range(len(rows))), f"{context}/{key}: non-contiguous predictions")
    return dict(grouped)


def float_equal(left: Any, right: Any, tolerance: float) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if not isinstance(left, (int, float)) or isinstance(left, bool):
        return False
    if not isinstance(right, (int, float)) or isinstance(right, bool):
        return False
    if not math.isfinite(float(left)) or not math.isfinite(float(right)):
        return False
    return abs(float(left) - float(right)) <= tolerance


def float_list_equal(left: Any, right: Any, tolerance: float) -> bool:
    if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
        return False
    return all(float_equal(a, b, tolerance) for a, b in zip(left, right))


def compare_pair_steps(
    reference: list[dict[str, Any]],
    repeat: list[dict[str, Any]],
    *,
    context: str,
    tolerance: float,
) -> None:
    require(len(reference) == len(repeat), f"{context}: prediction count differs")
    for index, (left, right) in enumerate(zip(reference, repeat)):
        step_context = f"{context}/prediction{index}"
        for field in EXACT_STEP_FIELDS:
            require(
                left.get(field) == right.get(field),
                f"{step_context}: field {field} differs: {left.get(field)!r} vs {right.get(field)!r}",
            )
        for field in FLOAT_STEP_FIELDS:
            require(
                float_equal(left.get(field), right.get(field), tolerance),
                f"{step_context}: float field {field} differs",
            )
        require(
            float_list_equal(left.get("conv_score_list", []), right.get("conv_score_list", []), tolerance),
            f"{step_context}: conv_score_list differs",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--repeat-root", type=Path, required=True)
    parser.add_argument("--tasks", default="0,5")
    parser.add_argument("--float-tolerance", type=float, default=1e-9)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require(not args.output.exists(), f"refusing to overwrite existing output: {args.output}")
    require(args.float_tolerance >= 0.0, "float tolerance must be non-negative")
    tasks = parse_tasks(args.tasks)

    reference_plan = load_json(args.reference_root / "run_plan.json")
    repeat_plan = load_json(args.repeat_root / "run_plan.json")
    require(reference_plan.get("evaluation_protocol_phase") == "smoke", "reference is not smoke")
    require(repeat_plan.get("evaluation_protocol_phase") == "smoke", "repeat is not smoke")
    require(reference_plan.get("seed") == repeat_plan.get("seed"), "base seeds differ")
    require(
        reference_plan.get("initial_state_manifest", {}).get("sha256")
        == repeat_plan.get("initial_state_manifest", {}).get("sha256"),
        "initial-state manifest digests differ",
    )
    require(
        reference_plan.get("checkpoint", {}).get("tree_sha256_before")
        == repeat_plan.get("checkpoint", {}).get("tree_sha256_before"),
        "checkpoint digests differ",
    )

    comparisons = []
    for task_id in tasks:
        require(task_id in reference_plan.get("tasks", []), f"reference lacks task {task_id}")
        require(task_id in repeat_plan.get("tasks", []), f"repeat lacks task {task_id}")
        for arm in ARMS:
            reference_run = run_entry(reference_plan, task_id=task_id, arm=arm)
            repeat_run = run_entry(repeat_plan, task_id=task_id, arm=arm)
            reference_dir = Path(reference_run["output_dir"])
            repeat_dir = Path(repeat_run["output_dir"])
            context = f"task{task_id}/{arm}"
            reference_episodes = episode_map(reference_dir / "result.json", context=context)
            repeat_episodes = episode_map(repeat_dir / "result.json", context=context)
            require(set(reference_episodes) == set(repeat_episodes), f"{context}: pair identities differ")
            reference_steps = step_map(reference_dir / "steps.jsonl", context=context)
            repeat_steps = step_map(repeat_dir / "steps.jsonl", context=context)
            require(set(reference_steps) == set(repeat_steps), f"{context}: step pair identities differ")
            require(set(reference_episodes) == set(reference_steps), f"{context}: episode/step pairs differ")
            for key in sorted(reference_episodes):
                require(
                    reference_episodes[key] == repeat_episodes[key],
                    f"{context}/{key}: success outcome differs",
                )
                compare_pair_steps(
                    reference_steps[key],
                    repeat_steps[key],
                    context=f"{context}/{key}",
                    tolerance=args.float_tolerance,
                )
            comparisons.append(
                {
                    "task_id": task_id,
                    "arm": arm,
                    "pair_count": len(reference_episodes),
                    "successes": sum(reference_episodes.values()),
                    "prediction_count": sum(len(rows) for rows in reference_steps.values()),
                }
            )

    payload = {
        "schema_version": 1,
        "role": "planning_repeatability_audit_not_performance_evidence",
        "behavioral_trace_repeatability_passed": True,
        "reference_root": str(args.reference_root.resolve()),
        "repeat_root": str(args.repeat_root.resolve()),
        "tasks": list(tasks),
        "arms": list(ARMS),
        "base_seed": reference_plan.get("seed"),
        "checkpoint_tree_sha256": reference_plan.get("checkpoint", {}).get("tree_sha256_before"),
        "initial_state_manifest_sha256": reference_plan.get("initial_state_manifest", {}).get("sha256"),
        "source_commits": {
            "reference": reference_plan.get("source_commit"),
            "repeat": repeat_plan.get("source_commit"),
            "equal": reference_plan.get("source_commit") == repeat_plan.get("source_commit"),
        },
        "float_tolerance": args.float_tolerance,
        "compared_exact_step_fields": list(EXACT_STEP_FIELDS),
        "compared_float_step_fields": list(FLOAT_STEP_FIELDS),
        "comparisons": comparisons,
        "limitations": [
            "latency was not compared",
            "returned action tensors were not hashed because shadow_full_depth was disabled",
            "a pass does not establish bitwise action equality",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Warm-start repeated preflight behavioral trace: PASS")
    print(
        f"Tasks={list(tasks)}, pairs compared={sum(item['pair_count'] for item in comparisons) // len(ARMS)}, "
        f"float_tolerance={args.float_tolerance:g}"
    )
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
