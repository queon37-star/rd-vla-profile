#!/usr/bin/env python3
"""Build a planning-only task-level p01/p10 profile from paired warm-start runs.

The input may contain one result JSON per task or a result JSON containing
multiple tasks. Pair identity is required to match exactly across arms:
`paired_trial_id`, `initial_state_id`, and `episode_seed` must all be present.
Legacy unpaired runs are rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = 1
EXPECTED_TASK_IDS = tuple(range(10))


class WarmStartProfileError(ValueError):
    """Raised when source runs cannot support a paired planning profile."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise WarmStartProfileError(message)


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


def load_task_name_map(manifest_path: Path) -> tuple[dict[str, int], str]:
    manifest = load_json(manifest_path)
    require(
        manifest.get("task_suite_name") == "libero_spatial",
        "manifest must be LIBERO Spatial",
    )
    tasks = manifest.get("tasks")
    require(isinstance(tasks, dict), "manifest is missing tasks")

    name_to_id: dict[str, int] = {}
    for task_id in EXPECTED_TASK_IDS:
        entry = tasks.get(str(task_id))
        require(isinstance(entry, dict), f"manifest is missing task {task_id}")
        task_name = entry.get("task_name")
        require(
            isinstance(task_name, str) and task_name,
            f"manifest task {task_id} has no task_name",
        )
        require(
            task_name not in name_to_id,
            f"duplicate task_name in manifest: {task_name}",
        )
        name_to_id[task_name] = task_id
    return name_to_id, sha256_file(manifest_path)


def pair_key(
    record: dict[str, Any], *, source: Path, task_name: str
) -> tuple[int, int, int]:
    values = []
    for field in ("paired_trial_id", "initial_state_id", "episode_seed"):
        value = record.get(field)
        require(
            isinstance(value, int) and not isinstance(value, bool),
            f"{source}: task={task_name} requires integer {field}; "
            "legacy/unpaired results are not admissible",
        )
        values.append(int(value))
    return values[0], values[1], values[2]


def collect_arm(
    paths: Iterable[Path],
    *,
    arm_name: str,
    task_name_to_id: dict[str, int],
) -> tuple[
    dict[int, dict[tuple[int, int, int], dict[str, Any]]],
    list[dict[str, Any]],
]:
    by_task: dict[int, dict[tuple[int, int, int], dict[str, Any]]] = defaultdict(dict)
    provenance = []

    for path in paths:
        payload = load_json(path)
        tasks = payload.get("tasks")
        require(
            isinstance(tasks, dict) and tasks,
            f"{path}: result JSON has no task records",
        )
        provenance.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "total_episodes": payload.get("total_episodes"),
                "total_successes": payload.get("total_successes"),
                "evaluation_protocol": payload.get("evaluation_protocol"),
            }
        )

        for task_name, records in tasks.items():
            require(
                task_name in task_name_to_id,
                f"{path}: unknown task name {task_name!r}",
            )
            task_id = task_name_to_id[task_name]
            require(
                isinstance(records, list),
                f"{path}: task {task_name} records must be a list",
            )
            for record in records:
                require(
                    isinstance(record, dict),
                    f"{path}: task {task_name} contains a non-object record",
                )
                key = pair_key(record, source=path, task_name=task_name)
                require(
                    key not in by_task[task_id],
                    f"duplicate {arm_name} pair for task {task_id}: {key}",
                )
                success = record.get("success")
                require(
                    isinstance(success, bool),
                    f"{path}: task {task_name} pair {key} has invalid success",
                )
                by_task[task_id][key] = {
                    "success": success,
                    "evaluation_protocol_phase": record.get(
                        "evaluation_protocol_phase"
                    ),
                    "source_path": str(path.resolve()),
                }

    require(by_task, f"no records found for arm {arm_name}")
    return dict(by_task), provenance


def summarize_profile(
    baseline: dict[int, dict[tuple[int, int, int], dict[str, Any]]],
    warm: dict[int, dict[tuple[int, int, int], dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    require(
        set(baseline) == set(EXPECTED_TASK_IDS),
        "baseline must contain exactly task IDs 0..9",
    )
    require(
        set(warm) == set(EXPECTED_TASK_IDS),
        "warm arm must contain exactly task IDs 0..9",
    )

    task_rows = []
    aggregate = {
        "both_success": 0,
        "baseline_only_success": 0,
        "warm_only_success": 0,
        "both_failure": 0,
        "total_pairs": 0,
    }

    for task_id in EXPECTED_TASK_IDS:
        baseline_pairs = baseline[task_id]
        warm_pairs = warm[task_id]
        require(
            set(baseline_pairs) == set(warm_pairs),
            f"task {task_id} pair identities differ across arms",
        )
        require(baseline_pairs, f"task {task_id} has no paired trials")

        counts = {
            "both_success": 0,
            "baseline_only_success": 0,
            "warm_only_success": 0,
            "both_failure": 0,
        }
        phases = set()
        for key in sorted(baseline_pairs):
            base_record = baseline_pairs[key]
            warm_record = warm_pairs[key]
            phases.add(base_record.get("evaluation_protocol_phase"))
            phases.add(warm_record.get("evaluation_protocol_phase"))
            base_success = bool(base_record["success"])
            warm_success = bool(warm_record["success"])
            if base_success and warm_success:
                category = "both_success"
            elif base_success:
                category = "baseline_only_success"
            elif warm_success:
                category = "warm_only_success"
            else:
                category = "both_failure"
            counts[category] += 1
            aggregate[category] += 1

        n_pairs = len(baseline_pairs)
        aggregate["total_pairs"] += n_pairs
        p01 = counts["warm_only_success"] / n_pairs
        p10 = counts["baseline_only_success"] / n_pairs
        require(
            math.isfinite(p01) and math.isfinite(p10),
            "non-finite profile probability",
        )
        task_rows.append(
            {
                "task_id": task_id,
                "n_pairs": n_pairs,
                "p01": p01,
                "p10": p10,
                "paired_difference": p01 - p10,
                "discordance": p01 + p10,
                **counts,
                "evaluation_protocol_phases": sorted(
                    str(value) for value in phases
                ),
            }
        )

    return task_rows, aggregate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, action="append", required=True)
    parser.add_argument("--warm", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--initial-state-manifest",
        type=Path,
        default=Path(
            "experiments/robot/libero/manifests/"
            "libero_spatial_official_50_v1.json"
        ),
    )
    parser.add_argument("--scenario-name", default="observed_warm_start_pilot")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require(
        not args.output.exists(),
        f"refusing to overwrite existing output: {args.output}",
    )

    task_name_to_id, manifest_sha256 = load_task_name_map(
        args.initial_state_manifest
    )
    baseline, baseline_sources = collect_arm(
        args.baseline,
        arm_name="baseline",
        task_name_to_id=task_name_to_id,
    )
    warm, warm_sources = collect_arm(
        args.warm,
        arm_name="warm",
        task_name_to_id=task_name_to_id,
    )
    task_rows, aggregate = summarize_profile(baseline, warm)

    total_pairs = aggregate["total_pairs"]
    aggregate_p01 = aggregate["warm_only_success"] / total_pairs
    aggregate_p10 = aggregate["baseline_only_success"] / total_pairs
    payload = {
        "schema_version": SCHEMA_VERSION,
        "role": "planning_only_not_confirmatory_evidence",
        "study": "midpoint_warm_start_primary_evaluation",
        "initial_state_manifest": {
            "path": str(args.initial_state_manifest.resolve()),
            "sha256": manifest_sha256,
        },
        "pair_contract": {
            "identity_fields": [
                "paired_trial_id",
                "initial_state_id",
                "episode_seed",
            ],
            "exact_cross_arm_identity_required": True,
            "legacy_unpaired_runs_allowed": False,
        },
        "sources": {
            "cold_initialized_adaptive": baseline_sources,
            "midpoint_warm_start_adaptive": warm_sources,
        },
        "aggregate": {
            **aggregate,
            "p01": aggregate_p01,
            "p10": aggregate_p10,
            "paired_difference": aggregate_p01 - aggregate_p10,
            "discordance": aggregate_p01 + aggregate_p10,
        },
        "scenarios": [
            {
                "name": args.scenario_name,
                "tasks": task_rows,
            }
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote: {args.output}")
    print(
        "Aggregate: "
        f"pairs={total_pairs}, p01={aggregate_p01:.4f}, "
        f"p10={aggregate_p10:.4f}, "
        f"difference={aggregate_p01 - aggregate_p10:+.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
