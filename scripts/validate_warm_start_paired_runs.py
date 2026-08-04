#!/usr/bin/env python3
"""Validate and summarize paired cold-vs-midpoint warm-start LIBERO runs.

The validator is shared by planning-only preflight runs and a possible
calibration pilot. It checks pairing, warm-state provenance, inactive
mechanisms, terminal-output reuse, and episode/prediction accounting. It also
reports descriptive warm-minus-cold efficiency differences for every pair.
It does not make a statistical success-preservation or efficiency claim.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Hashable, Iterable, Sequence


ARMS = ("cold_initialized_adaptive", "midpoint_warm_start_adaptive")
PAIR_FIELDS = ("paired_trial_id", "initial_state_id", "episode_seed")
PAIRED_EFFICIENCY_FIELDS = (
    "num_predictions",
    "recurrent_calls",
    "get_output_calls",
    "inference_time_ms",
    "mean_prediction_latency_ms",
    "mean_K",
)


class ValidationError(ValueError):
    """Raised when a paired run violates the frozen runtime contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"missing JSONL file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            require(
                isinstance(value, dict),
                f"{path}:{line_number} must contain a JSON object",
            )
            rows.append(value)
    require(rows, f"empty JSONL file: {path}")
    return rows


def finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


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


def pair_identity(key: tuple[int, int, int]) -> dict[str, int]:
    return {
        "paired_trial_id": int(key[0]),
        "initial_state_id": int(key[1]),
        "episode_seed": int(key[2]),
    }


def collect_episode_records(
    result: dict[str, Any], *, task_id: int, context: str
) -> dict[tuple[int, int, int], dict[str, Any]]:
    tasks = result.get("tasks")
    require(
        isinstance(tasks, dict) and len(tasks) == 1,
        f"{context}: expected one task in result JSON",
    )
    records = next(iter(tasks.values()))
    require(isinstance(records, list) and records, f"{context}: missing episode records")
    collected: dict[tuple[int, int, int], dict[str, Any]] = {}
    for record in records:
        require(isinstance(record, dict), f"{context}: episode record must be an object")
        key = pair_key(record, context=context)
        require(key not in collected, f"{context}: duplicate episode pair {key}")
        require(
            isinstance(record.get("success"), bool),
            f"{context}: invalid success for pair {key}",
        )
        collected[key] = record
    return collected


def validate_common_step(row: dict[str, Any], *, context: str) -> None:
    require(row.get("paired_rng") is True, f"{context}: paired_rng must be true")
    require(
        row.get("episode_seed_source") == "paired_protocol",
        f"{context}: invalid seed source",
    )
    require(
        row.get("environment_seed_applied") is True,
        f"{context}: environment seed was not applied",
    )
    require(
        row.get("reset_rng_each_episode") is True,
        f"{context}: reset_rng_each_episode must be true",
    )
    require(
        row.get("canonical_recurrence_strategy") == "adjacent_action_mse",
        f"{context}: wrong recurrence strategy",
    )
    require(row.get("use_latent_precheck") is False, f"{context}: latent pre-check enabled")
    require(
        row.get("latent_precheck_mode") == "off",
        f"{context}: latent pre-check mode is not off",
    )
    require(row.get("scalar_policy_requested") is False, f"{context}: scalar policy requested")
    require(row.get("scalar_policy_applied") is False, f"{context}: scalar policy applied")
    require(row.get("shadow_full_depth_enabled") is False, f"{context}: shadow recurrence enabled")
    require(row.get("numerical_retry_attempted") is False, f"{context}: numerical retry attempted")
    require(row.get("shadow_error") in (None, {}), f"{context}: shadow/non-finite error present")
    require(row.get("use_cached_final_output") is True, f"{context}: cached terminal output disabled")
    require(
        row.get("returned_cached_final_output") is True,
        f"{context}: cached terminal output not returned",
    )

    k = row.get("recurrent_iteration_count")
    get_output_calls = row.get("get_output_call_count")
    coda_calls = row.get("coda_call_count")
    require(isinstance(k, int) and k >= 2, f"{context}: invalid terminal K")
    require(get_output_calls == k, f"{context}: get_output calls {get_output_calls} != K {k}")
    require(coda_calls == k, f"{context}: Coda calls {coda_calls} != K {k}")
    require(
        finite_number(row.get("latency_ms")) and float(row["latency_ms"]) >= 0.0,
        f"{context}: invalid latency",
    )


def validate_arm_steps(
    rows: Iterable[dict[str, Any]],
    *,
    arm: str,
    task_id: int,
) -> tuple[dict[tuple[int, int, int], list[dict[str, Any]]], list[dict[str, Any]]]:
    by_pair: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    resets: list[dict[str, Any]] = []

    for row in rows:
        require(
            isinstance(row.get("task_id"), int) and int(row["task_id"]) == task_id,
            f"task {task_id}/{arm}: wrong task_id",
        )
        key = pair_key(row, context=f"task {task_id}/{arm}")
        prediction = row.get("prediction_step")
        require(
            isinstance(prediction, int) and prediction >= 0,
            f"task {task_id}/{arm}/{key}: invalid prediction index",
        )
        context = f"task {task_id}/{arm}/{key}/prediction{prediction}"
        validate_common_step(row, context=context)

        if arm == "cold_initialized_adaptive":
            require(row.get("warm_start_enabled") is False, f"{context}: baseline warm-start enabled")
            require(row.get("warm_start_used") is False, f"{context}: baseline used warm state")
            require(
                row.get("warm_start_state_provided") is False,
                f"{context}: baseline received warm state",
            )
            require(row.get("initial_state_origin") == "random", f"{context}: baseline origin is not random")
            require(row.get("actual_origin") == "COLD", f"{context}: baseline actual origin is not COLD")
        else:
            require(row.get("warm_start_enabled") is True, f"{context}: warm-start arm disabled")
            if prediction == 0:
                require(row.get("warm_start_used") is False, f"{context}: first prediction unexpectedly warm")
                require(
                    row.get("warm_start_state_provided") is False,
                    f"{context}: first prediction received cache",
                )
                require(row.get("initial_state_origin") == "random", f"{context}: first origin is not random")
                require(row.get("actual_origin") == "COLD", f"{context}: first actual origin is not COLD")
            else:
                require(
                    row.get("warm_start_state_provided") is True,
                    f"{context}: midpoint state was not provided",
                )
                require(row.get("warm_start_used") is True, f"{context}: midpoint state was not used")
                require(row.get("initial_state_origin") == "cached", f"{context}: warm origin is not cached")
                require(row.get("actual_origin") == "ACTUAL_WARM", f"{context}: actual origin is not ACTUAL_WARM")
                require(row.get("warm_start_source") == "midpoint", f"{context}: source is not midpoint")
                source_k = row.get("warm_start_source_K")
                source_iteration = row.get("warm_start_source_iteration")
                candidate_count = row.get("warm_start_candidate_state_count")
                require(isinstance(source_k, int) and source_k >= 2, f"{context}: invalid source K")
                require(
                    source_iteration == max(1, source_k // 2),
                    f"{context}: midpoint iteration mismatch",
                )
                require(
                    candidate_count == source_k,
                    f"{context}: candidate count does not equal source K",
                )

            if row.get("warm_start_reset") is True:
                resets.append(
                    {
                        "task_id": task_id,
                        "pair": list(key),
                        "prediction_step": prediction,
                        "reason": row.get("warm_start_reset_reason"),
                    }
                )

        by_pair[key].append(row)

    for key, pair_rows in by_pair.items():
        pair_rows.sort(key=lambda item: int(item["prediction_step"]))
        observed = [int(item["prediction_step"]) for item in pair_rows]
        require(
            observed == list(range(len(observed))),
            f"task {task_id}/{arm}/{key}: non-contiguous predictions",
        )
    return dict(by_pair), resets


def percentile(sorted_values: list[float], probability: float) -> float:
    require(sorted_values, "percentile requires non-empty data")
    require(0.0 <= probability <= 1.0, "probability must lie in [0, 1]")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def numeric_summary(values: Iterable[float]) -> dict[str, Any]:
    finite_values = [float(value) for value in values if finite_number(value)]
    if not finite_values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    ordered = sorted(finite_values)
    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "median": percentile(ordered, 0.50),
        "p95": percentile(ordered, 0.95),
        "min": ordered[0],
        "max": ordered[-1],
    }


def episode_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    require(rows, "episode metrics require at least one prediction")
    latencies = [float(row["latency_ms"]) for row in rows]
    ks = [int(row["recurrent_iteration_count"]) for row in rows]
    calls = [int(row["get_output_call_count"]) for row in rows]
    scheduled_actions = [int(row.get("executed_actions_from_prediction", 0)) for row in rows]
    return {
        "num_predictions": len(rows),
        "recurrent_calls": sum(ks),
        "get_output_calls": sum(calls),
        "inference_time_ms": sum(latencies),
        "mean_prediction_latency_ms": sum(latencies) / len(latencies),
        "mean_K": sum(ks) / len(ks),
        "scheduled_action_slots": sum(scheduled_actions),
    }


def mean(values: Iterable[float]) -> float | None:
    finite_values = [float(value) for value in values if finite_number(value)]
    return None if not finite_values else float(sum(finite_values) / len(finite_values))


def summarize_arm(
    episodes: dict[Hashable, dict[str, Any]],
    steps: dict[Hashable, list[dict[str, Any]]],
) -> dict[str, Any]:
    require(set(episodes) == set(steps), "episode and step-log pair identities differ")
    metrics = []
    successes = 0
    all_rows: list[dict[str, Any]] = []
    for key in sorted(episodes, key=str):
        successes += int(episodes[key]["success"])
        metrics.append(episode_metrics(steps[key]))
        all_rows.extend(steps[key])
    return {
        "pairs": len(episodes),
        "successes": successes,
        "success_rate": successes / len(episodes),
        "total_predictions": sum(int(item["num_predictions"]) for item in metrics),
        "total_recurrent_calls": sum(int(item["recurrent_calls"]) for item in metrics),
        "total_get_output_calls": sum(int(item["get_output_calls"]) for item in metrics),
        "total_inference_time_ms": sum(float(item["inference_time_ms"]) for item in metrics),
        "mean_predictions_per_episode": mean(item["num_predictions"] for item in metrics),
        "mean_recurrent_calls_per_episode": mean(item["recurrent_calls"] for item in metrics),
        "mean_get_output_calls_per_episode": mean(item["get_output_calls"] for item in metrics),
        "mean_inference_time_ms_per_episode": mean(item["inference_time_ms"] for item in metrics),
        "mean_prediction_latency_ms": mean(row["latency_ms"] for row in all_rows),
        "mean_K_per_prediction": mean(row["recurrent_iteration_count"] for row in all_rows),
        "mean_scheduled_action_slots_per_episode": mean(
            item["scheduled_action_slots"] for item in metrics
        ),
    }


def paired_episode_record(
    *,
    task_id: int,
    key: tuple[int, int, int],
    cold_episode: dict[str, Any],
    warm_episode: dict[str, Any],
    cold_rows: list[dict[str, Any]],
    warm_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    cold_metrics = episode_metrics(cold_rows)
    warm_metrics = episode_metrics(warm_rows)
    return {
        "task_id": int(task_id),
        **pair_identity(key),
        "cold_success": bool(cold_episode["success"]),
        "warm_success": bool(warm_episode["success"]),
        "success_difference_warm_minus_cold": (
            int(bool(warm_episode["success"])) - int(bool(cold_episode["success"]))
        ),
        "cold": cold_metrics,
        "warm": warm_metrics,
        "difference_warm_minus_cold": {
            field: float(warm_metrics[field]) - float(cold_metrics[field])
            for field in PAIRED_EFFICIENCY_FIELDS
        },
    }


def summarize_paired_efficiency(records: list[dict[str, Any]]) -> dict[str, Any]:
    require(records, "paired efficiency summary requires at least one pair")
    return {
        field: numeric_summary(
            record["difference_warm_minus_cold"][field] for record in records
        )
        for field in PAIRED_EFFICIENCY_FIELDS
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = load_json(args.run_root / "run_plan.json")
    runs = plan.get("runs")
    require(isinstance(runs, list) and runs, "run plan has no runs")

    expected_tasks = tuple(int(task) for task in plan.get("tasks", []))
    expected_episodes = int(plan.get("episodes_per_task", 0))
    require(expected_tasks and expected_episodes > 0, "invalid task/episode plan")

    per_task: dict[int, dict[str, Any]] = {}
    all_resets: list[dict[str, Any]] = []
    all_paired_episode_records: list[dict[str, Any]] = []
    aggregate_counts = {
        "both_success": 0,
        "cold_only_success": 0,
        "warm_only_success": 0,
        "both_failure": 0,
    }
    aggregate_arm_episodes: dict[str, dict[Hashable, dict[str, Any]]] = {
        arm: {} for arm in ARMS
    }
    aggregate_arm_steps: dict[str, dict[Hashable, list[dict[str, Any]]]] = {
        arm: {} for arm in ARMS
    }

    for task_id in expected_tasks:
        task_data: dict[str, Any] = {}
        for arm in ARMS:
            matches = [
                run
                for run in runs
                if int(run["task_id"]) == task_id and run["arm"] == arm
            ]
            require(len(matches) == 1, f"task {task_id}: expected one run for arm {arm}")
            output_dir = Path(matches[0]["output_dir"])
            result = load_json(output_dir / "result.json")
            protocol = result.get("evaluation_protocol")
            require(isinstance(protocol, dict), f"task {task_id}/{arm}: missing protocol metadata")
            require(protocol.get("paired_rng") is True, f"task {task_id}/{arm}: result is not paired")
            require(
                protocol.get("phase") == plan.get("evaluation_protocol_phase"),
                f"task {task_id}/{arm}: phase mismatch",
            )
            require(
                int(result.get("total_episodes", -1)) == expected_episodes,
                f"task {task_id}/{arm}: episode count mismatch",
            )
            episodes = collect_episode_records(
                result, task_id=task_id, context=f"task {task_id}/{arm}"
            )
            require(
                len(episodes) == expected_episodes,
                f"task {task_id}/{arm}: episode record count mismatch",
            )
            step_rows = load_jsonl(output_dir / "steps.jsonl")
            steps, resets = validate_arm_steps(step_rows, arm=arm, task_id=task_id)
            all_resets.extend(resets)
            task_data[arm] = {
                "episodes": episodes,
                "steps": steps,
                "summary": summarize_arm(episodes, steps),
            }
            for key, episode in episodes.items():
                aggregate_key = (task_id, *key)
                aggregate_arm_episodes[arm][aggregate_key] = episode
                aggregate_arm_steps[arm][aggregate_key] = steps[key]

        cold_episodes = task_data["cold_initialized_adaptive"]["episodes"]
        warm_episodes = task_data["midpoint_warm_start_adaptive"]["episodes"]
        cold_steps = task_data["cold_initialized_adaptive"]["steps"]
        warm_steps = task_data["midpoint_warm_start_adaptive"]["steps"]
        require(
            set(cold_episodes) == set(warm_episodes),
            f"task {task_id}: cross-arm pair identities differ",
        )
        paired_counts = dict.fromkeys(aggregate_counts, 0)
        task_pair_records = []
        for key in sorted(cold_episodes):
            cold_success = bool(cold_episodes[key]["success"])
            warm_success = bool(warm_episodes[key]["success"])
            if cold_success and warm_success:
                category = "both_success"
            elif cold_success:
                category = "cold_only_success"
            elif warm_success:
                category = "warm_only_success"
            else:
                category = "both_failure"
            paired_counts[category] += 1
            aggregate_counts[category] += 1
            pair_record = paired_episode_record(
                task_id=task_id,
                key=key,
                cold_episode=cold_episodes[key],
                warm_episode=warm_episodes[key],
                cold_rows=cold_steps[key],
                warm_rows=warm_steps[key],
            )
            task_pair_records.append(pair_record)
            all_paired_episode_records.append(pair_record)

        per_task[task_id] = {
            "paired_counts": paired_counts,
            "observed_initial_state_ids": sorted(
                int(key[1]) for key in cold_episodes
            ),
            "cold_initialized_adaptive": task_data[
                "cold_initialized_adaptive"
            ]["summary"],
            "midpoint_warm_start_adaptive": task_data[
                "midpoint_warm_start_adaptive"
            ]["summary"],
            "paired_efficiency_warm_minus_cold": summarize_paired_efficiency(
                task_pair_records
            ),
            "paired_episodes": task_pair_records,
        }

    require(not all_resets, f"warm-start reset/non-finite events found: {all_resets[:5]}")
    total_pairs = sum(aggregate_counts.values())
    p01 = aggregate_counts["warm_only_success"] / total_pairs
    p10 = aggregate_counts["cold_only_success"] / total_pairs
    aggregate_arm_summary = {
        arm: summarize_arm(aggregate_arm_episodes[arm], aggregate_arm_steps[arm])
        for arm in ARMS
    }
    aggregate_total_differences = {
        "total_predictions": (
            aggregate_arm_summary["midpoint_warm_start_adaptive"]["total_predictions"]
            - aggregate_arm_summary["cold_initialized_adaptive"]["total_predictions"]
        ),
        "total_recurrent_calls": (
            aggregate_arm_summary["midpoint_warm_start_adaptive"]["total_recurrent_calls"]
            - aggregate_arm_summary["cold_initialized_adaptive"]["total_recurrent_calls"]
        ),
        "total_get_output_calls": (
            aggregate_arm_summary["midpoint_warm_start_adaptive"]["total_get_output_calls"]
            - aggregate_arm_summary["cold_initialized_adaptive"]["total_get_output_calls"]
        ),
        "total_inference_time_ms": (
            aggregate_arm_summary["midpoint_warm_start_adaptive"]["total_inference_time_ms"]
            - aggregate_arm_summary["cold_initialized_adaptive"]["total_inference_time_ms"]
        ),
    }
    report = {
        "schema_version": 2,
        "role": plan.get("role"),
        "contract_passed": True,
        "statistical_claim_allowed": False,
        "run_root": str(args.run_root.resolve()),
        "tasks": list(expected_tasks),
        "episodes_per_task": expected_episodes,
        "total_pairs": total_pairs,
        "paired_counts": aggregate_counts,
        "planning_only_p01": p01,
        "planning_only_p10": p10,
        "planning_only_difference": p01 - p10,
        "planning_only_discordance": p01 + p10,
        "warm_start_resets": all_resets,
        "observed_initial_state_ids_by_task": {
            str(task_id): per_task[task_id]["observed_initial_state_ids"]
            for task_id in expected_tasks
        },
        "aggregate_arm_summary": aggregate_arm_summary,
        "aggregate_total_difference_warm_minus_cold": aggregate_total_differences,
        "paired_efficiency_warm_minus_cold": summarize_paired_efficiency(
            all_paired_episode_records
        ),
        "per_task": {str(key): value for key, value in per_task.items()},
    }
    output = args.output or (args.run_root / "validation_report.json")
    require(not output.exists(), f"refusing to overwrite validation report: {output}")
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paired_summary = report["paired_efficiency_warm_minus_cold"]
    print("Warm-start paired runtime contract: PASS")
    print(
        f"Pairs={total_pairs}, cold_only={aggregate_counts['cold_only_success']}, "
        f"warm_only={aggregate_counts['warm_only_success']}, "
        f"difference={p01 - p10:+.4f}"
    )
    print(
        "Descriptive paired efficiency, warm-cold mean: "
        f"predictions/episode={paired_summary['num_predictions']['mean']:+.4f}, "
        f"recurrent_calls/episode={paired_summary['recurrent_calls']['mean']:+.4f}, "
        f"mean_K={paired_summary['mean_K']['mean']:+.4f}, "
        f"inference_time/episode={paired_summary['inference_time_ms']['mean']:+.2f} ms"
    )
    print(f"Wrote: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
