"""Replay deployment-matched Action-Delta shadows with sequential first hits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_TASK_IDS = (0, 1, 2, 3, 6, 7, 8, 9)
EXPECTED_TRAJECTORY_COUNT = 80
EXPECTED_PREDICTION_COUNT = 1762
EXPECTED_ELIGIBLE_ROW_COUNT = 2555
EXPECTED_ROW_TRIGGER_COUNT = 616
EXPECTED_ROW_SAFE_TRIGGER_COUNT = 615
EXPECTED_ROW_FALSE_SAFE_COUNT = 1
DEFAULT_MIN_TERMINAL_ITERS = (5, 6, 7)
DEFAULT_MANIFEST = Path(
    "benchmark_results/coda_anchor_feasibility/deployment_matched_shadow/"
    "phaseA_8tasks_20260817_085319/shards/manifest.json"
)


class FirstHitReplayError(ValueError):
    """Raised when shadow provenance or sequential replay data are invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FirstHitReplayError(message)


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    _require(math.isfinite(result), f"{name} must be finite")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prediction_sort_key(prediction: Mapping[str, Any]) -> tuple[Any, ...]:
    identity = prediction["identity"]
    return (
        int(identity["task_id"]),
        str(identity["trajectory_id"]),
        int(identity["action_prediction_index"]),
        str(prediction["prediction_id"]),
    )


def _neighbor_fields(
    prefix: str, row: Mapping[str, Any] | None
) -> dict[str, int | float | None]:
    if row is None:
        return {
            f"{prefix}_eligible_terminal_iteration": None,
            f"{prefix}_eligible_gate_score": None,
            f"{prefix}_eligible_exact_adjacent_action_mse": None,
        }
    return {
        f"{prefix}_eligible_terminal_iteration": int(row["terminal_iteration"]),
        f"{prefix}_eligible_gate_score": float(row["gate_score"]),
        f"{prefix}_eligible_exact_adjacent_action_mse": float(
            row["exact_adjacent_action_mse"]
        ),
    }


def first_hit_event(
    prediction: Mapping[str, Any],
    *,
    threshold: float,
    min_terminal_iter: int,
) -> tuple[bool, dict[str, Any] | None]:
    """Return eligibility and the first production-like threshold hit."""

    _require(min_terminal_iter >= 2, "minimum terminal iteration must be >= 2")
    threshold = _finite_float(threshold, "gate threshold")
    identity = prediction.get("identity")
    _require(isinstance(identity, Mapping), "prediction identity is missing")
    rows = sorted(
        prediction.get("transitions", []),
        key=lambda row: int(row["terminal_iteration"]),
    )
    terminals = [int(row["terminal_iteration"]) for row in rows]
    _require(
        len(terminals) == len(set(terminals)),
        f"duplicate terminal iteration in prediction {prediction.get('prediction_id')}",
    )
    eligible = [
        row for row in rows if int(row["terminal_iteration"]) >= min_terminal_iter
    ]
    for eligible_index, row in enumerate(eligible):
        score = _finite_float(row["gate_score"], "gate score")
        row_threshold = _finite_float(row["gate_threshold"], "row gate threshold")
        _require(row_threshold == threshold, "row gate threshold differs from artifact")
        stored_trigger = row.get("predicted_trigger")
        _require(
            stored_trigger is bool(score <= threshold),
            "stored predicted-trigger label differs from authoritative score",
        )
        if score > threshold:
            continue

        exact_mse = _finite_float(
            row["exact_adjacent_action_mse"], "exact adjacent action MSE"
        )
        recurrence_threshold = _finite_float(
            row["recurrence_mse_threshold"], "recurrence MSE threshold"
        )
        exact_safe = bool(exact_mse < recurrence_threshold)
        false_safe = not exact_safe
        _require(row.get("exact_safe") is exact_safe, "stored exact-safe label mismatch")
        _require(row.get("false_safe") is false_safe, "stored false-safe label mismatch")
        residual = _finite_float(row["residual"], "residual")
        _require(residual == exact_mse - score, "stored residual mismatch")

        row_identity = row.get("identity")
        if row_identity is not None:
            for name in (
                "trajectory_id",
                "task_id",
                "episode_id",
                "initial_state_id",
                "episode_seed",
                "action_prediction_index",
                "environment_timestep",
            ):
                _require(
                    row_identity.get(name) == identity.get(name),
                    f"row/prediction identity mismatch for {name}",
                )

        event = {
            "min_terminal_iter": int(min_terminal_iter),
            "task_id": int(identity["task_id"]),
            "task_name": str(prediction.get("task_name", "")),
            "trajectory_id": str(identity["trajectory_id"]),
            "episode_id": int(identity["episode_id"]),
            "initial_state_id": int(identity["initial_state_id"]),
            "paired_trial_id": int(identity.get("paired_trial_id", -1)),
            "episode_seed": int(identity["episode_seed"]),
            "prediction_id": str(prediction["prediction_id"]),
            "action_prediction_index": int(identity["action_prediction_index"]),
            "environment_timestep": int(identity["environment_timestep"]),
            "anchor_iteration": int(row["anchor_iteration"]),
            "terminal_iteration": int(row["terminal_iteration"]),
            "gate_score": score,
            "gate_threshold": threshold,
            "exact_adjacent_action_mse": exact_mse,
            "recurrence_mse_threshold": recurrence_threshold,
            "exact_safe": exact_safe,
            "false_safe": false_safe,
            "residual": residual,
        }
        event.update(
            _neighbor_fields(
                "previous", eligible[eligible_index - 1] if eligible_index else None
            )
        )
        event.update(
            _neighbor_fields(
                "next",
                eligible[eligible_index + 1]
                if eligible_index + 1 < len(eligible)
                else None,
            )
        )
        return bool(eligible), event
    return bool(eligible), None


def _summarize(
    predictions: Sequence[Mapping[str, Any]],
    eligible_prediction_ids: set[str],
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    total = len(predictions)
    eligible = len(eligible_prediction_ids)
    activated = len(events)
    safe = sum(bool(event["exact_safe"]) for event in events)
    false_safe = activated - safe
    trajectories = {str(item["identity"]["trajectory_id"]) for item in predictions}
    false_safe_trajectories = {
        str(event["trajectory_id"]) for event in events if event["false_safe"]
    }
    distribution = Counter(str(event["terminal_iteration"]) for event in events)
    return {
        "trajectory_count": len(trajectories),
        "total_predictions": total,
        "predictions_eligible_for_gate_evaluation": eligible,
        "ineligible_predictions": total - eligible,
        "first_hit_activations": activated,
        "first_hit_safe_activations": safe,
        "first_hit_false_safe_activations": false_safe,
        "eligible_no_trigger_predictions": eligible - activated,
        "no_trigger_predictions": total - activated,
        "activation_rate": activated / total if total else None,
        "activation_rate_among_eligible_predictions": (
            activated / eligible if eligible else None
        ),
        "false_safe_rate_among_first_hit_activations": (
            false_safe / activated if activated else None
        ),
        "nominal_coda_calls_saved": activated,
        "false_safe_trajectory_count": len(false_safe_trajectories),
        "first_hit_terminal_iteration_distribution": dict(sorted(distribution.items())),
    }


def replay_predictions(
    predictions: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
    min_terminal_iter: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Replay one cadence over every action prediction with first-hit stopping."""

    ordered = sorted(predictions, key=_prediction_sort_key)
    prediction_ids = [str(item["prediction_id"]) for item in ordered]
    _require(len(prediction_ids) == len(set(prediction_ids)), "duplicate prediction ID")
    eligible_ids: set[str] = set()
    events: list[dict[str, Any]] = []
    for prediction in ordered:
        eligible, event = first_hit_event(
            prediction,
            threshold=threshold,
            min_terminal_iter=min_terminal_iter,
        )
        prediction_id = str(prediction["prediction_id"])
        if eligible:
            eligible_ids.add(prediction_id)
        if event is not None:
            events.append(event)

    by_task: dict[str, Any] = {}
    predictions_by_task: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    events_by_task: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for prediction in ordered:
        predictions_by_task[int(prediction["identity"]["task_id"])].append(prediction)
    for event in events:
        events_by_task[int(event["task_id"])].append(event)
    for task_id in sorted(predictions_by_task):
        task_predictions = predictions_by_task[task_id]
        task_prediction_ids = {str(item["prediction_id"]) for item in task_predictions}
        by_task[str(task_id)] = _summarize(
            task_predictions,
            eligible_ids & task_prediction_ids,
            events_by_task[task_id],
        )

    return {
        "min_terminal_iter": int(min_terminal_iter),
        "max_skip": 1,
        "global": _summarize(ordered, eligible_ids, events),
        "by_task": by_task,
    }, events


def _validate_source_manifest(
    manifest: Mapping[str, Any], predictions: Sequence[Mapping[str, Any]]
) -> tuple[float, dict[str, int]]:
    _require(manifest.get("complete") is True, "source manifest is not complete")
    tasks = tuple(int(value) for value in manifest.get("expected_task_ids", []))
    _require(tasks == EXPECTED_TASK_IDS, f"unexpected task partition: {tasks}")
    _require(len(predictions) == EXPECTED_PREDICTION_COUNT, "prediction count mismatch")
    trajectory_count = len(
        {str(item["identity"]["trajectory_id"]) for item in predictions}
    )
    _require(trajectory_count == EXPECTED_TRAJECTORY_COUNT, "trajectory count mismatch")
    row_count = sum(len(item.get("transitions", [])) for item in predictions)
    _require(row_count == EXPECTED_ELIGIBLE_ROW_COUNT, "eligible-row count mismatch")
    artifact = manifest.get("artifact_identity", {})
    threshold = _finite_float(artifact.get("threshold"), "artifact threshold")
    summary = manifest.get("summary", {}).get("aggregate", {})
    row_level = {
        "eligible_rows": int(summary.get("eligible_rows", -1)),
        "predicted_triggers": int(summary.get("predicted_triggers", -1)),
        "safe_triggers": int(summary.get("exact_safe_triggers", -1)),
        "false_safe_triggers": int(summary.get("false_safe_triggers", -1)),
    }
    _require(
        row_level
        == {
            "eligible_rows": EXPECTED_ELIGIBLE_ROW_COUNT,
            "predicted_triggers": EXPECTED_ROW_TRIGGER_COUNT,
            "safe_triggers": EXPECTED_ROW_SAFE_TRIGGER_COUNT,
            "false_safe_triggers": EXPECTED_ROW_FALSE_SAFE_COUNT,
        },
        f"row-level reference mismatch: {row_level}",
    )
    return threshold, row_level


def analyze_collection(
    manifest: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    *,
    source_manifest_path: Path,
    source_manifest_sha256: str,
    min_terminal_iters: Sequence[int] = DEFAULT_MIN_TERMINAL_ITERS,
) -> dict[str, Any]:
    threshold, row_level = _validate_source_manifest(manifest, predictions)
    cadence_replays: dict[str, Any] = {}
    cadence_events: dict[int, list[dict[str, Any]]] = {}
    for min_terminal_iter in min_terminal_iters:
        replay, events = replay_predictions(
            predictions,
            threshold=threshold,
            min_terminal_iter=int(min_terminal_iter),
        )
        cadence_replays[str(min_terminal_iter)] = replay
        cadence_events[int(min_terminal_iter)] = events
    _require(5 in cadence_events, "primary min-terminal-5 replay is required")
    primary = cadence_replays["5"]
    primary_events = cadence_events[5]
    return {
        "analysis_type": "deployment_matched_action_delta_sequential_first_hit",
        "diagnostic_only": True,
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": source_manifest_sha256,
        "source_dataset": {
            "task_ids": list(EXPECTED_TASK_IDS),
            "trajectory_count": EXPECTED_TRAJECTORY_COUNT,
            "prediction_count": len(predictions),
            "eligible_row_count": EXPECTED_ELIGIBLE_ROW_COUNT,
        },
        "frozen_policy": {
            "threshold": threshold,
            "min_terminal_iter": 5,
            "max_skip": 1,
            "decision": "first eligible gate_score <= frozen threshold",
        },
        "row_level_comparison": {
            **row_level,
            "warning": (
                "Row-level predicted-trigger counts include later transitions from "
                "the same action prediction and are not production Coda savings."
            ),
            "primary_first_hit_activations": primary["global"]["first_hit_activations"],
            "primary_nominal_coda_calls_saved": primary["global"]["nominal_coda_calls_saved"],
        },
        "primary_replay": primary,
        "cadence_replays": cadence_replays,
        "first_hit_events": primary_events,
        "first_hit_false_safe_events": [
            event for event in primary_events if event["false_safe"]
        ],
        "interpretation": {
            "no_trigger_denominator": (
                "All action predictions; ineligible predictions are explicitly "
                "reported and necessarily have no trigger."
            ),
            "nominal_saving": (
                "One skipped Coda per first-hit activation under max_skip=1; "
                "this is a replay count, not a measured latency claim."
            ),
            "cadence_diagnostics": (
                "Min-terminal 6 and 7 are descriptive only and are not selected "
                "or calibrated on this dataset."
            ),
        },
    }


EVENT_FIELDS = (
    "min_terminal_iter",
    "task_id",
    "task_name",
    "trajectory_id",
    "episode_id",
    "initial_state_id",
    "paired_trial_id",
    "episode_seed",
    "prediction_id",
    "action_prediction_index",
    "environment_timestep",
    "anchor_iteration",
    "terminal_iteration",
    "gate_score",
    "gate_threshold",
    "exact_adjacent_action_mse",
    "recurrence_mse_threshold",
    "exact_safe",
    "false_safe",
    "residual",
    "previous_eligible_terminal_iteration",
    "previous_eligible_gate_score",
    "previous_eligible_exact_adjacent_action_mse",
    "next_eligible_terminal_iteration",
    "next_eligible_gate_score",
    "next_eligible_exact_adjacent_action_mse",
)


SUMMARY_FIELDS = (
    "min_terminal_iter",
    "task_id",
    "trajectory_count",
    "total_predictions",
    "predictions_eligible_for_gate_evaluation",
    "ineligible_predictions",
    "first_hit_activations",
    "first_hit_safe_activations",
    "first_hit_false_safe_activations",
    "eligible_no_trigger_predictions",
    "no_trigger_predictions",
    "activation_rate",
    "activation_rate_among_eligible_predictions",
    "false_safe_rate_among_first_hit_activations",
    "nominal_coda_calls_saved",
    "false_safe_trajectory_count",
    "first_hit_terminal_iteration_distribution",
)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for source in rows:
            row = dict(source)
            for name, value in row.items():
                if isinstance(value, Mapping):
                    row[name] = json.dumps(value, sort_keys=True, separators=(",", ":"))
            writer.writerow({name: row.get(name) for name in fields})


def write_results(results: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(
        output_dir / "first_hit_events.csv",
        results["first_hit_events"],
        EVENT_FIELDS,
    )
    task_rows = []
    for cadence, replay in sorted(
        results["cadence_replays"].items(), key=lambda item: int(item[0])
    ):
        for task_id, summary in sorted(
            replay["by_task"].items(), key=lambda item: int(item[0])
        ):
            task_rows.append(
                {"min_terminal_iter": int(cadence), "task_id": int(task_id), **summary}
            )
    _write_csv(output_dir / "per_task_summary.csv", task_rows, SUMMARY_FIELDS)


def print_summary(results: Mapping[str, Any], output_dir: Path) -> None:
    print("Deployment-matched Action-Delta sequential first-hit replay")
    print(f"Threshold: {results['frozen_policy']['threshold']:.17g}")
    print(
        "cadence  total  eligible  activated  safe  false-safe  no-trigger  saves"
    )
    for cadence, replay in sorted(
        results["cadence_replays"].items(), key=lambda item: int(item[0])
    ):
        summary = replay["global"]
        print(
            f"{int(cadence):>7}  {summary['total_predictions']:>5}  "
            f"{summary['predictions_eligible_for_gate_evaluation']:>8}  "
            f"{summary['first_hit_activations']:>9}  "
            f"{summary['first_hit_safe_activations']:>4}  "
            f"{summary['first_hit_false_safe_activations']:>10}  "
            f"{summary['no_trigger_predictions']:>10}  "
            f"{summary['nominal_coda_calls_saved']:>5}"
        )
    primary = results["primary_replay"]["global"]
    print(
        "Primary first-hit terminal distribution: "
        + json.dumps(primary["first_hit_terminal_iteration_distribution"], sort_keys=True)
    )
    print(
        "Row-level triggers are not Coda savings: "
        f"{results['row_level_comparison']['predicted_triggers']} row hits vs "
        f"{primary['nominal_coda_calls_saved']} first-hit nominal saves."
    )
    false_safe = results["first_hit_false_safe_events"]
    print(f"First-hit false-safe events: {len(false_safe)}")
    for event in false_safe:
        print(json.dumps(event, indent=2, sort_keys=True))
    print(f"Wrote {output_dir / 'results.json'}")
    print(f"Wrote {output_dir / 'first_hit_events.csv'}")
    print(f"Wrote {output_dir / 'per_task_summary.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else manifest_path.parent.parent / "sequential_first_hit_analysis"
    )

    # Import lazily so unit tests of replay semantics do not load runtime packages.
    from experiments.robot.libero.action_delta_gate_shadow_collection import (
        load_action_delta_gate_shadow_collection,
    )

    manifest, predictions = load_action_delta_gate_shadow_collection(manifest_path)
    results = analyze_collection(
        manifest,
        predictions,
        source_manifest_path=manifest_path,
        source_manifest_sha256=_sha256_file(manifest_path),
    )
    write_results(results, output_dir)
    print_summary(results, output_dir)


if __name__ == "__main__":
    main()
