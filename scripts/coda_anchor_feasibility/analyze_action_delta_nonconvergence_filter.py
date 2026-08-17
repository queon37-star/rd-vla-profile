"""Feasibility analysis for a frozen Action-Delta non-convergence filter."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


TASK_IDS = (0, 1, 2, 3, 6, 7, 8, 9)
EXACT_CONVERGENCE_MSE = 0.001
MIN_TERMINAL_ITER = 5
SCORER_COST_MS = 0.36627289134678936
CODA_COST_MS = 1.864207385085098
EXPECTED_TRAJECTORIES = 80
EXPECTED_PREDICTIONS = 1762
EXPECTED_ROWS = 2555
DEFAULT_MANIFEST = Path(
    "benchmark_results/coda_anchor_feasibility/deployment_matched_shadow/"
    "phaseA_8tasks_20260817_085319/shards/manifest.json"
)
FIXED_THRESHOLDS = (
    0.0,
    0.00025,
    0.0005,
    0.000732466738008497,
    0.001,
    0.0015,
    0.002,
    0.003,
    0.005,
    0.0075,
    0.01,
    0.015,
    0.02,
)
QUANTILES = (0, 10, 20, 25, 33, 50, 60, 70, 75, 80, 85, 90, 92.5, 95, 97.5, 99, 99.5, 100)


class NonConvergenceAnalysisError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NonConvergenceAnalysisError(message)


def _finite(value: Any, name: str) -> float:
    result = float(value)
    _require(math.isfinite(result), f"{name} must be finite")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def row_metrics(rows: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    """Evaluate gate_score >= threshold without causal masking."""

    threshold = _finite(threshold, "non-convergence threshold")
    proposed = [row for row in rows if _finite(row["gate_score"], "gate score") >= threshold]
    true_nonconverged = [
        row
        for row in proposed
        if _finite(row["exact_adjacent_action_mse"], "exact MSE")
        >= EXACT_CONVERGENCE_MSE
    ]
    false_nonconvergence = len(proposed) - len(true_nonconverged)
    total_nonconverged = sum(
        _finite(row["exact_adjacent_action_mse"], "exact MSE")
        >= EXACT_CONVERGENCE_MSE
        for row in rows
    )
    total_safe = len(rows) - total_nonconverged
    return {
        "eligible_rows": len(rows),
        "exact_nonconverged_rows": int(total_nonconverged),
        "exact_safe_rows": int(total_safe),
        "proposed_skips": len(proposed),
        "true_nonconvergence_skips": len(true_nonconverged),
        "exact_safe_rows_incorrectly_skipped": false_nonconvergence,
        "precision": _safe_ratio(len(true_nonconverged), len(proposed)),
        "recall_exact_nonconverged": _safe_ratio(
            len(true_nonconverged), total_nonconverged
        ),
        "false_nonconvergence_rate_among_exact_safe": _safe_ratio(
            false_nonconvergence, total_safe
        ),
    }


def _native_mse_from_actions(current: Any, anchor: Any) -> float:
    _require(current.shape == anchor.shape, "action shape mismatch")
    _require(current.dtype == anchor.dtype, "action dtype mismatch")
    # Mirror the runtime adjacent-action expression in the saved action dtype.
    return float(((current - anchor) ** 2).mean().item())


def prepare_replay_data(
    predictions: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build compact rows and precompute altered-history exact comparisons."""

    compact_predictions: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []
    baseline_coda_calls = 0
    task_baseline_coda_calls: dict[int, int] = defaultdict(int)
    trajectory_ids: set[str] = set()
    for prediction in predictions:
        identity = prediction["identity"]
        task_id = int(identity["task_id"])
        trajectory_ids.add(str(identity["trajectory_id"]))
        production = prediction["production_parity"]
        coda_calls = int(production["exact_coda_call_count"])
        baseline_coda_calls += coda_calls
        task_baseline_coda_calls[task_id] += coda_calls
        source_rows = sorted(
            prediction.get("transitions", []),
            key=lambda row: int(row["terminal_iteration"]),
        )
        terminals = [int(row["terminal_iteration"]) for row in source_rows]
        _require(len(terminals) == len(set(terminals)), "duplicate terminal iteration")
        rows = []
        for index, source in enumerate(source_rows):
            terminal = int(source["terminal_iteration"])
            _require(terminal >= MIN_TERMINAL_ITER, "pre-eligibility row in shadow data")
            exact_mse = _finite(source["exact_adjacent_action_mse"], "exact MSE")
            exact_safe = exact_mse < EXACT_CONVERGENCE_MSE
            _require(source["exact_safe"] is exact_safe, "exact-safe label mismatch")
            two_step_mse = None
            if index:
                previous = source_rows[index - 1]
                _require(
                    terminal == int(previous["terminal_iteration"]) + 1,
                    "eligible rows must be terminal-contiguous",
                )
                _require(
                    bool(
                        (
                            previous["tensors"]["exact_terminal_action"]
                            == source["tensors"]["anchor_action"]
                        ).all().item()
                    ),
                    "consecutive exact actions do not join",
                )
                two_step_mse = _native_mse_from_actions(
                    source["tensors"]["exact_terminal_action"],
                    previous["tensors"]["anchor_action"],
                )
            row = {
                "task_id": task_id,
                "prediction_id": str(prediction["prediction_id"]),
                "trajectory_id": str(identity["trajectory_id"]),
                "terminal_iteration": terminal,
                "gate_score": _finite(source["gate_score"], "gate score"),
                "exact_adjacent_action_mse": exact_mse,
                "exact_safe": exact_safe,
                "two_step_exact_mse_after_previous_skip": two_step_mse,
            }
            rows.append(row)
            flat_rows.append(row)
        compact_predictions.append(
            {
                "task_id": task_id,
                "prediction_id": str(prediction["prediction_id"]),
                "trajectory_id": str(identity["trajectory_id"]),
                "baseline_coda_calls": coda_calls,
                "baseline_stop_reason": production.get("stop_reason"),
                "rows": rows,
            }
        )
    return compact_predictions, flat_rows, {
        "trajectory_count": len(trajectory_ids),
        "prediction_count": len(predictions),
        "baseline_coda_calls": baseline_coda_calls,
        "baseline_coda_calls_by_task": dict(task_baseline_coda_calls),
    }


def _empty_causal_counts() -> dict[str, Any]:
    return {
        "prediction_count": 0,
        "baseline_coda_calls": 0,
        "scorer_call_count": 0,
        "nominal_skipped_coda_calls": 0,
        "true_nonconvergence_skips": 0,
        "exact_safe_coda_calls_incorrectly_skipped": 0,
        "forced_exact_coda_calls": 0,
        "adjacent_history_difference_events": 0,
        "altered_history_stop_decision_changes": 0,
        "altered_history_cpu_mse_within_1e_6_of_threshold": 0,
        "censored_predictions": 0,
        "censored_after_skipped_terminal_exact_safe": 0,
        "censored_after_altered_history_nonconvergence": 0,
        "predictions_with_a_skip": 0,
        "predictions_with_history_difference": 0,
    }


def _finalize_causal(
    counts: Mapping[str, Any], *, total_exact_nonconverged: int, total_exact_safe: int
) -> dict[str, Any]:
    result = dict(counts)
    skips = int(result["nominal_skipped_coda_calls"])
    scorer_calls = int(result["scorer_call_count"])
    true_skips = int(result["true_nonconvergence_skips"])
    false_skips = int(result["exact_safe_coda_calls_incorrectly_skipped"])
    baseline_calls = int(result["baseline_coda_calls"])
    gross = skips * CODA_COST_MS
    conservative_gross = true_skips * CODA_COST_MS
    scorer_cost = scorer_calls * SCORER_COST_MS
    result.update(
        {
            "exact_nonconverged_eligible_rows": int(total_exact_nonconverged),
            "exact_safe_eligible_rows": int(total_exact_safe),
            "skip_precision": _safe_ratio(true_skips, skips),
            "exact_nonconvergence_capture": _safe_ratio(
                true_skips, total_exact_nonconverged
            ),
            "fraction_baseline_coda_calls_skipped": _safe_ratio(skips, baseline_calls),
            "fraction_exact_safe_eligible_calls_incorrectly_skipped": _safe_ratio(
                false_skips, total_exact_safe
            ),
            "gross_coda_time_saved_ms": gross,
            "conservative_true_nonconvergence_coda_time_saved_ms": conservative_gross,
            "scorer_cost_ms": scorer_cost,
            "nominal_net_time_saved_ms": gross - scorer_cost,
            "nominal_net_positive": bool(gross > scorer_cost),
            "conservative_nominal_net_time_saved_ms": conservative_gross
            - scorer_cost,
            "conservative_nominal_net_positive": bool(
                conservative_gross > scorer_cost
            ),
            "break_even_skip_probability_per_score": SCORER_COST_MS / CODA_COST_MS,
            "observed_skip_probability_per_score": _safe_ratio(skips, scorer_calls),
        }
    )
    return result


def causal_replay(
    compact_predictions: Sequence[Mapping[str, Any]], threshold: float
) -> dict[str, Any]:
    """Replay max_skip=1 with forced exact Coda and altered action history."""

    threshold = _finite(threshold, "non-convergence threshold")
    global_counts = _empty_causal_counts()
    task_counts: dict[int, dict[str, Any]] = defaultdict(_empty_causal_counts)
    flat_rows = [row for prediction in compact_predictions for row in prediction["rows"]]
    total_nonconverged = sum(not row["exact_safe"] for row in flat_rows)
    total_safe = len(flat_rows) - total_nonconverged
    task_row_totals: dict[int, tuple[int, int]] = {}
    for task_id in TASK_IDS:
        task_rows = [row for row in flat_rows if int(row["task_id"]) == task_id]
        nonconverged = sum(not row["exact_safe"] for row in task_rows)
        task_row_totals[task_id] = (nonconverged, len(task_rows) - nonconverged)

    for prediction in compact_predictions:
        task_id = int(prediction["task_id"])
        targets = (global_counts, task_counts[task_id])
        for target in targets:
            target["prediction_count"] += 1
            target["baseline_coda_calls"] += int(prediction["baseline_coda_calls"])
        forced_exact = False
        prediction_had_skip = False
        prediction_history_changed = False
        censored = False
        rows = prediction["rows"]
        for index, row in enumerate(rows):
            is_last = index + 1 == len(rows)
            if forced_exact:
                altered_mse = row["two_step_exact_mse_after_previous_skip"]
                _require(altered_mse is not None, "forced exact row lacks two-step MSE")
                altered_safe = float(altered_mse) < EXACT_CONVERGENCE_MSE
                native_safe = bool(row["exact_safe"])
                for target in targets:
                    target["forced_exact_coda_calls"] += 1
                    target["adjacent_history_difference_events"] += 1
                    target["altered_history_stop_decision_changes"] += int(
                        altered_safe != native_safe
                    )
                    target["altered_history_cpu_mse_within_1e_6_of_threshold"] += int(
                        abs(float(altered_mse) - EXACT_CONVERGENCE_MSE) <= 1e-6
                    )
                prediction_history_changed = True
                forced_exact = False
                if altered_safe:
                    break
                if is_last and prediction["baseline_stop_reason"] != "max_iter":
                    censored = True
                    for target in targets:
                        target["censored_after_altered_history_nonconvergence"] += 1
                    break
                continue

            for target in targets:
                target["scorer_call_count"] += 1
            if float(row["gate_score"]) >= threshold:
                prediction_had_skip = True
                for target in targets:
                    target["nominal_skipped_coda_calls"] += 1
                    if row["exact_safe"]:
                        target["exact_safe_coda_calls_incorrectly_skipped"] += 1
                    else:
                        target["true_nonconvergence_skips"] += 1
                if is_last:
                    censored = True
                    for target in targets:
                        target["censored_after_skipped_terminal_exact_safe"] += int(
                            row["exact_safe"]
                        )
                    break
                forced_exact = True
                continue

            if row["exact_safe"]:
                break

        for target in targets:
            target["predictions_with_a_skip"] += int(prediction_had_skip)
            target["predictions_with_history_difference"] += int(
                prediction_history_changed
            )
            target["censored_predictions"] += int(censored)

    by_task = {}
    for task_id in TASK_IDS:
        nonconverged, safe = task_row_totals[task_id]
        by_task[str(task_id)] = _finalize_causal(
            task_counts[task_id],
            total_exact_nonconverged=nonconverged,
            total_exact_safe=safe,
        )
    return {
        "threshold": threshold,
        "global": _finalize_causal(
            global_counts,
            total_exact_nonconverged=total_nonconverged,
            total_exact_safe=total_safe,
        ),
        "by_task": by_task,
    }


def _threshold_sweep(scores: np.ndarray) -> list[dict[str, Any]]:
    sources: dict[float, list[str]] = defaultdict(list)
    for value in FIXED_THRESHOLDS:
        sources[float(value)].append(f"fixed_{value:.17g}")
    for quantile in QUANTILES:
        value = float(np.percentile(scores, quantile))
        sources[value].append(f"quantile_p{quantile:g}")
    sources[float(np.nextafter(scores.max(), math.inf))].append("zero_skip_sentinel")
    return [
        {"threshold": threshold, "sources": labels}
        for threshold, labels in sorted(sources.items())
    ]


def _full_frontier_thresholds(scores: np.ndarray) -> list[float]:
    return sorted(
        {float(value) for value in scores}
        | {float(np.nextafter(scores.max(), math.inf))}
    )


def _row_metrics_by_task(rows: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    return {
        str(task_id): row_metrics(
            [row for row in rows if int(row["task_id"]) == task_id], threshold
        )
        for task_id in TASK_IDS
    }


def analyze(
    manifest: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    *,
    manifest_path: Path,
) -> dict[str, Any]:
    _require(manifest.get("complete") is True, "manifest is incomplete")
    _require(tuple(manifest["expected_task_ids"]) == TASK_IDS, "task partition mismatch")
    _require(len(predictions) == EXPECTED_PREDICTIONS, "prediction count mismatch")
    compact, rows, dataset = prepare_replay_data(predictions)
    _require(dataset["trajectory_count"] == EXPECTED_TRAJECTORIES, "trajectory count mismatch")
    _require(len(rows) == EXPECTED_ROWS, "eligible-row count mismatch")
    scores = np.asarray([row["gate_score"] for row in rows], dtype=np.float64)
    artifact_threshold = float(manifest["artifact_identity"]["threshold"])

    broad_sweep = []
    for definition in _threshold_sweep(scores):
        threshold = definition["threshold"]
        broad_sweep.append(
            {
                **definition,
                "row_level": row_metrics(rows, threshold),
                "row_level_by_task": _row_metrics_by_task(rows, threshold),
                "causal_max_skip_1": causal_replay(compact, threshold),
            }
        )

    frontier = []
    for threshold in _full_frontier_thresholds(scores):
        row_result = row_metrics(rows, threshold)
        replay = causal_replay(compact, threshold)["global"]
        frontier.append(
            {
                "threshold": threshold,
                "row_proposed_skips": row_result["proposed_skips"],
                "row_precision": row_result["precision"],
                "row_recall_exact_nonconverged": row_result[
                    "recall_exact_nonconverged"
                ],
                "causal_scorer_calls": replay["scorer_call_count"],
                "causal_nominal_skips": replay["nominal_skipped_coda_calls"],
                "causal_true_nonconvergence_skips": replay[
                    "true_nonconvergence_skips"
                ],
                "causal_exact_safe_incorrect_skips": replay[
                    "exact_safe_coda_calls_incorrectly_skipped"
                ],
                "causal_skip_precision": replay["skip_precision"],
                "causal_nonconvergence_capture": replay[
                    "exact_nonconvergence_capture"
                ],
                "fraction_coda_calls_skipped": replay[
                    "fraction_baseline_coda_calls_skipped"
                ],
                "fraction_exact_safe_calls_incorrectly_skipped": replay[
                    "fraction_exact_safe_eligible_calls_incorrectly_skipped"
                ],
                "gross_coda_time_saved_ms": replay["gross_coda_time_saved_ms"],
                "conservative_true_nonconvergence_coda_time_saved_ms": replay[
                    "conservative_true_nonconvergence_coda_time_saved_ms"
                ],
                "scorer_cost_ms": replay["scorer_cost_ms"],
                "nominal_net_time_saved_ms": replay["nominal_net_time_saved_ms"],
                "nominal_net_positive": replay["nominal_net_positive"],
                "conservative_nominal_net_time_saved_ms": replay[
                    "conservative_nominal_net_time_saved_ms"
                ],
                "conservative_nominal_net_positive": replay[
                    "conservative_nominal_net_positive"
                ],
                "adjacent_history_difference_events": replay[
                    "adjacent_history_difference_events"
                ],
                "altered_history_stop_decision_changes": replay[
                    "altered_history_stop_decision_changes"
                ],
                "censored_predictions": replay["censored_predictions"],
            }
        )

    simultaneous = [
        point
        for point in frontier
        if point["causal_skip_precision"] is not None
        and point["causal_skip_precision"] >= 0.99
        and point["fraction_coda_calls_skipped"] >= 0.05
        and point["conservative_nominal_net_positive"]
    ]
    zero_direct_error_region = [
        point
        for point in simultaneous
        if point["causal_exact_safe_incorrect_skips"] == 0
    ]
    return {
        "analysis_type": "frozen_action_delta_nonconvergence_filter_feasibility",
        "diagnostic_only": True,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256_file(manifest_path),
        "source_dataset": {
            **dataset,
            "task_ids": list(TASK_IDS),
            "eligible_row_count": len(rows),
            "exact_nonconverged_row_count": sum(not row["exact_safe"] for row in rows),
            "exact_safe_row_count": sum(row["exact_safe"] for row in rows),
        },
        "frozen_artifact": {
            **dict(manifest["artifact_identity"]),
            "original_convergence_threshold_preserved": artifact_threshold,
        },
        "policy_interpretation": {
            "proposal": "skip exact Coda only when gate_score >= diagnostic q",
            "exact_stop_only": f"executed exact Coda MSE < {EXACT_CONVERGENCE_MSE}",
            "max_skip": 1,
            "post_skip": "force exact Coda next; do not score the forced iteration",
            "min_terminal_iter": MIN_TERMINAL_ITER,
        },
        "cost_anchors": {
            "scorer_cost_ms": SCORER_COST_MS,
            "coda_cost_ms": CODA_COST_MS,
            "break_even_skip_probability_per_score": SCORER_COST_MS / CODA_COST_MS,
            "interpretation": "Established Task-4 optimized-scorer/Coda measurements",
        },
        "broad_sweep": broad_sweep,
        "risk_coverage_frontier": frontier,
        "simultaneous_region_screen": {
            "descriptive_criteria_not_policy_selection": {
                "minimum_skip_precision": 0.99,
                "minimum_fraction_baseline_coda_calls_skipped": 0.05,
                "requires_positive_true_nonconvergence_only_nominal_net_latency": True,
            },
            "matching_frontier_point_count": len(simultaneous),
            "exists": bool(simultaneous),
            "points": simultaneous,
            "zero_direct_false_nonconvergence_subregion": {
                "matching_frontier_point_count": len(zero_direct_error_region),
                "exists": bool(zero_direct_error_region),
                "minimum_threshold": (
                    min(point["threshold"] for point in zero_direct_error_region)
                    if zero_direct_error_region
                    else None
                ),
                "maximum_threshold": (
                    max(point["threshold"] for point in zero_direct_error_region)
                    if zero_direct_error_region
                    else None
                ),
                "points": zero_direct_error_region,
                "warning": (
                    "Zero directly skipped exact-safe rows does not imply causal "
                    "equivalence; altered-history and censored counts remain nonzero."
                ),
            },
        },
        "limitations": [
            "Thresholds are descriptive sweeps on development Tasks 0,1,2,3,6,7,8,9; none is selected.",
            "No Task 4/5 data is used.",
            "A skip changes adjacent exact-action history, so Warm-only trajectory equivalence is not claimed.",
            "Altered-history MSE is replayed on CPU from saved exact action tensors using the runtime expression.",
            "When altered history requires a state beyond the Warm-only terminal row, replay is censored and counted.",
            "Coda savings and latency are nominal observed-prefix estimates; censored continuations are not invented.",
        ],
    }


FRONTIER_FIELDS = (
    "threshold",
    "row_proposed_skips",
    "row_precision",
    "row_recall_exact_nonconverged",
    "causal_scorer_calls",
    "causal_nominal_skips",
    "causal_true_nonconvergence_skips",
    "causal_exact_safe_incorrect_skips",
    "causal_skip_precision",
    "causal_nonconvergence_capture",
    "fraction_coda_calls_skipped",
    "fraction_exact_safe_calls_incorrectly_skipped",
    "gross_coda_time_saved_ms",
    "conservative_true_nonconvergence_coda_time_saved_ms",
    "scorer_cost_ms",
    "nominal_net_time_saved_ms",
    "nominal_net_positive",
    "conservative_nominal_net_time_saved_ms",
    "conservative_nominal_net_positive",
    "adjacent_history_difference_events",
    "altered_history_stop_decision_changes",
    "censored_predictions",
)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_outputs(results: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(
        output_dir / "risk_coverage_frontier.csv",
        results["risk_coverage_frontier"],
        FRONTIER_FIELDS,
    )
    per_task_rows = []
    for point in results["broad_sweep"]:
        replay = point["causal_max_skip_1"]
        for task_id in TASK_IDS:
            row_result = point["row_level_by_task"][str(task_id)]
            causal = replay["by_task"][str(task_id)]
            per_task_rows.append(
                {
                    "threshold": point["threshold"],
                    "threshold_sources": "|".join(point["sources"]),
                    "task_id": task_id,
                    **{f"row_{name}": value for name, value in row_result.items()},
                    **{f"causal_{name}": value for name, value in causal.items()},
                }
            )
    fields = tuple(per_task_rows[0]) if per_task_rows else ()
    _write_csv(output_dir / "per_task_summary.csv", per_task_rows, fields)


def print_summary(results: Mapping[str, Any], output_dir: Path) -> None:
    dataset = results["source_dataset"]
    print("Frozen Action-Delta conservative non-convergence feasibility")
    print(
        f"Rows={dataset['eligible_row_count']} nonconverged={dataset['exact_nonconverged_row_count']} "
        f"safe={dataset['exact_safe_row_count']} baseline_coda={dataset['baseline_coda_calls']}"
    )
    print("q  skips  true  false-safe  precision  capture  coda-fraction  net-ms  censored")
    for point in results["broad_sweep"]:
        causal = point["causal_max_skip_1"]["global"]
        precision = causal["skip_precision"]
        capture = causal["exact_nonconvergence_capture"]
        print(
            f"{point['threshold']:.9g}  {causal['nominal_skipped_coda_calls']:>5}  "
            f"{causal['true_nonconvergence_skips']:>4}  "
            f"{causal['exact_safe_coda_calls_incorrectly_skipped']:>10}  "
            f"{precision if precision is not None else float('nan'):.5f}  "
            f"{capture if capture is not None else float('nan'):.5f}  "
            f"{causal['fraction_baseline_coda_calls_skipped']:.5f}  "
            f"{causal['nominal_net_time_saved_ms']:.3f}  "
            f"{causal['censored_predictions']:>8}"
        )
    screen = results["simultaneous_region_screen"]
    print(
        "Descriptive >=99% precision, >=5% nominal Coda reduction, "
        "true-nonconvergence-only positive-net region: "
        f"{screen['exists']} ({screen['matching_frontier_point_count']} frontier points)"
    )
    for name in ("results.json", "risk_coverage_frontier.csv", "per_task_summary.csv"):
        print(f"Wrote {output_dir / name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else manifest_path.parent.parent / "nonconvergence_filter_analysis"
    )
    from experiments.robot.libero.action_delta_gate_shadow_collection import (
        load_action_delta_gate_shadow_collection,
    )

    manifest, predictions = load_action_delta_gate_shadow_collection(manifest_path)
    results = analyze(manifest, predictions, manifest_path=manifest_path)
    write_outputs(results, output_dir)
    print_summary(results, output_dir)


if __name__ == "__main__":
    main()
