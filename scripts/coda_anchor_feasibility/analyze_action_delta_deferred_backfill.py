"""Analyze adjacent-history-preserving deferred Action-Delta confirmation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


TASK_IDS = (0, 1, 2, 3, 6, 7, 8, 9)
DIAGNOSTIC_Q = 0.0015
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


class DeferredBackfillAnalysisError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DeferredBackfillAnalysisError(message)


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_predictions(
    predictions: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    trajectory_ids: set[str] = set()
    row_count = 0
    baseline_calls = 0
    for prediction in predictions:
        identity = prediction["identity"]
        production = prediction["production_parity"]
        task_id = int(identity["task_id"])
        trajectory_ids.add(str(identity["trajectory_id"]))
        baseline_k = int(production["K_t"])
        exact_calls = int(production["exact_coda_call_count"])
        _require(exact_calls == baseline_k, "baseline exact-Coda count must equal K")
        source_rows = sorted(
            prediction.get("transitions", []),
            key=lambda row: int(row["terminal_iteration"]),
        )
        rows: list[dict[str, Any]] = []
        previous_terminal = None
        for source in source_rows:
            terminal = int(source["terminal_iteration"])
            _require(terminal >= MIN_TERMINAL_ITER, "pre-eligibility transition")
            if previous_terminal is not None:
                _require(
                    terminal == previous_terminal + 1,
                    "eligible transition trace is not terminal-contiguous",
                )
            score = float(source["gate_score"])
            exact_mse = float(source["exact_adjacent_action_mse"])
            _require(math.isfinite(score), "non-finite gate score")
            _require(math.isfinite(exact_mse), "non-finite exact adjacent MSE")
            exact_safe = exact_mse < EXACT_CONVERGENCE_MSE
            _require(bool(source["exact_safe"]) == exact_safe, "exact-safe mismatch")
            rows.append(
                {
                    "terminal_iteration": terminal,
                    "gate_score": score,
                    "exact_adjacent_action_mse": exact_mse,
                    "exact_safe": exact_safe,
                }
            )
            previous_terminal = terminal
        if rows:
            _require(
                rows[0]["terminal_iteration"] == MIN_TERMINAL_ITER,
                "eligible trace must begin at terminal 5",
            )
            _require(
                rows[-1]["terminal_iteration"] == baseline_k,
                "eligible trace must end at baseline K",
            )
        row_count += len(rows)
        baseline_calls += exact_calls
        compact.append(
            {
                "task_id": task_id,
                "prediction_id": str(prediction["prediction_id"]),
                "trajectory_id": str(identity["trajectory_id"]),
                "baseline_k": baseline_k,
                "baseline_coda_calls": exact_calls,
                "baseline_stop_reason": str(production["stop_reason"]),
                "baseline_adaptive_stop": bool(production["adaptive_stop"]),
                "rows": rows,
            }
        )
    return compact, {
        "trajectory_count": len(trajectory_ids),
        "prediction_count": len(compact),
        "eligible_row_count": row_count,
        "baseline_coda_calls": baseline_calls,
    }


def replay_prediction(
    prediction: Mapping[str, Any],
    *,
    threshold: float = DIAGNOSTIC_Q,
    require_exact_terminal_output: bool = False,
) -> dict[str, Any]:
    """Replay one prediction while retaining baseline-adjacent confirmations.

    A terminal high-score run needs no convergence confirmation. The optional
    exact-terminal-output variant executes its final Coda solely to preserve
    the baseline terminal action contract; this is reported separately because
    the requested policy specifies stopping semantics, not return semantics.
    """

    rows = list(prediction["rows"])
    baseline_k = int(prediction["baseline_k"])
    baseline_calls = int(prediction["baseline_coda_calls"])
    preeligible_calls = baseline_calls - len(rows)
    _require(preeligible_calls >= 0, "negative pre-eligibility Coda count")

    scorer_calls = 0
    eligible_exact_calls = 0
    deferred_calls = 0
    backfilled_calls = 0
    policy_k = None
    assumption_violations = 0
    runs: list[dict[str, Any]] = []
    open_run: list[dict[str, Any]] = []

    def close_run(*, followed_by_low: bool, trace_end: bool) -> None:
        nonlocal eligible_exact_calls, backfilled_calls, open_run
        if not open_run:
            return
        backfilled = 0
        if followed_by_low:
            # Only a[k-1] is required alongside the current low-score a[k].
            backfilled = 1
        elif trace_end and require_exact_terminal_output:
            # This preserves the exact returned terminal action, not stopping.
            backfilled = 1
        eligible_exact_calls += backfilled
        backfilled_calls += backfilled
        runs.append(
            {
                "length": len(open_run),
                "start_terminal_iteration": int(
                    open_run[0]["terminal_iteration"]
                ),
                "end_terminal_iteration": int(open_run[-1]["terminal_iteration"]),
                "followed_by_low_score": bool(followed_by_low),
                "ends_at_baseline_trace": bool(trace_end),
                "backfilled_coda_calls": backfilled,
                "eliminated_coda_calls": len(open_run) - backfilled,
            }
        )
        open_run = []

    for row in rows:
        scorer_calls += 1
        if float(row["gate_score"]) >= threshold:
            deferred_calls += 1
            open_run.append(row)
            if bool(row["exact_safe"]):
                # The requested replay assumes this cannot happen. Retain the
                # violation so K agreement is not overstated at another q/data.
                assumption_violations += 1
            continue

        if open_run:
            close_run(followed_by_low=True, trace_end=False)
        eligible_exact_calls += 1
        if bool(row["exact_safe"]):
            policy_k = int(row["terminal_iteration"])
            break

    if open_run:
        close_run(followed_by_low=False, trace_end=True)

    if policy_k is None and not rows:
        # Cold-origin or pre-eligibility baseline predictions never apply the
        # diagnostic filter and therefore remain exactly unchanged.
        policy_k = baseline_k
    elif policy_k is None and assumption_violations == 0:
        # No adjacent exact confirmation stopped the policy. The recorded trace
        # then ends at the same recurrence maximum as baseline.
        _require(
            not bool(prediction["baseline_adaptive_stop"]),
            "adaptive baseline ended without a reproduced exact-safe confirmation",
        )
        policy_k = baseline_k

    actual_calls = preeligible_calls + eligible_exact_calls
    eliminated_calls = baseline_calls - actual_calls
    _require(
        eliminated_calls == deferred_calls - backfilled_calls,
        "deferred/backfill Coda accounting mismatch",
    )
    k_agrees = bool(policy_k == baseline_k and assumption_violations == 0)
    return {
        "baseline_k": baseline_k,
        "policy_k": policy_k,
        "baseline_coda_calls": baseline_calls,
        "actual_coda_calls": actual_calls,
        "scorer_calls": scorer_calls,
        "deferred_coda_calls": deferred_calls,
        "deferred_calls_later_backfilled": backfilled_calls,
        "truly_eliminated_coda_calls": eliminated_calls,
        "high_score_exact_safe_assumption_violations": assumption_violations,
        "baseline_k_agrees": k_agrees,
        "runs": runs,
    }


def _empty_counts() -> dict[str, int]:
    return {
        "prediction_count": 0,
        "baseline_coda_calls": 0,
        "actual_coda_calls": 0,
        "scorer_calls": 0,
        "deferred_coda_calls": 0,
        "deferred_calls_later_backfilled": 0,
        "truly_eliminated_coda_calls": 0,
        "high_score_exact_safe_assumption_violations": 0,
        "baseline_k_agreement_count": 0,
        "predictions_with_terminal_mismatch": 0,
    }


def _finalize(counts: Mapping[str, int]) -> dict[str, Any]:
    result = dict(counts)
    baseline_calls = int(result["baseline_coda_calls"])
    deferred = int(result["deferred_coda_calls"])
    eliminated = int(result["truly_eliminated_coda_calls"])
    scorer_calls = int(result["scorer_calls"])
    predictions = int(result["prediction_count"])
    scorer_cost = scorer_calls * SCORER_COST_MS
    coda_savings = eliminated * CODA_COST_MS
    result.update(
        {
            "backfill_rate": _safe_ratio(
                int(result["deferred_calls_later_backfilled"]), deferred
            ),
            "coda_reduction": _safe_ratio(eliminated, baseline_calls),
            "estimated_scorer_cost_ms": scorer_cost,
            "estimated_coda_savings_ms": coda_savings,
            "estimated_net_latency_saving_ms": coda_savings - scorer_cost,
            "estimated_net_latency_positive": bool(coda_savings > scorer_cost),
            "baseline_k_agreement_rate": _safe_ratio(
                int(result["baseline_k_agreement_count"]), predictions
            ),
        }
    )
    return result


def _run_distribution(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    lengths = Counter(int(run["length"]) for run in runs)
    followed = Counter(
        int(run["length"])
        for run in runs
        if bool(run["followed_by_low_score"])
    )
    ended = Counter(
        int(run["length"])
        for run in runs
        if bool(run["ends_at_baseline_trace"])
    )
    return {
        "region_count": len(runs),
        "high_score_row_count": sum(int(run["length"]) for run in runs),
        "single_iteration_region_count": lengths.get(1, 0),
        "single_iteration_regions_followed_by_low_score": followed.get(1, 0),
        "single_iteration_regions_with_zero_saving_after_required_backfill": sum(
            int(run["length"] == 1 and run["backfilled_coda_calls"] == 1)
            for run in runs
        ),
        "by_length": {
            str(length): {
                "region_count": count,
                "high_score_rows": length * count,
                "followed_by_low_score": followed.get(length, 0),
                "ends_at_baseline_trace": ended.get(length, 0),
                "backfilled_coda_calls": sum(
                    int(run["backfilled_coda_calls"])
                    for run in runs
                    if int(run["length"]) == length
                ),
                "eliminated_coda_calls": sum(
                    int(run["eliminated_coda_calls"])
                    for run in runs
                    if int(run["length"]) == length
                ),
            }
            for length, count in sorted(lengths.items())
        },
    }


def replay_dataset(
    compact: Sequence[Mapping[str, Any]],
    *,
    threshold: float = DIAGNOSTIC_Q,
    require_exact_terminal_output: bool = False,
) -> dict[str, Any]:
    global_counts = _empty_counts()
    task_counts = {task_id: _empty_counts() for task_id in TASK_IDS}
    global_runs: list[dict[str, Any]] = []
    task_runs: dict[int, list[dict[str, Any]]] = defaultdict(list)
    mismatch_predictions: list[dict[str, Any]] = []
    for prediction in compact:
        replay = replay_prediction(
            prediction,
            threshold=threshold,
            require_exact_terminal_output=require_exact_terminal_output,
        )
        task_id = int(prediction["task_id"])
        for counts in (global_counts, task_counts[task_id]):
            counts["prediction_count"] += 1
            for name in (
                "baseline_coda_calls",
                "actual_coda_calls",
                "scorer_calls",
                "deferred_coda_calls",
                "deferred_calls_later_backfilled",
                "truly_eliminated_coda_calls",
                "high_score_exact_safe_assumption_violations",
            ):
                counts[name] += int(replay[name])
            counts["baseline_k_agreement_count"] += int(replay["baseline_k_agrees"])
            counts["predictions_with_terminal_mismatch"] += int(
                not replay["baseline_k_agrees"]
            )
        for run in replay["runs"]:
            record = {**run, "task_id": task_id}
            global_runs.append(record)
            task_runs[task_id].append(record)
        if not replay["baseline_k_agrees"]:
            mismatch_predictions.append(
                {
                    "task_id": task_id,
                    "trajectory_id": prediction["trajectory_id"],
                    "prediction_id": prediction["prediction_id"],
                    "baseline_k": replay["baseline_k"],
                    "policy_k": replay["policy_k"],
                    "assumption_violations": replay[
                        "high_score_exact_safe_assumption_violations"
                    ],
                }
            )
    return {
        "threshold": threshold,
        "require_exact_terminal_output": require_exact_terminal_output,
        "global": _finalize(global_counts),
        "by_task": {
            str(task_id): {
                **_finalize(task_counts[task_id]),
                "high_score_run_length_distribution": _run_distribution(
                    task_runs[task_id]
                ),
            }
            for task_id in TASK_IDS
        },
        "high_score_run_length_distribution": _run_distribution(global_runs),
        "terminal_mismatch_predictions": mismatch_predictions,
    }


def analyze(
    manifest: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    *,
    manifest_path: Path,
) -> dict[str, Any]:
    _require(manifest.get("complete") is True, "shadow manifest is incomplete")
    _require(tuple(manifest["expected_task_ids"]) == TASK_IDS, "task partition mismatch")
    compact, dataset = compact_predictions(predictions)
    _require(dataset["trajectory_count"] == EXPECTED_TRAJECTORIES, "trajectory count mismatch")
    _require(dataset["prediction_count"] == EXPECTED_PREDICTIONS, "prediction count mismatch")
    _require(dataset["eligible_row_count"] == EXPECTED_ROWS, "eligible-row count mismatch")

    primary = replay_dataset(compact)
    exact_terminal_variant = replay_dataset(
        compact, require_exact_terminal_output=True
    )

    # Reuse the already-tested causal max_skip=1 replay for a direct comparison.
    from scripts.coda_anchor_feasibility.analyze_action_delta_nonconvergence_filter import (
        causal_replay,
        prepare_replay_data,
    )

    current_compact, _, current_dataset = prepare_replay_data(predictions)
    _require(
        current_dataset["baseline_coda_calls"] == dataset["baseline_coda_calls"],
        "baseline Coda accounting differs from existing replay",
    )
    current = causal_replay(current_compact, DIAGNOSTIC_Q)
    current_global = current["global"]
    return {
        "analysis_type": "action_delta_adjacent_history_deferred_backfill",
        "diagnostic_only": True,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256_file(manifest_path),
        "source_dataset": {
            **dataset,
            "task_ids": list(TASK_IDS),
        },
        "frozen_predictor": {
            **dict(manifest["artifact_identity"]),
            "diagnostic_high_side_threshold": DIAGNOSTIC_Q,
            "retrained": False,
        },
        "cost_anchors": {
            "scorer_cost_ms": SCORER_COST_MS,
            "coda_cost_ms": CODA_COST_MS,
        },
        "policy": {
            "minimum_terminal_iteration": MIN_TERMINAL_ITER,
            "predictor_scored_on_adjacent_latent_states": True,
            "high_score_assumed_exact_nonconverged": True,
            "low_score_after_defer_executes_adjacent_backfill_pair": True,
            "only_adjacent_exact_action_mse_controls_stopping": True,
        },
        "deferred_backfill": primary,
        "exact_terminal_output_variant": exact_terminal_variant,
        "current_max_skip_1_comparison": {
            "baseline_coda_calls": int(current_global["baseline_coda_calls"]),
            "nominal_actual_coda_calls": int(
                current_global["baseline_coda_calls"]
                - current_global["nominal_skipped_coda_calls"]
            ),
            "nominal_coda_calls_saved": int(
                current_global["nominal_skipped_coda_calls"]
            ),
            "scorer_calls": int(current_global["scorer_call_count"]),
            "adjacent_history_difference_events": int(
                current_global["adjacent_history_difference_events"]
            ),
            "altered_history_stop_decision_changes": int(
                current_global["altered_history_stop_decision_changes"]
            ),
            "predictions_censored_after_history_change": int(
                current_global["censored_predictions"]
            ),
            "estimated_scorer_cost_ms": float(current_global["scorer_cost_ms"]),
            "estimated_coda_savings_ms": float(
                current_global["gross_coda_time_saved_ms"]
            ),
            "estimated_net_latency_saving_ms": float(
                current_global["nominal_net_time_saved_ms"]
            ),
            "warning": (
                "Nominal observed-prefix accounting: forced exact Coda compares "
                "against a non-adjacent last-executed anchor, so baseline K is not preserved."
            ),
            "by_task": {
                str(task_id): {
                    "baseline_coda_calls": int(
                        current["by_task"][str(task_id)]["baseline_coda_calls"]
                    ),
                    "nominal_coda_calls_saved": int(
                        current["by_task"][str(task_id)][
                            "nominal_skipped_coda_calls"
                        ]
                    ),
                    "scorer_calls": int(
                        current["by_task"][str(task_id)]["scorer_call_count"]
                    ),
                    "adjacent_history_difference_events": int(
                        current["by_task"][str(task_id)][
                            "adjacent_history_difference_events"
                        ]
                    ),
                    "altered_history_stop_decision_changes": int(
                        current["by_task"][str(task_id)][
                            "altered_history_stop_decision_changes"
                        ]
                    ),
                    "predictions_censored_after_history_change": int(
                        current["by_task"][str(task_id)][
                            "censored_predictions"
                        ]
                    ),
                }
                for task_id in TASK_IDS
            },
        },
        "limitations": [
            "This is an offline replay on development Tasks 0,1,2,3,6,7,8,9 only.",
            "All q=0.0015 high-score rows are verified exact-nonconverged in this dataset; this is not a guarantee on new trajectories.",
            "The primary policy preserves exact adjacent stopping semantics but does not define an exact returned action when a high-score run reaches max iteration.",
            "The exact-terminal-output variant separately counts one final Coda for such terminal runs.",
            "Latency is estimated from established scorer/Coda cost anchors; LIBERO was not run.",
        ],
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = tuple(rows[0]) if rows else ()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(results: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    per_task = []
    for task_id in TASK_IDS:
        summary = results["deferred_backfill"]["by_task"][str(task_id)]
        per_task.append(
            {
                "task_id": task_id,
                **{
                    key: value
                    for key, value in summary.items()
                    if key != "high_score_run_length_distribution"
                },
            }
        )
    _write_csv(output_dir / "per_task_summary.csv", per_task)
    distribution = results["deferred_backfill"][
        "high_score_run_length_distribution"
    ]["by_length"]
    _write_csv(
        output_dir / "run_length_distribution.csv",
        [
            {"run_length": int(length), **values}
            for length, values in distribution.items()
        ],
    )


def print_summary(results: Mapping[str, Any], output_dir: Path) -> None:
    primary = results["deferred_backfill"]["global"]
    print("Action-Delta deferred/backfill adjacent-history replay")
    print(
        f"predictions={primary['prediction_count']} baseline_coda={primary['baseline_coda_calls']} "
        f"actual_coda={primary['actual_coda_calls']} eliminated={primary['truly_eliminated_coda_calls']}"
    )
    print(
        f"deferred={primary['deferred_coda_calls']} backfilled={primary['deferred_calls_later_backfilled']} "
        f"backfill_rate={primary['backfill_rate']:.6f} coda_reduction={primary['coda_reduction']:.6f}"
    )
    print(
        f"scorer_calls={primary['scorer_calls']} scorer_ms={primary['estimated_scorer_cost_ms']:.3f} "
        f"coda_saved_ms={primary['estimated_coda_savings_ms']:.3f} "
        f"net_ms={primary['estimated_net_latency_saving_ms']:.3f}"
    )
    print(
        f"baseline_K_agreement={primary['baseline_k_agreement_count']}/{primary['prediction_count']} "
        f"mismatches={primary['predictions_with_terminal_mismatch']}"
    )
    print("task  predictions  baseline  actual  eliminated  backfilled  reduction  K-match")
    for task_id in TASK_IDS:
        row = results["deferred_backfill"]["by_task"][str(task_id)]
        print(
            f"{task_id:>4}  {row['prediction_count']:>11}  {row['baseline_coda_calls']:>8}  "
            f"{row['actual_coda_calls']:>6}  {row['truly_eliminated_coda_calls']:>10}  "
            f"{row['deferred_calls_later_backfilled']:>10}  {row['coda_reduction']:.5f}  "
            f"{row['baseline_k_agreement_count']}/{row['prediction_count']}"
        )
    print(
        "Run lengths: "
        + json.dumps(
            results["deferred_backfill"]["high_score_run_length_distribution"][
                "by_length"
            ],
            sort_keys=True,
        )
    )
    for name in ("results.json", "per_task_summary.csv", "run_length_distribution.csv"):
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
        else manifest_path.parent.parent / "deferred_backfill_analysis"
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
