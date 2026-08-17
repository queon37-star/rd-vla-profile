"""Sweep deferred/backfill cadence on a full deployment-matched shadow trace."""

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
REPLAY_MIN_TERMINAL_ITERS = (2, 3, 4, 5)
DIAGNOSTIC_Q = 0.0015
EXACT_CONVERGENCE_MSE = 0.001
SCORER_COST_MS = 0.36627289134678936
CODA_COST_MS = 1.864207385085098
EXPECTED_ARTIFACT_SHA256 = (
    "b4f9e938c72108f164b9d997a86eb4f5d9ea15b146754b9af83f9735e1ecfcf8"
)
DEFAULT_MANIFEST = Path(
    "benchmark_results/coda_anchor_feasibility/deployment_matched_shadow/"
    "phaseA_terminal2_8tasks/shards/manifest.json"
)


class DeferredBackfillAnalysisError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DeferredBackfillAnalysisError(message)


def _safe_ratio(
    numerator: int | float, denominator: int | float
) -> float | None:
    return float(numerator / denominator) if denominator else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_min_terminal_iteration(manifest: Mapping[str, Any]) -> int:
    value = manifest.get("min_terminal_iteration")
    configuration = manifest.get("configuration") or {}
    gate = configuration.get("gate") if isinstance(configuration, Mapping) else None
    configured = gate.get("min_terminal_iteration") if isinstance(gate, Mapping) else None
    if configured is None and isinstance(configuration, Mapping):
        configured = configuration.get("gate_min_terminal_iteration")
    if value is None:
        value = configured
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 2,
        "source manifest does not declare a valid minimum terminal iteration",
    )
    if configured is not None:
        _require(value == configured, "source manifest minimum/configuration mismatch")
    return int(value)


def compact_predictions(
    predictions: Sequence[Mapping[str, Any]],
    *,
    source_min_terminal_iter: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate source identity and retain only replay-relevant scalars."""

    compact: list[dict[str, Any]] = []
    trajectory_ids: set[str] = set()
    source_minima: set[int] = set()
    row_count = 0
    baseline_calls = 0
    for prediction in predictions:
        identity = prediction["identity"]
        production = prediction["production_parity"]
        task_id = int(identity["task_id"])
        _require(task_id in TASK_IDS, f"forbidden task in shadow data: {task_id}")
        trajectory_id = str(identity["trajectory_id"])
        trajectory_ids.add(trajectory_id)
        baseline_k = int(production["K_t"])
        exact_calls = int(production["exact_coda_call_count"])
        _require(exact_calls == baseline_k, "baseline exact-Coda count must equal K")
        minimum = int(prediction["min_terminal_iteration"])
        _require(minimum >= 2, "source minimum terminal iteration must be >= 2")
        source_minima.add(minimum)
        if source_min_terminal_iter is not None:
            _require(
                minimum == source_min_terminal_iter,
                "prediction minimum differs from source manifest",
            )

        source_rows = sorted(
            prediction.get("transitions", []),
            key=lambda row: int(row["terminal_iteration"]),
        )
        collection_applied = bool(prediction.get("collection_applied"))
        observed_terminals = [int(row["terminal_iteration"]) for row in source_rows]
        expected_terminals = (
            list(range(minimum, baseline_k + 1)) if collection_applied else []
        )
        _require(
            observed_terminals == expected_terminals,
            "source transition trace is not contiguous from configured minimum "
            f"through K: prediction={prediction['prediction_id']}, "
            f"expected={expected_terminals}, actual={observed_terminals}",
        )

        rows: list[dict[str, Any]] = []
        for source in source_rows:
            terminal = int(source["terminal_iteration"])
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

        row_count += len(rows)
        baseline_calls += exact_calls
        compact.append(
            {
                "task_id": task_id,
                "prediction_id": str(prediction["prediction_id"]),
                "trajectory_id": trajectory_id,
                "baseline_k": baseline_k,
                "baseline_coda_calls": exact_calls,
                "baseline_stop_reason": str(production["stop_reason"]),
                "baseline_adaptive_stop": bool(production["adaptive_stop"]),
                "collection_applied": collection_applied,
                "source_min_terminal_iteration": minimum,
                "rows": rows,
            }
        )

    _require(len(source_minima) <= 1, "mixed source minimum terminal iterations")
    observed_minimum = next(iter(source_minima), source_min_terminal_iter)
    return compact, {
        "trajectory_count": len(trajectory_ids),
        "prediction_count": len(compact),
        "source_min_terminal_iteration": observed_minimum,
        "eligible_row_count": row_count,
        "baseline_coda_calls": baseline_calls,
    }


def slice_prediction(
    prediction: Mapping[str, Any], min_terminal_iter: int
) -> dict[str, Any]:
    _require(
        isinstance(min_terminal_iter, int)
        and not isinstance(min_terminal_iter, bool)
        and min_terminal_iter >= 2,
        "replay minimum terminal iteration must be an integer >= 2",
    )
    source_minimum = int(
        prediction.get("source_min_terminal_iteration", min_terminal_iter)
    )
    _require(
        min_terminal_iter >= source_minimum,
        "replay minimum precedes the source collection minimum",
    )
    sliced = dict(prediction)
    sliced["rows"] = [
        dict(row)
        for row in prediction.get("rows", [])
        if int(row["terminal_iteration"]) >= min_terminal_iter
    ]
    sliced["replay_min_terminal_iteration"] = int(min_terminal_iter)
    return sliced


def replay_prediction(
    prediction: Mapping[str, Any],
    *,
    threshold: float = DIAGNOSTIC_Q,
    min_terminal_iter: int | None = None,
    require_exact_terminal_output: bool = True,
) -> dict[str, Any]:
    """Replay one trace with exact adjacent confirmation after every high run."""

    if min_terminal_iter is not None:
        prediction = slice_prediction(prediction, min_terminal_iter)
    rows = list(prediction.get("rows", []))
    baseline_k = int(prediction["baseline_k"])
    baseline_calls = int(prediction["baseline_coda_calls"])
    preeligible_calls = baseline_calls - len(rows)
    _require(preeligible_calls >= 0, "negative pre-eligibility Coda count")

    scorer_calls = 0
    eligible_exact_calls = 0
    deferred_calls = 0
    backfilled_calls = 0
    terminal_exact_fallback_calls = 0
    policy_k: int | None = None
    violations: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    open_run: list[dict[str, Any]] = []

    def close_run(*, followed_by_low: bool, terminal_fallback: bool) -> None:
        nonlocal eligible_exact_calls, backfilled_calls
        nonlocal terminal_exact_fallback_calls, open_run
        if not open_run:
            return
        backfilled = int(followed_by_low)
        exact_terminal = int(terminal_fallback)
        recovered = backfilled + exact_terminal
        eligible_exact_calls += recovered
        backfilled_calls += backfilled
        terminal_exact_fallback_calls += exact_terminal
        runs.append(
            {
                "length": len(open_run),
                "start_terminal_iteration": int(open_run[0]["terminal_iteration"]),
                "end_terminal_iteration": int(open_run[-1]["terminal_iteration"]),
                "followed_by_low_score": bool(followed_by_low),
                "ends_at_baseline_trace": bool(terminal_fallback),
                "backfilled_coda_calls": backfilled,
                "terminal_exact_fallback_coda_calls": exact_terminal,
                "recovered_deferred_coda_calls": recovered,
                "eliminated_coda_calls": len(open_run) - recovered,
            }
        )
        open_run = []

    for row in rows:
        scorer_calls += 1
        if float(row["gate_score"]) >= threshold:
            deferred_calls += 1
            open_run.append(row)
            if bool(row["exact_safe"]):
                violations.append(
                    {
                        "terminal_iteration": int(row["terminal_iteration"]),
                        "gate_score": float(row["gate_score"]),
                        "exact_adjacent_action_mse": float(
                            row["exact_adjacent_action_mse"]
                        ),
                    }
                )
            continue

        if open_run:
            close_run(followed_by_low=True, terminal_fallback=False)
        eligible_exact_calls += 1
        if bool(row["exact_safe"]):
            policy_k = int(row["terminal_iteration"])
            break

    if open_run:
        _require(
            require_exact_terminal_output,
            "replay ended with an unresolved deferred terminal action",
        )
        close_run(followed_by_low=False, terminal_fallback=True)

    if policy_k is None and not violations:
        if rows and bool(prediction["baseline_adaptive_stop"]):
            _require(
                bool(rows[-1]["exact_safe"]),
                "adaptive baseline trace does not end at an exact-safe row",
            )
        policy_k = baseline_k

    actual_calls = preeligible_calls + eligible_exact_calls
    eliminated_calls = baseline_calls - actual_calls
    recovered_calls = backfilled_calls + terminal_exact_fallback_calls
    _require(
        eliminated_calls == deferred_calls - recovered_calls,
        "deferred/backfill Coda accounting mismatch",
    )
    k_agrees = bool(policy_k == baseline_k and not violations)
    return {
        "baseline_k": baseline_k,
        "policy_k": policy_k,
        "baseline_coda_calls": baseline_calls,
        "actual_coda_calls": actual_calls,
        "eligible_row_count": len(rows),
        "scorer_calls": scorer_calls,
        "deferred_coda_calls": deferred_calls,
        "backfilled_coda_calls": backfilled_calls,
        "deferred_calls_later_backfilled": backfilled_calls,
        "terminal_exact_fallback_coda_calls": terminal_exact_fallback_calls,
        "recovered_deferred_coda_calls": recovered_calls,
        "truly_eliminated_coda_calls": eliminated_calls,
        "high_score_exact_safe_assumption_violations": len(violations),
        "violation_rows": violations,
        "baseline_k_agrees": k_agrees,
        "runs": runs,
    }


def _empty_counts() -> dict[str, int]:
    return {
        "prediction_count": 0,
        "eligible_row_count": 0,
        "baseline_coda_calls": 0,
        "actual_coda_calls": 0,
        "scorer_calls": 0,
        "deferred_coda_calls": 0,
        "backfilled_coda_calls": 0,
        "terminal_exact_fallback_coda_calls": 0,
        "recovered_deferred_coda_calls": 0,
        "truly_eliminated_coda_calls": 0,
        "high_score_exact_safe_assumption_violations": 0,
        "baseline_k_agreement_count": 0,
        "predictions_with_terminal_mismatch": 0,
    }


def _finalize(
    counts: Mapping[str, int],
    *,
    trajectory_count: int,
    deferred_run_count: int,
    single_length_run_count: int,
) -> dict[str, Any]:
    result = dict(counts)
    baseline_calls = int(result["baseline_coda_calls"])
    deferred = int(result["deferred_coda_calls"])
    eliminated = int(result["truly_eliminated_coda_calls"])
    scorer_calls = int(result["scorer_calls"])
    predictions = int(result["prediction_count"])
    scorer_cost = scorer_calls * SCORER_COST_MS
    coda_savings = eliminated * CODA_COST_MS
    cost_ratio = _safe_ratio(eliminated, scorer_calls)
    result.update(
        {
            "trajectory_count": int(trajectory_count),
            "deferred_run_count": int(deferred_run_count),
            "single_length_run_count": int(single_length_run_count),
            "backfill_rate": _safe_ratio(
                int(result["backfilled_coda_calls"]), deferred
            ),
            "recovered_deferred_rate": _safe_ratio(
                int(result["recovered_deferred_coda_calls"]), deferred
            ),
            "coda_reduction": _safe_ratio(eliminated, baseline_calls),
            "coda_reduction_percent": (
                100.0 * eliminated / baseline_calls if baseline_calls else None
            ),
            "violation_rate_among_high_score_rows": _safe_ratio(
                int(result["high_score_exact_safe_assumption_violations"]),
                deferred,
            ),
            "baseline_k_agreement_rate": _safe_ratio(
                int(result["baseline_k_agreement_count"]), predictions
            ),
            "eliminated_coda_per_score": cost_ratio,
            "break_even_scorer_to_coda_cost_ratio": cost_ratio,
            "break_even_scorer_cost_ms_per_call": (
                cost_ratio * CODA_COST_MS if cost_ratio is not None else None
            ),
            "historical_estimated_scorer_cost_ms": scorer_cost,
            "historical_estimated_coda_savings_ms": coda_savings,
            "historical_estimated_net_latency_saving_ms": coda_savings
            - scorer_cost,
            "historical_estimated_net_latency_positive": bool(
                coda_savings > scorer_cost
            ),
        }
    )
    return result


def _run_distribution(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    lengths = Counter(int(run["length"]) for run in runs)
    return {
        "run_count": len(runs),
        "high_score_row_count": sum(int(run["length"]) for run in runs),
        "single_length_run_count": lengths.get(1, 0),
        "by_length": {
            str(length): {
                "run_count": count,
                "high_score_rows": length * count,
                "backfilled_coda_calls": sum(
                    int(run["backfilled_coda_calls"])
                    for run in runs
                    if int(run["length"]) == length
                ),
                "terminal_exact_fallback_coda_calls": sum(
                    int(run["terminal_exact_fallback_coda_calls"])
                    for run in runs
                    if int(run["length"]) == length
                ),
                "truly_eliminated_coda_calls": sum(
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
    min_terminal_iter: int,
    threshold: float = DIAGNOSTIC_Q,
) -> dict[str, Any]:
    global_counts = _empty_counts()
    task_counts = {task_id: _empty_counts() for task_id in TASK_IDS}
    global_runs: list[dict[str, Any]] = []
    task_runs: dict[int, list[dict[str, Any]]] = defaultdict(list)
    global_trajectories: set[str] = set()
    task_trajectories: dict[int, set[str]] = defaultdict(set)
    violations: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []

    count_fields = tuple(
        name for name in _empty_counts() if name != "prediction_count"
    )
    for source_prediction in compact:
        prediction = slice_prediction(source_prediction, min_terminal_iter)
        replay = replay_prediction(prediction, threshold=threshold)
        task_id = int(prediction["task_id"])
        trajectory_id = str(prediction["trajectory_id"])
        global_trajectories.add(trajectory_id)
        task_trajectories[task_id].add(trajectory_id)
        for counts in (global_counts, task_counts[task_id]):
            counts["prediction_count"] += 1
            for name in count_fields:
                if name == "baseline_k_agreement_count":
                    counts[name] += int(replay["baseline_k_agrees"])
                elif name == "predictions_with_terminal_mismatch":
                    counts[name] += int(not replay["baseline_k_agrees"])
                else:
                    counts[name] += int(replay[name])

        for run in replay["runs"]:
            record = {
                **run,
                "task_id": task_id,
                "trajectory_id": trajectory_id,
                "prediction_id": prediction["prediction_id"],
            }
            global_runs.append(record)
            task_runs[task_id].append(record)
        for violation in replay["violation_rows"]:
            violations.append(
                {
                    "task_id": task_id,
                    "trajectory_id": trajectory_id,
                    "prediction_id": prediction["prediction_id"],
                    **violation,
                }
            )
        if not replay["baseline_k_agrees"]:
            mismatches.append(
                {
                    "task_id": task_id,
                    "trajectory_id": trajectory_id,
                    "prediction_id": prediction["prediction_id"],
                    "baseline_k": replay["baseline_k"],
                    "policy_k": replay["policy_k"],
                    "violation_count": replay[
                        "high_score_exact_safe_assumption_violations"
                    ],
                }
            )

    global_distribution = _run_distribution(global_runs)
    task_distributions = {
        task_id: _run_distribution(task_runs[task_id]) for task_id in TASK_IDS
    }
    return {
        "min_terminal_iteration": int(min_terminal_iter),
        "threshold": float(threshold),
        "global": _finalize(
            global_counts,
            trajectory_count=len(global_trajectories),
            deferred_run_count=global_distribution["run_count"],
            single_length_run_count=global_distribution[
                "single_length_run_count"
            ],
        ),
        "by_task": {
            str(task_id): {
                **_finalize(
                    task_counts[task_id],
                    trajectory_count=len(task_trajectories[task_id]),
                    deferred_run_count=task_distributions[task_id]["run_count"],
                    single_length_run_count=task_distributions[task_id][
                        "single_length_run_count"
                    ],
                ),
                "run_length_distribution": task_distributions[task_id],
            }
            for task_id in TASK_IDS
        },
        "run_length_distribution": global_distribution,
        "high_score_exact_safe_violation_rows": violations,
        "terminal_mismatch_predictions": mismatches,
    }


def _margin(new: Mapping[str, Any], old: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "added_scorer_calls": int(new["scorer_calls"] - old["scorer_calls"]),
        "added_deferred_rows": int(
            new["deferred_coda_calls"] - old["deferred_coda_calls"]
        ),
        "added_truly_eliminated_coda_calls": int(
            new["truly_eliminated_coda_calls"]
            - old["truly_eliminated_coda_calls"]
        ),
        "added_safety_violations": int(
            new["high_score_exact_safe_assumption_violations"]
            - old["high_score_exact_safe_assumption_violations"]
        ),
        "coda_reduction_percentage_point_change": float(
            (new["coda_reduction_percent"] or 0.0)
            - (old["coda_reduction_percent"] or 0.0)
        ),
        "historical_estimated_net_latency_change_ms": float(
            new["historical_estimated_net_latency_saving_ms"]
            - old["historical_estimated_net_latency_saving_ms"]
        ),
    }


def _marginal_effects(cadences: Mapping[str, Any]) -> dict[str, Any]:
    effects: dict[str, Any] = {}
    for old_min, new_min in ((5, 4), (4, 3), (3, 2)):
        old = cadences[str(old_min)]
        new = cadences[str(new_min)]
        effects[f"min{old_min}_to_min{new_min}"] = {
            "from_min_terminal_iteration": old_min,
            "to_min_terminal_iteration": new_min,
            "global": _margin(new["global"], old["global"]),
            "by_task": {
                str(task_id): _margin(
                    new["by_task"][str(task_id)],
                    old["by_task"][str(task_id)],
                )
                for task_id in TASK_IDS
            },
        }
    return effects


def analyze(
    manifest: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    *,
    manifest_path: Path,
    min_terminal_iters: Sequence[int] = REPLAY_MIN_TERMINAL_ITERS,
) -> dict[str, Any]:
    _require(manifest.get("complete") is True, "shadow manifest is incomplete")
    _require(
        tuple(int(value) for value in manifest["expected_task_ids"]) == TASK_IDS,
        "task partition mismatch",
    )
    _require(
        manifest.get("artifact_identity", {}).get("sha256")
        == EXPECTED_ARTIFACT_SHA256,
        "frozen Action-Delta artifact identity mismatch",
    )
    source_minimum = _manifest_min_terminal_iteration(manifest)
    requested = tuple(int(value) for value in min_terminal_iters)
    _require(bool(requested), "at least one replay minimum is required")
    _require(
        all(value >= source_minimum for value in requested),
        "requested replay minimum precedes source collection minimum",
    )
    compact, dataset = compact_predictions(
        predictions, source_min_terminal_iter=source_minimum
    )
    _require(
        dataset["prediction_count"] == int(manifest["prediction_count"]),
        "prediction count differs from source manifest identity",
    )
    _require(
        dataset["eligible_row_count"] == int(manifest["transition_count"]),
        "transition count differs from source manifest identity",
    )
    aggregate = manifest.get("summary", {}).get("aggregate", {})
    _require(
        dataset["trajectory_count"] == int(aggregate.get("trajectories", -1)),
        "trajectory count differs from source manifest summary",
    )

    cadences = {
        str(minimum): replay_dataset(
            compact,
            min_terminal_iter=minimum,
            threshold=DIAGNOSTIC_Q,
        )
        for minimum in requested
    }
    min5 = cadences.get("5")
    comparisons = {}
    for minimum in requested:
        summary = cadences[str(minimum)]["global"]
        baseline_reduction = (
            min5["global"]["coda_reduction_percent"] if min5 else None
        )
        absolute_gain = (
            summary["coda_reduction_percent"] - baseline_reduction
            if baseline_reduction is not None
            else None
        )
        relative_gain = (
            absolute_gain / baseline_reduction
            if absolute_gain is not None and baseline_reduction
            else None
        )
        comparisons[str(minimum)] = {
            "zero_high_score_exact_safe_violations": bool(
                summary["high_score_exact_safe_assumption_violations"] == 0
            ),
            "full_baseline_k_agreement": bool(
                summary["baseline_k_agreement_count"]
                == summary["prediction_count"]
            ),
            "coda_reduction_percentage_point_gain_vs_min5": absolute_gain,
            "coda_reduction_relative_gain_vs_min5": relative_gain,
            "materially_greater_coda_reduction_than_min5": (
                bool(
                    absolute_gain >= 1.0
                    and relative_gain is not None
                    and relative_gain >= 0.10
                )
                if minimum != 5 and absolute_gain is not None
                else False
            ),
            "historical_estimated_action_head_net_positive": bool(
                summary["historical_estimated_net_latency_positive"]
            ),
        }

    return {
        "analysis_type": "action_delta_deferred_backfill_minimum_sweep",
        "diagnostic_only": True,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256_file(manifest_path),
        "source_dataset": {
            **dataset,
            "task_ids": list(TASK_IDS),
            "manifest_dataset_identity_sha256": manifest.get(
                "dataset_identity_sha256"
            ),
        },
        "frozen_predictor": {
            **dict(manifest["artifact_identity"]),
            "diagnostic_high_side_threshold": DIAGNOSTIC_Q,
            "retrained": False,
        },
        "exact_convergence_mse_threshold": EXACT_CONVERGENCE_MSE,
        "historical_cost_anchors": {
            "scorer_cost_ms": SCORER_COST_MS,
            "coda_cost_ms": CODA_COST_MS,
        },
        "break_even_definition": {
            "anchor_independent_required_scorer_to_coda_cost_ratio": (
                "truly_eliminated_coda_calls / scorer_calls"
            ),
            "break_even_scorer_cost_ms_per_call": (
                "ratio * historical_coda_cost_ms"
            ),
        },
        "policy": {
            "predictor_scored_on_adjacent_latent_states": True,
            "high_score_rule": "gate_score >= 0.0015",
            "low_after_run_backfills_only_immediately_previous_state": True,
            "only_adjacent_exact_action_mse_controls_stopping": True,
            "exact_terminal_action_required": True,
        },
        "cadences": cadences,
        "marginal_effects": (
            _marginal_effects(cadences)
            if all(str(value) in cadences for value in REPLAY_MIN_TERMINAL_ITERS)
            else {}
        ),
        "development_checks": comparisons,
        "material_reduction_definition": (
            ">=1.0 absolute percentage point and >=10% relative gain vs min=5; "
            "descriptive only, not policy selection"
        ),
        "limitations": [
            "Development Tasks 0,1,2,3,6,7,8,9 only; Tasks 4/5 remain held out.",
            "q=0.0015 is frozen; no threshold or model was fitted.",
            "A high-score exact-safe row is surfaced as a violation and K mismatch.",
            "Latency uses historical fixed anchors; LIBERO was not run.",
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
    distributions = []
    for cadence, replay in results["cadences"].items():
        for task_id in TASK_IDS:
            summary = replay["by_task"][str(task_id)]
            per_task.append(
                {
                    "min_terminal_iteration": int(cadence),
                    "task_id": task_id,
                    **{
                        key: value
                        for key, value in summary.items()
                        if key != "run_length_distribution"
                    },
                }
            )
        for length, values in replay["run_length_distribution"]["by_length"].items():
            distributions.append(
                {
                    "min_terminal_iteration": int(cadence),
                    "scope": "global",
                    "task_id": "",
                    "run_length": int(length),
                    **values,
                }
            )
            for task_id in TASK_IDS:
                task_values = replay["by_task"][str(task_id)][
                    "run_length_distribution"
                ]["by_length"].get(length)
                if task_values is not None:
                    distributions.append(
                        {
                            "min_terminal_iteration": int(cadence),
                            "scope": "task",
                            "task_id": task_id,
                            "run_length": int(length),
                            **task_values,
                        }
                    )
    margins = []
    for name, effect in results["marginal_effects"].items():
        margins.append(
            {
                "comparison": name,
                "scope": "global",
                "task_id": "",
                **effect["global"],
            }
        )
        for task_id in TASK_IDS:
            margins.append(
                {
                    "comparison": name,
                    "scope": "task",
                    "task_id": task_id,
                    **effect["by_task"][str(task_id)],
                }
            )
    _write_csv(output_dir / "per_task_summary.csv", per_task)
    _write_csv(output_dir / "run_length_distribution.csv", distributions)
    _write_csv(output_dir / "marginal_effects.csv", margins)


def print_summary(results: Mapping[str, Any], output_dir: Path) -> None:
    print("Action-Delta deferred/backfill minimum-terminal sweep")
    print(
        "min  scores  deferred  runs  backfill  eliminated  reduction  "
        "violations  K-match  net-ms  break-even-score-ms"
    )
    for minimum in REPLAY_MIN_TERMINAL_ITERS:
        if str(minimum) not in results["cadences"]:
            continue
        row = results["cadences"][str(minimum)]["global"]
        print(
            f"{minimum:>3}  {row['scorer_calls']:>6}  {row['deferred_coda_calls']:>8}  "
            f"{row['deferred_run_count']:>4}  {row['backfilled_coda_calls']:>8}  "
            f"{row['truly_eliminated_coda_calls']:>10}  "
            f"{row['coda_reduction_percent']:>8.3f}%  "
            f"{row['high_score_exact_safe_assumption_violations']:>10}  "
            f"{row['baseline_k_agreement_count']}/{row['prediction_count']}  "
            f"{row['historical_estimated_net_latency_saving_ms']:>7.2f}  "
            f"{row['break_even_scorer_cost_ms_per_call'] or 0.0:.6f}"
        )
    for name, effect in results["marginal_effects"].items():
        row = effect["global"]
        print(
            f"{name}: +scores={row['added_scorer_calls']} "
            f"+deferred={row['added_deferred_rows']} "
            f"+eliminated={row['added_truly_eliminated_coda_calls']} "
            f"+violations={row['added_safety_violations']} "
            f"delta_reduction_pp={row['coda_reduction_percentage_point_change']:.3f} "
            f"delta_net_ms={row['historical_estimated_net_latency_change_ms']:.3f}"
        )
    for name in (
        "results.json",
        "per_task_summary.csv",
        "run_length_distribution.csv",
        "marginal_effects.csv",
    ):
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
        else manifest_path.parent.parent / "deferred_backfill_minimum_sweep"
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
