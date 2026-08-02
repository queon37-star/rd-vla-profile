"""Audit frozen adaptive-Coda OOF results and replay safety wrappers.

This module is deliberately read-only with respect to the fitted OOF models.  It
loads their recorded fold-specific thresholds and triggers, combines trigger
iterations for predeclared wrappers, and replays the original action-MSE labels.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from scripts.adaptive_coda_gate_oof import (
    METRIC_FIELDS,
    aggregate_replays,
    replay_trigger,
    score_gate_predictions,
)
from scripts.analyze_latent_dynamics_features import canonical_json


SCHEMA_VERSION = 1
EXPECTED_WORKLOAD_IDENTITY = (
    "11e9625e136e2c1c08255a020b10a4b6645f8136a9c49d6bbf383f30d987b268"
)
EXPECTED_SOURCE_COMMIT = "975da9093b960b36910c5b3a84723d23bbed3873"
EXPECTED_PREDICTION_COUNT = 2298
SOURCE_FIXED_POLICY = "fixed_raw_mse_beta_0_05"
AUDIT_POLICIES = (
    "coda_every_iteration",
    SOURCE_FIXED_POLICY,
    "raw_mse_logistic",
    "iteration_raw_mse",
    "combined",
)
LEARNED_AUDIT_POLICIES = (
    "raw_mse_logistic",
    "iteration_raw_mse",
    "combined",
)
WRAPPER_COMPONENTS = {
    "combined": ("combined",),
    "fixed_raw_mse": (SOURCE_FIXED_POLICY,),
    "combined_or_fixed": ("combined", SOURCE_FIXED_POLICY),
    "combined_or_iteration_raw_mse": ("combined", "iteration_raw_mse"),
    "combined_or_raw_mse_logistic": ("combined", "raw_mse_logistic"),
    "combined_or_fixed_or_iteration_raw_mse": (
        "combined",
        SOURCE_FIXED_POLICY,
        "iteration_raw_mse",
    ),
}
PROMOTION_GATE = {
    "exact_K_preservation_rate_min": 0.95,
    "mean_delta_K_max": 0.1,
    "p95_delta_K_max": 1.0,
    "forced_trigger_rate_required": 0.0,
}


class AdaptiveCodaSafetyError(ValueError):
    """Raised when an input or replay safety invariant is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AdaptiveCodaSafetyError(message)


def prediction_key(item: Mapping[str, Any]) -> tuple[str, int, int]:
    return (
        str(item["task_id"]),
        int(item["episode_id"]),
        int(item["prediction_id"]),
    )


def prediction_key_text(key: Sequence[Any]) -> str:
    return f"{key[0]}:{int(key[1])}:{int(key[2])}"


def exact_required_count(count: int, required_rate: float = 0.95) -> int:
    """Return the smallest integer count satisfying the requested rate."""

    _require(count >= 0, "prediction count must be non-negative")
    _require(0.0 <= required_rate <= 1.0, "required rate must be in [0, 1]")
    return int(math.ceil(count * required_rate))


def combine_trigger_k(component_replays: Sequence[Mapping[str, Any]]) -> int:
    _require(bool(component_replays), "a wrapper requires at least one component")
    keys = {prediction_key(item) for item in component_replays}
    _require(len(keys) == 1, "wrapper components refer to different predictions")
    return min(int(item["trigger_k"]) for item in component_replays)


def _parse_bool(value: str) -> bool:
    _require(value in {"True", "False"}, f"invalid CSV boolean: {value!r}")
    return value == "True"


def load_replay_csv(path: Path) -> dict[str, list[dict[str, Any]]]:
    integer_fields = {
        "outer_fold",
        "task_id",
        "episode_id",
        "prediction_id",
        "baseline_k",
        "activation_target",
        "trigger_k",
        "first_action_mse_check_k",
        "terminal_k",
        "delta_k",
        "trigger_delay",
        "early_trigger_distance",
        "baseline_coda_calls",
        "scheduled_coda_calls",
    }
    boolean_fields = {
        "forced_trigger",
        "exact_k_preserved",
        "delta_k_gt_0",
        "max_iteration",
    }
    json_fields = {
        "executed_coda_iterations_json": "executed_coda_iterations",
        "executed_action_mse_checks_json": "executed_action_mse_checks",
    }
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = dict(raw)
            for field in integer_fields:
                row[field] = int(row[field])
            for field in boolean_fields:
                row[field] = _parse_bool(row[field])
            for source, destination in json_fields.items():
                row[destination] = json.loads(row.pop(source))
            row["task_id"] = str(row["task_id"])
            by_policy[row["policy"]].append(row)
    for rows in by_policy.values():
        rows.sort(key=prediction_key)
    return dict(by_policy)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_oof_artifacts(input_dir: Path) -> dict[str, Any]:
    """Load and validate the immutable adaptive-gate OOF artifact bundle."""

    required = {
        "metric_report.json",
        "model_summary.json",
        "oof_prediction_replays.csv",
        "output_hashes.json",
    }
    missing = sorted(name for name in required if not (input_dir / name).is_file())
    _require(not missing, f"missing OOF artifacts: {missing}")
    hash_report = json.loads((input_dir / "output_hashes.json").read_text())
    for name, expected in hash_report["files"].items():
        _require((input_dir / name).is_file(), f"hashed source artifact is missing: {name}")
        _require(_sha256(input_dir / name) == expected, f"source artifact hash mismatch: {name}")
    report = json.loads((input_dir / "metric_report.json").read_text())
    models = json.loads((input_dir / "model_summary.json").read_text())
    _require(
        report["inputs"]["workload_identity_sha256"] == EXPECTED_WORKLOAD_IDENTITY,
        "frozen workload identity mismatch",
    )
    _require(
        report["inputs"]["source_git_commit"] == EXPECTED_SOURCE_COMMIT,
        "adaptive-gate source commit mismatch",
    )
    _require(
        report["prediction_count"] == EXPECTED_PREDICTION_COUNT,
        "OOF prediction count mismatch",
    )
    _require(models["global_model_fitted"] is False, "unexpected global fitted model")
    _require(models["global_threshold_fitted"] is False, "unexpected global threshold")
    replays = load_replay_csv(input_dir / "oof_prediction_replays.csv")
    for policy in AUDIT_POLICIES:
        _require(policy in replays, f"missing OOF policy: {policy}")
        _require(
            len(replays[policy]) == EXPECTED_PREDICTION_COUNT,
            f"{policy}: OOF replay count mismatch",
        )
    reference_keys = {prediction_key(item) for item in replays["coda_every_iteration"]}
    for policy in AUDIT_POLICIES:
        keys = [prediction_key(item) for item in replays[policy]]
        _require(len(keys) == len(set(keys)), f"{policy}: duplicate OOF prediction key")
        _require(set(keys) == reference_keys, f"{policy}: OOF workload differs")
    return {
        "input_dir": input_dir,
        "metric_report": report,
        "model_summary": models,
        "replays": replays,
        "source_hashes": hash_report,
    }


def index_replays(
    replays_by_policy: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[tuple[str, int, int], Mapping[str, Any]]]:
    return {
        policy: {prediction_key(item): item for item in rows}
        for policy, rows in replays_by_policy.items()
    }


def _histogram(values: Iterable[int]) -> dict[str, int]:
    counts = Counter(int(value) for value in values)
    return {str(key): counts[key] for key in sorted(counts)}


def _delta_buckets(replays: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    values = [int(item["delta_k"]) for item in replays]
    return {
        "0": sum(value == 0 for value in values),
        "1": sum(value == 1 for value in values),
        "2": sum(value == 2 for value in values),
        "3": sum(value == 3 for value in values),
        ">=4": sum(value >= 4 for value in values),
    }


def _audit_breakdown(replays: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(replays)
    delayed = [item for item in replays if int(item["delta_k"]) > 0]
    exact_count = sum(int(item["delta_k"]) == 0 for item in replays)
    return {
        "prediction_count": count,
        "exact_K_count": exact_count,
        "exact_K_rate": exact_count / count,
        "delta_K_gt_0_count": len(delayed),
        "delta_K_gt_0_rate": len(delayed) / count,
        "delta_K_histogram": _histogram(int(item["delta_k"]) for item in replays),
        "delta_K_buckets": _delta_buckets(replays),
        "total_additional_recurrent_iterations": sum(
            int(item["delta_k"]) for item in replays
        ),
        "mean_delta_K_conditional_gt_0": (
            float(np.mean([int(item["delta_k"]) for item in delayed]))
            if delayed
            else 0.0
        ),
        "trigger_delay_histogram": _histogram(
            int(item["trigger_delay"]) for item in replays
        ),
    }


def _breakdown_by(
    replays: Sequence[Mapping[str, Any]], field: str, *, numeric: bool = False
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for replay in replays:
        groups[str(replay[field])].append(replay)
    key_fn = (lambda value: int(value)) if numeric else (lambda value: value)
    return {
        key: _audit_breakdown(groups[key]) for key in sorted(groups, key=key_fn)
    }


def failure_sets(
    fixed_replays: Sequence[Mapping[str, Any]],
    combined_replays: Sequence[Mapping[str, Any]],
) -> dict[str, set[tuple[str, int, int]]]:
    fixed = {
        prediction_key(item) for item in fixed_replays if int(item["delta_k"]) > 0
    }
    combined = {
        prediction_key(item)
        for item in combined_replays
        if int(item["delta_k"]) > 0
    }
    return {
        "fixed": fixed,
        "combined": combined,
        "shared": fixed & combined,
        "fixed_only": fixed - combined,
        "combined_only": combined - fixed,
        "union": fixed | combined,
    }


def selected_thresholds_by_fold(model_summary: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for fold in model_summary["outer_folds"]:
        fold_id = str(int(fold["outer_fold"]))
        result[fold_id] = {
            "fixed_raw_mse": {
                "threshold": float(fold["fixed_raw_mse_reference"]["threshold"]),
                "threshold_hex": fold["fixed_raw_mse_reference"]["threshold_hex"],
                "selection_scope": "outer_training_only_frozen",
            }
        }
        for policy in LEARNED_AUDIT_POLICIES:
            selection = fold["learned_models"][policy]["threshold_selection"]
            result[fold_id][policy] = {
                "threshold": float(selection["selected_threshold"]),
                "threshold_hex": selection["selected_threshold_hex"],
                "selection_scope": "inner_cross_fitted_outer_training_only_frozen",
            }
    return result


def _vector_stability(vectors: Sequence[Sequence[float]]) -> dict[str, Any]:
    matrix = np.asarray(vectors, dtype=np.float64)
    _require(matrix.ndim == 2 and np.isfinite(matrix).all(), "invalid stability vector")
    cosine_values = []
    for left in range(len(matrix)):
        for right in range(left + 1, len(matrix)):
            denominator = float(np.linalg.norm(matrix[left]) * np.linalg.norm(matrix[right]))
            cosine_values.append(
                float(np.dot(matrix[left], matrix[right]) / denominator)
                if denominator
                else 1.0
            )
    return {
        "mean_by_parameter": matrix.mean(axis=0).tolist(),
        "std_by_parameter": matrix.std(axis=0).tolist(),
        "min_by_parameter": matrix.min(axis=0).tolist(),
        "max_by_parameter": matrix.max(axis=0).tolist(),
        "pairwise_cosine": {
            "minimum": min(cosine_values),
            "mean": float(np.mean(cosine_values)),
            "maximum": max(cosine_values),
        },
    }


def model_stability_by_fold(model_summary: Mapping[str, Any]) -> dict[str, Any]:
    """Expose frozen fold parameters and summarize their cross-fold stability."""

    output: dict[str, Any] = {}
    for policy in LEARNED_AUDIT_POLICIES:
        fold_rows = []
        for fold in model_summary["outer_folds"]:
            artifact = fold["learned_models"][policy]["outer_training_refit"]
            selection = fold["learned_models"][policy]["threshold_selection"]
            fold_rows.append(
                {
                    "outer_fold": int(fold["outer_fold"]),
                    "training_task_ids": artifact["training_task_ids"],
                    "threshold": float(selection["selected_threshold"]),
                    "threshold_hex": selection["selected_threshold_hex"],
                    "feature_names": artifact["feature_names"],
                    "expanded_feature_names": artifact["preprocessor"][
                        "expanded_feature_names"
                    ],
                    "bias": float(artifact["model"]["bias"]),
                    "coefficients": [float(value) for value in artifact["model"]["weights"]],
                    "imputation_medians": artifact["preprocessor"]["imputation_medians"],
                    "scaling_mean": artifact["preprocessor"]["scaling_mean"],
                    "scaling_scale": artifact["preprocessor"]["scaling_scale"],
                }
            )
        expanded_names = fold_rows[0]["expanded_feature_names"]
        _require(
            all(row["expanded_feature_names"] == expanded_names for row in fold_rows),
            f"{policy}: expanded features changed across folds",
        )
        feature_names = fold_rows[0]["feature_names"]
        _require(
            all(row["feature_names"] == feature_names for row in fold_rows),
            f"{policy}: features changed across folds",
        )
        coefficients = [row["coefficients"] for row in fold_rows]
        coefficient_stability = _vector_stability(coefficients)
        coefficient_stability["parameter_names"] = expanded_names
        preprocessing = {
            "imputation_medians": {
                "parameter_names": feature_names,
                **_vector_stability([row["imputation_medians"] for row in fold_rows]),
            },
            "scaling_mean": {
                "parameter_names": expanded_names,
                **_vector_stability([row["scaling_mean"] for row in fold_rows]),
            },
            "scaling_scale": {
                "parameter_names": expanded_names,
                **_vector_stability([row["scaling_scale"] for row in fold_rows]),
            },
        }
        thresholds = [row["threshold"] for row in fold_rows]
        output[policy] = {
            "by_fold": fold_rows,
            "coefficient_stability": coefficient_stability,
            "bias_stability": {
                "minimum": min(row["bias"] for row in fold_rows),
                "mean": float(np.mean([row["bias"] for row in fold_rows])),
                "maximum": max(row["bias"] for row in fold_rows),
                "std": float(np.std([row["bias"] for row in fold_rows])),
            },
            "threshold_stability": {
                "minimum": min(thresholds),
                "mean": float(np.mean(thresholds)),
                "maximum": max(thresholds),
                "std": float(np.std(thresholds)),
            },
            "preprocessing_stability": preprocessing,
        }
    return output


def reconstruct_frozen_scores(
    predictions: Sequence[Mapping[str, Any]], model_summary: Mapping[str, Any]
) -> dict[str, dict[tuple[str, int, int], dict[int, float]]]:
    """Score held-out rows with serialized outer refits; never fit or select."""

    by_policy: dict[str, dict[tuple[str, int, int], dict[int, float]]] = {
        policy: {} for policy in LEARNED_AUDIT_POLICIES
    }
    for fold in model_summary["outer_folds"]:
        held_out_tasks = {str(item) for item in fold["outer_held_out_task_ids"]}
        held_out = [item for item in predictions if str(item["task_id"]) in held_out_tasks]
        _require(bool(held_out), f"outer fold {fold['outer_fold']}: no held-out predictions")
        for policy in LEARNED_AUDIT_POLICIES:
            fitted = fold["learned_models"][policy]["outer_training_refit"]
            for scored in score_gate_predictions(held_out, fitted):
                key = prediction_key(scored["prediction"])
                _require(key not in by_policy[policy], f"{policy}: duplicate OOF score")
                by_policy[policy][key] = scored["scores_by_k"]
    expected = {prediction_key(item) for item in predictions}
    for policy, values in by_policy.items():
        _require(set(values) == expected, f"{policy}: incomplete frozen OOF scores")
    return by_policy


def score_margins(
    predictions: Sequence[Mapping[str, Any]],
    replay_index: Mapping[str, Mapping[tuple[str, int, int], Mapping[str, Any]]],
    scores: Mapping[str, Mapping[tuple[str, int, int], Mapping[int, float]]],
    thresholds: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, dict[tuple[str, int, int], dict[str, float]]]:
    prediction_index = {prediction_key(item): item for item in predictions}
    margins: dict[str, dict[tuple[str, int, int], dict[str, float]]] = {
        policy: {} for policy in LEARNED_AUDIT_POLICIES
    }
    for policy in LEARNED_AUDIT_POLICIES:
        for key, replay in replay_index[policy].items():
            fold = str(int(replay["outer_fold"]))
            threshold = float(thresholds[fold][policy]["threshold"])
            target = int(prediction_index[key]["activation_target"])
            trigger = int(replay["trigger_k"])
            margins[policy][key] = {
                "activation_target_score": float(scores[policy][key][target]),
                "activation_target_margin": float(scores[policy][key][target]) - threshold,
                "first_trigger_score": float(scores[policy][key][trigger]),
                "first_trigger_margin": float(scores[policy][key][trigger]) - threshold,
                "selected_threshold": threshold,
            }
    return margins


def _margin_summary(
    values: Mapping[tuple[str, int, int], Mapping[str, float]],
    failure_keys: set[tuple[str, int, int]],
) -> dict[str, Any]:
    def summarize(rows: Sequence[Mapping[str, float]]) -> dict[str, Any]:
        return {
            field: {
                "minimum": min(float(row[field]) for row in rows),
                "mean": float(np.mean([float(row[field]) for row in rows])),
                "median": float(np.median([float(row[field]) for row in rows])),
                "maximum": max(float(row[field]) for row in rows),
            }
            for field in ("activation_target_margin", "first_trigger_margin")
        }

    all_rows = list(values.values())
    failed_rows = [values[key] for key in sorted(failure_keys) if key in values]
    return {
        "margin_definition": "serialized outer-OOF probability score minus frozen fold threshold",
        "all_predictions": summarize(all_rows),
        "delayed_predictions": summarize(failed_rows) if failed_rows else None,
    }


def build_failure_audit(
    replays_by_policy: Mapping[str, Sequence[Mapping[str, Any]]],
    model_summary: Mapping[str, Any],
    *,
    margins: Mapping[str, Mapping[tuple[str, int, int], Mapping[str, float]]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    replay_index = index_replays(replays_by_policy)
    fixed = replays_by_policy[SOURCE_FIXED_POLICY]
    combined = replays_by_policy["combined"]
    sets = failure_sets(fixed, combined)
    policy_audits = {}
    for policy in AUDIT_POLICIES:
        rows = replays_by_policy[policy]
        policy_audits[policy] = {
            **_audit_breakdown(rows),
            "by_task": _breakdown_by(rows, "task_id", numeric=True),
            "by_outer_fold": _breakdown_by(rows, "outer_fold", numeric=True),
            "by_difficulty": _breakdown_by(rows, "difficulty"),
        }
        if margins and policy in margins:
            failures = {
                prediction_key(item) for item in rows if int(item["delta_k"]) > 0
            }
            policy_audits[policy]["score_margins"] = _margin_summary(
                margins[policy], failures
            )

    combined_failures = [
        item for item in combined if prediction_key(item) in sets["combined"]
    ]
    task_counts = Counter(str(item["task_id"]) for item in combined_failures)
    fold_counts = Counter(str(item["outer_fold"]) for item in combined_failures)

    def concentration(counts: Counter[str]) -> dict[str, Any]:
        if not counts:
            return {
                "largest_group": None,
                "largest_group_failure_count": 0,
                "largest_group_failure_fraction": 0.0,
                "majority_in_one_group": False,
                "majority_definition": "strictly more than 50% of combined delayed predictions",
            }
        group, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
        total = sum(counts.values())
        return {
            "largest_group": group,
            "largest_group_failure_count": count,
            "largest_group_failure_fraction": count / total,
            "majority_in_one_group": count / total > 0.5,
            "majority_definition": "strictly more than 50% of combined delayed predictions",
        }

    required = exact_required_count(len(combined))
    actual = sum(int(item["delta_k"]) == 0 for item in combined)
    thresholds = selected_thresholds_by_fold(model_summary)
    audit = {
        "schema_version": SCHEMA_VERSION,
        "status": "calibration_development_failure_audit",
        "prediction_count": len(combined),
        "failure_definition": "terminal delta_K > 0 relative to Coda-every-iteration baseline K",
        "policy_audits": policy_audits,
        "fixed_combined_failure_sets": {
            "fixed_failure_count": len(sets["fixed"]),
            "combined_failure_count": len(sets["combined"]),
            "shared_failure_count": len(sets["shared"]),
            "fixed_only_failure_count": len(sets["fixed_only"]),
            "combined_only_failure_count": len(sets["combined_only"]),
            "union_failure_count": len(sets["union"]),
            "shared_prediction_keys": [prediction_key_text(key) for key in sorted(sets["shared"])],
            "fixed_only_prediction_keys": [
                prediction_key_text(key) for key in sorted(sets["fixed_only"])
            ],
            "combined_only_prediction_keys": [
                prediction_key_text(key) for key in sorted(sets["combined_only"])
            ],
        },
        "exact_95_percent_requirement": {
            "required_exact_prediction_count": required,
            "actual_combined_exact_prediction_count": actual,
            "shortfall_count": max(0, required - actual),
            "required_rate": 0.95,
        },
        "combined_failure_concentration": {
            "by_task": concentration(task_counts),
            "by_outer_fold": concentration(fold_counts),
        },
        "selected_thresholds_by_outer_fold": thresholds,
        "model_and_preprocessing_stability": model_stability_by_fold(model_summary),
        "models_refit": False,
        "thresholds_reselected": False,
    }

    failure_rows = []
    for key in sorted(sets["union"]):
        fixed_row = replay_index[SOURCE_FIXED_POLICY][key]
        combined_row = replay_index["combined"][key]
        if key in sets["shared"]:
            category = "shared"
        elif key in sets["fixed_only"]:
            category = "fixed_only"
        else:
            category = "combined_only"
        combined_margin = margins["combined"][key] if margins else {}
        failure_rows.append(
            {
                "task_id": key[0],
                "episode_id": key[1],
                "prediction_id": key[2],
                "outer_fold": combined_row["outer_fold"],
                "difficulty": combined_row["difficulty"],
                "failure_category": category,
                "baseline_k": combined_row["baseline_k"],
                "activation_target": combined_row["activation_target"],
                "fixed_trigger_k": fixed_row["trigger_k"],
                "fixed_terminal_k": fixed_row["terminal_k"],
                "fixed_delta_k": fixed_row["delta_k"],
                "combined_trigger_k": combined_row["trigger_k"],
                "combined_terminal_k": combined_row["terminal_k"],
                "combined_delta_k": combined_row["delta_k"],
                "combined_selected_threshold": combined_margin.get("selected_threshold"),
                "combined_activation_target_score": combined_margin.get("activation_target_score"),
                "combined_activation_target_margin": combined_margin.get(
                    "activation_target_margin"
                ),
                "combined_first_trigger_score": combined_margin.get("first_trigger_score"),
                "combined_first_trigger_margin": combined_margin.get("first_trigger_margin"),
            }
        )
    return audit, failure_rows


def _extended_metrics(replays: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = aggregate_replays(replays)
    exact_count = sum(bool(item["exact_k_preserved"]) for item in replays)
    delayed_count = sum(int(item["delta_k"]) > 0 for item in replays)
    metrics.update(
        {
            "exact_K_count": exact_count,
            "delta_K_gt_0_count": delayed_count,
            "reduced_coda_calls": (
                metrics["baseline_total_coda_calls"]
                - metrics["scheduled_total_coda_calls"]
            ),
        }
    )
    return metrics


def _group_metrics(
    replays: Sequence[Mapping[str, Any]], field: str, *, numeric: bool = False
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for replay in replays:
        groups[str(replay[field])].append(replay)
    key_fn = (lambda value: int(value)) if numeric else (lambda value: value)
    return {
        key: _extended_metrics(groups[key]) for key in sorted(groups, key=key_fn)
    }


def replay_predeclared_wrappers(
    predictions: Sequence[Mapping[str, Any]],
    source_replays: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Combine frozen trigger iterations, then rerun the exact label sequence."""

    prediction_index = {prediction_key(item): item for item in predictions}
    for prediction in predictions:
        baseline_k = int(prediction["baseline_k"])
        for transition in prediction["transitions"]:
            iteration = int(transition["k"])
            expected_source = (
                "production_iteration_mse"
                if iteration <= baseline_k
                else "shadow_fp32_adjacent_action_mse"
            )
            _require(
                transition.get("action_mse_source") == expected_source,
                f"{prediction_key(prediction)} k={iteration}: action-label phase mismatch",
            )
    source_index = index_replays(source_replays)
    expected_keys = set(prediction_index)
    for component in {item for values in WRAPPER_COMPONENTS.values() for item in values}:
        _require(component in source_index, f"missing wrapper component: {component}")
        _require(set(source_index[component]) == expected_keys, f"{component}: workload mismatch")

    all_wrapper_replays: dict[str, list[dict[str, Any]]] = {}
    invariant_audit = []
    for wrapper, components in WRAPPER_COMPONENTS.items():
        rows = []
        for key in sorted(expected_keys):
            component_rows = [source_index[component][key] for component in components]
            trigger_k = combine_trigger_k(component_rows)
            forced = all(bool(item["forced_trigger"]) for item in component_rows)
            replay = replay_trigger(prediction_index[key], trigger_k, forced_trigger=forced)
            replay.update(
                {
                    "policy": wrapper,
                    "outer_fold": int(component_rows[0]["outer_fold"]),
                    "component_policies": list(components),
                    "component_trigger_k": {
                        component: int(item["trigger_k"])
                        for component, item in zip(components, component_rows)
                    },
                    "component_terminal_k": {
                        component: int(item["terminal_k"])
                        for component, item in zip(components, component_rows)
                    },
                }
            )
            for component, component_row in zip(components, component_rows):
                _require(
                    replay["terminal_k"] <= int(component_row["terminal_k"]),
                    f"{wrapper} {key}: terminal K exceeds {component}",
                )
            if len(components) == 1:
                source = component_rows[0]
                for field in (
                    "trigger_k",
                    "forced_trigger",
                    "first_action_mse_check_k",
                    "terminal_k",
                    "delta_k",
                    "exact_k_preserved",
                    "scheduled_coda_calls",
                    "executed_coda_iterations",
                    "executed_action_mse_checks",
                ):
                    _require(
                        replay[field] == source[field],
                        f"{wrapper} {key}: singleton replay differs in {field}",
                    )
            rows.append(replay)
        for component in components:
            wrapper_exact = sum(bool(item["exact_k_preserved"]) for item in rows)
            component_exact = sum(
                bool(source_index[component][key]["exact_k_preserved"])
                for key in expected_keys
            )
            _require(
                wrapper_exact >= component_exact,
                f"{wrapper}: exact-K count below component {component}",
            )
        invariant_audit.append(
            {
                "wrapper": wrapper,
                "components": list(components),
                "prediction_count": len(rows),
                "terminal_K_no_greater_than_each_component": True,
                "exact_K_count_no_lower_than_each_component": True,
                "coda_calls_recomputed_from_combined_trigger": True,
            }
        )
        all_wrapper_replays[wrapper] = rows

    baseline_rows = source_replays["coda_every_iteration"]
    fixed_rows = all_wrapper_replays["fixed_raw_mse"]
    combined_rows = all_wrapper_replays["combined"]
    baseline_metrics = _extended_metrics(baseline_rows)
    fixed_metrics = _extended_metrics(fixed_rows)
    combined_metrics = _extended_metrics(combined_rows)
    wrapper_results = {}
    passing = []
    for wrapper, rows in all_wrapper_replays.items():
        metrics = _extended_metrics(rows)
        checks = {
            "exact_K_rate": (
                metrics["exact_K_preservation_rate"]
                >= PROMOTION_GATE["exact_K_preservation_rate_min"]
            ),
            "mean_delta_K": metrics["mean_delta_K"] <= PROMOTION_GATE["mean_delta_K_max"],
            "p95_delta_K": metrics["p95_delta_K"] <= PROMOTION_GATE["p95_delta_K_max"],
            "forced_trigger_rate": (
                metrics["forced_trigger_rate"]
                == PROMOTION_GATE["forced_trigger_rate_required"]
            ),
            "no_max_iteration_increase": (
                metrics["max_iteration_rate"] <= baseline_metrics["max_iteration_rate"]
            ),
            "reduction_strictly_greater_than_corrected_fixed": (
                metrics["coda_call_reduction"] > fixed_metrics["coda_call_reduction"]
            ),
        }
        passed = all(checks.values())
        if passed:
            passing.append(wrapper)
        wrapper_results[wrapper] = {
            "metrics": metrics,
            "task_metrics": _group_metrics(rows, "task_id", numeric=True),
            "fold_metrics": _group_metrics(rows, "outer_fold", numeric=True),
            "difficulty_metrics": _group_metrics(rows, "difficulty"),
            "additional_coda_calls_relative_to_combined": (
                metrics["scheduled_total_coda_calls"]
                - combined_metrics["scheduled_total_coda_calls"]
            ),
            "saved_coda_calls_relative_to_fixed": (
                fixed_metrics["scheduled_total_coda_calls"]
                - metrics["scheduled_total_coda_calls"]
            ),
            "promotion_checks": checks,
            "passes_full_promotion_gate": passed,
        }
    return {
        "wrapper_results": wrapper_results,
        "all_replays": all_wrapper_replays,
        "reference_metrics": {
            "coda_every_iteration": baseline_metrics,
            "corrected_fixed_raw_mse": fixed_metrics,
            "combined": combined_metrics,
        },
        "invariant_audit": invariant_audit,
        "promotion_assessment": {
            "gate": dict(PROMOTION_GATE),
            "passing_wrappers": passing,
            "any_wrapper_passes": bool(passing),
            "component_thresholds_modified": False,
            "deployment_threshold_recorded": False,
            "next_step_if_passing": (
                "freeze the passing wrapper and evaluate it on the independent screening partition"
            ),
        },
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _flatten_metrics(
    policy: str, group_field: str, group: str, metrics: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "policy": policy,
        group_field: group,
        **{
            field: metrics[field]
            for field in (
                *METRIC_FIELDS,
                "exact_K_count",
                "delta_K_gt_0_count",
                "reduced_coda_calls",
                "prediction_count",
            )
            if field in metrics
        },
    }


def _compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_safety_outputs(
    output_dir: Path,
    *,
    failure_audit: Mapping[str, Any],
    failure_rows: Sequence[Mapping[str, Any]],
    wrapper_evaluation: Mapping[str, Any],
    inputs: Mapping[str, Any],
    overwrite: bool = False,
) -> dict[str, str]:
    filenames = (
        "failure_audit.json",
        "failure_predictions.csv",
        "wrapper_metrics.csv",
        "wrapper_prediction_replays.csv",
        "task_metrics.csv",
        "fold_metrics.csv",
        "difficulty_metrics.csv",
        "metric_report.json",
        "output_hashes.json",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [name for name in filenames if (output_dir / name).exists()]
    _require(overwrite or not existing, f"refusing to overwrite outputs: {existing}")
    (output_dir / "failure_audit.json").write_text(canonical_json(failure_audit), encoding="utf-8")
    failure_fields = (
        tuple(failure_rows[0])
        if failure_rows
        else (
            "task_id",
            "episode_id",
            "prediction_id",
            "outer_fold",
            "difficulty",
            "failure_category",
        )
    )
    _write_csv(output_dir / "failure_predictions.csv", failure_rows, failure_fields)

    wrapper_rows = []
    for policy in WRAPPER_COMPONENTS:
        result = wrapper_evaluation["wrapper_results"][policy]
        wrapper_rows.append(
            {
                "policy": policy,
                **{
                    field: result["metrics"][field]
                    for field in (
                        *METRIC_FIELDS,
                        "exact_K_count",
                        "delta_K_gt_0_count",
                        "reduced_coda_calls",
                        "prediction_count",
                    )
                    if field in result["metrics"]
                },
                "additional_coda_calls_relative_to_combined": result[
                    "additional_coda_calls_relative_to_combined"
                ],
                "saved_coda_calls_relative_to_fixed": result[
                    "saved_coda_calls_relative_to_fixed"
                ],
                "passes_full_promotion_gate": result["passes_full_promotion_gate"],
            }
        )
    _write_csv(output_dir / "wrapper_metrics.csv", wrapper_rows, tuple(wrapper_rows[0]))

    replay_rows = []
    for policy in WRAPPER_COMPONENTS:
        for replay in wrapper_evaluation["all_replays"][policy]:
            replay_rows.append(
                {
                    "policy": policy,
                    "outer_fold": replay["outer_fold"],
                    "task_id": replay["task_id"],
                    "episode_id": replay["episode_id"],
                    "prediction_id": replay["prediction_id"],
                    "difficulty": replay["difficulty"],
                    "baseline_k": replay["baseline_k"],
                    "activation_target": replay["activation_target"],
                    "trigger_k": replay["trigger_k"],
                    "forced_trigger": replay["forced_trigger"],
                    "first_action_mse_check_k": replay["first_action_mse_check_k"],
                    "terminal_k": replay["terminal_k"],
                    "stop_reason": replay["stop_reason"],
                    "delta_k": replay["delta_k"],
                    "exact_k_preserved": replay["exact_k_preserved"],
                    "delta_k_gt_0": replay["delta_k_gt_0"],
                    "trigger_delay": replay["trigger_delay"],
                    "early_trigger_distance": replay["early_trigger_distance"],
                    "max_iteration": replay["max_iteration"],
                    "baseline_coda_calls": replay["baseline_coda_calls"],
                    "scheduled_coda_calls": replay["scheduled_coda_calls"],
                    "component_trigger_k_json": _compact_json(replay["component_trigger_k"]),
                    "component_terminal_k_json": _compact_json(replay["component_terminal_k"]),
                    "executed_coda_iterations_json": _compact_json(
                        replay["executed_coda_iterations"]
                    ),
                    "executed_action_mse_checks_json": _compact_json(
                        replay["executed_action_mse_checks"]
                    ),
                }
            )
    _write_csv(output_dir / "wrapper_prediction_replays.csv", replay_rows, tuple(replay_rows[0]))

    for filename, field, numeric in (
        ("task_metrics.csv", "task_id", True),
        ("fold_metrics.csv", "outer_fold", True),
        ("difficulty_metrics.csv", "difficulty", False),
    ):
        rows = []
        result_key = {
            "task_id": "task_metrics",
            "outer_fold": "fold_metrics",
            "difficulty": "difficulty_metrics",
        }[field]
        for policy in WRAPPER_COMPONENTS:
            groups = wrapper_evaluation["wrapper_results"][policy][result_key]
            key_fn = (lambda value: int(value)) if numeric else (lambda value: value)
            for group in sorted(groups, key=key_fn):
                rows.append(_flatten_metrics(policy, field, group, groups[group]))
        _write_csv(output_dir / filename, rows, tuple(rows[0]))

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "calibration_development_safety_wrapper_evaluation",
        "inputs": dict(inputs),
        "workload_identity_sha256": EXPECTED_WORKLOAD_IDENTITY,
        "prediction_count": failure_audit["prediction_count"],
        "wrapper_components": {key: list(value) for key, value in WRAPPER_COMPONENTS.items()},
        "trigger_combination": (
            "minimum frozen component trigger K, followed by exact recorded-label replay"
        ),
        "action_label_phase_semantics": {
            "k_at_or_before_baseline_K": "authoritative BF16 production iteration_mse label",
            "k_after_baseline_K": "FP32 shadow-tail adjacent_action_mse label",
        },
        "wrapper_results": wrapper_evaluation["wrapper_results"],
        "reference_metrics": wrapper_evaluation["reference_metrics"],
        "invariant_audit": wrapper_evaluation["invariant_audit"],
        "promotion_assessment": wrapper_evaluation["promotion_assessment"],
        "failure_audit_file": "failure_audit.json",
        "runtime_inference_modified": False,
        "models_refit": False,
        "component_thresholds_modified": False,
        "outer_oof_results_modified": False,
        "deployment_threshold_recorded": False,
        "calibration_development_only": True,
        "outputs": {name: name for name in filenames},
    }
    (output_dir / "metric_report.json").write_text(canonical_json(report), encoding="utf-8")
    hashes = {name: _sha256(output_dir / name) for name in filenames[:-1]}
    (output_dir / "output_hashes.json").write_text(
        canonical_json(
            {
                "schema_version": SCHEMA_VERSION,
                "hash_algorithm": "sha256",
                "files": hashes,
            }
        ),
        encoding="utf-8",
    )
    return hashes
