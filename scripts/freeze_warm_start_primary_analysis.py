#!/usr/bin/env python3
"""Freeze the warm-start primary statistical analysis before final rollout.

This script validates the planning artifacts and writes a fail-closed analysis
freeze. It does not authorize GPU rollout. Runtime source, checkpoint, and
environment identities are frozen in a later authorization step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


MARGIN = 0.05
PROFILE_P_VALUE_CUTOFF = 0.045
PAIRS_PER_TASK = 47
TASK_COUNT = 10
TOTAL_PRIMARY_PAIRS = PAIRS_PER_TASK * TASK_COUNT
FINAL_BASE_SEED = 47007
PRIMARY_PHASES = ("calibration", "screening", "final")


class AnalysisFreezeError(ValueError):
    """Raised when a planning artifact violates the frozen analysis contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisFreezeError(message)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_ref(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def cutoff_row(calibration: dict[str, Any], cutoff: float) -> dict[str, Any]:
    rows = calibration.get("cutoff_summaries")
    require(isinstance(rows, list), "cutoff calibration has no cutoff_summaries")
    matches = [
        row
        for row in rows
        if isinstance(row, dict) and abs(float(row.get("cutoff", -1.0)) - cutoff) <= 1e-12
    ]
    require(len(matches) == 1, f"expected one cutoff row for {cutoff}")
    return matches[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cutoff-calibration",
        type=Path,
        default=Path(
            "benchmark_results/warm_start_power/profile_cutoff_calibration_v1.json"
        ),
    )
    parser.add_argument(
        "--state-accounting",
        type=Path,
        default=Path("benchmark_results/warm_start_primary/state_accounting_v1.json"),
    )
    parser.add_argument(
        "--extended-preflight-validation",
        type=Path,
        default=Path(
            "benchmark_results/warm_start_primary/extended_preflight_seed17007/"
            "validation_report_v2.json"
        ),
    )
    parser.add_argument(
        "--repeatability-audit",
        type=Path,
        default=Path(
            "benchmark_results/warm_start_primary/preflight_repeatability_v1.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require(not args.output.exists(), f"refusing to overwrite output: {args.output}")

    calibration = load_json(args.cutoff_calibration)
    accounting = load_json(args.state_accounting)
    preflight = load_json(args.extended_preflight_validation)
    repeatability = load_json(args.repeatability_audit)

    require(
        calibration.get("role")
        == "planning_profile_cutoff_calibration_not_final_authorization",
        "wrong cutoff-calibration role",
    )
    require(calibration.get("final_run_authorized") is False, "calibration unexpectedly authorizes rollout")
    require(calibration.get("cutoff_candidate_available") is True, "no qualified cutoff candidate")
    require(
        abs(float(calibration.get("largest_qualified_cutoff", -1.0)) - PROFILE_P_VALUE_CUTOFF)
        <= 1e-12,
        "largest qualified cutoff is not 0.045",
    )
    analysis = calibration.get("analysis_contract")
    require(isinstance(analysis, dict), "cutoff calibration has no analysis_contract")
    require(abs(float(analysis.get("non_inferiority_margin", -1.0)) - MARGIN) <= 1e-12, "margin mismatch")
    require(int(analysis.get("pairs_per_task", -1)) == PAIRS_PER_TASK, "pairs/task mismatch")
    require(int(analysis.get("total_pairs", -1)) == TOTAL_PRIMARY_PAIRS, "total pairs mismatch")

    selected = cutoff_row(calibration, PROFILE_P_VALUE_CUTOFF)
    nominal = cutoff_row(calibration, 0.05)
    require(selected.get("cutoff_qualified") is True, "cutoff 0.045 did not qualify")
    require(selected.get("all_null_scenarios_qualified") is True, "cutoff 0.045 failed size grid")
    require(
        selected.get("all_zero_effect_power_scenarios_qualified") is True,
        "cutoff 0.045 failed zero-effect power grid",
    )
    require(nominal.get("cutoff_qualified") is False, "nominal cutoff 0.05 unexpectedly qualified")

    require(
        accounting.get("role") == "planning_state_accounting_not_final_authorization",
        "wrong state-accounting role",
    )
    state_counts = accounting.get("state_accounting")
    require(isinstance(state_counts, dict), "state accounting is incomplete")
    require(
        int(state_counts.get("maximum_warm_start_outcome_unseen_states_per_task", -1))
        == PAIRS_PER_TASK,
        "state accounting does not expose 47 outcome-unseen states/task",
    )
    require(
        state_counts.get("preflight_states_allowed_in_primary_confirmatory_analysis") is False,
        "preflight states must be excluded from primary analysis",
    )

    require(preflight.get("schema_version") == 2, "extended preflight validation v2 required")
    require(preflight.get("contract_passed") is True, "extended preflight did not pass")
    require(preflight.get("total_pairs") == 30, "extended preflight must contain 30 pairs")
    require(preflight.get("warm_start_resets") == [], "warm-start resets were observed")

    require(repeatability.get("contract_passed") is True, "repeatability audit did not pass")
    require(repeatability.get("tasks") == [0, 5], "repeatability audit must cover tasks 0 and 5")
    require(repeatability.get("pairs_compared") == 6, "repeatability audit must compare six pairs")

    tasks = accounting.get("tasks")
    require(isinstance(tasks, dict) and set(tasks) == {str(i) for i in range(TASK_COUNT)}, "invalid task accounting")
    excluded_state_ids = {
        task_id: list(tasks[task_id]["observed_in_warm_start_preflight"])
        for task_id in sorted(tasks, key=int)
    }
    included_state_ids = {
        task_id: list(tasks[task_id]["warm_start_outcome_unseen"])
        for task_id in sorted(tasks, key=int)
    }
    require(all(len(ids) == 3 for ids in excluded_state_ids.values()), "expected three excluded states/task")
    require(all(len(ids) == PAIRS_PER_TASK for ids in included_state_ids.values()), "expected 47 included states/task")

    payload = {
        "schema_version": 2,
        "study": "midpoint_warm_start_primary_evaluation",
        "status": "primary_analysis_frozen_runtime_not_authorized",
        "primary_estimand": {
            "name": "equal_weight_mean_paired_success_difference_across_fixed_tasks",
            "direction": "midpoint_warm_start_minus_cold_initialized_adaptive",
            "task_count": TASK_COUNT,
            "pairs_per_task": PAIRS_PER_TASK,
            "total_primary_pairs": TOTAL_PRIMARY_PAIRS,
            "p01": "cold failure and warm success",
            "p10": "cold success and warm failure",
        },
        "noninferiority": {
            "margin_absolute_success_probability": MARGIN,
            "pass_rule": "pooled_profile_likelihood_one_sided_p_value_strictly_less_than_0.045",
            "method": "pooled_paired_trinomial_profile_likelihood",
            "profile_p_value_cutoff": PROFILE_P_VALUE_CUTOFF,
            "nominal_alpha_context": 0.05,
            "cutoff_interpretation": (
                "finite-sample simulation-calibrated decision cutoff for a nominal "
                "one-sided 5% profile-likelihood test; not a newly claimed universal alpha"
            ),
            "selected_cutoff_planning_results": {
                "worst_null_estimated_type_i": selected["worst_null_estimated_type_i"],
                "worst_null_wilson_95_high": selected["worst_null_wilson_95_high"],
                "minimum_zero_effect_power": selected["minimum_zero_effect_power"],
                "minimum_zero_effect_power_wilson_95_low": selected[
                    "minimum_zero_effect_power_wilson_95_low"
                ],
                "minimum_minus_one_point_power_sensitivity": selected[
                    "minimum_minus_one_point_power"
                ],
            },
            "nominal_cutoff_0_05_rejected": {
                "worst_null_estimated_type_i": nominal["worst_null_estimated_type_i"],
                "worst_null_wilson_95_high": nominal["worst_null_wilson_95_high"],
            },
        },
        "margin_rationale": {
            "substantive_threshold": (
                "an absolute loss greater than five successful episodes per 100 paired "
                "rollouts is considered practically unacceptable"
            ),
            "continuity_with_prior_project_protocol": (
                "the same 5 percentage-point practical non-inferiority threshold was "
                "used in the earlier scalar-stopping preservation evaluation"
            ),
            "selection_disclosure": (
                "5, 3, and 2 percentage-point margins were evaluated before the final "
                "warm-start outcomes; 5 percentage points was the only candidate with "
                "adequate planned power at 47 pairs/task and is also the maximum loss "
                "accepted on substantive grounds"
            ),
            "forbidden_interpretation": "the margin was not chosen because the final outcome passed it",
        },
        "state_allocation": {
            "official_states_per_task": 50,
            "excluded_preflight_states_per_task": 3,
            "included_primary_states_per_task": PAIRS_PER_TASK,
            "excluded_initial_state_ids_by_task": excluded_state_ids,
            "included_initial_state_ids_by_task": included_state_ids,
            "acquisition_plan": {
                "runtime_phases": list(PRIMARY_PHASES),
                "episodes_executed_per_task": 50,
                "episodes_executed_total_both_arms": 1000,
                "primary_analysis_excludes_predeclared_first_three_calibration_states": True,
                "reason": (
                    "reuse the already validated 10/10/30 protocol without modifying "
                    "core state-selection code; exclude the three preflight-observed states"
                ),
            },
        },
        "paired_seed_plan": {
            "base_seed": FINAL_BASE_SEED,
            "phase_names": list(PRIMARY_PHASES),
            "arm_identity_in_seed_derivation": False,
            "fresh_relative_to_preflight_seed_17007": True,
            "seed_domain_note": (
                "existing deterministic namespace is retained; the new base seed and "
                "phase identity create a disjoint final seed domain"
            ),
        },
        "secondary_efficiency_endpoints": {
            "confirmatory_status": "descriptive_secondary",
            "latency_scope": "synchronized online policy-query latency around get_action",
            "metrics": [
                "predictions_per_episode",
                "mean_K_per_prediction",
                "recurrent_calls_per_episode",
                "get_output_calls_per_episode",
                "mean_prediction_latency_ms",
                "summed_inference_time_ms_per_episode",
            ],
            "trajectory_warning": (
                "episode cost differences combine per-query cost, query count, episode "
                "length, and closed-loop trajectory differences"
            ),
        },
        "artifact_provenance": {
            "cutoff_calibration": artifact_ref(args.cutoff_calibration),
            "state_accounting": artifact_ref(args.state_accounting),
            "extended_preflight_validation": artifact_ref(args.extended_preflight_validation),
            "repeatability_audit": artifact_ref(args.repeatability_audit),
        },
        "forbidden_post_freeze_changes": [
            "noninferiority_margin",
            "profile_p_value_cutoff",
            "primary_method",
            "included_state_ids",
            "excluded_preflight_state_ids",
            "warm_start_midpoint_definition",
            "Action-MSE threshold 0.001",
            "maximum recurrence depth 32",
            "num_exec_actions 5",
            "task-specific rescue rules",
            "staleness gates",
            "latent or scalar stopping",
        ],
        "runtime_authorization_requirements": {
            "source_commit_frozen": False,
            "checkpoint_tree_digest_frozen": False,
            "environment_snapshot_frozen": False,
            "final_launcher_validated": False,
            "final_result_analyzer_validated": False,
            "authorized": False,
        },
        "final_run_authorized": False,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote: {args.output}")
    print(
        "Frozen primary analysis: margin=0.05, method=pooled profile likelihood, "
        "p-value cutoff=0.045, primary pairs=47/task (470 total)"
    )
    print(
        "Final acquisition plan: execute 50 states/task over calibration+screening+final, "
        "exclude the predeclared 3 preflight states/task from primary analysis"
    )
    print("Final run authorized: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
