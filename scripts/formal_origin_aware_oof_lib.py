"""Formal multi-scenario OOF replay for the frozen origin-aware calibration set."""

from __future__ import annotations

import copy
from dataclasses import asdict
from typing import Any, Dict, Mapping, Sequence

from scripts.origin_aware_replay_lib import (
    CONFIRMATION_MODES,
    FIXED_WARM_THRESHOLDS,
    MAX_SKIP_VALUES,
    WARM_QUANTILES,
    CostModel,
    SchedulerConfig,
    SelectionConstraints,
    ShadowPrediction,
    aggregate_evaluations,
    evaluate_replay,
    run_task_level_oof_selection,
)


class FormalOOFValidationError(ValueError):
    """Raised when a formal OOF input or result violates the frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FormalOOFValidationError(message)


def validate_cost_sensitivity_manifest(manifest: Mapping[str, Any]) -> list[Dict[str, Any]]:
    """Validate and normalize the frozen cost-sensitivity scenarios."""

    _require(manifest.get("schema_version") == 1, "unsupported cost manifest schema")
    _require(
        manifest.get("scope") == "baseline-conditioned offline pruning only",
        "cost manifest scope mismatch",
    )
    raw_scenarios = manifest.get("scenarios")
    _require(isinstance(raw_scenarios, list) and raw_scenarios, "cost scenarios are required")
    scenarios = []
    identifiers = set()
    primary_count = 0
    candidate_favorable_count = 0
    for index, raw in enumerate(raw_scenarios):
        _require(isinstance(raw, Mapping), f"cost scenario {index} must be an object")
        identifier = raw.get("id")
        _require(isinstance(identifier, str) and identifier, f"cost scenario {index} has no id")
        _require(identifier not in identifiers, f"duplicate cost scenario id: {identifier}")
        identifiers.add(identifier)
        _require(isinstance(raw.get("description"), str), f"{identifier}: description is required")
        _require(isinstance(raw.get("primary"), bool), f"{identifier}: primary must be boolean")
        _require(
            isinstance(raw.get("candidate_favorable"), bool),
            f"{identifier}: candidate_favorable must be boolean",
        )
        primary_count += int(raw["primary"])
        candidate_favorable_count += int(raw["candidate_favorable"])
        cost_mapping = raw.get("cost_model")
        _require(isinstance(cost_mapping, Mapping), f"{identifier}: cost_model is required")
        try:
            costs = CostModel(**dict(cost_mapping))
        except (TypeError, ValueError) as exc:
            raise FormalOOFValidationError(f"{identifier}: invalid cost model: {exc}") from exc
        scenarios.append(
            {
                "id": identifier,
                "description": raw["description"],
                "primary": raw["primary"],
                "candidate_favorable": raw["candidate_favorable"],
                "costs": costs,
            }
        )
    _require(primary_count == 1, "cost manifest must define exactly one primary scenario")
    _require(candidate_favorable_count >= 1, "at least one candidate-favorable scenario is required")

    gate = manifest.get("promotion_gate")
    _require(isinstance(gate, Mapping), "promotion_gate is required")
    minimum = gate.get("min_action_head_improvement")
    _require(
        isinstance(minimum, (int, float))
        and not isinstance(minimum, bool)
        and 0 < float(minimum) < 1,
        "promotion gate must be a fraction strictly between zero and one",
    )
    return scenarios


def _add_full_data_refit_metrics(
    selection: Dict[str, Any],
    predictions: Sequence[ShadowPrediction],
    costs: CostModel,
    constraints: SelectionConstraints,
) -> None:
    """Attach diagnostics for each refit config without treating them as OOF estimates."""

    for selected in selection["selected_refit_configs"]:
        config_mapping = selected["config"]
        config = SchedulerConfig(
            warm_threshold=float(config_mapping["warm_threshold"]),
            max_skip_iters=int(config_mapping["max_skip_iters"]),
            confirmation_mode=config_mapping["confirmation_mode"],
            cold_threshold=float(config_mapping["cold_threshold"]),
        )
        evaluations = [evaluate_replay(prediction, config, costs) for prediction in predictions]
        selected["full_data_refit_metrics_diagnostic"] = aggregate_evaluations(
            evaluations, constraints
        )


def run_formal_cost_sensitivity_oof(
    predictions: Sequence[ShadowPrediction],
    fold_manifest: Mapping[str, Any],
    cost_manifest: Mapping[str, Any],
    *,
    constraints: SelectionConstraints = SelectionConstraints(),
    top_n: int = 6,
) -> Dict[str, Any]:
    """Run the fixed family grid under every cost scenario and make a microbenchmark shortlist."""

    _require(len(predictions) > 0, "formal OOF requires predictions")
    scenarios = validate_cost_sensitivity_manifest(cost_manifest)
    scenario_reports = []
    passing_family_sets = []
    for scenario in scenarios:
        selection = run_task_level_oof_selection(
            predictions,
            fold_manifest,
            scenario["costs"],
            constraints=constraints,
            fixed_thresholds=FIXED_WARM_THRESHOLDS,
            quantiles=WARM_QUANTILES,
            max_skip_values=MAX_SKIP_VALUES,
            confirmation_modes=CONFIRMATION_MODES,
            top_n=top_n,
        )
        _add_full_data_refit_metrics(selection, predictions, scenario["costs"], constraints)
        passing_ids = {
            report["family_id"]
            for report in selection["family_reports"]
            if report["oof_metrics"]["passes_safety_constraints"]
        }
        passing_family_sets.append(passing_ids)
        scenario_reports.append(
            {
                "id": scenario["id"],
                "description": scenario["description"],
                "primary": scenario["primary"],
                "candidate_favorable": scenario["candidate_favorable"],
                "selection": selection,
            }
        )
    first_passing = passing_family_sets[0]
    _require(
        all(families == first_passing for families in passing_family_sets[1:]),
        "safety-passing family set changed across cost-only scenarios",
    )

    primary = next(report for report in scenario_reports if report["primary"])
    shortlist = copy.deepcopy(primary["selection"]["selected_refit_configs"])
    _require(
        len(shortlist) == top_n,
        f"formal OOF requires {top_n} distinct safety-passing refit configs",
    )
    for candidate in shortlist:
        candidate["status"] = "gpu_schedule_microbenchmark_required"

    gate = float(cost_manifest["promotion_gate"]["min_action_head_improvement"])
    sensitivity_summary = []
    for report in scenario_reports:
        selected = report["selection"]["selected_refit_configs"]
        best = max(
            selected,
            key=lambda item: item["oof_metrics"]["predicted_action_head_improvement"],
        )
        improvement = float(best["oof_metrics"]["predicted_action_head_improvement"])
        sensitivity_summary.append(
            {
                "scenario_id": report["id"],
                "primary": report["primary"],
                "candidate_favorable": report["candidate_favorable"],
                "best_family_id": best["source_family_id"],
                "best_predicted_variable_action_head_improvement": improvement,
                "meets_5pct_model_gate": improvement >= gate,
            }
        )
    favorable_best = max(
        item["best_predicted_variable_action_head_improvement"]
        for item in sensitivity_summary
        if item["candidate_favorable"]
    )
    return {
        "schema_version": 1,
        "scope": (
            "OOF estimates are conditional on baseline observations and incoming midpoint caches. "
            "Cost scenarios rank candidates for GPU microbenchmarking only."
        ),
        "family_grid": {
            "fixed_warm_thresholds": list(FIXED_WARM_THRESHOLDS),
            "warm_quantiles": list(WARM_QUANTILES),
            "max_skip_values": list(MAX_SKIP_VALUES),
            "confirmation_modes": list(CONFIRMATION_MODES),
            "size": primary["selection"]["family_grid_size"],
        },
        "constraints": asdict(constraints),
        "passing_family_count": len(first_passing),
        "primary_scenario_id": primary["id"],
        "microbenchmark_shortlist": shortlist,
        "microbenchmark_shortlist_count": len(shortlist),
        "sensitivity_summary": sensitivity_summary,
        "tested_candidate_favorable_best_improvement": favorable_best,
        "linear_model_5pct_gate_met": favorable_best >= gate,
        "online_screening_allowed": False,
        "next_required_gate": (
            "Run the exact GPU action-head schedule microbenchmark. Online screening remains "
            "disallowed until measured action-head and converted E2E pruning criteria pass."
        ),
        "scenario_reports": scenario_reports,
    }
