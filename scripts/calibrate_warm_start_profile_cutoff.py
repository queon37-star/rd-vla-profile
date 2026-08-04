#!/usr/bin/env python3
"""Calibrate a conservative pooled profile-likelihood cutoff for warm-start NI.

This planning-only script fixes the non-inferiority margin at 5 percentage
points and evaluates a predeclared grid of profile-likelihood p-value cutoffs.
It uses the same 10 fixed LIBERO Spatial tasks and 47 outcome-unseen pairs/task
as the planned final study.  It never authorizes a rollout or selects the
substantive margin automatically.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from validate_warm_start_interval_methods import (
    build_scenarios,
    one_sided_profile_p_value,
    parse_float_list,
    require,
    simulate_counts,
    validate_state_accounting,
    wilson_interval,
)


MARGIN = 0.05
TASK_COUNT = 10
DEFAULT_PAIRS_PER_TASK = 47
DEFAULT_DISCORDANCE = (0.05, 0.08, 0.10, 0.15)
DEFAULT_POWER_DIFFERENCE = (0.00, -0.01)
DEFAULT_CUTOFFS = (0.030, 0.035, 0.040, 0.045, 0.050)
TARGET_TYPE_I = 0.05
TYPE_I_WILSON_UPPER_LIMIT = 0.06
TARGET_POWER = 0.80


def profile_p_values(counts: np.ndarray, *, margin: float) -> np.ndarray:
    aggregate = np.sum(counts, axis=1)
    values = np.empty(aggregate.shape[0], dtype=np.float64)
    for index, row in enumerate(aggregate):
        minus, zero, plus = (int(value) for value in row)
        values[index] = one_sided_profile_p_value(minus, zero, plus, -margin)
    return values


def rate_record(decisions: np.ndarray) -> dict[str, Any]:
    declarations = int(np.sum(decisions))
    total = int(decisions.size)
    rate = declarations / total
    low, high = wilson_interval(declarations, total)
    return {
        "declarations": declarations,
        "outer_replicates": total,
        "estimated_rate": rate,
        "monte_carlo_standard_error": math.sqrt(rate * (1.0 - rate) / total),
        "wilson_95_low": low,
        "wilson_95_high": high,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--state-accounting",
        type=Path,
        default=Path("benchmark_results/warm_start_primary/state_accounting_v1.json"),
    )
    parser.add_argument("--pairs-per-task", type=int, default=DEFAULT_PAIRS_PER_TASK)
    parser.add_argument(
        "--discordance",
        default=",".join(str(value) for value in DEFAULT_DISCORDANCE),
    )
    parser.add_argument(
        "--power-difference",
        default=",".join(str(value) for value in DEFAULT_POWER_DIFFERENCE),
    )
    parser.add_argument(
        "--cutoffs",
        default=",".join(str(value) for value in DEFAULT_CUTOFFS),
    )
    parser.add_argument("--outer-replicates", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=37007)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require(not args.output.exists(), f"refusing to overwrite existing output: {args.output}")
    require(args.outer_replicates > 0, "outer-replicates must be positive")
    state_accounting = validate_state_accounting(args.state_accounting, args.pairs_per_task)

    cutoffs = tuple(sorted(parse_float_list(args.cutoffs)))
    require(all(0.0 < value <= 0.05 for value in cutoffs), "cutoffs must lie in (0, 0.05]")
    discordance = parse_float_list(args.discordance)
    power_difference = parse_float_list(args.power_difference)
    require(0.0 in power_difference, "power scenarios must include true difference 0")

    scenarios = build_scenarios(
        margins=(MARGIN,),
        discordance_values=discordance,
        power_difference_values=power_difference,
    )
    seed_sequence = np.random.SeedSequence(args.seed)
    scenario_seeds = seed_sequence.spawn(len(scenarios))

    scenario_rows: list[dict[str, Any]] = []
    for index, scenario in enumerate(scenarios):
        counts = simulate_counts(
            scenario,
            pairs_per_task=args.pairs_per_task,
            outer_replicates=args.outer_replicates,
            rng=np.random.default_rng(scenario_seeds[index]),
        )
        p_values = profile_p_values(counts, margin=MARGIN)
        for cutoff in cutoffs:
            row = {
                "scenario": scenario.name,
                "scenario_role": scenario.role,
                "construction": scenario.construction,
                "aggregate_discordance": scenario.aggregate_discordance,
                "aggregate_true_difference": scenario.aggregate_difference,
                "cutoff": cutoff,
                **rate_record(p_values < cutoff),
            }
            if scenario.role == "null_boundary":
                row["qualification"] = (
                    "null_qualified"
                    if row["wilson_95_high"] <= TYPE_I_WILSON_UPPER_LIMIT
                    else "null_not_qualified"
                )
            elif abs(scenario.aggregate_difference) <= 1e-12:
                row["qualification"] = (
                    "zero_effect_power_qualified"
                    if row["wilson_95_low"] >= TARGET_POWER
                    else "zero_effect_power_not_qualified"
                )
            else:
                row["qualification"] = "sensitivity_only"
            scenario_rows.append(row)

    cutoff_summaries: list[dict[str, Any]] = []
    for cutoff in cutoffs:
        rows = [row for row in scenario_rows if row["cutoff"] == cutoff]
        null_rows = [row for row in rows if row["scenario_role"] == "null_boundary"]
        zero_rows = [
            row
            for row in rows
            if row["scenario_role"] == "power"
            and abs(row["aggregate_true_difference"]) <= 1e-12
        ]
        minus_one_rows = [
            row
            for row in rows
            if row["scenario_role"] == "power"
            and abs(row["aggregate_true_difference"] + 0.01) <= 1e-12
        ]
        all_null_qualified = all(row["qualification"] == "null_qualified" for row in null_rows)
        all_zero_power_qualified = all(
            row["qualification"] == "zero_effect_power_qualified" for row in zero_rows
        )
        cutoff_summaries.append(
            {
                "cutoff": cutoff,
                "all_null_scenarios_qualified": all_null_qualified,
                "all_zero_effect_power_scenarios_qualified": all_zero_power_qualified,
                "cutoff_qualified": all_null_qualified and all_zero_power_qualified,
                "worst_null_estimated_type_i": max(row["estimated_rate"] for row in null_rows),
                "worst_null_wilson_95_high": max(row["wilson_95_high"] for row in null_rows),
                "minimum_zero_effect_power": min(row["estimated_rate"] for row in zero_rows),
                "minimum_zero_effect_power_wilson_95_low": min(row["wilson_95_low"] for row in zero_rows),
                "minimum_minus_one_point_power": (
                    min(row["estimated_rate"] for row in minus_one_rows)
                    if minus_one_rows
                    else None
                ),
            }
        )

    qualified = [row for row in cutoff_summaries if row["cutoff_qualified"]]
    largest_qualified = max((row["cutoff"] for row in qualified), default=None)
    payload = {
        "schema_version": 1,
        "role": "planning_profile_cutoff_calibration_not_final_authorization",
        "study": "midpoint_warm_start_primary_evaluation",
        "state_accounting": state_accounting,
        "analysis_contract": {
            "estimand": "equal-weight mean paired success difference across ten fixed tasks",
            "non_inferiority_margin": MARGIN,
            "margin_status": "candidate_only_substantive_justification_still_required",
            "method": "pooled paired-trinomial profile-likelihood test",
            "candidate_p_value_cutoffs": list(cutoffs),
            "target_type_i_error": TARGET_TYPE_I,
            "null_qualification_rule": (
                "Wilson 95% upper bound for simulated type-I error <= 0.06 "
                "for every predeclared null scenario"
            ),
            "target_power": TARGET_POWER,
            "power_qualification_rule": (
                "Wilson 95% lower bound for power >= 0.80 for every true-difference-0 scenario"
            ),
            "minus_one_point_power": "reported as sensitivity, not a qualification gate",
            "pairs_per_task": args.pairs_per_task,
            "total_pairs": TASK_COUNT * args.pairs_per_task,
            "outer_replicates": args.outer_replicates,
            "seed": args.seed,
        },
        "scenario_results": scenario_rows,
        "cutoff_summaries": cutoff_summaries,
        "largest_qualified_cutoff": largest_qualified,
        "cutoff_candidate_available": largest_qualified is not None,
        "important_limitations": [
            "cutoff calibration is valid only for the predeclared simulation grid",
            "pooled profile likelihood remains a working model under task heterogeneity",
            "margin 5 percentage points is not justified merely because it is statistically feasible",
            "preflight outcomes are excluded from the final primary analysis",
        ],
        "final_run_authorized": False,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote: {args.output}")
    for row in cutoff_summaries:
        print(
            f"cutoff={row['cutoff']:.3f}: qualified={row['cutoff_qualified']}, "
            f"worst_type_I={row['worst_null_estimated_type_i']:.4f}, "
            f"worst_upper={row['worst_null_wilson_95_high']:.4f}, "
            f"min_power_d0={row['minimum_zero_effect_power']:.4f}, "
            f"min_power_d-0.01={row['minimum_minus_one_point_power']:.4f}"
        )
    print(f"Largest qualified cutoff: {largest_qualified}")
    print("Final run authorized: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
