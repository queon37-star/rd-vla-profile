#!/usr/bin/env python3
"""Summarize warm-start interval-validation JSON by method and margin.

This is a reporting utility. It does not select a method, margin, or authorize
any rollout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "JSON root must be an object")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top-null", type=int, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require(args.top_null > 0, "--top-null must be positive")
    payload = load_json(args.input)
    require(
        payload.get("role") == "planning_interval_validation_not_final_authorization",
        "input is not a warm-start interval-validation result",
    )
    scenarios = payload.get("scenarios")
    results = payload.get("results")
    require(isinstance(scenarios, list) and scenarios, "input has no scenarios")
    require(isinstance(results, list) and results, "input has no results")

    scenario_by_name = {
        str(item["name"]): item
        for item in scenarios
        if isinstance(item, dict) and item.get("name") is not None
    }
    methods = sorted({str(row["method"]) for row in results})
    margins = sorted({float(row["margin"]) for row in results}, reverse=True)

    method_rows: dict[str, Any] = {}
    for method in methods:
        rows = [row for row in results if row.get("method") == method]
        null_rows = [row for row in rows if row.get("scenario_role") == "null_boundary"]
        ordered_null = sorted(
            null_rows,
            key=lambda row: (
                float(row.get("estimated_rate", -1.0)),
                float(row.get("wilson_95_high", -1.0)),
            ),
            reverse=True,
        )
        margin_rows: dict[str, Any] = {}
        for margin in margins:
            selected = [row for row in rows if float(row["margin"]) == margin]
            selected_null = [row for row in selected if row["scenario_role"] == "null_boundary"]
            selected_power = [row for row in selected if row["scenario_role"] == "power"]

            power_by_true_difference: dict[str, Any] = {}
            differences = sorted(
                {
                    float(scenario_by_name[row["scenario"]]["task_p01"][0])
                    - float(scenario_by_name[row["scenario"]]["task_p10"][0])
                    for row in selected_power
                    if row["scenario"] in scenario_by_name
                    and len(set(scenario_by_name[row["scenario"]]["task_p01"])) == 1
                    and len(set(scenario_by_name[row["scenario"]]["task_p10"])) == 1
                },
                reverse=True,
            )
            for difference in differences:
                matching = []
                for row in selected_power:
                    scenario = scenario_by_name.get(row["scenario"])
                    if not scenario:
                        continue
                    p01 = tuple(float(value) for value in scenario["task_p01"])
                    p10 = tuple(float(value) for value in scenario["task_p10"])
                    if len(set(p01)) != 1 or len(set(p10)) != 1:
                        continue
                    observed_difference = p01[0] - p10[0]
                    if abs(observed_difference - difference) <= 1e-12:
                        matching.append(row)
                if matching:
                    power_by_true_difference[f"{difference:+.3f}"] = {
                        "minimum_estimated_power_across_q": min(
                            float(row["estimated_rate"]) for row in matching
                        ),
                        "maximum_estimated_power_across_q": max(
                            float(row["estimated_rate"]) for row in matching
                        ),
                        "all_power_adequate": all(
                            row["status"] == "power_adequate" for row in matching
                        ),
                        "rows": len(matching),
                    }

            margin_rows[f"{margin:.3f}"] = {
                "null_scenarios": len(selected_null),
                "worst_estimated_type_i_error": (
                    max(float(row["estimated_rate"]) for row in selected_null)
                    if selected_null
                    else None
                ),
                "worst_type_i_wilson_95_high": (
                    max(float(row["wilson_95_high"]) for row in selected_null)
                    if selected_null
                    else None
                ),
                "all_null_scenarios_size_controlled": bool(selected_null)
                and all(
                    row["status"] == "size_controlled_by_project_rule"
                    for row in selected_null
                ),
                "null_status_counts": {
                    status: sum(1 for row in selected_null if row["status"] == status)
                    for status in (
                        "size_controlled_by_project_rule",
                        "size_liberal_by_project_rule",
                        "size_inconclusive",
                    )
                },
                "homogeneous_power_by_true_difference": power_by_true_difference,
            }

        top_null = []
        for row in ordered_null[: args.top_null]:
            scenario = scenario_by_name.get(row["scenario"], {})
            top_null.append(
                {
                    "scenario": row["scenario"],
                    "margin": row["margin"],
                    "estimated_type_i_error": row["estimated_rate"],
                    "wilson_95": [row["wilson_95_low"], row["wilson_95_high"]],
                    "status": row["status"],
                    "aggregate_difference": (
                        sum(float(x) - float(y) for x, y in zip(
                            scenario.get("task_p01", []),
                            scenario.get("task_p10", []),
                        )) / len(scenario.get("task_p01", []))
                        if scenario.get("task_p01")
                        else None
                    ),
                    "aggregate_discordance": (
                        sum(float(x) + float(y) for x, y in zip(
                            scenario.get("task_p01", []),
                            scenario.get("task_p10", []),
                        )) / len(scenario.get("task_p01", []))
                        if scenario.get("task_p01")
                        else None
                    ),
                    "construction": scenario.get("construction"),
                }
            )

        method_rows[method] = {
            "top_null_scenarios": top_null,
            "by_margin": margin_rows,
            "eligible_only_if_every_null_scenario_controlled": all(
                row["status"] == "size_controlled_by_project_rule"
                for row in null_rows
            ),
        }

    summary = {
        "schema_version": 1,
        "role": "interval_validation_summary_not_final_authorization",
        "source": str(args.input.resolve()),
        "analysis_contract": payload.get("analysis_contract"),
        "methods": method_rows,
        "interpretation_rules": [
            "Development runs do not select a primary method or margin.",
            "A method is eligible only when every prespecified null-boundary scenario is size-controlled by the project rule in the planning-grade run.",
            "Minimum power over all alternatives is not a useful standalone decision metric; inspect power separately by margin and assumed true difference.",
            "The non-inferiority margin must be justified scientifically and cannot be chosen only because it attains power.",
        ],
        "final_run_authorized": False,
    }

    if args.output:
        require(not args.output.exists(), f"refusing to overwrite: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote: {args.output}")

    for method, item in method_rows.items():
        print(method)
        for row in item["top_null_scenarios"]:
            print(
                "  null "
                f"{row['scenario']}: rate={row['estimated_type_i_error']:.4f}, "
                f"Wilson95=[{row['wilson_95'][0]:.4f}, {row['wilson_95'][1]:.4f}], "
                f"status={row['status']}"
            )
        for margin, row in item["by_margin"].items():
            print(
                f"  margin={margin}: worst_size={row['worst_estimated_type_i_error']}, "
                f"all_size_controlled={row['all_null_scenarios_size_controlled']}, "
                f"status={row['null_status_counts']}, "
                f"power={row['homogeneous_power_by_true_difference']}"
            )
    print("Final run authorized: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
