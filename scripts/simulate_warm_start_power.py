#!/usr/bin/env python3
"""Plan paired non-inferiority power for the warm-start primary evaluation.

The simulated paired outcome is D = warm_success - baseline_success:
    +1: warm-only success (p01)
    -1: baseline-only success (p10)
     0: concordant pair

Power is estimated for the same fixed-task analysis planned for the final study:
a pooled paired success difference with a task-stratified percentile bootstrap
that resamples pairs within each of the ten fixed LIBERO Spatial tasks.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


SCHEMA_VERSION = 1
DEFAULT_PAIRS_PER_TASK = (10, 20, 30, 40, 50)
DEFAULT_DISCORDANCE = (0.03, 0.05, 0.08, 0.10)
DEFAULT_TRUE_DIFFERENCE = (0.00, -0.01, -0.02)
DEFAULT_MARGINS = (0.05, 0.03, 0.02)


class PowerPlanningError(ValueError):
    """Raised when a requested power scenario is internally inconsistent."""


@dataclass(frozen=True)
class Scenario:
    name: str
    task_p01: tuple[float, ...]
    task_p10: tuple[float, ...]
    source: str

    @property
    def task_count(self) -> int:
        return len(self.task_p01)

    @property
    def aggregate_p01(self) -> float:
        return float(np.mean(self.task_p01))

    @property
    def aggregate_p10(self) -> float:
        return float(np.mean(self.task_p10))

    @property
    def aggregate_difference(self) -> float:
        return self.aggregate_p01 - self.aggregate_p10

    @property
    def aggregate_discordance(self) -> float:
        return self.aggregate_p01 + self.aggregate_p10


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PowerPlanningError(message)


def parse_float_list(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    require(bool(values), "at least one numeric value is required")
    require(all(math.isfinite(value) for value in values), "all values must be finite")
    return values


def parse_int_list(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    require(bool(values), "at least one integer value is required")
    require(all(value > 0 for value in values), "all integer values must be positive")
    return values


def homogeneous_scenarios(
    *,
    task_count: int,
    discordance_values: Iterable[float],
    true_difference_values: Iterable[float],
) -> list[Scenario]:
    scenarios: list[Scenario] = []
    for discordance in discordance_values:
        require(0.0 <= discordance <= 1.0, "discordance must lie in [0, 1]")
        for difference in true_difference_values:
            require(
                abs(difference) <= discordance + 1e-12,
                "absolute true difference cannot exceed total discordance",
            )
            p01 = (discordance + difference) / 2.0
            p10 = (discordance - difference) / 2.0
            require(p01 >= 0.0 and p10 >= 0.0, "derived directional discordance is negative")
            require(p01 + p10 <= 1.0 + 1e-12, "derived probabilities exceed one")
            scenarios.append(
                Scenario(
                    name=f"homogeneous_q{discordance:.3f}_d{difference:+.3f}",
                    task_p01=(p01,) * task_count,
                    task_p10=(p10,) * task_count,
                    source="homogeneous_grid",
                )
            )
    return scenarios


def load_profile_scenarios(path: Path) -> list[Scenario]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), "profile JSON root must be an object")
    entries = payload.get("scenarios")
    require(isinstance(entries, list) and entries, "profile JSON must contain a non-empty scenarios list")

    scenarios: list[Scenario] = []
    for index, entry in enumerate(entries):
        require(isinstance(entry, dict), f"scenario {index} must be an object")
        tasks = entry.get("tasks")
        require(isinstance(tasks, list) and tasks, f"scenario {index} must contain tasks")
        ordered = sorted(tasks, key=lambda item: int(item["task_id"]))
        task_ids = [int(item["task_id"]) for item in ordered]
        require(task_ids == list(range(len(task_ids))), f"scenario {index} task IDs must be contiguous from zero")
        p01 = tuple(float(item["p01"]) for item in ordered)
        p10 = tuple(float(item["p10"]) for item in ordered)
        for task_id, warm_only, baseline_only in zip(task_ids, p01, p10):
            require(
                math.isfinite(warm_only) and math.isfinite(baseline_only),
                f"scenario {index} task {task_id} has a non-finite probability",
            )
            require(
                warm_only >= 0.0 and baseline_only >= 0.0 and warm_only + baseline_only <= 1.0,
                f"scenario {index} task {task_id} has invalid directional discordance",
            )
        scenarios.append(
            Scenario(
                name=str(entry.get("name") or f"profile_{index}"),
                task_p01=p01,
                task_p10=p10,
                source=str(path),
            )
        )
    return scenarios


def simulate_study(scenario: Scenario, pairs_per_task: int, rng: np.random.Generator) -> np.ndarray:
    """Return a task x pair matrix with values in {-1, 0, +1}."""

    outcomes = np.zeros((scenario.task_count, pairs_per_task), dtype=np.int8)
    for task_id, (p01, p10) in enumerate(zip(scenario.task_p01, scenario.task_p10)):
        draws = rng.random(pairs_per_task)
        outcomes[task_id, draws < p01] = 1
        outcomes[task_id, (draws >= p01) & (draws < p01 + p10)] = -1
    return outcomes


def stratified_bootstrap_lower_bound(
    outcomes: np.ndarray,
    *,
    alpha: float,
    bootstrap_replicates: int,
    rng: np.random.Generator,
) -> float:
    """Percentile lower bound from resampling pairs independently within each fixed task.

    Multinomial resampling of the empirical {-1, 0, +1} counts is equivalent
    to drawing pair indices with replacement, but is much faster.
    """

    require(outcomes.ndim == 2, "outcomes must be a task x pair matrix")
    task_count, pairs_per_task = outcomes.shape
    require(task_count > 0 and pairs_per_task > 0, "outcomes cannot be empty")

    bootstrap_sums = np.zeros(bootstrap_replicates, dtype=np.int64)
    for task_id in range(task_count):
        row = outcomes[task_id]
        minus_count = int(np.sum(row == -1))
        plus_count = int(np.sum(row == 1))
        zero_count = pairs_per_task - minus_count - plus_count
        probabilities = np.array(
            [minus_count, zero_count, plus_count], dtype=np.float64
        ) / pairs_per_task
        resampled = rng.multinomial(
            pairs_per_task, probabilities, size=bootstrap_replicates
        )
        bootstrap_sums += resampled[:, 2] - resampled[:, 0]

    bootstrap_differences = bootstrap_sums / float(task_count * pairs_per_task)
    return float(np.quantile(bootstrap_differences, alpha, method="linear"))


def estimate_power(
    scenario: Scenario,
    *,
    pairs_per_task: int,
    margin: float,
    alpha: float,
    outer_replicates: int,
    bootstrap_replicates: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    passed = 0
    observed_differences = np.empty(outer_replicates, dtype=np.float64)
    lower_bounds = np.empty(outer_replicates, dtype=np.float64)

    for replicate in range(outer_replicates):
        outcomes = simulate_study(scenario, pairs_per_task, rng)
        observed_differences[replicate] = float(np.mean(outcomes))
        lower_bounds[replicate] = stratified_bootstrap_lower_bound(
            outcomes,
            alpha=alpha,
            bootstrap_replicates=bootstrap_replicates,
            rng=rng,
        )
        if lower_bounds[replicate] > -margin:
            passed += 1

    power = passed / outer_replicates
    mc_standard_error = math.sqrt(power * (1.0 - power) / outer_replicates)
    return {
        "pairs_per_task": pairs_per_task,
        "total_pairs": scenario.task_count * pairs_per_task,
        "margin": margin,
        "estimated_power": power,
        "monte_carlo_standard_error": mc_standard_error,
        "mean_observed_difference": float(np.mean(observed_differences)),
        "mean_one_sided_lower_bound": float(np.mean(lower_bounds)),
        "lower_bound_p05": float(np.quantile(lower_bounds, 0.05)),
        "lower_bound_p50": float(np.quantile(lower_bounds, 0.50)),
        "lower_bound_p95": float(np.quantile(lower_bounds, 0.95)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-count", type=int, default=10)
    parser.add_argument(
        "--pairs-per-task",
        default=",".join(str(value) for value in DEFAULT_PAIRS_PER_TASK),
    )
    parser.add_argument(
        "--discordance",
        default=",".join(str(value) for value in DEFAULT_DISCORDANCE),
        help="Aggregate p01+p10 values for homogeneous scenarios.",
    )
    parser.add_argument(
        "--true-difference",
        default=",".join(str(value) for value in DEFAULT_TRUE_DIFFERENCE),
        help="Warm-minus-baseline success differences for homogeneous scenarios.",
    )
    parser.add_argument(
        "--margins",
        default=",".join(str(value) for value in DEFAULT_MARGINS),
        help="Positive non-inferiority margins; the lower-bound gate is > -margin.",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--target-power", type=float, default=0.80)
    parser.add_argument("--outer-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--profile-json",
        type=Path,
        help="Optional explicit task-level p01/p10 scenarios. When supplied, the homogeneous grid is skipped.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require(args.task_count > 0, "task-count must be positive")
    require(0.0 < args.alpha < 0.5, "alpha must lie in (0, 0.5)")
    require(0.0 < args.target_power < 1.0, "target-power must lie in (0, 1)")
    require(args.outer_replicates > 0, "outer-replicates must be positive")
    require(args.bootstrap_replicates > 0, "bootstrap-replicates must be positive")

    pairs_per_task_values = parse_int_list(args.pairs_per_task)
    require(
        all(value <= 50 for value in pairs_per_task_values),
        "pairs-per-task > 50 reuses official initial states and requires a clustered repeated-state design not modeled by this script",
    )
    margins = parse_float_list(args.margins)
    require(all(0.0 < margin < 1.0 for margin in margins), "margins must lie in (0, 1)")

    if args.profile_json is not None:
        scenarios = load_profile_scenarios(args.profile_json)
        require(
            all(scenario.task_count == args.task_count for scenario in scenarios),
            "profile task count does not match --task-count",
        )
    else:
        scenarios = homogeneous_scenarios(
            task_count=args.task_count,
            discordance_values=parse_float_list(args.discordance),
            true_difference_values=parse_float_list(args.true_difference),
        )

    rng = np.random.default_rng(args.seed)
    results: list[dict[str, Any]] = []
    required: list[dict[str, Any]] = []

    for scenario in scenarios:
        for margin in margins:
            margin_rows = []
            for pairs_per_task in pairs_per_task_values:
                row = estimate_power(
                    scenario,
                    pairs_per_task=pairs_per_task,
                    margin=margin,
                    alpha=args.alpha,
                    outer_replicates=args.outer_replicates,
                    bootstrap_replicates=args.bootstrap_replicates,
                    rng=rng,
                )
                margin_rows.append(row)
                results.append(
                    {
                        "scenario": scenario.name,
                        "scenario_source": scenario.source,
                        "aggregate_p01": scenario.aggregate_p01,
                        "aggregate_p10": scenario.aggregate_p10,
                        "aggregate_true_difference": scenario.aggregate_difference,
                        "aggregate_discordance": scenario.aggregate_discordance,
                        **row,
                    }
                )

            passing = [
                row for row in margin_rows if row["estimated_power"] >= args.target_power
            ]
            required.append(
                {
                    "scenario": scenario.name,
                    "margin": margin,
                    "target_power": args.target_power,
                    "required_pairs_per_task": (
                        min(row["pairs_per_task"] for row in passing)
                        if passing
                        else None
                    ),
                    "required_total_pairs": (
                        min(row["total_pairs"] for row in passing)
                        if passing
                        else None
                    ),
                    "status": (
                        "reached_with_unique_official_states"
                        if passing
                        else "not_reached_with_pairs_per_task_at_most_50"
                    ),
                }
            )

    output = {
        "schema_version": SCHEMA_VERSION,
        "analysis": {
            "estimand": "paired success difference = warm-start - cold-initialized baseline",
            "primary_gate": "one-sided task-stratified percentile-bootstrap lower bound > -margin",
            "task_population": "ten fixed LIBERO Spatial tasks",
            "resampling": "pairs resampled within task; tasks are not resampled",
            "alpha": args.alpha,
            "target_power": args.target_power,
            "outer_replicates": args.outer_replicates,
            "bootstrap_replicates": args.bootstrap_replicates,
            "seed": args.seed,
            "p01_definition": "baseline failure, warm-start success",
            "p10_definition": "baseline success, warm-start failure",
            "unique_official_state_limit_per_task": 50,
        },
        "scenarios": [
            {
                **asdict(scenario),
                "aggregate_p01": scenario.aggregate_p01,
                "aggregate_p10": scenario.aggregate_p10,
                "aggregate_true_difference": scenario.aggregate_difference,
                "aggregate_discordance": scenario.aggregate_discordance,
            }
            for scenario in scenarios
        ],
        "results": results,
        "required_sample_sizes": required,
    }

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote: {args.output}")
    for item in required:
        print(
            f"{item['scenario']} margin={item['margin']:.3f}: "
            f"pairs/task={item['required_pairs_per_task']} "
            f"status={item['status']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
