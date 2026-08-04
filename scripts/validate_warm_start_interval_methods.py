#!/usr/bin/env python3
"""Validate candidate paired non-inferiority procedures for warm-start.

The fixed study has ten LIBERO Spatial tasks and at most 47
warm-start-outcome-unseen pairs per task.  This script evaluates candidate
one-sided non-inferiority procedures under homogeneous and task-heterogeneous
paired-binary scenarios.

The simulated pair outcome is D = warm_success - cold_success:
    +1: warm-only success (p01)
    -1: cold-only success (p10)
     0: concordant pair

This is a planning diagnostic.  It never authorizes the final rollout and it
does not select a non-inferiority margin automatically.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Sequence

import numpy as np


SCHEMA_VERSION = 1
TASK_COUNT = 10
DEFAULT_PAIRS_PER_TASK = 47
DEFAULT_MARGINS = (0.05, 0.03, 0.02)
DEFAULT_DISCORDANCE = (0.05, 0.08, 0.10, 0.15)
DEFAULT_POWER_DIFFERENCE = (0.00, -0.01, -0.02)
SUPPORTED_METHODS = (
    "pooled_profile_likelihood",
    "stratified_wald",
    "stratified_percentile_bootstrap",
)


class IntervalValidationError(ValueError):
    """Raised when the planning simulation contract is invalid."""


@dataclass(frozen=True)
class Scenario:
    name: str
    role: str
    margin: float
    task_p01: tuple[float, ...]
    task_p10: tuple[float, ...]
    construction: str

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
        raise IntervalValidationError(message)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def parse_float_list(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    require(values, "at least one numeric value is required")
    require(all(math.isfinite(value) for value in values), "all values must be finite")
    return values


def parse_methods(text: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in text.split(",") if item.strip())
    require(values, "at least one method is required")
    require(len(values) == len(set(values)), "methods must be unique")
    unknown = sorted(set(values) - set(SUPPORTED_METHODS))
    require(not unknown, f"unsupported methods: {unknown}")
    return values


def probabilities_from_q_delta(q: float, delta: float) -> tuple[float, float]:
    require(0.0 <= q <= 1.0, "discordance must lie in [0, 1]")
    require(abs(delta) <= q + 1e-12, "absolute difference cannot exceed discordance")
    p01 = (q + delta) / 2.0
    p10 = (q - delta) / 2.0
    require(p01 >= -1e-12 and p10 >= -1e-12, "derived directional probability is negative")
    require(p01 + p10 <= 1.0 + 1e-12, "derived directional probabilities exceed one")
    return max(0.0, p01), max(0.0, p10)


def scenario_from_q_delta(
    *,
    name: str,
    role: str,
    margin: float,
    q_by_task: Iterable[float],
    delta_by_task: Iterable[float],
    construction: str,
) -> Scenario:
    q_values = tuple(float(value) for value in q_by_task)
    delta_values = tuple(float(value) for value in delta_by_task)
    require(len(q_values) == TASK_COUNT, "scenario must contain ten task discordances")
    require(len(delta_values) == TASK_COUNT, "scenario must contain ten task differences")
    p01 = []
    p10 = []
    for q, delta in zip(q_values, delta_values):
        warm_only, cold_only = probabilities_from_q_delta(q, delta)
        p01.append(warm_only)
        p10.append(cold_only)
    scenario = Scenario(
        name=name,
        role=role,
        margin=margin,
        task_p01=tuple(p01),
        task_p10=tuple(p10),
        construction=construction,
    )
    if role == "null_boundary":
        require(
            abs(scenario.aggregate_difference + margin) <= 1e-12,
            "null-boundary scenario must have aggregate difference = -margin",
        )
    elif role == "power":
        require(
            scenario.aggregate_difference > -margin + 1e-12,
            "power scenario must lie strictly above the non-inferiority boundary",
        )
    else:
        raise IntervalValidationError(f"unsupported scenario role: {role}")
    return scenario


def build_scenarios(
    *,
    margins: Iterable[float],
    discordance_values: Iterable[float],
    power_difference_values: Iterable[float],
) -> list[Scenario]:
    scenarios: list[Scenario] = []
    seen: set[str] = set()

    def add(scenario: Scenario) -> None:
        require(scenario.name not in seen, f"duplicate scenario name: {scenario.name}")
        seen.add(scenario.name)
        scenarios.append(scenario)

    for margin in margins:
        require(0.0 < margin < 1.0, "margins must lie in (0, 1)")
        boundary = -margin
        for q in discordance_values:
            if q + 1e-12 < margin:
                continue

            add(
                scenario_from_q_delta(
                    name=f"null_m{margin:.3f}_q{q:.3f}_homogeneous",
                    role="null_boundary",
                    margin=margin,
                    q_by_task=(q,) * TASK_COUNT,
                    delta_by_task=(boundary,) * TASK_COUNT,
                    construction="homogeneous q and homogeneous boundary difference",
                )
            )

            low_q = max(margin, 0.60 * q)
            high_q = 2.0 * q - low_q
            if high_q <= 1.0 + 1e-12 and abs(high_q - low_q) > 1e-12:
                add(
                    scenario_from_q_delta(
                        name=f"null_m{margin:.3f}_q{q:.3f}_qheterogeneous",
                        role="null_boundary",
                        margin=margin,
                        q_by_task=(low_q,) * 5 + (high_q,) * 5,
                        delta_by_task=(boundary,) * TASK_COUNT,
                        construction=(
                            "five low-discordance and five high-discordance tasks; "
                            "common boundary difference"
                        ),
                    )
                )

            directional_room = q - margin
            amplitude = min(0.01, max(0.0, 0.50 * directional_room))
            if amplitude > 1e-12:
                add(
                    scenario_from_q_delta(
                        name=f"null_m{margin:.3f}_q{q:.3f}_dheterogeneous",
                        role="null_boundary",
                        margin=margin,
                        q_by_task=(q,) * TASK_COUNT,
                        delta_by_task=(boundary + amplitude,) * 5
                        + (boundary - amplitude,) * 5,
                        construction=(
                            "homogeneous discordance; balanced task-specific differences "
                            "around the aggregate null boundary"
                        ),
                    )
                )

            for difference in power_difference_values:
                if difference <= boundary + 1e-12 or abs(difference) > q + 1e-12:
                    continue
                add(
                    scenario_from_q_delta(
                        name=(
                            f"power_m{margin:.3f}_q{q:.3f}_"
                            f"d{difference:+.3f}_homogeneous"
                        ),
                        role="power",
                        margin=margin,
                        q_by_task=(q,) * TASK_COUNT,
                        delta_by_task=(difference,) * TASK_COUNT,
                        construction="homogeneous q and homogeneous true difference",
                    )
                )

    require(scenarios, "scenario grid is empty")
    return scenarios


def simulate_counts(
    scenario: Scenario,
    *,
    pairs_per_task: int,
    outer_replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return replicate x task x {minus, zero, plus} trinomial counts."""

    counts = np.empty((outer_replicates, TASK_COUNT, 3), dtype=np.int32)
    for task_id, (p01, p10) in enumerate(zip(scenario.task_p01, scenario.task_p10)):
        probabilities = np.array([p10, 1.0 - p01 - p10, p01], dtype=np.float64)
        counts[:, task_id, :] = rng.multinomial(
            pairs_per_task,
            probabilities,
            size=outer_replicates,
        )
    return counts


def stratified_wald_decisions(
    counts: np.ndarray,
    *,
    pairs_per_task: int,
    margin: float,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One-sided fixed-task Wald lower bound with within-task sample variance."""

    minus = counts[:, :, 0].astype(np.float64)
    plus = counts[:, :, 2].astype(np.float64)
    sums = plus - minus
    sum_squares = plus + minus
    task_means = sums / pairs_per_task
    estimate = np.mean(task_means, axis=1)

    sample_variance = (
        sum_squares - np.square(sums) / pairs_per_task
    ) / (pairs_per_task - 1)
    sample_variance = np.maximum(sample_variance, 0.0)
    variance_of_fixed_task_mean = np.sum(
        sample_variance / pairs_per_task,
        axis=1,
    ) / float(TASK_COUNT**2)
    standard_error = np.sqrt(np.maximum(variance_of_fixed_task_mean, 0.0))
    z_value = NormalDist().inv_cdf(1.0 - alpha)
    lower_bound = estimate - z_value * standard_error
    return lower_bound > -margin, lower_bound


def empirical_bootstrap_lower_bound(
    task_counts: np.ndarray,
    *,
    pairs_per_task: int,
    alpha: float,
    bootstrap_replicates: int,
    rng: np.random.Generator,
) -> float:
    bootstrap_sums = np.zeros(bootstrap_replicates, dtype=np.int64)
    for task_id in range(TASK_COUNT):
        probabilities = task_counts[task_id].astype(np.float64) / pairs_per_task
        resampled = rng.multinomial(
            pairs_per_task,
            probabilities,
            size=bootstrap_replicates,
        )
        bootstrap_sums += resampled[:, 2] - resampled[:, 0]
    differences = bootstrap_sums / float(TASK_COUNT * pairs_per_task)
    return float(np.quantile(differences, alpha, method="linear"))


def stratified_bootstrap_decisions(
    counts: np.ndarray,
    *,
    pairs_per_task: int,
    margin: float,
    alpha: float,
    bootstrap_replicates: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    outer_replicates = counts.shape[0]
    lower_bounds = np.empty(outer_replicates, dtype=np.float64)
    for replicate in range(outer_replicates):
        lower_bounds[replicate] = empirical_bootstrap_lower_bound(
            counts[replicate],
            pairs_per_task=pairs_per_task,
            alpha=alpha,
            bootstrap_replicates=bootstrap_replicates,
            rng=rng,
        )
    return lower_bounds > -margin, lower_bounds


def multinomial_log_likelihood(counts: tuple[int, int, int], probabilities: tuple[float, float, float]) -> float:
    value = 0.0
    for count, probability in zip(counts, probabilities):
        if count == 0:
            continue
        if probability <= 0.0:
            return -math.inf
        value += count * math.log(probability)
    return value


def saturated_log_likelihood(minus: int, zero: int, plus: int) -> float:
    total = minus + zero + plus
    require(total > 0, "profile likelihood requires at least one pair")
    return multinomial_log_likelihood(
        (minus, zero, plus),
        (minus / total, zero / total, plus / total),
    )


def constrained_q_mle(minus: int, zero: int, plus: int, delta: float) -> float:
    """Maximize paired-trinomial likelihood over q with fixed p01-p10=delta."""

    lower = abs(delta)
    upper = 1.0
    eps = 1e-12

    def derivative(q: float) -> float:
        value = 0.0
        plus_denominator = q + delta
        minus_denominator = q - delta
        zero_denominator = 1.0 - q
        if plus:
            if plus_denominator <= 0.0:
                return math.inf
            value += plus / plus_denominator
        if minus:
            if minus_denominator <= 0.0:
                return math.inf
            value += minus / minus_denominator
        if zero:
            if zero_denominator <= 0.0:
                return -math.inf
            value -= zero / zero_denominator
        return value

    interior_low = min(upper, lower + eps * max(1.0, lower))
    interior_high = max(lower, upper - eps)
    low_derivative = derivative(interior_low)
    high_derivative = derivative(interior_high)

    if low_derivative <= 0.0:
        return lower
    if high_derivative >= 0.0:
        return upper

    lo = interior_low
    hi = interior_high
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if derivative(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def profile_log_likelihood(minus: int, zero: int, plus: int, delta: float) -> float:
    q_hat = constrained_q_mle(minus, zero, plus, delta)
    probabilities = (
        (q_hat - delta) / 2.0,
        1.0 - q_hat,
        (q_hat + delta) / 2.0,
    )
    return multinomial_log_likelihood((minus, zero, plus), probabilities)


def one_sided_profile_p_value(minus: int, zero: int, plus: int, delta0: float) -> float:
    total = minus + zero + plus
    estimate = (plus - minus) / total
    if estimate <= delta0:
        return 1.0
    unrestricted = saturated_log_likelihood(minus, zero, plus)
    constrained = profile_log_likelihood(minus, zero, plus, delta0)
    if not math.isfinite(constrained):
        return 0.0
    likelihood_ratio = max(0.0, 2.0 * (unrestricted - constrained))
    signed_root = math.sqrt(likelihood_ratio)
    return max(0.0, min(1.0, 1.0 - NormalDist().cdf(signed_root)))


def pooled_profile_decisions(
    counts: np.ndarray,
    *,
    margin: float,
    alpha: float,
) -> tuple[np.ndarray, None]:
    aggregate = np.sum(counts, axis=1)
    decisions = np.zeros(aggregate.shape[0], dtype=bool)
    for replicate, row in enumerate(aggregate):
        minus, zero, plus = (int(value) for value in row)
        decisions[replicate] = (
            one_sided_profile_p_value(minus, zero, plus, -margin) < alpha
        )
    return decisions, None


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    require(total > 0, "Wilson interval requires a positive total")
    z_value = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = successes / total
    denominator = 1.0 + z_value * z_value / total
    center = (proportion + z_value * z_value / (2.0 * total)) / denominator
    half_width = (
        z_value
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z_value * z_value / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def method_result(
    *,
    scenario: Scenario,
    method: str,
    decisions: np.ndarray,
    lower_bounds: np.ndarray | None,
    alpha: float,
    size_tolerance: float,
    target_power: float,
) -> dict[str, Any]:
    declarations = int(np.sum(decisions))
    total = int(decisions.size)
    rate = declarations / total
    interval_low, interval_high = wilson_interval(declarations, total)

    if scenario.role == "null_boundary":
        threshold = alpha + size_tolerance
        if interval_high <= threshold:
            status = "size_controlled_by_project_rule"
        elif interval_low > threshold:
            status = "size_liberal_by_project_rule"
        else:
            status = "size_inconclusive"
        metric_name = "type_i_error"
    else:
        if interval_low >= target_power:
            status = "power_adequate"
        elif interval_high < target_power:
            status = "power_inadequate"
        else:
            status = "power_inconclusive"
        metric_name = "power"

    result = {
        "scenario": scenario.name,
        "scenario_role": scenario.role,
        "method": method,
        "margin": scenario.margin,
        "metric": metric_name,
        "estimated_rate": rate,
        "monte_carlo_standard_error": math.sqrt(rate * (1.0 - rate) / total),
        "wilson_95_low": interval_low,
        "wilson_95_high": interval_high,
        "declarations": declarations,
        "outer_replicates": total,
        "status": status,
    }
    if lower_bounds is not None:
        result["mean_lower_bound"] = float(np.mean(lower_bounds))
        result["lower_bound_p05"] = float(np.quantile(lower_bounds, 0.05))
        result["lower_bound_p50"] = float(np.quantile(lower_bounds, 0.50))
        result["lower_bound_p95"] = float(np.quantile(lower_bounds, 0.95))
    return result


def summarize_methods(results: list[dict[str, Any]], methods: Iterable[str]) -> list[dict[str, Any]]:
    summaries = []
    for method in methods:
        rows = [row for row in results if row["method"] == method]
        null_rows = [row for row in rows if row["scenario_role"] == "null_boundary"]
        power_rows = [row for row in rows if row["scenario_role"] == "power"]
        summaries.append(
            {
                "method": method,
                "null_scenario_count": len(null_rows),
                "worst_estimated_type_i_error": (
                    max(row["estimated_rate"] for row in null_rows) if null_rows else None
                ),
                "worst_type_i_wilson_95_high": (
                    max(row["wilson_95_high"] for row in null_rows) if null_rows else None
                ),
                "size_status_counts": {
                    status: sum(1 for row in null_rows if row["status"] == status)
                    for status in (
                        "size_controlled_by_project_rule",
                        "size_liberal_by_project_rule",
                        "size_inconclusive",
                    )
                },
                "power_scenario_count": len(power_rows),
                "minimum_estimated_power": (
                    min(row["estimated_rate"] for row in power_rows) if power_rows else None
                ),
                "power_status_counts": {
                    status: sum(1 for row in power_rows if row["status"] == status)
                    for status in (
                        "power_adequate",
                        "power_inadequate",
                        "power_inconclusive",
                    )
                },
            }
        )
    return summaries


def validate_state_accounting(path: Path, pairs_per_task: int) -> dict[str, Any]:
    payload = load_json(path)
    require(
        payload.get("role") == "planning_state_accounting_not_final_authorization",
        "wrong state-accounting role",
    )
    accounting = payload.get("state_accounting")
    require(isinstance(accounting, dict), "state-accounting payload is incomplete")
    available = int(accounting.get("maximum_warm_start_outcome_unseen_states_per_task", -1))
    require(available == 47, "state accounting must expose exactly 47 outcome-unseen states/task")
    require(
        pairs_per_task <= available,
        f"requested {pairs_per_task} pairs/task exceeds the {available} outcome-unseen states",
    )
    require(payload.get("final_run_authorized") is False, "state accounting unexpectedly authorizes final run")
    return {
        "path": str(path.resolve()),
        "maximum_pairs_per_task": available,
        "final_run_authorized": False,
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
        "--methods",
        default="pooled_profile_likelihood,stratified_wald",
        help=(
            "Comma-separated candidate methods. Bootstrap is opt-in because it "
            "adds an expensive nested simulation."
        ),
    )
    parser.add_argument(
        "--margins",
        default=",".join(str(value) for value in DEFAULT_MARGINS),
    )
    parser.add_argument(
        "--discordance",
        default=",".join(str(value) for value in DEFAULT_DISCORDANCE),
    )
    parser.add_argument(
        "--power-difference",
        default=",".join(str(value) for value in DEFAULT_POWER_DIFFERENCE),
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--size-tolerance", type=float, default=0.01)
    parser.add_argument("--target-power", type=float, default=0.80)
    parser.add_argument("--outer-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17007)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require(not args.output.exists(), f"refusing to overwrite existing output: {args.output}")
    require(args.pairs_per_task >= 2, "pairs-per-task must be at least two")
    require(0.0 < args.alpha < 0.5, "alpha must lie in (0, 0.5)")
    require(args.size_tolerance >= 0.0, "size-tolerance must be non-negative")
    require(0.0 < args.target_power < 1.0, "target-power must lie in (0, 1)")
    require(args.outer_replicates > 0, "outer-replicates must be positive")
    require(args.bootstrap_replicates > 0, "bootstrap-replicates must be positive")

    state_accounting = validate_state_accounting(args.state_accounting, args.pairs_per_task)
    methods = parse_methods(args.methods)
    margins = parse_float_list(args.margins)
    discordance = parse_float_list(args.discordance)
    power_difference = parse_float_list(args.power_difference)
    scenarios = build_scenarios(
        margins=margins,
        discordance_values=discordance,
        power_difference_values=power_difference,
    )

    if "stratified_percentile_bootstrap" in methods:
        work = len(scenarios) * args.outer_replicates * args.bootstrap_replicates
        require(
            work <= 100_000_000,
            "requested nested-bootstrap workload is too large; reduce scenarios, "
            "outer replicates, or bootstrap replicates",
        )

    seed_sequence = np.random.SeedSequence(args.seed)
    scenario_seeds = seed_sequence.spawn(len(scenarios) * 2)
    results: list[dict[str, Any]] = []

    for scenario_index, scenario in enumerate(scenarios):
        data_rng = np.random.default_rng(scenario_seeds[2 * scenario_index])
        bootstrap_rng = np.random.default_rng(scenario_seeds[2 * scenario_index + 1])
        counts = simulate_counts(
            scenario,
            pairs_per_task=args.pairs_per_task,
            outer_replicates=args.outer_replicates,
            rng=data_rng,
        )

        for method in methods:
            if method == "pooled_profile_likelihood":
                decisions, lower_bounds = pooled_profile_decisions(
                    counts,
                    margin=scenario.margin,
                    alpha=args.alpha,
                )
            elif method == "stratified_wald":
                decisions, lower_bounds = stratified_wald_decisions(
                    counts,
                    pairs_per_task=args.pairs_per_task,
                    margin=scenario.margin,
                    alpha=args.alpha,
                )
            elif method == "stratified_percentile_bootstrap":
                decisions, lower_bounds = stratified_bootstrap_decisions(
                    counts,
                    pairs_per_task=args.pairs_per_task,
                    margin=scenario.margin,
                    alpha=args.alpha,
                    bootstrap_replicates=args.bootstrap_replicates,
                    rng=bootstrap_rng,
                )
            else:
                raise AssertionError(method)

            results.append(
                method_result(
                    scenario=scenario,
                    method=method,
                    decisions=decisions,
                    lower_bounds=lower_bounds,
                    alpha=args.alpha,
                    size_tolerance=args.size_tolerance,
                    target_power=args.target_power,
                )
            )

    output = {
        "schema_version": SCHEMA_VERSION,
        "role": "planning_interval_validation_not_final_authorization",
        "study": "midpoint_warm_start_primary_evaluation",
        "state_accounting": state_accounting,
        "analysis_contract": {
            "estimand": "equal-weight mean paired success difference across ten fixed tasks",
            "difference_direction": "warm-start minus cold-initialized adaptive baseline",
            "pairs_per_task": args.pairs_per_task,
            "total_pairs": TASK_COUNT * args.pairs_per_task,
            "alpha_one_sided": args.alpha,
            "target_power": args.target_power,
            "size_project_rule": (
                "Wilson 95% upper bound for simulated type-I error <= alpha + size_tolerance"
            ),
            "size_tolerance": args.size_tolerance,
            "methods": list(methods),
            "outer_replicates": args.outer_replicates,
            "bootstrap_replicates": (
                args.bootstrap_replicates
                if "stratified_percentile_bootstrap" in methods
                else None
            ),
            "seed": args.seed,
            "final_method_selected": False,
            "final_margin_selected": False,
        },
        "important_limitations": [
            "pooled profile likelihood assumes a pooled paired-trinomial working model; task heterogeneity is assessed only through simulation",
            "stratified Wald is asymptotic and may be unstable under sparse discordance",
            "percentile bootstrap is an empirical sensitivity method and is not automatically the primary procedure",
            "scenario validation does not prove calibration outside the simulated probability grid",
        ],
        "scenarios": [asdict(scenario) for scenario in scenarios],
        "results": results,
        "method_summaries": summarize_methods(results, methods),
        "final_run_authorized": False,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote: {args.output}")
    print(
        f"Validated methods={list(methods)}, scenarios={len(scenarios)}, "
        f"outer_replicates={args.outer_replicates}, pairs/task={args.pairs_per_task}"
    )
    for summary in output["method_summaries"]:
        print(
            f"{summary['method']}: "
            f"worst_type_I={summary['worst_estimated_type_i_error']}, "
            f"worst_type_I_upper={summary['worst_type_i_wilson_95_high']}, "
            f"min_power={summary['minimum_estimated_power']}, "
            f"size_status={summary['size_status_counts']}, "
            f"power_status={summary['power_status_counts']}"
        )
    print("Final run authorized: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
