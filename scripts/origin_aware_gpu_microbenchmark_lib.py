"""Validation, ordering, and statistics for saved-workload GPU schedule replay."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


BASELINE_CONDITION_ID = "clean_warm_only"
SUPPORTED_SCOPES = ("actual_warm", "all_sampled", "cold")


class GPUMicrobenchmarkValidationError(ValueError):
    """Raised when a benchmark input or measured block violates the frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GPUMicrobenchmarkValidationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path: Path) -> dict[str, Any]:
    source = Path(path)
    _require(source.is_file(), f"missing JSON input: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GPUMicrobenchmarkValidationError(f"invalid JSON input {source}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON root must be an object: {source}")
    return value


@dataclass(frozen=True)
class BenchmarkCondition:
    condition_id: str
    kind: str
    rank: int | None = None
    warm_threshold: float | None = None
    cold_threshold: float = 0.2
    max_skip_iters: int | None = None
    confirmation_mode: str | None = None

    @property
    def exact_key(self) -> tuple[Any, ...]:
        if self.kind == "baseline":
            return (self.kind,)
        return (
            self.kind,
            float(self.warm_threshold).hex(),
            float(self.cold_threshold).hex(),
            int(self.max_skip_iters),
            self.confirmation_mode,
        )


def validate_protocol_manifest(manifest: Mapping[str, Any]) -> None:
    _require(manifest.get("schema_version") == 1, "unsupported protocol schema version")
    _require(manifest.get("device") == "cuda:0", "formal protocol device must be cuda:0")
    for field, minimum in (
        ("seed", 0),
        ("warmup_rounds_per_origin", 1),
        ("measurement_repeats", 3),
    ):
        value = manifest.get(field)
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
            f"protocol field {field!r} must be an integer >= {minimum}",
        )
    timer = manifest.get("timer")
    _require(isinstance(timer, Mapping), "protocol timer must be an object")
    _require(
        timer.get("kind") == "cpu_wall_clock_outer_cuda_sync",
        "formal timer must use CPU wall-clock with outer CUDA synchronization",
    )
    aggregation = manifest.get("aggregation")
    _require(isinstance(aggregation, Mapping), "protocol aggregation must be an object")
    _require(
        aggregation.get("primary_scope") == "actual_warm",
        "formal primary scope must be actual_warm",
    )
    bootstrap = manifest.get("bootstrap")
    _require(isinstance(bootstrap, Mapping), "protocol bootstrap must be an object")
    _require(
        isinstance(bootstrap.get("draws"), int) and bootstrap["draws"] >= 1000,
        "formal bootstrap requires at least 1000 draws",
    )
    alpha = bootstrap.get("alpha")
    _require(
        isinstance(alpha, (int, float)) and not isinstance(alpha, bool) and 0 < alpha < 0.5,
        "bootstrap alpha must be in (0, 0.5)",
    )
    promotion = manifest.get("promotion")
    _require(isinstance(promotion, Mapping), "protocol promotion must be an object")
    gate = promotion.get("minimum_primary_action_head_improvement")
    _require(
        isinstance(gate, (int, float)) and not isinstance(gate, bool) and 0 < gate < 1,
        "promotion improvement gate must be in (0, 1)",
    )
    _require(
        promotion.get("require_simultaneous_one_sided_lower_bound") is True,
        "formal promotion must require the simultaneous lower bound",
    )
    _require(
        promotion.get("require_zero_schedule_mismatches") is True,
        "formal promotion must require zero schedule mismatches",
    )


def conditions_from_shortlist(shortlist: Mapping[str, Any]) -> list[BenchmarkCondition]:
    _require(shortlist.get("schema_version") == 1, "unsupported shortlist schema version")
    _require(
        shortlist.get("status") == "gpu_schedule_microbenchmark_required",
        "shortlist is not awaiting GPU schedule microbenchmarking",
    )
    _require(shortlist.get("online_screening_allowed") is False, "input shortlist is already promoted")
    raw_candidates = shortlist.get("candidates")
    _require(isinstance(raw_candidates, list) and len(raw_candidates) == 6, "shortlist must contain six candidates")

    conditions = [BenchmarkCondition(condition_id=BASELINE_CONDITION_ID, kind="baseline")]
    seen = set()
    for expected_rank, candidate in enumerate(raw_candidates, start=1):
        _require(isinstance(candidate, Mapping), f"candidate {expected_rank} must be an object")
        _require(candidate.get("rank") == expected_rank, "candidate ranks must be contiguous from one")
        warm_threshold = candidate.get("warm_threshold")
        warm_hex = candidate.get("warm_threshold_hex")
        _require(
            isinstance(warm_threshold, (int, float)) and not isinstance(warm_threshold, bool),
            f"candidate {expected_rank} warm threshold is invalid",
        )
        _require(
            isinstance(warm_hex, str) and float.fromhex(warm_hex) == float(warm_threshold),
            f"candidate {expected_rank} warm threshold hex mismatch",
        )
        condition = BenchmarkCondition(
            condition_id=f"candidate_rank_{expected_rank}",
            kind="origin_aware",
            rank=expected_rank,
            warm_threshold=float(warm_threshold),
            cold_threshold=float(candidate.get("cold_threshold")),
            max_skip_iters=int(candidate.get("max_skip_iters")),
            confirmation_mode=candidate.get("confirmation_mode"),
        )
        _require(condition.cold_threshold == 0.2, "candidate cold threshold must remain 0.2")
        _require(condition.max_skip_iters >= 1, "candidate max_skip_iters must be >= 1")
        _require(
            condition.confirmation_mode in {"next_iter", "backfill_pair"},
            "candidate confirmation mode is unsupported",
        )
        _require(condition.exact_key not in seen, "shortlist contains duplicate numerical configs")
        seen.add(condition.exact_key)
        conditions.append(condition)
    return conditions


def balanced_condition_order(
    condition_ids: Sequence[str], *, block_index: int, repeat_index: int, seed: int
) -> list[str]:
    """Return a deterministic, nearly position-balanced complete-block order."""

    ids = list(condition_ids)
    _require(ids and len(ids) == len(set(ids)), "condition IDs must be non-empty and unique")
    _require(block_index >= 0 and repeat_index >= 0, "block and repeat indices must be non-negative")
    base = list(ids)
    random.Random(int(seed)).shuffle(base)
    global_index = int(block_index) + int(repeat_index)
    cycle, shift = divmod(global_index, len(base))
    if cycle % 2:
        base.reverse()
    return base[shift:] + base[:shift]


def _scope_accepts(scope: str, origin: str) -> bool:
    if scope == "actual_warm":
        return origin == "ACTUAL_WARM"
    if scope == "cold":
        return origin == "COLD"
    if scope == "all_sampled":
        return origin in {"ACTUAL_WARM", "COLD"}
    raise GPUMicrobenchmarkValidationError(f"unsupported aggregation scope: {scope}")


def _prepare_episode_arrays(
    measurements: Sequence[Mapping[str, Any]],
    condition_ids: Sequence[str],
    *,
    scope: str,
    expected_repeats: int,
    required_task_ids: Iterable[int] | None,
    episodes_per_task: int | None,
) -> dict[int, np.ndarray]:
    _require(scope in SUPPORTED_SCOPES, f"unsupported scope: {scope}")
    condition_ids = list(condition_ids)
    condition_set = set(condition_ids)
    grouped: dict[tuple[int, int, int, str], list[float]] = {}
    workload_origins: dict[tuple[int, int, int], str] = {}

    for index, measurement in enumerate(measurements):
        condition_id = measurement.get("condition_id")
        _require(condition_id in condition_set, f"measurement {index}: unknown condition")
        task_id = measurement.get("task_id")
        episode_id = measurement.get("episode_id")
        prediction_step = measurement.get("prediction_step")
        repeat_index = measurement.get("repeat_index")
        origin = measurement.get("actual_origin")
        latency_ms = measurement.get("latency_ms")
        _require(
            all(isinstance(value, int) and not isinstance(value, bool) for value in (task_id, episode_id, prediction_step, repeat_index)),
            f"measurement {index}: integer identity fields are invalid",
        )
        _require(origin in {"COLD", "ACTUAL_WARM"}, f"measurement {index}: invalid origin")
        _require(
            isinstance(latency_ms, (int, float))
            and not isinstance(latency_ms, bool)
            and math.isfinite(float(latency_ms))
            and float(latency_ms) > 0,
            f"measurement {index}: latency must be finite and positive",
        )
        workload_key = (int(task_id), int(episode_id), int(prediction_step))
        prior_origin = workload_origins.setdefault(workload_key, origin)
        _require(prior_origin == origin, f"measurement {index}: workload origin changed")
        key = (*workload_key, str(condition_id))
        grouped.setdefault(key, []).append(float(latency_ms))

    workload_medians: dict[tuple[int, int, int, str], float] = {}
    for key, values in grouped.items():
        _require(
            len(values) == expected_repeats,
            f"workload/condition {key} has {len(values)} repeats, expected {expected_repeats}",
        )
        workload_medians[key] = float(np.median(np.asarray(values, dtype=np.float64)))

    episode_values: dict[tuple[int, int], dict[str, list[float]]] = {}
    eligible_workloads = 0
    for workload_key, origin in workload_origins.items():
        if not _scope_accepts(scope, origin):
            continue
        eligible_workloads += 1
        task_id, episode_id, prediction_step = workload_key
        values = episode_values.setdefault(
            (task_id, episode_id), {condition_id: [] for condition_id in condition_ids}
        )
        for condition_id in condition_ids:
            key = (task_id, episode_id, prediction_step, condition_id)
            _require(key in workload_medians, f"missing measured condition for workload {workload_key}")
            values[condition_id].append(workload_medians[key])
    _require(eligible_workloads > 0, f"scope {scope} contains no workloads")

    task_rows: dict[int, list[list[float]]] = {}
    for (task_id, episode_id), values in sorted(episode_values.items()):
        row = []
        counts = set()
        for condition_id in condition_ids:
            condition_values = values[condition_id]
            _require(condition_values, f"task {task_id} episode {episode_id}: empty condition")
            counts.add(len(condition_values))
            row.append(float(np.mean(condition_values)))
        _require(len(counts) == 1, f"task {task_id} episode {episode_id}: unequal workload counts")
        task_rows.setdefault(task_id, []).append(row)

    if required_task_ids is not None:
        required = tuple(int(task_id) for task_id in required_task_ids)
        _require(tuple(sorted(task_rows)) == tuple(sorted(required)), "measured task IDs do not match protocol")
    arrays = {
        task_id: np.asarray(rows, dtype=np.float64)
        for task_id, rows in sorted(task_rows.items())
    }
    if episodes_per_task is not None:
        for task_id, array in arrays.items():
            _require(
                array.shape == (episodes_per_task, len(condition_ids)),
                f"task {task_id}: expected {episodes_per_task} episodes, got {array.shape[0]}",
            )
    return arrays


def summarize_scope(
    measurements: Sequence[Mapping[str, Any]],
    conditions: Sequence[BenchmarkCondition],
    *,
    scope: str,
    expected_repeats: int,
    bootstrap_draws: int,
    bootstrap_seed: int,
    alpha: float,
    required_task_ids: Iterable[int] | None = None,
    episodes_per_task: int | None = None,
) -> dict[str, Any]:
    _require(bootstrap_draws >= 1, "bootstrap_draws must be positive")
    _require(0 < alpha < 0.5, "alpha must be in (0, 0.5)")
    condition_ids = [condition.condition_id for condition in conditions]
    _require(condition_ids[0] == BASELINE_CONDITION_ID, "baseline must be the first condition")
    arrays = _prepare_episode_arrays(
        measurements,
        condition_ids,
        scope=scope,
        expected_repeats=expected_repeats,
        required_task_ids=required_task_ids,
        episodes_per_task=episodes_per_task,
    )
    task_ids = list(arrays)
    task_means = np.stack([arrays[task_id].mean(axis=0) for task_id in task_ids])
    point_latency = task_means.mean(axis=0)
    _require(point_latency[0] > 0, "baseline macro latency must be positive")
    point_improvement = 1.0 - point_latency[1:] / point_latency[0]

    rng = np.random.default_rng(int(bootstrap_seed))
    boot_latency = np.zeros((bootstrap_draws, len(condition_ids)), dtype=np.float64)
    for task_id in task_ids:
        array = arrays[task_id]
        indices = rng.integers(0, array.shape[0], size=(bootstrap_draws, array.shape[0]))
        boot_latency += array[indices].mean(axis=1) / len(task_ids)
    _require(np.all(boot_latency[:, 0] > 0), "bootstrap baseline latency must be positive")
    boot_improvement = 1.0 - boot_latency[:, 1:] / boot_latency[:, [0]]

    downward_error = point_improvement[None, :] - boot_improvement
    max_downward_error = downward_error.max(axis=1)
    critical = float(np.quantile(max_downward_error, 1.0 - alpha, method="higher"))
    simultaneous_lower = point_improvement - critical
    individual_lower = np.quantile(boot_improvement, alpha, axis=0, method="lower")
    individual_upper = np.quantile(boot_improvement, 1.0 - alpha, axis=0, method="higher")

    condition_reports = []
    for index, condition in enumerate(conditions):
        report = {
            **asdict(condition),
            "task_macro_latency_ms": float(point_latency[index]),
            "task_latency_ms": {
                str(task_id): float(task_means[task_index, index])
                for task_index, task_id in enumerate(task_ids)
            },
        }
        if index == 0:
            report.update(
                {
                    "improvement_vs_baseline": 0.0,
                    "individual_interval": None,
                    "simultaneous_one_sided_lower_bound": None,
                }
            )
        else:
            report.update(
                {
                    "improvement_vs_baseline": float(point_improvement[index - 1]),
                    "individual_interval": [
                        float(individual_lower[index - 1]),
                        float(individual_upper[index - 1]),
                    ],
                    "simultaneous_one_sided_lower_bound": float(
                        simultaneous_lower[index - 1]
                    ),
                }
            )
        condition_reports.append(report)
    return {
        "scope": scope,
        "task_ids": task_ids,
        "episodes_per_task": {str(task_id): int(arrays[task_id].shape[0]) for task_id in task_ids},
        "bootstrap_draws": int(bootstrap_draws),
        "bootstrap_seed": int(bootstrap_seed),
        "alpha": float(alpha),
        "simultaneous_critical_downward_error": critical,
        "conditions": condition_reports,
    }


def build_benchmark_summary(
    measurements: Sequence[Mapping[str, Any]],
    conditions: Sequence[BenchmarkCondition],
    protocol: Mapping[str, Any],
    *,
    schedule_mismatch_count: int,
    required_task_ids: Iterable[int] = range(10),
    episodes_per_task: int = 10,
) -> dict[str, Any]:
    validate_protocol_manifest(protocol)
    repeats = int(protocol["measurement_repeats"])
    bootstrap = protocol["bootstrap"]
    scopes = {
        scope: summarize_scope(
            measurements,
            conditions,
            scope=scope,
            expected_repeats=repeats,
            bootstrap_draws=int(bootstrap["draws"]),
            bootstrap_seed=int(bootstrap["seed"]),
            alpha=float(bootstrap["alpha"]),
            required_task_ids=required_task_ids,
            episodes_per_task=episodes_per_task,
        )
        for scope in SUPPORTED_SCOPES
    }
    primary_scope = protocol["aggregation"]["primary_scope"]
    primary = scopes[primary_scope]
    gate = float(protocol["promotion"]["minimum_primary_action_head_improvement"])
    eligible = []
    for condition_report in primary["conditions"][1:]:
        lower = condition_report["simultaneous_one_sided_lower_bound"]
        passed = schedule_mismatch_count == 0 and lower >= gate
        condition_report["promotion_gate_passed"] = bool(passed)
        if passed:
            eligible.append(condition_report)
    eligible.sort(key=lambda item: item["improvement_vs_baseline"], reverse=True)
    limit = int(protocol["promotion"]["maximum_screening_candidates"])
    promoted = eligible[:limit]
    return {
        "primary_scope": primary_scope,
        "scopes": scopes,
        "schedule_mismatch_count": int(schedule_mismatch_count),
        "minimum_primary_action_head_improvement": gate,
        "eligible_candidate_count": len(eligible),
        "screening_candidates": [item["condition_id"] for item in promoted],
        "online_screening_allowed": bool(promoted),
        "interpretation": (
            "Selection-conditioned saved-workload result only; closed-loop latency and "
            "task success require online screening."
        ),
    }
