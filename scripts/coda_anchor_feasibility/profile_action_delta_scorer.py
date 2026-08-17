"""Diagnostic-only CUDA micro-profiler for the frozen Action-Delta scorer."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from prismatic.models.action_delta_gate import (
    PreparedActionDeltaGate,
    evaluate_action_delta_gate,
    load_action_delta_gate_artifact,
    prepare_action_delta_gate_shadow,
    score_action_delta_gate,
)


DIAGNOSTIC_ONLY = True
EXPECTED_ARTIFACT_SHA256 = (
    "b4f9e938c72108f164b9d997a86eb4f5d9ea15b146754b9af83f9735e1ecfcf8"
)
DEVELOPMENT_TASK_IDS = (0, 1, 2, 3, 6, 7, 8, 9)
HIGH_SIDE_THRESHOLD = 0.0015
RUNTIME_SHAPE = (1, 8, 896)
OBSERVED_DEV8_SCORER_MS = 0.27
EMPIRICAL_DEV8_BREAK_EVEN_SCORER_MS = 0.547
HISTORICAL_CODA_COST_MS = 1.864207385085098
DEV8_PREDICTIONS = 1762
DEV8_BASELINE_CODA_CALLS = 9300
DEV8_ELIMINATED_CODA_CALLS = 2833
EXPECTED_DEV8_TRANSITIONS = 7139
DEFAULT_MANIFEST = Path(
    "benchmark_results/coda_anchor_feasibility/deployment_matched_shadow/"
    "phaseA_dev8_min2_20260817_175619/manifest.json"
)
DEFAULT_ARTIFACT = Path(
    "benchmark_results/coda_anchor_feasibility/action_delta_gate_fold4/"
    "action_delta_gate.pt"
)
DEFAULT_OUTPUT_DIR = Path(
    "benchmark_results/coda_anchor_feasibility/"
    "action_delta_scorer_compile_experiment"
)
COMPONENT_NAMES = (
    "latent_subtraction_fp32_conversion",
    "bf16_quantization_fp32_restore",
    "normalization",
    "linear",
    "output_denormalization",
    "square_mean_reduction",
)


class ActionDeltaScorerProfileError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ActionDeltaScorerProfileError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latency_statistics(
    values_ms: Sequence[float], *, warmup_count: int = 0
) -> dict[str, Any]:
    values = np.asarray(values_ms, dtype=np.float64)
    _require(values.size > 0, "latency sample is empty")
    _require(bool(np.isfinite(values).all()), "latency sample is non-finite")
    return {
        "count": int(values.size),
        "warmup_count": int(warmup_count),
        "mean_ms": float(values.mean()),
        "median_ms": float(np.percentile(values, 50)),
        "p90_ms": float(np.percentile(values, 90)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "std_ms": float(values.std()),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
    }


def tensor_action_delta_score(
    anchor_state: torch.Tensor,
    current_state: torch.Tensor,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor,
    y_std: torch.Tensor,
    y_mean: torch.Tensor,
) -> torch.Tensor:
    """Pure tensor scorer used only by the diagnostic compile experiment."""

    delta = current_state.float() - anchor_state.float()
    delta = delta.to(torch.bfloat16).float()
    x = (delta - x_mean) / x_std
    pred_norm = F.linear(x, linear_weight, linear_bias)
    pred_delta = pred_norm * y_std + y_mean
    return pred_delta.square().mean()


def _gate_tensor_arguments(gate: PreparedActionDeltaGate) -> tuple[torch.Tensor, ...]:
    return (
        gate.x_mean,
        gate.x_std,
        gate.linear_weight,
        gate.linear_bias,
        gate.y_std,
        gate.y_mean,
    )


def call_tensor_scorer(
    scorer: Any,
    gate: PreparedActionDeltaGate,
    anchor_state: torch.Tensor,
    current_state: torch.Tensor,
) -> torch.Tensor:
    return scorer(
        anchor_state,
        current_state,
        *_gate_tensor_arguments(gate),
    )


def snapshot_compiled_scalar(score: torch.Tensor) -> torch.Tensor:
    """Own a compiled scalar before a later CUDA-Graph invocation can overwrite it."""

    return score.detach().clone().cpu()


def decomposed_action_delta_score(
    gate: PreparedActionDeltaGate,
    anchor_state: torch.Tensor,
    current_state: torch.Tensor,
    *,
    return_intermediates: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Reproduce production math stage-by-stage without altering the scorer."""

    delta_fp32 = current_state.float() - anchor_state.float()
    delta_bfloat16 = delta_fp32.to(torch.bfloat16)
    delta = delta_bfloat16.float()
    normalized = (delta - gate.x_mean) / gate.x_std
    pred_norm = F.linear(normalized, gate.linear_weight, gate.linear_bias)
    pred_delta = pred_norm * gate.y_std + gate.y_mean
    score = pred_delta.square().mean()
    if not return_intermediates:
        return score
    return score, {
        "delta_fp32_before_quantization": delta_fp32,
        "delta_bfloat16": delta_bfloat16,
        "delta_bfloat16_restored_fp32": delta,
        "normalized": normalized,
        "pred_norm": pred_norm,
        "pred_delta": pred_delta,
    }


def validate_decomposition(
    gate: PreparedActionDeltaGate,
    anchors: torch.Tensor,
    currents: torch.Tensor,
) -> dict[str, Any]:
    """Verify score/decision parity and tensor immutability for every sample."""

    _require(tuple(anchors.shape[1:]) == RUNTIME_SHAPE, "anchor runtime shape mismatch")
    _require(tuple(currents.shape[1:]) == RUNTIME_SHAPE, "current runtime shape mismatch")
    _require(anchors.shape == currents.shape, "anchor/current sample shape mismatch")
    anchors_before = anchors.clone()
    currents_before = currents.clone()
    gate_before = {
        name: getattr(gate, name).clone()
        for name in (
            "x_mean",
            "x_std",
            "y_mean",
            "y_std",
            "linear_weight",
            "linear_bias",
        )
    }
    maximum_absolute_difference = 0.0
    high_side_decision_mismatches = 0
    evaluate_score_mismatches = 0
    with torch.inference_mode():
        for index in range(len(anchors)):
            anchor = anchors[index]
            current = currents[index]
            decomposed = decomposed_action_delta_score(gate, anchor, current)
            production_tensor = score_action_delta_gate(gate, anchor, current)
            decomposed_value = float(decomposed.item())
            production_tensor_value = float(production_tensor.item())
            evaluate_value, _low_side_gate_decision = evaluate_action_delta_gate(
                gate, anchor, current
            )
            maximum_absolute_difference = max(
                maximum_absolute_difference,
                abs(decomposed_value - production_tensor_value),
                abs(decomposed_value - evaluate_value),
            )
            evaluate_score_mismatches += int(evaluate_value != production_tensor_value)
            high_side_decision_mismatches += int(
                (decomposed_value >= HIGH_SIDE_THRESHOLD)
                != (evaluate_value >= HIGH_SIDE_THRESHOLD)
            )

    _require(maximum_absolute_difference == 0.0, "decomposed score parity failed")
    _require(evaluate_score_mismatches == 0, "evaluate/score tensor parity failed")
    _require(high_side_decision_mismatches == 0, "high-side decision parity failed")
    _require(torch.equal(anchors, anchors_before), "anchor states were mutated")
    _require(torch.equal(currents, currents_before), "current states were mutated")
    for name, before in gate_before.items():
        _require(torch.equal(getattr(gate, name), before), f"gate tensor mutated: {name}")
    return {
        "sample_count": int(len(anchors)),
        "maximum_absolute_score_difference": maximum_absolute_difference,
        "evaluate_score_mismatches": evaluate_score_mismatches,
        "high_side_decision_mismatches": high_side_decision_mismatches,
        "anchor_tensor_mutated": False,
        "current_tensor_mutated": False,
        "gate_tensor_mutated": False,
    }


def _manifest_minimum(manifest: Mapping[str, Any]) -> int:
    value = manifest.get("min_terminal_iteration")
    if value is None:
        value = (manifest.get("configuration") or {}).get("gate", {}).get(
            "min_terminal_iteration"
        )
    _require(value == 2, "micro-profile source must be the min-terminal-2 dataset")
    return int(value)


def load_real_transition_sample(
    manifest_path: Path,
    *,
    sample_count: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]], dict[str, Any]]:
    """Load an evenly spaced, hash-verified sample from real shadow shards."""

    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("complete") is True, "shadow manifest is incomplete")
    _require(
        tuple(int(value) for value in manifest.get("expected_task_ids", ()))
        == DEVELOPMENT_TASK_IDS,
        "shadow manifest is not the development-task partition",
    )
    _manifest_minimum(manifest)
    _require(
        manifest.get("artifact_identity", {}).get("sha256")
        == EXPECTED_ARTIFACT_SHA256,
        "shadow manifest artifact identity mismatch",
    )
    descriptors = [
        descriptor
        for descriptor in manifest.get("predictions", [])
        if int(descriptor.get("eligible_row_count", 0)) > 0
    ]
    cumulative: list[int] = []
    total = 0
    for descriptor in descriptors:
        task_id = int(descriptor["task_id"])
        _require(task_id in DEVELOPMENT_TASK_IDS, "forbidden task in shadow sample")
        total += int(descriptor["eligible_row_count"])
        cumulative.append(total)
    _require(total == int(manifest["transition_count"]), "manifest row accounting mismatch")
    _require(
        isinstance(sample_count, int)
        and not isinstance(sample_count, bool)
        and 1 <= sample_count <= total,
        f"sample count must be in [1, {total}]",
    )
    positions = np.linspace(0, total - 1, num=sample_count).round().astype(np.int64)
    requests: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for output_index, global_position in enumerate(positions.tolist()):
        descriptor_index = bisect.bisect_right(cumulative, global_position)
        previous_end = cumulative[descriptor_index - 1] if descriptor_index else 0
        local_transition_index = int(global_position - previous_end)
        descriptor = descriptors[descriptor_index]
        requests[str(descriptor["shard_path"])].append(
            (output_index, int(descriptor["shard_index"]), local_transition_index)
        )

    anchors: list[torch.Tensor | None] = [None] * sample_count
    currents: list[torch.Tensor | None] = [None] * sample_count
    identities: list[dict[str, Any] | None] = [None] * sample_count
    descriptor_by_shard = {
        str(descriptor["shard_path"]): descriptor for descriptor in descriptors
    }
    accessed_shards = []
    for shard_name, shard_requests in requests.items():
        descriptor = descriptor_by_shard[shard_name]
        shard_path = manifest_path.parent / shard_name
        actual_hash = sha256_file(shard_path)
        _require(actual_hash == descriptor["shard_sha256"], "shadow shard hash mismatch")
        shard = torch.load(shard_path, map_location="cpu", weights_only=True)
        predictions = shard["predictions"]
        for output_index, prediction_index, transition_index in shard_requests:
            prediction = predictions[prediction_index]
            transition = prediction["transitions"][transition_index]
            task_id = int(transition["identity"]["task_id"])
            _require(task_id in DEVELOPMENT_TASK_IDS, "forbidden sampled transition")
            anchor = transition["tensors"]["anchor_state"].detach().clone().contiguous()
            current = transition["tensors"]["current_state"].detach().clone().contiguous()
            _require(tuple(anchor.shape) == RUNTIME_SHAPE, "sampled anchor shape mismatch")
            _require(tuple(current.shape) == RUNTIME_SHAPE, "sampled current shape mismatch")
            anchors[output_index] = anchor
            currents[output_index] = current
            identities[output_index] = {
                "global_transition_index": int(positions[output_index]),
                "task_id": task_id,
                "trajectory_id": transition["identity"]["trajectory_id"],
                "prediction_id": prediction["prediction_id"],
                "terminal_iteration": int(transition["terminal_iteration"]),
                "shard_path": shard_name,
            }
        accessed_shards.append(
            {"path": str(shard_path), "sha256": actual_hash}
        )
    _require(all(value is not None for value in anchors), "missing sampled anchor")
    _require(all(value is not None for value in currents), "missing sampled current")
    _require(all(value is not None for value in identities), "missing sample identity")
    return (
        torch.stack(anchors),
        torch.stack(currents),
        list(identities),
        {
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "dataset_identity_sha256": manifest.get("dataset_identity_sha256"),
            "source_transition_count": total,
            "sample_count": sample_count,
            "selection": "evenly_spaced_global_transition_indices",
            "development_task_ids": list(DEVELOPMENT_TASK_IDS),
            "accessed_shards": accessed_shards,
        },
    )


def compare_score_vectors(
    eager_scores: torch.Tensor,
    candidate_scores: torch.Tensor,
    identities: Sequence[Mapping[str, Any]],
    *,
    threshold: float = HIGH_SIDE_THRESHOLD,
) -> dict[str, Any]:
    """Compare candidate scores and high-side decisions without tolerance masking."""

    eager = eager_scores.detach().to(device="cpu", dtype=torch.float64).flatten()
    candidate = (
        candidate_scores.detach().to(device="cpu", dtype=torch.float64).flatten()
    )
    _require(eager.shape == candidate.shape, "score comparison shape mismatch")
    _require(len(eager) == len(identities), "score identity count mismatch")
    _require(
        bool(torch.isfinite(eager).all()),
        "authoritative eager scores are non-finite",
    )
    _require(
        bool(torch.isfinite(candidate).all()), "candidate scores are non-finite"
    )
    absolute = (candidate - eager).abs()
    relative = torch.where(
        eager.abs() > 0,
        absolute / eager.abs(),
        torch.where(
            absolute == 0,
            torch.zeros_like(absolute),
            torch.full_like(absolute, math.inf),
        ),
    )
    exact = candidate == eager
    eager_decision = eager >= threshold
    candidate_decision = candidate >= threshold
    eager_margin = (eager - threshold).abs()
    candidate_margin = (candidate - threshold).abs()
    largest_index = int(absolute.argmax().item())
    eager_margin_index = int(eager_margin.argmin().item())
    candidate_margin_index = int(candidate_margin.argmin().item())
    return {
        "transition_count": int(eager.numel()),
        "exact_score_match_count": int(exact.sum().item()),
        "exact_score_match_rate": float(exact.double().mean().item()),
        "scores_bitwise_identical": bool(exact.all().item()),
        "maximum_absolute_score_difference": float(absolute.max().item()),
        "mean_absolute_score_difference": float(absolute.mean().item()),
        "maximum_relative_score_difference": float(relative.max().item()),
        "high_side_decision_match_count": int(
            (eager_decision == candidate_decision).sum().item()
        ),
        "high_side_decision_mismatch_count": int(
            (eager_decision != candidate_decision).sum().item()
        ),
        "minimum_eager_score_distance_to_threshold": float(
            eager_margin.min().item()
        ),
        "minimum_candidate_score_distance_to_threshold": float(
            candidate_margin.min().item()
        ),
        "largest_score_difference_transition": {
            "identity": dict(identities[largest_index]),
            "eager_score": float(eager[largest_index].item()),
            "candidate_score": float(candidate[largest_index].item()),
            "absolute_difference": float(absolute[largest_index].item()),
            "relative_difference": float(relative[largest_index].item()),
        },
        "smallest_eager_threshold_margin_transition": {
            "identity": dict(identities[eager_margin_index]),
            "eager_score": float(eager[eager_margin_index].item()),
            "candidate_score": float(candidate[eager_margin_index].item()),
            "distance_to_threshold": float(eager_margin[eager_margin_index].item()),
        },
        "smallest_candidate_threshold_margin_transition": {
            "identity": dict(identities[candidate_margin_index]),
            "eager_score": float(eager[candidate_margin_index].item()),
            "candidate_score": float(candidate[candidate_margin_index].item()),
            "distance_to_threshold": float(
                candidate_margin[candidate_margin_index].item()
            ),
        },
    }


def compile_tensor_candidates(
    gate: PreparedActionDeltaGate,
    anchor: torch.Tensor,
    current: torch.Tensor,
) -> dict[str, dict[str, Any]]:
    """Compile each requested candidate once and force its lazy first invocation."""

    candidates: dict[str, dict[str, Any]] = {}
    configurations = {
        "compile_default": {},
        "compile_reduce_overhead": {"mode": "reduce-overhead"},
    }
    for name, options in configurations.items():
        started = time.perf_counter_ns()
        try:
            compiled = torch.compile(
                tensor_action_delta_score,
                fullgraph=True,
                dynamic=False,
                **options,
            )
            with torch.inference_mode():
                call_tensor_scorer(compiled, gate, anchor, current).item()
            torch.cuda.synchronize()
            candidates[name] = {
                "status": "available",
                "callable": compiled,
                "configuration": {
                    "fullgraph": True,
                    "dynamic": False,
                    **options,
                },
                "compile_and_first_call_ms_excluded_from_timing": (
                    time.perf_counter_ns() - started
                )
                / 1e6,
            }
        except Exception as error:  # pragma: no cover - CUDA/compiler dependent
            candidates[name] = {
                "status": "failed",
                "configuration": {
                    "fullgraph": True,
                    "dynamic": False,
                    **options,
                },
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(limit=8),
            }
    return candidates


def evaluate_candidate_parity(
    gate: PreparedActionDeltaGate,
    anchors: torch.Tensor,
    currents: torch.Tensor,
    identities: Sequence[Mapping[str, Any]],
    candidates: Mapping[str, Mapping[str, Any]],
) -> tuple[torch.Tensor, dict[str, dict[str, Any]]]:
    """Evaluate eager and every compiled candidate over all real transitions."""

    _require(
        len(anchors) == EXPECTED_DEV8_TRANSITIONS,
        "full parity row count mismatch",
    )
    eager_values: list[torch.Tensor] = []
    eager_tensor_values: list[torch.Tensor] = []
    candidate_values: dict[str, list[torch.Tensor]] = {
        name: []
        for name, candidate in candidates.items()
        if candidate["status"] == "available"
    }
    tracked_tensors = (anchors, currents, *_gate_tensor_arguments(gate))
    versions_before = tuple(tensor._version for tensor in tracked_tensors)
    with torch.inference_mode():
        for index in range(len(anchors)):
            anchor = anchors[index]
            current = currents[index]
            eager_values.append(score_action_delta_gate(gate, anchor, current))
            eager_tensor_values.append(
                call_tensor_scorer(tensor_action_delta_score, gate, anchor, current)
            )
            for name in candidate_values:
                compiled_score = call_tensor_scorer(
                    candidates[name]["callable"], gate, anchor, current
                )
                candidate_values[name].append(
                    snapshot_compiled_scalar(compiled_score)
                )
        eager_scores = torch.stack(eager_values).cpu()
        eager_tensor_scores = torch.stack(eager_tensor_values).cpu()
        compiled_scores = {
            name: torch.stack(values)
            for name, values in candidate_values.items()
        }
    versions_after = tuple(tensor._version for tensor in tracked_tensors)
    mutation_free = versions_before == versions_after
    eager_tensor_comparison = compare_score_vectors(
        eager_scores, eager_tensor_scores, identities
    )
    eager_tensor_comparison["input_and_artifact_tensor_mutation_free"] = mutation_free
    results: dict[str, dict[str, Any]] = {
        "eager_tensor": {"status": "available", **eager_tensor_comparison}
    }
    for name, candidate in candidates.items():
        if candidate["status"] != "available":
            results[name] = {
                "status": "failed",
                "error_type": candidate["error_type"],
                "error": candidate["error"],
            }
            continue
        comparison = compare_score_vectors(
            eager_scores, compiled_scores[name], identities
        )
        comparison["input_and_artifact_tensor_mutation_free"] = mutation_free
        results[name] = {"status": "available", **comparison}
    _require(mutation_free, "compile parity evaluation mutated an input or artifact tensor")
    return eager_scores, results


def _attach_repeats(
    pooled: Sequence[float], per_repeat: Sequence[Sequence[float]], *, warmup: int
) -> dict[str, Any]:
    result = latency_statistics(pooled, warmup_count=warmup * len(per_repeat))
    result["repeat_count"] = len(per_repeat)
    result["per_repeat_mean_ms"] = [
        float(np.asarray(values, dtype=np.float64).mean()) for values in per_repeat
    ]
    return result


def benchmark_faithful_runtime(
    gate: PreparedActionDeltaGate,
    anchors: torch.Tensor,
    currents: torch.Tensor,
    *,
    warmup: int,
    repetitions: int,
    repeats: int,
) -> tuple[dict[str, Any], list[float]]:
    pooled: list[float] = []
    per_repeat: list[list[float]] = []
    sample_count = len(anchors)
    with torch.inference_mode():
        for repeat in range(repeats):
            torch.cuda.synchronize()
            for index in range(warmup):
                position = (repeat * warmup + index) % sample_count
                evaluate_action_delta_gate(gate, anchors[position], currents[position])
            torch.cuda.synchronize()
            values = []
            for index in range(repetitions):
                position = (repeat * repetitions + index) % sample_count
                start = time.perf_counter_ns()
                evaluate_action_delta_gate(gate, anchors[position], currents[position])
                values.append((time.perf_counter_ns() - start) / 1e6)
            torch.cuda.synchronize()
            pooled.extend(values)
            per_repeat.append(values)
    return _attach_repeats(pooled, per_repeat, warmup=warmup), pooled


def benchmark_tensor_cuda_events(
    scorer: Any,
    gate: PreparedActionDeltaGate,
    anchors: torch.Tensor,
    currents: torch.Tensor,
    *,
    warmup: int,
    repetitions: int,
    repeats: int,
) -> tuple[dict[str, Any], list[float]]:
    """Measure tensor execution with CUDA events, excluding host scalar transfer."""

    pooled: list[float] = []
    per_repeat: list[list[float]] = []
    sample_count = len(anchors)
    with torch.inference_mode():
        for repeat in range(repeats):
            for index in range(warmup):
                position = (repeat * warmup + index) % sample_count
                call_tensor_scorer(
                    scorer, gate, anchors[position], currents[position]
                )
            torch.cuda.synchronize()
            rows = []
            for index in range(repetitions):
                position = (repeat * repetitions + index) % sample_count
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                call_tensor_scorer(
                    scorer, gate, anchors[position], currents[position]
                )
                end.record()
                rows.append((start, end))
            torch.cuda.synchronize()
            values = [float(start.elapsed_time(end)) for start, end in rows]
            pooled.extend(values)
            per_repeat.append(values)
    return _attach_repeats(pooled, per_repeat, warmup=warmup), pooled


def benchmark_tensor_runtime_wall(
    scorer: Any,
    gate: PreparedActionDeltaGate,
    anchors: torch.Tensor,
    currents: torch.Tensor,
    *,
    warmup: int,
    repetitions: int,
    repeats: int,
) -> tuple[dict[str, Any], list[float]]:
    """Measure runtime-like scorer invocation immediately followed by ``item``."""

    pooled: list[float] = []
    per_repeat: list[list[float]] = []
    sample_count = len(anchors)
    with torch.inference_mode():
        for repeat in range(repeats):
            for index in range(warmup):
                position = (repeat * warmup + index) % sample_count
                call_tensor_scorer(
                    scorer, gate, anchors[position], currents[position]
                ).item()
            torch.cuda.synchronize()
            values = []
            for index in range(repetitions):
                position = (repeat * repetitions + index) % sample_count
                start = time.perf_counter_ns()
                call_tensor_scorer(
                    scorer, gate, anchors[position], currents[position]
                ).item()
                values.append((time.perf_counter_ns() - start) / 1e6)
            pooled.extend(values)
            per_repeat.append(values)
    return _attach_repeats(pooled, per_repeat, warmup=warmup), pooled


def benchmark_components(
    gate: PreparedActionDeltaGate,
    anchors: torch.Tensor,
    currents: torch.Tensor,
    *,
    warmup: int,
    repetitions: int,
    repeats: int,
) -> dict[str, Any]:
    pooled = {name: [] for name in COMPONENT_NAMES}
    repeat_values = {name: [] for name in COMPONENT_NAMES}
    sample_count = len(anchors)
    with torch.inference_mode():
        for repeat in range(repeats):
            for index in range(warmup):
                position = (repeat * warmup + index) % sample_count
                decomposed_action_delta_score(
                    gate, anchors[position], currents[position]
                ).item()
            torch.cuda.synchronize()
            event_rows = []
            for index in range(repetitions):
                position = (repeat * repetitions + index) % sample_count
                anchor = anchors[position]
                current = currents[position]
                events = [torch.cuda.Event(enable_timing=True) for _ in range(7)]
                events[0].record()
                delta_fp32 = current.float() - anchor.float()
                events[1].record()
                delta = delta_fp32.to(torch.bfloat16).float()
                events[2].record()
                normalized = (delta - gate.x_mean) / gate.x_std
                events[3].record()
                pred_norm = F.linear(
                    normalized, gate.linear_weight, gate.linear_bias
                )
                events[4].record()
                pred_delta = pred_norm * gate.y_std + gate.y_mean
                events[5].record()
                score = pred_delta.square().mean()
                events[6].record()
                event_rows.append((events, score))
            torch.cuda.synchronize()
            current_repeat = {name: [] for name in COMPONENT_NAMES}
            for events, _score in event_rows:
                for stage_index, name in enumerate(COMPONENT_NAMES):
                    elapsed = float(
                        events[stage_index].elapsed_time(events[stage_index + 1])
                    )
                    pooled[name].append(elapsed)
                    current_repeat[name].append(elapsed)
            for name in COMPONENT_NAMES:
                repeat_values[name].append(current_repeat[name])
    return {
        name: _attach_repeats(
            pooled[name], repeat_values[name], warmup=warmup
        )
        for name in COMPONENT_NAMES
    }


def benchmark_host_sync(
    gate: PreparedActionDeltaGate,
    anchors: torch.Tensor,
    currents: torch.Tensor,
    *,
    warmup: int,
    repetitions: int,
    repeats: int,
) -> dict[str, Any]:
    natural_pooled: list[float] = []
    wait_pooled: list[float] = []
    post_pooled: list[float] = []
    natural_repeats: list[list[float]] = []
    wait_repeats: list[list[float]] = []
    post_repeats: list[list[float]] = []
    sample_count = len(anchors)
    with torch.inference_mode():
        for repeat in range(repeats):
            for index in range(warmup):
                position = (repeat * warmup + index) % sample_count
                score_action_delta_gate(
                    gate, anchors[position], currents[position]
                ).item()
            torch.cuda.synchronize()
            natural = []
            for index in range(repetitions):
                position = (repeat * repetitions + index) % sample_count
                score = score_action_delta_gate(
                    gate, anchors[position], currents[position]
                )
                start = time.perf_counter_ns()
                score.item()
                natural.append((time.perf_counter_ns() - start) / 1e6)
            torch.cuda.synchronize()
            waits = []
            posts = []
            for index in range(repetitions):
                position = (repeat * repetitions + index) % sample_count
                score = score_action_delta_gate(
                    gate, anchors[position], currents[position]
                )
                wait_start = time.perf_counter_ns()
                torch.cuda.synchronize()
                waits.append((time.perf_counter_ns() - wait_start) / 1e6)
                item_start = time.perf_counter_ns()
                score.item()
                posts.append((time.perf_counter_ns() - item_start) / 1e6)
            natural_pooled.extend(natural)
            wait_pooled.extend(waits)
            post_pooled.extend(posts)
            natural_repeats.append(natural)
            wait_repeats.append(waits)
            post_repeats.append(posts)
    return {
        "natural_item_wall": _attach_repeats(
            natural_pooled, natural_repeats, warmup=warmup
        ),
        "pre_item_sync_wait": _attach_repeats(
            wait_pooled, wait_repeats, warmup=warmup
        ),
        "post_sync_item": _attach_repeats(
            post_pooled, post_repeats, warmup=warmup
        ),
    }


def benchmark_tensor_host_sync(
    scorer: Any,
    gate: PreparedActionDeltaGate,
    anchors: torch.Tensor,
    currents: torch.Tensor,
    *,
    warmup: int,
    repetitions: int,
    repeats: int,
) -> dict[str, Any]:
    """Attribute queued device work versus scalar extraction for a tensor scorer."""

    natural_pooled: list[float] = []
    wait_pooled: list[float] = []
    post_pooled: list[float] = []
    natural_repeats: list[list[float]] = []
    wait_repeats: list[list[float]] = []
    post_repeats: list[list[float]] = []
    sample_count = len(anchors)
    with torch.inference_mode():
        for repeat in range(repeats):
            for index in range(warmup):
                position = (repeat * warmup + index) % sample_count
                call_tensor_scorer(
                    scorer, gate, anchors[position], currents[position]
                ).item()
            torch.cuda.synchronize()
            natural = []
            for index in range(repetitions):
                position = (repeat * repetitions + index) % sample_count
                score = call_tensor_scorer(
                    scorer, gate, anchors[position], currents[position]
                )
                start = time.perf_counter_ns()
                score.item()
                natural.append((time.perf_counter_ns() - start) / 1e6)
            torch.cuda.synchronize()
            waits = []
            posts = []
            for index in range(repetitions):
                position = (repeat * repetitions + index) % sample_count
                score = call_tensor_scorer(
                    scorer, gate, anchors[position], currents[position]
                )
                wait_start = time.perf_counter_ns()
                torch.cuda.synchronize()
                waits.append((time.perf_counter_ns() - wait_start) / 1e6)
                item_start = time.perf_counter_ns()
                score.item()
                posts.append((time.perf_counter_ns() - item_start) / 1e6)
            natural_pooled.extend(natural)
            wait_pooled.extend(waits)
            post_pooled.extend(posts)
            natural_repeats.append(natural)
            wait_repeats.append(waits)
            post_repeats.append(posts)
    return {
        "natural_item_wall": _attach_repeats(
            natural_pooled, natural_repeats, warmup=warmup
        ),
        "pre_item_sync_wait": _attach_repeats(
            wait_pooled, wait_repeats, warmup=warmup
        ),
        "post_sync_item": _attach_repeats(
            post_pooled, post_repeats, warmup=warmup
        ),
    }


def profile_kernel_launches(
    scorer: Any,
    gate: PreparedActionDeltaGate,
    anchor: torch.Tensor,
    current: torch.Tensor,
) -> dict[str, Any]:
    """Collect a one-call qualitative CUDA-kernel diagnostic."""

    try:
        with torch.inference_mode():
            for _ in range(10):
                call_tensor_scorer(scorer, gate, anchor, current).item()
            torch.cuda.synchronize()
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as profile:
                call_tensor_scorer(scorer, gate, anchor, current).item()
                torch.cuda.synchronize()
        cuda_events = [
            event
            for event in profile.events()
            if "cuda" in str(event.device_type).lower()
        ]
        kernel_names = [str(event.name) for event in cuda_events]
        lowered = [name.lower() for name in kernel_names]
        gemm_names = [
            name
            for name, lower in zip(kernel_names, lowered)
            if any(
                token in lower
                for token in ("gemm", "cublas", "addmm", " matmul", " mm")
            )
        ]
        reduction_names = [
            name
            for name, lower in zip(kernel_names, lowered)
            if any(token in lower for token in ("reduce", "reduction", "triton_red"))
        ]
        fused_names = [
            name for name, lower in zip(kernel_names, lowered) if "fused" in lower
        ]
        return {
            "status": "available",
            "approximate_gpu_kernel_count_per_scorer": len(kernel_names),
            "kernel_names": kernel_names,
            "elementwise_fusion_evidence": bool(fused_names),
            "elementwise_stages_fused": bool(fused_names),
            "fused_kernel_names": fused_names,
            "linear_appears_to_remain_separate": bool(gemm_names),
            "linear_kernel_names": gemm_names,
            "reduction_appears_separate": bool(reduction_names),
            "reduction_fused_with_elementwise": bool(
                set(reduction_names).intersection(fused_names)
            ),
            "reduction_kernel_names": reduction_names,
            "interpretation": (
                "Kernel names are a one-call qualitative diagnostic; profiler time is "
                "not used as primary latency. Generic generated names can make fusion "
                "classification inconclusive."
            ),
        }
    except Exception as error:  # pragma: no cover - profiler/CUDA dependent
        return {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
        }


def policy_economics(
    faithful_mean_ms: float, *, scorer_calls: int
) -> dict[str, Any]:
    gross_saving = DEV8_ELIMINATED_CODA_CALLS * HISTORICAL_CODA_COST_MS
    break_even = gross_saving / scorer_calls
    scenarios = {}
    for reduction_percent in (25, 50, 75):
        reduced_cost = faithful_mean_ms * (1.0 - reduction_percent / 100.0)
        total_scorer = scorer_calls * reduced_cost
        scenarios[str(reduction_percent)] = {
            "hypothetical_scorer_cost_ms_per_call": reduced_cost,
            "estimated_changed_path_scorer_latency_ms": total_scorer,
            "estimated_gross_coda_saving_ms": gross_saving,
            "estimated_net_latency_saving_ms": gross_saving - total_scorer,
            "measured_runtime_improvement": False,
        }
    return {
        "predictions": DEV8_PREDICTIONS,
        "baseline_coda_calls": DEV8_BASELINE_CODA_CALLS,
        "eliminated_coda_calls": DEV8_ELIMINATED_CODA_CALLS,
        "scorer_calls": int(scorer_calls),
        "historical_coda_cost_ms": HISTORICAL_CODA_COST_MS,
        "gross_coda_saving_ms": gross_saving,
        "break_even_scorer_cost_ms_per_call": break_even,
        "break_even_scorer_to_coda_cost_ratio": (
            DEV8_ELIMINATED_CODA_CALLS / scorer_calls
        ),
        "empirical_paired_runtime_break_even_scorer_cost_ms_per_call": (
            EMPIRICAL_DEV8_BREAK_EVEN_SCORER_MS
        ),
        "empirical_break_even_source": (
            "previously measured paired-runtime changed-path values"
        ),
        "measured_faithful_scorer_cost_ms_per_call": faithful_mean_ms,
        "measured_faithful_is_theoretically_net_positive": faithful_mean_ms
        < break_even,
        "hypothetical_reductions": scenarios,
        "warning": (
            "Hypothetical reductions are arithmetic estimates, not measured runtime improvements."
        ),
    }


def compiled_candidate_economics(
    measured_standalone_wall_ms: float, *, scorer_calls: int
) -> dict[str, Any]:
    gross_historical = DEV8_ELIMINATED_CODA_CALLS * HISTORICAL_CODA_COST_MS
    projected_scorer_total = scorer_calls * measured_standalone_wall_ms
    return {
        "measured_standalone_compiled_scorer_wall_ms_per_call": (
            measured_standalone_wall_ms
        ),
        "scorer_calls": scorer_calls,
        "eliminated_coda_calls": DEV8_ELIMINATED_CODA_CALLS,
        "projected_dev8_scorer_time_ms": projected_scorer_total,
        "projected_gross_coda_time_saving_using_historical_cost_ms": (
            gross_historical
        ),
        "projected_net_using_historical_coda_cost_ms": (
            gross_historical - projected_scorer_total
        ),
        "historical_cost_break_even_ms_per_call": gross_historical / scorer_calls,
        "empirical_paired_runtime_break_even_ms_per_call": (
            EMPIRICAL_DEV8_BREAK_EVEN_SCORER_MS
        ),
        "projected_headroom_vs_empirical_break_even_ms_per_call": (
            EMPIRICAL_DEV8_BREAK_EVEN_SCORER_MS - measured_standalone_wall_ms
        ),
        "projected_total_headroom_vs_empirical_break_even_ms": scorer_calls
        * (EMPIRICAL_DEV8_BREAK_EVEN_SCORER_MS - measured_standalone_wall_ms),
        "is_below_empirical_paired_runtime_break_even": (
            measured_standalone_wall_ms < EMPIRICAL_DEV8_BREAK_EVEN_SCORER_MS
        ),
        "warning": (
            "This projects dev8 latency from a standalone micro-profile. It is not "
            "a measured LIBERO or paired-runtime speedup."
        ),
    }


def classify_compiled_candidate(
    parity: Mapping[str, Any],
    *,
    candidate_wall_ms: float,
    eager_tensor_wall_ms: float,
) -> dict[str, Any]:
    _require(eager_tensor_wall_ms > 0, "eager tensor wall latency must be positive")
    improvement = 1.0 - candidate_wall_ms / eager_tensor_wall_ms
    decision_parity = (
        int(parity.get("transition_count", -1)) == EXPECTED_DEV8_TRANSITIONS
        and int(parity.get("high_side_decision_mismatch_count", -1)) == 0
        and int(parity.get("high_side_decision_match_count", -1))
        == EXPECTED_DEV8_TRANSITIONS
    )
    mutation_free = parity.get("input_and_artifact_tensor_mutation_free") is True
    accepted = decision_parity and mutation_free and improvement >= 0.20
    if accepted and improvement >= 0.25:
        classification = "ACCEPT_FOR_RUNTIME_TRIAL_STRONG_CANDIDATE"
    elif accepted:
        classification = "ACCEPT_FOR_RUNTIME_TRIAL"
    else:
        classification = "REJECT_FOR_RUNTIME_TRIAL"
    return {
        "classification": classification,
        "runtime_like_wall_improvement_fraction_vs_eager_tensor": improvement,
        "runtime_like_wall_improvement_percent_vs_eager_tensor": improvement * 100.0,
        "decision_parity_7139_of_7139": decision_parity,
        "mutation_free": mutation_free,
        "minimum_required_improvement_percent": 20.0,
        "strong_candidate_improvement_percent": 25.0,
    }


def build_summary_text(results: Mapping[str, Any]) -> str:
    methods = results["timing_methods"]
    eager_tensor_wall = methods["eager_tensor"]["runtime_like_wall"]["mean_ms"]
    lines = [
        "Action-Delta scorer TorchInductor diagnostic experiment",
        "",
        f"device: {results['device']['name']}",
        f"samples: {results['input']['sample_count']}",
        f"full parity transitions: {results['input']['source_transition_count']}",
        f"repeats: {results['configuration']['repeats']}",
        "",
        "method                         wall_ms    speedup    decision_parity",
        "-----------------------------------------------------------------",
    ]
    eager_production = methods["eager_production"]["runtime_like_wall"]["mean_ms"]
    lines.append(
        f"{'eager production':<30} {eager_production:>8.6f}    "
        f"{eager_tensor_wall / eager_production:>7.3f}x    authoritative"
    )
    lines.append(
        f"{'eager tensor':<30} {eager_tensor_wall:>8.6f}    1.000x    authoritative"
    )
    for name, label in (
        ("compile_default", "compile default"),
        ("compile_reduce_overhead", "compile reduce-overhead"),
    ):
        method = methods[name]
        if method["status"] != "available":
            lines.append(f"{label:<30} {'FAILED':>8}    n/a       n/a")
            continue
        wall = method["runtime_like_wall"]["mean_ms"]
        parity = results["compiled_candidates"][name]["parity"]
        lines.append(
            f"{label:<30} {wall:>8.6f}    "
            f"{eager_tensor_wall / wall:>7.3f}x    "
            f"{parity['high_side_decision_match_count']}/"
            f"{parity['transition_count']}"
        )
    fastest = results.get("fastest_compiled_candidate")
    lines.extend(["", f"fastest candidate: {fastest or 'none'}"])
    if fastest:
        candidate = results["compiled_candidates"][fastest]
        parity = candidate["parity"]
        host = results["fastest_compiled_host_sync"]
        eager_kernels = results["kernel_launch_diagnostic"]["eager_tensor"]
        compiled_kernels = results["kernel_launch_diagnostic"][fastest]
        lines.extend(
            [
                "score exact match: "
                f"{parity['exact_score_match_count']}/{parity['transition_count']} "
                f"(bitwise={parity['scores_bitwise_identical']})",
                "decision parity: "
                f"{parity['high_side_decision_match_count']}/"
                f"{parity['transition_count']}",
                "max absolute score difference: "
                f"{parity['maximum_absolute_score_difference']:.12g}",
                "kernel count eager: "
                f"{eager_kernels.get('approximate_gpu_kernel_count_per_scorer', 'unavailable')}",
                "kernel count compiled: "
                f"{compiled_kernels.get('approximate_gpu_kernel_count_per_scorer', 'unavailable')}",
                "natural item: "
                f"{host['natural_item_wall']['mean_ms']:.9f} ms",
                "pre-item wait: "
                f"{host['pre_item_sync_wait']['mean_ms']:.9f} ms",
                "post-sync item: "
                f"{host['post_sync_item']['mean_ms']:.9f} ms",
                "runtime trial recommendation: "
                f"{candidate['acceptance']['classification']}",
            ]
        )
    lines.extend(
        [
            "",
            "Break-even diagnostics:",
            "  historical cost-model break-even: "
            f"{results['policy_economics']['break_even_scorer_cost_ms_per_call']:.9f} ms/call",
            "  empirical paired-runtime break-even: "
            f"{EMPIRICAL_DEV8_BREAK_EVEN_SCORER_MS:.9f} ms/call",
        ]
    )
    lines.extend(
        [
            "",
            "Compiled latency is a standalone measured micro-profile; projected dev8 values",
            "are not measured LIBERO speedups. Compilation and first-call time are excluded.",
            "CUDA profiler instrumentation is qualitative and not a primary timing source.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(
    results: Mapping[str, Any],
    output_dir: Path,
    *,
    per_call_latency_ms: Mapping[str, Sequence[float]] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.txt").write_text(
        build_summary_text(results), encoding="utf-8"
    )
    if per_call_latency_ms is not None:
        with (output_dir / "eager_vs_compiled_per_call.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=("method", "call_index", "latency_ms")
            )
            writer.writeheader()
            for method, values in per_call_latency_ms.items():
                for index, value in enumerate(values):
                    writer.writerow(
                        {
                            "method": method,
                            "call_index": index,
                            "latency_ms": value,
                        }
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-count", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--write-per-call-csv", action="store_true")
    args = parser.parse_args()
    _require(torch.cuda.is_available(), "CUDA is required for scorer profiling")
    _require(args.warmup >= 100, "warmup must be at least 100")
    _require(args.repetitions >= 1000, "repetitions must be at least 1000")
    _require(args.repeats >= 5, "repeat count must be at least 5")

    cpu_all_anchors, cpu_all_currents, all_identities, input_provenance = (
        load_real_transition_sample(
            args.manifest, sample_count=EXPECTED_DEV8_TRANSITIONS
        )
    )
    _require(
        int(input_provenance["source_transition_count"])
        == EXPECTED_DEV8_TRANSITIONS,
        "compile experiment requires all 7,139 dev-shadow transitions",
    )
    _require(
        1 <= args.sample_count <= EXPECTED_DEV8_TRANSITIONS,
        "timing sample count is out of range",
    )
    artifact_hash = sha256_file(args.artifact)
    _require(artifact_hash == EXPECTED_ARTIFACT_SHA256, "artifact SHA-256 mismatch")
    artifact_manifest, payload = load_action_delta_gate_artifact(
        args.artifact,
        expected_sha256=EXPECTED_ARTIFACT_SHA256,
    )
    gate = prepare_action_delta_gate_shadow(payload, device="cuda", task_id=0)
    all_anchors = cpu_all_anchors.to(device="cuda")
    all_currents = cpu_all_currents.to(device="cuda")
    _require(tuple(all_anchors.shape[1:]) == RUNTIME_SHAPE, "CUDA anchor shape mismatch")
    _require(tuple(all_currents.shape[1:]) == RUNTIME_SHAPE, "CUDA current shape mismatch")
    timing_indices = torch.from_numpy(
        np.linspace(
            0,
            EXPECTED_DEV8_TRANSITIONS - 1,
            num=args.sample_count,
        )
        .round()
        .astype(np.int64)
    ).to(device="cuda")
    anchors = all_anchors.index_select(0, timing_indices)
    currents = all_currents.index_select(0, timing_indices)
    timing_identity_indices = timing_indices.cpu().tolist()
    identities = [all_identities[index] for index in timing_identity_indices]

    validation = validate_decomposition(gate, anchors, currents)
    compiled_runtime = compile_tensor_candidates(
        gate, anchors[0], currents[0]
    )
    _authoritative_scores, all_transition_parity = evaluate_candidate_parity(
        gate,
        all_anchors,
        all_currents,
        all_identities,
        compiled_runtime,
    )
    faithful, faithful_calls = benchmark_faithful_runtime(
        gate,
        anchors,
        currents,
        warmup=args.warmup,
        repetitions=args.repetitions,
        repeats=args.repeats,
    )
    eager_tensor_event, _eager_tensor_event_calls = benchmark_tensor_cuda_events(
        tensor_action_delta_score,
        gate,
        anchors,
        currents,
        warmup=args.warmup,
        repetitions=args.repetitions,
        repeats=args.repeats,
    )
    eager_tensor_wall, eager_tensor_wall_calls = benchmark_tensor_runtime_wall(
        tensor_action_delta_score,
        gate,
        anchors,
        currents,
        warmup=args.warmup,
        repetitions=args.repetitions,
        repeats=args.repeats,
    )
    timing_methods: dict[str, Any] = {
        "eager_production": {
            "status": "available",
            "runtime_like_wall": faithful,
            "includes_production_validation_and_host_decision": True,
        },
        "eager_tensor": {
            "status": "available",
            "tensor_only_cuda_event": eager_tensor_event,
            "runtime_like_wall": eager_tensor_wall,
        },
    }
    per_call_latency: dict[str, Sequence[float]] = {
        "eager_production_wall": faithful_calls,
        "eager_tensor_wall": eager_tensor_wall_calls,
    }
    compiled_candidate_results: dict[str, Any] = {}
    for name, compiled in compiled_runtime.items():
        if compiled["status"] != "available":
            timing_methods[name] = {
                "status": "failed",
                "error_type": compiled["error_type"],
                "error": compiled["error"],
            }
            compiled_candidate_results[name] = {
                key: value for key, value in compiled.items() if key != "callable"
            }
            compiled_candidate_results[name]["parity"] = all_transition_parity[name]
            compiled_candidate_results[name]["acceptance"] = {
                "classification": "REJECT_FOR_RUNTIME_TRIAL",
                "reason": "compilation failed",
            }
            continue
        event_timing, _event_calls = benchmark_tensor_cuda_events(
            compiled["callable"],
            gate,
            anchors,
            currents,
            warmup=args.warmup,
            repetitions=args.repetitions,
            repeats=args.repeats,
        )
        wall_timing, wall_calls = benchmark_tensor_runtime_wall(
            compiled["callable"],
            gate,
            anchors,
            currents,
            warmup=args.warmup,
            repetitions=args.repetitions,
            repeats=args.repeats,
        )
        timing_methods[name] = {
            "status": "available",
            "tensor_only_cuda_event": event_timing,
            "runtime_like_wall": wall_timing,
        }
        per_call_latency[f"{name}_wall"] = wall_calls
        parity = all_transition_parity[name]
        compiled_candidate_results[name] = {
            key: value for key, value in compiled.items() if key != "callable"
        }
        compiled_candidate_results[name].update(
            {
                "parity": parity,
                "timing": timing_methods[name],
                "economics": compiled_candidate_economics(
                    wall_timing["mean_ms"],
                    scorer_calls=EXPECTED_DEV8_TRANSITIONS,
                ),
                "acceptance": classify_compiled_candidate(
                    parity,
                    candidate_wall_ms=wall_timing["mean_ms"],
                    eager_tensor_wall_ms=eager_tensor_wall["mean_ms"],
                ),
            }
        )
    components = benchmark_components(
        gate,
        anchors,
        currents,
        warmup=args.warmup,
        repetitions=args.repetitions,
        repeats=args.repeats,
    )
    eager_host_sync = benchmark_host_sync(
        gate,
        anchors,
        currents,
        warmup=args.warmup,
        repetitions=args.repetitions,
        repeats=args.repeats,
    )
    available_compiled = [
        name
        for name in ("compile_default", "compile_reduce_overhead")
        if timing_methods[name]["status"] == "available"
    ]
    fastest_compiled = (
        min(
            available_compiled,
            key=lambda name: timing_methods[name]["runtime_like_wall"]["mean_ms"],
        )
        if available_compiled
        else None
    )
    fastest_host_sync = None
    if fastest_compiled is not None:
        fastest_host_sync = benchmark_tensor_host_sync(
            compiled_runtime[fastest_compiled]["callable"],
            gate,
            anchors,
            currents,
            warmup=args.warmup,
            repetitions=args.repetitions,
            repeats=args.repeats,
        )
    kernel_diagnostic: dict[str, Any] = {
        "eager_tensor": profile_kernel_launches(
            tensor_action_delta_score, gate, anchors[0], currents[0]
        )
    }
    for name in available_compiled:
        kernel_diagnostic[name] = profile_kernel_launches(
            compiled_runtime[name]["callable"], gate, anchors[0], currents[0]
        )
    scorer_calls = int(input_provenance["source_transition_count"])
    results = {
        "schema_version": 2,
        "diagnostic_only": DIAGNOSTIC_ONLY,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": {
            "name": torch.cuda.get_device_name(torch.cuda.current_device()),
            "index": int(torch.cuda.current_device()),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "artifact": {
            "path": str(args.artifact.resolve()),
            "sha256": artifact_hash,
            "manifest": artifact_manifest,
        },
        "input": {
            **input_provenance,
            "loaded_transition_count": input_provenance["sample_count"],
            "sample_count": args.sample_count,
            "timing_selection": "evenly_spaced_full_dev8_transition_indices",
            "full_parity_transition_count": EXPECTED_DEV8_TRANSITIONS,
            "runtime_shape": list(RUNTIME_SHAPE),
            "sample_task_distribution": dict(
                sorted(
                    {
                        str(task_id): sum(
                            int(item["task_id"] == task_id) for item in identities
                        )
                        for task_id in DEVELOPMENT_TASK_IDS
                    }.items()
                )
            ),
        },
        "configuration": {
            "warmup_per_repeat": args.warmup,
            "measured_calls_per_repeat": args.repetitions,
            "repeats": args.repeats,
            "high_side_threshold": HIGH_SIDE_THRESHOLD,
            "bf16_quantization_contract": (
                "(current.float() - anchor.float()).to(torch.bfloat16).float()"
            ),
            "compile_candidates": {
                "compile_default": {
                    "fullgraph": True,
                    "dynamic": False,
                },
                "compile_reduce_overhead": {
                    "fullgraph": True,
                    "dynamic": False,
                    "mode": "reduce-overhead",
                },
            },
            "compilation_first_call_and_warmup_excluded": True,
            "autograd_enabled": False,
        },
        "score_validation": validation,
        "all_transition_eager_tensor_parity": all_transition_parity["eager_tensor"],
        "timing_methods": timing_methods,
        "compiled_candidates": compiled_candidate_results,
        "fastest_compiled_candidate": fastest_compiled,
        "component_breakdown": components,
        "eager_production_host_sync_decomposition": eager_host_sync,
        "fastest_compiled_host_sync": fastest_host_sync,
        "kernel_launch_diagnostic": kernel_diagnostic,
        "observed_dev8_runtime_scorer_mean_ms_for_comparison": (
            OBSERVED_DEV8_SCORER_MS
        ),
        "policy_economics": policy_economics(
            faithful["mean_ms"], scorer_calls=scorer_calls
        ),
        "production_source_provenance": {
            "action_delta_gate_path": str(
                Path("prismatic/models/action_delta_gate.py").resolve()
            ),
            "action_delta_gate_sha256": sha256_file(
                Path("prismatic/models/action_delta_gate.py")
            ),
            "action_heads_path": str(Path("prismatic/models/action_heads.py").resolve()),
            "action_heads_sha256": sha256_file(
                Path("prismatic/models/action_heads.py")
            ),
            "modified_by_compile_experiment": False,
        },
        "interpretation_limits": [
            "No LIBERO control path was executed or modified.",
            "CUDA-event component timings are diagnostic and not additive wall time.",
            "Hypothetical reductions are arithmetic estimates, not measured improvements.",
            "Projected dev8 compiled economics are not a measured LIBERO speedup.",
            "Production runtime scorer and action_heads.py are not modified by this experiment.",
        ],
    }
    write_outputs(
        results,
        args.output_dir.resolve(),
        per_call_latency_ms=(per_call_latency if args.write_per_call_csv else None),
    )
    print(build_summary_text(results), end="")


if __name__ == "__main__":
    main()
