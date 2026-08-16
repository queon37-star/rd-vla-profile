"""CUDA microbenchmark for legacy and optimized Action-Delta Gate scoring."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from prismatic.models.action_delta_gate import (
    NonFiniteActionDeltaGateError,
    evaluate_action_delta_gate,
    load_action_delta_gate_artifact,
    prepare_action_delta_gate,
    score_action_delta_gate,
)


def legacy_evaluate(gate, anchor_state, current_state):
    """The pre-optimization six-synchronization decision path."""

    for name, state in (("anchor_state", anchor_state), ("current_state", current_state)):
        if not bool(torch.isfinite(state).all().item()):
            raise NonFiniteActionDeltaGateError(f"{name} is non-finite")

    delta = (
        current_state.float() - anchor_state.float()
    ).to(torch.bfloat16).float()
    x = (delta - gate.x_mean) / gate.x_std
    pred_norm = F.linear(x, gate.linear_weight, gate.linear_bias)
    pred_delta = pred_norm * gate.y_std + gate.y_mean
    score = pred_delta.square().mean()

    if not bool(torch.isfinite(x).all().item()):
        raise NonFiniteActionDeltaGateError("normalized input is non-finite")
    if not bool(torch.isfinite(pred_delta).all().item()):
        raise NonFiniteActionDeltaGateError("prediction is non-finite")
    if not bool(torch.isfinite(score).item()):
        raise NonFiniteActionDeltaGateError("score is non-finite")
    score_value = float(score.item())
    return score_value, score_value <= gate.threshold


def percentile_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "mean_ms": float(values.mean()),
    }


def benchmark_synchronized(fn, gate, anchor, currents, warmup, repetitions):
    for index in range(warmup):
        fn(gate, anchor, currents[index % len(currents):index % len(currents) + 1])

    latencies = []
    for index in range(repetitions):
        current = currents[index % len(currents):index % len(currents) + 1]
        start = time.perf_counter_ns()
        fn(gate, anchor, current)
        latencies.append((time.perf_counter_ns() - start) / 1e6)
    return percentile_summary(latencies)


def benchmark_tensor_only(gate, anchor, currents, warmup, repetitions):
    for index in range(warmup):
        current = currents[index % len(currents):index % len(currents) + 1]
        score_action_delta_gate(gate, anchor, current)
    torch.cuda.synchronize()

    event_pairs = []
    for index in range(repetitions):
        current = currents[index % len(currents):index % len(currents) + 1]
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        score_action_delta_gate(gate, anchor, current)
        end.record()
        event_pairs.append((start, end))
    torch.cuda.synchronize()
    return percentile_summary([start.elapsed_time(end) for start, end in event_pairs])


def validate_score_parity(gate, anchor, currents):
    max_score_difference = 0.0
    decision_mismatches = 0
    for index in range(len(currents)):
        current = currents[index:index + 1]
        legacy_score, legacy_decision = legacy_evaluate(gate, anchor, current)
        optimized_score, optimized_decision = evaluate_action_delta_gate(
            gate, anchor, current
        )
        max_score_difference = max(
            max_score_difference, abs(legacy_score - optimized_score)
        )
        decision_mismatches += int(legacy_decision != optimized_decision)
    if max_score_difference != 0.0 or decision_mismatches != 0:
        raise RuntimeError(
            "legacy/optimized gate parity failed: "
            f"max_score_difference={max_score_difference}, "
            f"decision_mismatches={decision_mismatches}"
        )
    return {
        "max_score_difference": max_score_difference,
        "decision_mismatches": decision_mismatches,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("benchmark_results/coda_anchor_feasibility/action_delta_gate_fold4"),
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("benchmark_results/coda_anchor_feasibility/action_delta_cache.pt"),
    )
    parser.add_argument("--warmup", type=int, default=250)
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--sample-count", type=int, default=512)
    parser.add_argument("--coda-ms", type=float, default=1.95)
    parser.add_argument("--trigger-probability", type=float, default=0.15)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Action-Delta Gate microbenchmark")

    manifest_path = args.artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _, payload = load_action_delta_gate_artifact(
        args.artifact,
        expected_sha256=manifest["artifact_sha256"],
    )
    gate = prepare_action_delta_gate(payload, device="cuda", task_id=4)
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    fold4 = torch.where(cache["folds"] == 4)[0]
    if len(fold4) < args.sample_count:
        raise RuntimeError("not enough fold-4 transitions for the requested sample count")
    positions = torch.linspace(
        0,
        len(fold4) - 1,
        steps=args.sample_count,
        dtype=torch.float64,
    ).round().long()
    selected = fold4[positions]
    currents = cache["delta_states"][selected].to(device="cuda")
    anchor = torch.zeros(1, 8, 896, device="cuda", dtype=currents.dtype)

    score_parity = validate_score_parity(gate, anchor, currents)
    legacy = benchmark_synchronized(
        legacy_evaluate, gate, anchor, currents, args.warmup, args.repetitions
    )
    optimized = benchmark_synchronized(
        evaluate_action_delta_gate,
        gate,
        anchor,
        currents,
        args.warmup,
        args.repetitions,
    )
    tensor_only = benchmark_tensor_only(
        gate, anchor, currents, args.warmup, args.repetitions
    )
    break_even_ms = args.coda_ms * args.trigger_probability

    result = {
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "artifact_sha256": manifest["artifact_sha256"],
        "transition_source": str(args.cache),
        "sample_count": args.sample_count,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "score_parity": score_parity,
        "legacy_six_sync": legacy,
        "optimized_one_sync": optimized,
        "tensor_only_cuda_events": tensor_only,
        "estimated_host_decision_overhead_p50_ms": max(
            0.0, optimized["p50_ms"] - tensor_only["p50_ms"]
        ),
        "break_even": {
            "coda_ms": args.coda_ms,
            "trigger_probability": args.trigger_probability,
            "maximum_gate_ms": break_even_ms,
            "optimized_p50_is_break_even": optimized["p50_ms"] < break_even_ms,
            "optimized_mean_is_break_even": optimized["mean_ms"] < break_even_ms,
            "optimized_p95_is_break_even": optimized["p95_ms"] < break_even_ms,
            "expected_net_savings_ms_per_score": (
                break_even_ms - optimized["p50_ms"]
            ),
        },
    }
    if not all(
        math.isfinite(value)
        for section in (legacy, optimized, tensor_only)
        for value in section.values()
    ):
        raise RuntimeError("microbenchmark produced a non-finite latency")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
