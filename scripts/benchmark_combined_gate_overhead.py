#!/usr/bin/env python3
"""Profile the frozen Combined adaptive-Coda gate on captured GPU workloads."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import numpy as np  # noqa: E402

from prismatic.models.action_head_workload import load_action_head_workload  # noqa: E402
from scripts.combined_gate_microbenchmark_lib import (  # noqa: E402
    DEFAULT_ORDER_SEED,
    EXPECTED_TRACE_IDENTITY,
    OPERATIONS,
    SCHEMA_VERSION,
    STATE_CASES,
    CombinedGateMicrobenchmarkError,
    StateCase,
    assert_case_parity,
    bootstrap_projection,
    combined_current_diagnostic,
    combined_optimized_decision,
    combined_optimized_tensor,
    compare_independent_runs,
    deterministic_operation_order,
    feature_family_tensors,
    load_actual_warm_workload_descriptors,
    load_fixed_raw_mse_thresholds,
    load_scheduler_replays,
    load_serialized_combined_models,
    model_to_device,
    optimized_host_transfer_audit,
    project_policy_latency,
    raw_mse_decision,
    raw_mse_tensor,
    select_stratified_workloads,
    sha256_file,
    summarize_latency_samples,
    write_json,
)
from scripts.run_origin_aware_gpu_microbenchmark import load_benchmark_modules  # noqa: E402


DEFAULT_WORKLOAD_MANIFEST = (
    REPO_ROOT / "benchmark_results/learned_convergence_probe/20260801_seed7/dataset/manifest.json"
)
DEFAULT_CHECKPOINT = REPO_ROOT / "outputs/12_24-24_24_Spatial_40k"
DEFAULT_MODEL_ARTIFACTS = (
    REPO_ROOT
    / "benchmark_results/latent_dynamics_features/calibration_b17fe9d/adaptive_coda_gate_oof_975da90"
)
OUTPUT_FILENAMES = (
    "benchmark_config.json",
    "workload_inventory.json",
    "correctness_report.json",
    "latency_samples.csv",
    "latency_summary.json",
    "profiler_summary.json",
    "memory_summary.json",
    "break_even_analysis.json",
    "metric_report.json",
    "output_hashes.json",
)


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def _git_status() -> list[str]:
    return subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
    ).splitlines()


def _nvidia_query(fields: str) -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return result.stdout.strip().splitlines()[0] if result.stdout.strip() else None


def _gpu_telemetry() -> dict[str, Any]:
    raw = _nvidia_query("temperature.gpu,power.draw")
    if raw is None:
        return {"available": False, "temperature_c": None, "power_w": None}
    values = [value.strip() for value in raw.split(",")]
    try:
        return {
            "available": True,
            "temperature_c": float(values[0]),
            "power_w": float(values[1]),
        }
    except (ValueError, IndexError):
        return {"available": False, "temperature_c": None, "power_w": None}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload-manifest", type=Path, default=DEFAULT_WORKLOAD_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-artifacts", type=Path, default=DEFAULT_MODEL_ARTIFACTS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workload-count", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--order-seed", type=int, default=DEFAULT_ORDER_SEED)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--compare-to", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _prepare_gpu_tensors(payload: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    tensors = {
        name: None if value is None else value.to(device=device, non_blocking=False)
        for name, value in payload["tensors"].items()
    }
    for name, tensor in tensors.items():
        if tensor is not None and not tensor.is_contiguous():
            raise CombinedGateMicrobenchmarkError(f"{name}: GPU layout is not contiguous")
    return tensors


def _prepare_context(action_head, tensors: Mapping[str, torch.Tensor]):
    model = action_head.model
    actions = tensors["actions_hidden_states"]
    h_t = actions[:, :, : action_head.num_task_tokens, :]
    h_a = actions[:, :, action_head.num_task_tokens :, :]
    p = tensors["proprio_features"]
    x = model.action_queries.unsqueeze(0).expand(actions.shape[0], -1, -1).to(actions.dtype)
    for index, layer in enumerate(model.prelude):
        x = layer(
            x,
            h_a[:, model.prelude_vlm_layers[index]],
            h_t[:, model.prelude_vlm_layers[index]],
            p,
        )
    return x, h_a, h_t, p


def _generate_state_cases(action_head, tensors: Mapping[str, torch.Tensor]):
    model = action_head.model
    prelude, h_a, h_t, p = _prepare_context(action_head, tensors)
    anchor = tensors["selected_initial_state"]
    state1 = model._run_one_iteration(anchor, prelude, h_a, h_t, p)
    state2 = model._run_one_iteration(state1, prelude, h_a, h_t, p)
    state3 = model._run_one_iteration(state2, prelude, h_a, h_t, p)
    cases = (
        StateCase("k2", 2, state2, state1, None, anchor),
        StateCase("k_ge_3", 3, state3, state2, state2.float() - state1.float(), anchor),
    )
    return cases, (prelude, h_a, h_t, p)


def _warm_sk1(action_head, context, anchor):
    _prelude, h_a, h_t, p = context
    return action_head.model(
        h_a,
        h_t,
        p,
        num_iter=1,
        convergence_strategy=None,
        warm_start_state=anchor,
        enable_warm_start=True,
        warm_start_source="midpoint",
        warm_start_min_iter=2,
        validate_warm_start_finite=False,
        profile_coda_cost=False,
        use_cached_final_output=False,
        use_latent_precheck=False,
        latent_precheck_mode="off",
        shadow_full_depth=False,
        capture_action_head_workload=False,
    )


def _operation_callables(
    action_head,
    state_case: StateCase,
    context,
    serialized_model,
    tensor_model,
    fixed_threshold: float,
) -> dict[str, Callable[[], Any]]:
    prelude, h_a, h_t, p = context
    model = action_head.model
    return {
        "raw_mse_tensor": lambda: raw_mse_tensor(
            state_case.current_state, state_case.previous_state
        ),
        "raw_mse_decision": lambda: raw_mse_decision(
            state_case.current_state, state_case.previous_state, fixed_threshold
        ),
        "combined_current_diagnostic": lambda: combined_current_diagnostic(
            state_case.current_state,
            state_case.previous_state,
            state_case.previous_update,
            state_case.warm_anchor,
            serialized_model,
        ),
        "combined_optimized_tensor": lambda: combined_optimized_tensor(
            state_case.current_state,
            state_case.previous_state,
            state_case.previous_update,
            state_case.warm_anchor,
            tensor_model,
        ),
        "combined_optimized_decision": lambda: combined_optimized_decision(
            state_case.current_state,
            state_case.previous_state,
            state_case.previous_update,
            state_case.warm_anchor,
            tensor_model,
        ),
        "coda_get_output": lambda: model._get_output(
            state_case.current_state, h_a, h_t, p, profile=False
        ),
        "recurrent_one_iteration": lambda: model._run_one_iteration(
            state_case.current_state, prelude, h_a, h_t, p
        ),
        "warm_start_sk1_action_head": lambda: _warm_sk1(
            action_head, context, state_case.warm_anchor
        ),
    }


def _measure_call(
    function: Callable[[], Any], device: torch.device
) -> tuple[float, float]:
    torch.cuda.synchronize(device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter_ns()
    start_event.record()
    result = function()
    end_event.record()
    torch.cuda.synchronize(device)
    wall_ms = (time.perf_counter_ns() - wall_start) / 1_000_000.0
    device_ms = float(start_event.elapsed_time(end_event))
    del result
    if not all(math.isfinite(value) and value >= 0 for value in (device_ms, wall_ms)):
        raise CombinedGateMicrobenchmarkError("non-finite timing result")
    return device_ms, wall_ms


def _measure_peak_memory(function: Callable[[], Any], device: torch.device) -> dict[str, int]:
    torch.cuda.synchronize(device)
    baseline = int(torch.cuda.memory_allocated(device))
    torch.cuda.reset_peak_memory_stats(device)
    result = function()
    torch.cuda.synchronize(device)
    peak = int(torch.cuda.max_memory_allocated(device))
    del result
    return {
        "baseline_allocated_bytes": baseline,
        "peak_allocated_bytes": peak,
        "incremental_peak_bytes": max(0, peak - baseline),
    }


def _profile_operation(
    operation: str,
    function: Callable[[], Any],
    device: torch.device,
    *,
    history_available: bool,
) -> dict[str, Any]:
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    torch.cuda.synchronize(device)
    with torch.profiler.profile(activities=activities, record_shapes=False) as profile:
        with torch.profiler.record_function(f"combined_gate/{operation}"):
            result = function()
        torch.cuda.synchronize(device)
    del result
    averages = profile.key_averages()
    cuda_ops = [item for item in averages if item.self_cuda_time_total > 0]
    major = sorted(cuda_ops, key=lambda item: item.self_cuda_time_total, reverse=True)[:10]
    cuda_events = [
        event
        for event in profile.events()
        if str(event.device_type).lower().endswith("cuda")
    ]
    known_items = {
        "raw_mse_tensor": 0,
        "raw_mse_decision": 1,
        "combined_current_diagnostic": 20 if history_available else 16,
        "combined_optimized_tensor": 0,
        "combined_optimized_decision": 1,
        "coda_get_output": 0,
        "recurrent_one_iteration": 0,
        "warm_start_sk1_action_head": 0,
    }[operation]
    return {
        "operation": operation,
        "cuda_kernel_count": len(cuda_events),
        "cuda_operator_count": len(cuda_ops),
        "cpu_operator_count": len(averages),
        "known_tensor_item_or_host_decision_count": known_items,
        "major_cuda_operators": [
            {
                "name": item.key,
                "call_count": int(item.count),
                "self_cuda_time_total_us": float(item.self_cuda_time_total),
            }
            for item in major
        ],
    }


def _workload_inventory_row(descriptor, payload, fold_id: int) -> dict[str, Any]:
    return {
        "workload_id": descriptor["workload_id"],
        "identity": descriptor["identity"],
        "actual_origin": descriptor["actual_origin"],
        "outer_fold": fold_id,
        "path": str(Path(descriptor["path"]).resolve()),
        "sha256": descriptor["sha256"],
        "tensor_metadata": payload["tensor_metadata"],
        "layout_validated": True,
    }


def _latency_values(samples, operation: str) -> list[float]:
    return [
        float(row["wall_time_ms"])
        for row in samples
        if row["operation"] == operation
    ]


def _build_break_even(samples, artifact_dir: Path, draws: int, seed: int):
    replays = load_scheduler_replays(artifact_dir)
    common = {
        "coda": _latency_values(samples, "coda_get_output"),
        "recurrent": _latency_values(samples, "recurrent_one_iteration"),
        "baseline": _latency_values(samples, "warm_start_sk1_action_head"),
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "latency_projection_not_end_to_end_LIBERO",
        "formula": (
            "saved_coda_calls_per_prediction*coda_latency - "
            "mean_delta_K*recurrent_iteration_latency - "
            "gate_evaluations_per_prediction*gate_decision_latency"
        ),
        "latency_statistic": "arithmetic mean of synchronized per-call wall samples",
        "policies": {},
    }
    gates = {
        "fixed_raw_mse": _latency_values(samples, "raw_mse_decision"),
        "combined": _latency_values(samples, "combined_optimized_decision"),
    }
    for policy in ("fixed_raw_mse", "combined"):
        latency = {**common, "gate": gates[policy]}
        projection = project_policy_latency(
            replays[policy],
            coda_latency_ms=float(np.mean(latency["coda"])),
            recurrent_iteration_latency_ms=float(np.mean(latency["recurrent"])),
            gate_decision_latency_ms=float(np.mean(latency["gate"])),
            baseline_action_head_latency_ms=float(np.mean(latency["baseline"])),
        )
        projection["bootstrap"] = bootstrap_projection(
            replays[policy], latency, draws=draws, seed=seed
        )
        result["policies"][policy] = projection
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_outputs(
    output_dir: Path,
    reports: Mapping[str, Any],
    latency_samples: Sequence[Mapping[str, Any]],
    *,
    overwrite: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [name for name in OUTPUT_FILENAMES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite benchmark outputs: {existing}")
    for name, value in reports.items():
        write_json(output_dir / name, value)
    _write_csv(output_dir / "latency_samples.csv", latency_samples)
    hashes = {
        name: sha256_file(output_dir / name) for name in OUTPUT_FILENAMES[:-1]
    }
    write_json(
        output_dir / "output_hashes.json",
        {"schema_version": SCHEMA_VERSION, "hash_algorithm": "sha256", "files": hashes},
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.warmup < 1 or args.repeats < 1 or args.trials < 3:
        raise ValueError("benchmark requires warmup>=1, repeats>=1 and trials>=3")
    commit = _git_commit()
    output_dir = args.output_dir or (
        REPO_ROOT / f"benchmark_results/combined_gate_overhead/{commit[:7]}"
    )
    descriptors, workload_provenance = load_actual_warm_workload_descriptors(
        args.workload_manifest
    )
    selected = select_stratified_workloads(descriptors, args.workload_count)
    models, task_to_fold, model_provenance = load_serialized_combined_models(
        args.model_artifacts
    )
    fixed_thresholds = load_fixed_raw_mse_thresholds(args.model_artifacts)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Combined gate real microbenchmark requires CUDA; no report was written"
        )
    device = torch.device("cuda:0")
    torch.manual_seed(args.order_seed)
    torch.cuda.manual_seed_all(args.order_seed)
    random.seed(args.order_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    start_telemetry = _gpu_telemetry()
    action_head, _proprio_projector, checkpoint = load_benchmark_modules(
        args.checkpoint, device
    )
    tensor_models = {
        fold_id: model_to_device(model, device) for fold_id, model in models.items()
    }
    samples = []
    inventory = []
    correctness_rows = []
    memory_rows = []
    profiler_rows = []
    decomposition = []
    with torch.inference_mode():
        for workload_index, descriptor in enumerate(selected):
            payload = load_action_head_workload(
                descriptor["path"],
                expected_sha256=descriptor["sha256"],
                expected_identity=descriptor["identity"],
                expected_origin="ACTUAL_WARM",
            )
            tensors = _prepare_gpu_tensors(payload, device)
            task_id = int(descriptor["identity"]["task_id"])
            fold_id = task_to_fold[task_id]
            serialized = models[fold_id]
            tensor_model = tensor_models[fold_id]
            inventory.append(_workload_inventory_row(descriptor, payload, fold_id))
            cases, context = _generate_state_cases(action_head, tensors)
            before_coda = {
                case.case_id: action_head.model._get_output(
                    case.current_state, context[1], context[2], context[3]
                ).detach().clone()
                for case in cases
            }
            for case_index, state_case in enumerate(cases):
                correctness = assert_case_parity(state_case, serialized, tensor_model)
                correctness.update(
                    {
                        "workload_id": descriptor["workload_id"],
                        "task_id": task_id,
                        "outer_fold": fold_id,
                    }
                )
                correctness_rows.append(correctness)
                operations = _operation_callables(
                    action_head,
                    state_case,
                    context,
                    serialized,
                    tensor_model,
                    fixed_thresholds[fold_id],
                )
                for operation in OPERATIONS:
                    for _ in range(args.warmup):
                        operations[operation]()
                    torch.cuda.synchronize(device)
                for trial_index in range(args.trials):
                    for repeat_index in range(args.repeats):
                        order = deterministic_operation_order(
                            workload_index=workload_index,
                            trial_index=trial_index * args.repeats + repeat_index,
                            case_index=case_index,
                            seed=args.order_seed,
                        )
                        for position, operation in enumerate(order):
                            device_ms, wall_ms = _measure_call(operations[operation], device)
                            samples.append(
                                {
                                    "workload_id": descriptor["workload_id"],
                                    "task_id": task_id,
                                    "outer_fold": fold_id,
                                    "case_id": state_case.case_id,
                                    "iteration": state_case.iteration,
                                    "operation": operation,
                                    "trial_index": trial_index,
                                    "repeat_index": repeat_index,
                                    "order_position": position,
                                    "cuda_event_ms": device_ms,
                                    "wall_time_ms": wall_ms,
                                }
                            )
                if workload_index == 0:
                    for operation in OPERATIONS:
                        memory_rows.append(
                            {
                                "operation": operation,
                                "case_id": state_case.case_id,
                                **_measure_peak_memory(operations[operation], device),
                            }
                        )
                        profiler_rows.append(
                            _profile_operation(
                                operation,
                                operations[operation],
                                device,
                                history_available=(state_case.previous_update is not None),
                            )
                            | {"case_id": state_case.case_id}
                        )
                    for family, function in feature_family_tensors(
                        state_case.current_state,
                        state_case.previous_state,
                        state_case.previous_update,
                        state_case.warm_anchor,
                        tensor_model,
                    ).items():
                        device_ms, wall_ms = _measure_call(function, device)
                        decomposition.append(
                            {
                                "feature_family": family,
                                "case_id": state_case.case_id,
                                "cuda_event_ms": device_ms,
                                "synchronized_wall_time_ms": wall_ms,
                                "standalone_decomposition_not_additive": True,
                            }
                        )
            for name, tensor in tensors.items():
                if tensor is not None and not torch.equal(tensor.detach().cpu(), payload["tensors"][name]):
                    raise CombinedGateMicrobenchmarkError(f"{descriptor['workload_id']}: input tensor mutated: {name}")
            for state_case in cases:
                after = action_head.model._get_output(
                    state_case.current_state, context[1], context[2], context[3]
                )
                if not torch.equal(before_coda[state_case.case_id], after):
                    raise CombinedGateMicrobenchmarkError(
                        f"{descriptor['workload_id']}:{state_case.case_id}: Coda output changed"
                    )
            del tensors, payload, cases, context
            if (workload_index + 1) % 10 == 0 or workload_index + 1 == len(selected):
                print(f"Profiled workloads: {workload_index + 1}/{len(selected)}", flush=True)
    latency_summary = summarize_latency_samples(samples)
    break_even = _build_break_even(
        samples, args.model_artifacts, args.bootstrap_draws, args.order_seed
    )
    end_telemetry = _gpu_telemetry()
    properties = torch.cuda.get_device_properties(device)
    environment = {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "cuda_device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_total_memory_bytes": int(properties.total_memory),
        "driver_version": _nvidia_query("driver_version"),
        "start_gpu_telemetry": start_telemetry,
        "end_gpu_telemetry": end_telemetry,
    }
    config = {
        "schema_version": SCHEMA_VERSION,
        "git_commit": commit,
        "git_status_porcelain": _git_status(),
        "workload_count": len(selected),
        "required_primary_workload_count": 200,
        "formal_200_workload_run": len(selected) == 200,
        "warmup_count": args.warmup,
        "repeat_count": args.repeats,
        "trial_count": args.trials,
        "random_operation_order_seed": args.order_seed,
        "bootstrap_draws": args.bootstrap_draws,
        "torch_inference_mode": True,
        "inputs_resident_on_gpu_before_timing": True,
        "timing_excludes_model_loading_disk_io_transfer_and_report_writing": True,
        "cuda_event_and_wall_time_are_distinct": True,
        "environment": environment,
    }
    inventory_report = {
        "schema_version": SCHEMA_VERSION,
        "selection": "deterministic task-round-robin stratification",
        "origin": "ACTUAL_WARM",
        "workload_provenance": workload_provenance,
        "selected_count": len(inventory),
        "counts_by_task": dict(sorted(Counter(str(row["identity"]["task_id"]) for row in inventory).items())),
        "workloads": inventory,
        "full_latent_states_serialized": False,
    }
    correctness_report = {
        "schema_version": SCHEMA_VERSION,
        "tolerance": {"rtol": 1e-5, "atol": 1e-6},
        "optimized_host_transfer_audit": optimized_host_transfer_audit(),
        "all_cases_passed": all(row["parity_passed"] for row in correctness_rows),
        "all_inputs_bitwise_unchanged": all(row["inputs_bitwise_unchanged"] for row in correctness_rows),
        "coda_outputs_unchanged": True,
        "case_count": len(correctness_rows),
        "cases": correctness_rows,
    }
    profiler_summary = {
        "schema_version": SCHEMA_VERSION,
        "separate_from_latency_samples": True,
        "operations": profiler_rows,
        "feature_family_decomposition": decomposition,
    }
    memory_summary = {
        "schema_version": SCHEMA_VERSION,
        "shared_model_parameters_excluded_from_incremental_interpretation": True,
        "representative_workload_only": True,
        "operations": memory_rows,
    }
    stability = None
    if args.compare_to:
        previous_summary = json.loads(
            (args.compare_to / "latency_summary.json").read_text(encoding="utf-8")
        )
        stability = compare_independent_runs(latency_summary, previous_summary)
        previous_break_even = json.loads(
            (args.compare_to / "break_even_analysis.json").read_text(encoding="utf-8")
        )
        stability["same_projected_net_sign"] = {
            policy: math.copysign(
                1, break_even["policies"][policy]["projected_net_saving_ms"]
            )
            == math.copysign(
                1,
                previous_break_even["policies"][policy]["projected_net_saving_ms"],
            )
            for policy in ("fixed_raw_mse", "combined")
        }
        previous_config = json.loads(
            (args.compare_to / "benchmark_config.json").read_text(encoding="utf-8")
        )
        previous_inventory = json.loads(
            (args.compare_to / "workload_inventory.json").read_text(encoding="utf-8")
        )
        previous_correctness = json.loads(
            (args.compare_to / "correctness_report.json").read_text(encoding="utf-8")
        )
        previous_metric_report = json.loads(
            (args.compare_to / "metric_report.json").read_text(encoding="utf-8")
        )
        config_identity_fields = (
            "workload_count",
            "required_primary_workload_count",
            "warmup_count",
            "repeat_count",
            "trial_count",
            "random_operation_order_seed",
            "bootstrap_draws",
        )
        config_identity_match = all(
            config[field] == previous_config[field] for field in config_identity_fields
        )
        workload_identity_match = inventory_report["workloads"] == previous_inventory[
            "workloads"
        ]
        model_identity_match = (
            model_provenance == previous_metric_report["model_artifacts"]
        )
        correctness_identity_match = correctness_report == previous_correctness
        stability["identity_checks"] = {
            "config_identity_match": config_identity_match,
            "workload_identity_match": workload_identity_match,
            "model_identity_match": model_identity_match,
            "deterministic_correctness_match": correctness_identity_match,
        }
        stability["identity_checks_passed"] = all(
            stability["identity_checks"].values()
        )
        if not stability["identity_checks_passed"]:
            raise CombinedGateMicrobenchmarkError(
                "independent-run workload/model/config/correctness identity mismatch"
            )
    metric_report = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "profiling_only_combined_gate_microbenchmark"
            if len(selected) == 200
            else "profiling_only_development_subset_incomplete_primary_workload"
        ),
        "formal_200_workload_run": len(selected) == 200,
        "trace_identity_sha256": EXPECTED_TRACE_IDENTITY,
        "model_artifacts": model_provenance,
        "checkpoint": checkpoint,
        "workload_manifest": workload_provenance,
        "operation_count": len(OPERATIONS),
        "state_cases": list(STATE_CASES),
        "latency_projection_only_not_end_to_end_LIBERO": True,
        "runtime_inference_modified": False,
        "runtime_gate_implemented": False,
        "models_refit": False,
        "thresholds_reselected": False,
        "full_latent_states_serialized": False,
        "stability_comparison": stability,
        "outputs": {name: name for name in OUTPUT_FILENAMES},
    }
    reports = {
        "benchmark_config.json": config,
        "workload_inventory.json": inventory_report,
        "correctness_report.json": correctness_report,
        "latency_summary.json": latency_summary,
        "profiler_summary.json": profiler_summary,
        "memory_summary.json": memory_summary,
        "break_even_analysis.json": break_even,
        "metric_report.json": metric_report,
    }
    _write_outputs(
        output_dir, reports, samples, overwrite=args.overwrite
    )
    print(f"Wrote Combined gate microbenchmark: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
