#!/usr/bin/env python3
"""Run the frozen Top-6 scheduler shortlist on captured GPU action-head workloads."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import subprocess
import sys
import time
import types
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prismatic.models.action_head_workload import load_action_head_workload  # noqa: E402
from prismatic.models.action_heads import (  # noqa: E402
    ActionHeadRecurrent,
    RecurrentConfigInternal,
)
from prismatic.models.projectors import ProprioProjector  # noqa: E402
from scripts.origin_aware_calibration_lib import validate_calibration_run  # noqa: E402
from scripts.origin_aware_gpu_microbenchmark_lib import (  # noqa: E402
    BASELINE_CONDITION_ID,
    BenchmarkCondition,
    GPUMicrobenchmarkValidationError,
    balanced_condition_order,
    build_benchmark_summary,
    conditions_from_shortlist,
    load_json_object,
    sha256_file,
    validate_protocol_manifest,
)
from scripts.origin_aware_replay_lib import (  # noqa: E402
    SchedulerConfig,
    parse_shadow_prediction,
    replay_prediction,
)


DEFAULT_RUN_ROOT = (
    REPO_ROOT
    / "benchmark_results/origin_aware_calibration/20260801_ca1b7d3_seed7_10x10"
)
DEFAULT_SHORTLIST = (
    REPO_ROOT
    / "experiments/robot/libero/manifests/origin_aware_oof_seed7_shortlist_v1.json"
)
DEFAULT_PROTOCOL = (
    REPO_ROOT
    / "experiments/robot/libero/manifests/origin_aware_gpu_microbenchmark_v1.json"
)
DEFAULT_INITIAL_STATE_MANIFEST = (
    REPO_ROOT / "experiments/robot/libero/manifests/libero_spatial_official_50_v1.json"
)
DEFAULT_CHECKPOINT = REPO_ROOT / "outputs/12_24-24_24_Spatial_40k"


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GPUMicrobenchmarkValidationError(
                f"invalid JSONL record at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise GPUMicrobenchmarkValidationError(
                f"JSONL record must be an object at {path}:{line_number}"
            )
        records.append(value)
    return records


def _exactly_one(path: Path, pattern: str) -> Path:
    matches = sorted(path.glob(pattern))
    if len(matches) != 1:
        raise GPUMicrobenchmarkValidationError(
            f"expected exactly one {pattern!r} under {path}, found {len(matches)}"
        )
    return matches[0]


def _strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }


def load_benchmark_modules(checkpoint_dir: Path, device: torch.device):
    config_path = _exactly_one(checkpoint_dir, "action_head_config--*.json")
    action_head_path = _exactly_one(checkpoint_dir, "action_head--*_checkpoint.pt")
    proprio_path = _exactly_one(checkpoint_dir, "proprio_projector--*_checkpoint.pt")
    saved_config = load_json_object(config_path)
    saved_config = dict(saved_config)
    saved_config.pop("_type", None)
    for field in ("prelude_vlm_layers", "recurrent_vlm_layers", "coda_vlm_layers"):
        if field in saved_config:
            saved_config[field] = tuple(saved_config[field])
    config = RecurrentConfigInternal(**saved_config)

    action_head = ActionHeadRecurrent(
        hidden_dim=config.hidden_dim,
        action_dim=config.action_dim,
        cfg=config,
    )
    action_head = action_head.to(torch.bfloat16).to(device).eval()
    action_state = torch.load(action_head_path, map_location="cpu", weights_only=True)
    action_head.load_state_dict(_strip_module_prefix(action_state), strict=True)

    proprio_projector = ProprioProjector(llm_dim=config.hidden_dim, proprio_dim=8)
    proprio_projector = proprio_projector.to(torch.bfloat16).to(device).eval()
    proprio_state = torch.load(proprio_path, map_location="cpu", weights_only=True)
    proprio_projector.load_state_dict(_strip_module_prefix(proprio_state), strict=True)
    return action_head, proprio_projector, {
        "action_head_config": {
            "path": str(config_path.resolve()),
            "sha256": sha256_file(config_path),
        },
        "action_head_checkpoint": {
            "path": str(action_head_path.resolve()),
            "sha256": sha256_file(action_head_path),
        },
        "proprio_projector_checkpoint": {
            "path": str(proprio_path.resolve()),
            "sha256": sha256_file(proprio_path),
        },
        "recurrent_config": saved_config,
    }


def load_workload_descriptors(run_root: Path) -> list[dict[str, Any]]:
    descriptors = []
    for task_id in range(10):
        task_dir = run_root / f"task{task_id}"
        for record in _load_jsonl(task_dir / "steps.jsonl"):
            if record.get("action_head_workload_captured") is not True:
                continue
            identity = {
                "task_id": int(record["task_id"]),
                "episode_id": int(record["episode_id"]),
                "paired_trial_id": int(record["paired_trial_id"]),
                "prediction_step": int(record["prediction_step"]),
                "initial_state_id": int(record["initial_state_id"]),
                "episode_seed": int(record["episode_seed"]),
            }
            path = Path(record["action_head_workload_file"])
            if not path.is_absolute():
                path = task_dir / path
            actual_origin = parse_shadow_prediction(record).actual_origin
            descriptors.append(
                {
                    "identity": identity,
                    "actual_origin": actual_origin,
                    "path": path.resolve(),
                    "sha256": record["action_head_workload_sha256"],
                    "record": record,
                }
            )
    descriptors.sort(
        key=lambda item: (
            item["identity"]["task_id"],
            item["identity"]["episode_id"],
            item["identity"]["prediction_step"],
        )
    )
    if len(descriptors) != 200:
        raise GPUMicrobenchmarkValidationError(
            f"formal calibration must provide 200 workload descriptors, got {len(descriptors)}"
        )
    return descriptors


def validate_shortlist_provenance(shortlist: Mapping[str, Any]) -> dict[str, Any]:
    source = shortlist.get("source")
    if not isinstance(source, Mapping):
        raise GPUMicrobenchmarkValidationError("shortlist source must be an object")
    validated = {}
    for path_field, hash_field in (
        ("formal_report", "formal_report_sha256"),
        ("calibration_validation", "calibration_validation_sha256"),
    ):
        path = _resolve_repo_path(source[path_field])
        expected = source[hash_field]
        actual = sha256_file(path)
        if actual != expected:
            raise GPUMicrobenchmarkValidationError(
                f"shortlist provenance hash mismatch for {path_field}: {actual} != {expected}"
            )
        validated[path_field] = {"path": str(path.resolve()), "sha256": actual}
    formal_report = load_json_object(Path(validated["formal_report"]["path"]))
    if formal_report.get("online_screening_allowed") is not False:
        raise GPUMicrobenchmarkValidationError("formal OOF report unexpectedly permits screening")
    if formal_report.get("microbenchmark_shortlist_count") != 6:
        raise GPUMicrobenchmarkValidationError("formal OOF report shortlist count mismatch")
    return validated


def _condition_kwargs(condition: BenchmarkCondition, incoming_warm_state: torch.Tensor | None):
    common = {
        "phase": "Inference",
        "convergence_strategy": "adjacent_action_mse",
        "kl_thresh": 0.001,
        "cos_thresh": 0.999,
        "max_iter": 32,
        "warm_start_state": incoming_warm_state,
        "enable_warm_start": True,
        "warm_start_source": "midpoint",
        "warm_start_min_iter": 2,
        "validate_warm_start_finite": True,
        "profile_coda_cost": False,
        "use_cached_final_output": True,
        "latent_precheck_min_iter": 2,
        "latent_precheck_force_interval": 0,
        "latent_precheck_trace_level": "off",
        "shadow_full_depth": False,
        "capture_action_head_workload": False,
    }
    if condition.kind == "baseline":
        common.update(
            {
                "use_latent_precheck": False,
                "latent_precheck_mode": "off",
                "latent_precheck_warm_thresh": None,
                "latent_precheck_max_skip_iters": 0,
                "latent_precheck_confirmation_mode": "next_iter",
                "nonfinite_policy": "legacy",
            }
        )
    else:
        common.update(
            {
                "use_latent_precheck": True,
                "latent_precheck_mode": "origin_aware",
                "latent_precheck_warm_thresh": condition.warm_threshold,
                "latent_precheck_max_skip_iters": condition.max_skip_iters,
                "latent_precheck_confirmation_mode": condition.confirmation_mode,
                "nonfinite_policy": "cold_retry_once",
            }
        )
    return common


@contextmanager
def captured_cold_initial_state(model, selected_initial_state: torch.Tensor, actual_origin: str):
    """Preserve production cold-init cost while replaying the captured state value."""

    if actual_origin != "COLD":
        yield
        return
    had_instance_attribute = "init_state" in model.__dict__
    original_instance_attribute = model.__dict__.get("init_state")
    original_init = model.init_state

    def replay_init(_self, batch_size: int, device, dtype):
        generated = original_init(batch_size, device, dtype)
        expected = selected_initial_state.to(device=device, dtype=dtype)
        if tuple(generated.shape) != tuple(expected.shape):
            raise GPUMicrobenchmarkValidationError("captured cold initial-state shape mismatch")
        return expected.detach().clone()

    model.init_state = types.MethodType(replay_init, model)
    try:
        yield
    finally:
        if had_instance_attribute:
            model.init_state = original_instance_attribute
        else:
            delattr(model, "init_state")


def _expected_schedule(record: Mapping[str, Any], condition: BenchmarkCondition) -> dict[str, Any]:
    prediction = parse_shadow_prediction(record)
    if condition.kind == "baseline":
        return {
            "K_t": int(prediction.baseline_k),
            "coda_calls": int(prediction.baseline_k),
            "latent_gate_calls": 0,
            "action_comparisons": max(0, int(prediction.baseline_k) - 1),
            "actual_origin": prediction.actual_origin,
        }
    replayed = replay_prediction(
        prediction,
        SchedulerConfig(
            warm_threshold=float(condition.warm_threshold),
            cold_threshold=float(condition.cold_threshold),
            max_skip_iters=int(condition.max_skip_iters),
            confirmation_mode=str(condition.confirmation_mode),
        ),
    )
    return {
        "K_t": int(replayed.terminal_k),
        "coda_calls": int(replayed.decode_calls),
        "latent_gate_calls": int(replayed.latent_gate_calls),
        "action_comparisons": int(replayed.action_comparisons),
        "actual_origin": prediction.actual_origin,
    }


def _actual_schedule(action_head: ActionHeadRecurrent, condition: BenchmarkCondition, K_t: int):
    debug = action_head.model.last_recurrence_debug
    if not isinstance(debug, Mapping):
        raise GPUMicrobenchmarkValidationError("action head did not publish recurrence debug metadata")
    if condition.kind == "baseline":
        return {
            "K_t": int(K_t),
            "coda_calls": int(K_t),
            "latent_gate_calls": 0,
            "action_comparisons": len(debug.get("iteration_mse", [])),
            "actual_origin": "ACTUAL_WARM" if debug.get("warm_start_state_used") else "COLD",
        }
    return {
        "K_t": int(K_t),
        "coda_calls": int(debug["latent_precheck_call_count"]),
        "latent_gate_calls": int(debug["latent_metric_count"]),
        "action_comparisons": int(debug["adjacent_comparison_pair_count"]),
        "actual_origin": debug["latent_precheck_origin"],
        "numerical_retry_attempted": bool(debug.get("numerical_retry_attempted")),
        "final_state_coda_executed": bool(debug.get("final_state_coda_executed")),
        "cached_final_matches_returned": bool(debug.get("cached_final_matches_returned")),
    }


def _schedule_mismatch(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> dict[str, Any] | None:
    compared_fields = ("K_t", "coda_calls", "latent_gate_calls", "action_comparisons", "actual_origin")
    differences = {
        field: {"expected": expected.get(field), "actual": actual.get(field)}
        for field in compared_fields
        if expected.get(field) != actual.get(field)
    }
    if actual.get("numerical_retry_attempted") is True:
        differences["numerical_retry_attempted"] = {"expected": False, "actual": True}
    for field in ("final_state_coda_executed", "cached_final_matches_returned"):
        if field in actual and actual[field] is not True:
            differences[field] = {"expected": True, "actual": actual[field]}
    return differences or None


def _execute(
    action_head: ActionHeadRecurrent,
    proprio_projector: ProprioProjector,
    tensors: Mapping[str, torch.Tensor | None],
    condition: BenchmarkCondition,
):
    result = action_head.predict_action(
        tensors["actions_hidden_states"],
        proprio=tensors["proprio_input"],
        proprio_projector=proprio_projector,
        **_condition_kwargs(condition, tensors["incoming_warm_start_state"]),
    )
    output, K_t, final_score = result
    return output, int(K_t), final_score, _actual_schedule(action_head, condition, int(K_t))


def _timed_execute(
    action_head: ActionHeadRecurrent,
    proprio_projector: ProprioProjector,
    tensors: Mapping[str, torch.Tensor | None],
    condition: BenchmarkCondition,
    device: torch.device,
):
    torch.cuda.synchronize(device)
    start_ns = time.perf_counter_ns()
    output, K_t, final_score, schedule = _execute(
        action_head, proprio_projector, tensors, condition
    )
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
    if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
        raise GPUMicrobenchmarkValidationError("measured latency must be finite and positive")
    if not bool(torch.isfinite(output).all().item()):
        raise GPUMicrobenchmarkValidationError("action-head replay returned a non-finite output")
    return elapsed_ms, output, K_t, final_score, schedule


def _prepare_tensors(payload: Mapping[str, Any], device: torch.device):
    tensors = {
        name: (None if tensor is None else tensor.to(device=device, non_blocking=False))
        for name, tensor in payload["tensors"].items()
    }
    for name, tensor in tensors.items():
        if tensor is not None and not tensor.is_contiguous():
            raise GPUMicrobenchmarkValidationError(f"GPU workload tensor {name} is non-contiguous")
    return tensors


def _validate_projector(
    proprio_projector: ProprioProjector,
    tensors: Mapping[str, torch.Tensor | None],
    device: torch.device,
):
    projected = proprio_projector(tensors["proprio_input"].reshape(1, -1).to(torch.bfloat16)).unsqueeze(1)
    torch.cuda.synchronize(device)
    expected = tensors["proprio_features"]
    if not torch.equal(projected, expected):
        max_error = float(torch.max(torch.abs(projected.float() - expected.float())).item())
        raise GPUMicrobenchmarkValidationError(
            f"captured proprio feature mismatch; max absolute error={max_error}"
        )


def _run_workload(
    *,
    descriptor: Mapping[str, Any],
    block_index: int,
    action_head: ActionHeadRecurrent,
    proprio_projector: ProprioProjector,
    conditions: list[BenchmarkCondition],
    device: torch.device,
    repeats: int,
    order_seed: int,
    measured: bool,
):
    payload = load_action_head_workload(
        descriptor["path"],
        expected_sha256=descriptor["sha256"],
        expected_identity=descriptor["identity"],
        expected_origin=descriptor["actual_origin"],
    )
    tensors = _prepare_tensors(payload, device)
    _validate_projector(proprio_projector, tensors, device)
    condition_by_id = {condition.condition_id: condition for condition in conditions}
    condition_ids = list(condition_by_id)
    measurements = []
    mismatches = []
    with captured_cold_initial_state(
        action_head.model, tensors["selected_initial_state"], descriptor["actual_origin"]
    ):
        for repeat_index in range(repeats):
            order = balanced_condition_order(
                condition_ids,
                block_index=block_index * repeats,
                repeat_index=repeat_index,
                seed=order_seed,
            )
            for order_position, condition_id in enumerate(order):
                condition = condition_by_id[condition_id]
                expected_schedule = _expected_schedule(descriptor["record"], condition)
                if measured:
                    elapsed_ms, output, K_t, final_score, actual_schedule = _timed_execute(
                        action_head, proprio_projector, tensors, condition, device
                    )
                else:
                    output, K_t, final_score, actual_schedule = _execute(
                        action_head, proprio_projector, tensors, condition
                    )
                    torch.cuda.synchronize(device)
                    elapsed_ms = None
                mismatch = _schedule_mismatch(expected_schedule, actual_schedule)
                if mismatch is not None:
                    mismatches.append(
                        {
                            **descriptor["identity"],
                            "condition_id": condition_id,
                            "repeat_index": repeat_index,
                            "differences": mismatch,
                        }
                    )
                if measured:
                    measurements.append(
                        {
                            **descriptor["identity"],
                            "actual_origin": descriptor["actual_origin"],
                            "condition_id": condition_id,
                            "repeat_index": repeat_index,
                            "order_position": order_position,
                            "latency_ms": elapsed_ms,
                            "K_t": K_t,
                            "final_score": (
                                None if final_score is None else float(final_score)
                            ),
                            "output_finite": bool(torch.isfinite(output).all().item()),
                            "schedule": actual_schedule,
                        }
                    )
    del tensors
    del payload
    return measurements, mismatches


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--shortlist", type=Path, default=DEFAULT_SHORTLIST)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--initial-state-manifest", type=Path, default=DEFAULT_INITIAL_STATE_MANIFEST
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-seed", type=int, default=7)
    parser.add_argument("--max-workloads", type=int)
    parser.add_argument("--measurement-repeats", type=int)
    parser.add_argument("--warmup-rounds", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite GPU microbenchmark report: {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("formal GPU schedule microbenchmark requires CUDA")

    shortlist = load_json_object(args.shortlist)
    protocol = load_json_object(args.protocol)
    conditions = conditions_from_shortlist(shortlist)
    if args.measurement_repeats is not None:
        protocol = copy.deepcopy(protocol)
        protocol["measurement_repeats"] = int(args.measurement_repeats)
    if args.warmup_rounds is not None:
        protocol = copy.deepcopy(protocol)
        protocol["warmup_rounds_per_origin"] = int(args.warmup_rounds)
    validate_protocol_manifest(protocol)
    provenance = validate_shortlist_provenance(shortlist)

    calibration_validation = validate_calibration_run(
        str(args.run_root), str(args.initial_state_manifest), base_seed=args.base_seed
    )
    if calibration_validation.get("complete_10_task_gate") is not True:
        raise GPUMicrobenchmarkValidationError("benchmark requires complete ten-task calibration")
    descriptors = load_workload_descriptors(args.run_root)
    formal_workload_count = len(descriptors)
    if args.max_workloads is not None:
        if args.max_workloads < 2:
            raise ValueError("--max-workloads must be at least 2")
        descriptors = descriptors[: args.max_workloads]

    device = torch.device(protocol["device"])
    torch.manual_seed(int(protocol["seed"]))
    torch.cuda.manual_seed_all(int(protocol["seed"]))
    random.seed(int(protocol["seed"]))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    action_head, proprio_projector, checkpoint_inputs = load_benchmark_modules(
        args.checkpoint, device
    )

    warmup_rounds = int(protocol["warmup_rounds_per_origin"])
    representative = {}
    for descriptor in descriptors:
        representative.setdefault(descriptor["actual_origin"], descriptor)
    if set(representative) != {"COLD", "ACTUAL_WARM"}:
        raise GPUMicrobenchmarkValidationError("selected workloads must include both origins")

    all_mismatches = []
    with torch.inference_mode():
        for warmup_index, origin in enumerate(("COLD", "ACTUAL_WARM")):
            _, mismatches = _run_workload(
                descriptor=representative[origin],
                block_index=warmup_index,
                action_head=action_head,
                proprio_projector=proprio_projector,
                conditions=conditions,
                device=device,
                repeats=warmup_rounds,
                order_seed=int(protocol["seed"]),
                measured=False,
            )
            all_mismatches.extend(mismatches)

        measurements = []
        for block_index, descriptor in enumerate(descriptors):
            block_measurements, mismatches = _run_workload(
                descriptor=descriptor,
                block_index=block_index,
                action_head=action_head,
                proprio_projector=proprio_projector,
                conditions=conditions,
                device=device,
                repeats=int(protocol["measurement_repeats"]),
                order_seed=int(protocol["seed"]),
                measured=True,
            )
            measurements.extend(block_measurements)
            all_mismatches.extend(mismatches)
            if (block_index + 1) % 20 == 0 or block_index + 1 == len(descriptors):
                print(f"Measured workloads: {block_index + 1}/{len(descriptors)}", flush=True)

    unique_mismatches = {}
    for mismatch in all_mismatches:
        key = (
            mismatch["task_id"],
            mismatch["episode_id"],
            mismatch["prediction_step"],
            mismatch["condition_id"],
        )
        unique_mismatches.setdefault(key, mismatch)
    mismatch_records = list(unique_mismatches.values())
    is_formal = (
        len(descriptors) == formal_workload_count
        and args.measurement_repeats is None
        and args.warmup_rounds is None
    )
    measured_task_ids = sorted({item["identity"]["task_id"] for item in descriptors})
    summary = build_benchmark_summary(
        measurements,
        conditions,
        protocol,
        schedule_mismatch_count=len(mismatch_records),
        required_task_ids=range(10) if is_formal else measured_task_ids,
        episodes_per_task=10 if is_formal else None,
    )
    if not is_formal:
        summary["online_screening_allowed"] = False
        summary["screening_candidates"] = []
        summary["interpretation"] = "Development subset only; promotion is disabled."

    torch.cuda.synchronize(device)
    device_properties = torch.cuda.get_device_properties(device)
    report = {
        "schema_version": 1,
        "formal_run": is_formal,
        "code_git_commit": _git_commit(),
        "protocol": protocol,
        "conditions": [asdict(condition) for condition in conditions],
        "inputs": {
            "run_root": str(args.run_root.resolve()),
            "shortlist": {
                "path": str(args.shortlist.resolve()),
                "sha256": sha256_file(args.shortlist),
            },
            "protocol_manifest": {
                "path": str(args.protocol.resolve()),
                "sha256": sha256_file(args.protocol),
            },
            "initial_state_manifest": {
                "path": str(args.initial_state_manifest.resolve()),
                "sha256": sha256_file(args.initial_state_manifest),
            },
            "shortlist_provenance": provenance,
            "checkpoint": checkpoint_inputs,
        },
        "environment": {
            "torch_version": torch.__version__,
            "cuda_runtime_version": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "device_total_memory": int(device_properties.total_memory),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        },
        "calibration_validation": calibration_validation,
        "workloads": {
            "formal_available": formal_workload_count,
            "measured": len(descriptors),
            "cold": sum(item["actual_origin"] == "COLD" for item in descriptors),
            "actual_warm": sum(
                item["actual_origin"] == "ACTUAL_WARM" for item in descriptors
            ),
        },
        "schedule_mismatches": mismatch_records,
        "measurements": measurements,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    primary = summary["scopes"][summary["primary_scope"]]
    print(f"Schedule mismatches: {len(mismatch_records)}")
    for candidate in primary["conditions"][1:]:
        print(
            f"{candidate['condition_id']}: improvement="
            f"{100 * candidate['improvement_vs_baseline']:.3f}%, simultaneous lower="
            f"{100 * candidate['simultaneous_one_sided_lower_bound']:.3f}%"
        )
    print(f"Online screening allowed: {summary['online_screening_allowed']}")
    print(f"Screening candidates: {summary['screening_candidates']}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
