#!/usr/bin/env python3
"""Validate fixed-depth legacy and terminal-only output equivalence before timing."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prismatic.models.action_head_workload import load_action_head_workload  # noqa: E402
from scripts.origin_aware_calibration_lib import validate_calibration_run  # noqa: E402
from scripts.origin_aware_gpu_microbenchmark_lib import (  # noqa: E402
    GPUMicrobenchmarkValidationError,
    load_json_object,
    sha256_file,
)
from scripts.run_fixed_coda_microprofile import (  # noqa: E402
    DEFAULT_PROTOCOL,
    _condition_kwargs,
    _conditions,
    _schedule_mismatch,
    _validate_protocol,
)
from scripts.run_origin_aware_gpu_microbenchmark import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_INITIAL_STATE_MANIFEST,
    DEFAULT_RUN_ROOT,
    _prepare_tensors,
    _validate_projector,
    captured_cold_initial_state,
    load_benchmark_modules,
    load_workload_descriptors,
)


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _execute(action_head, proprio_projector, tensors, condition):
    output, returned_k, _ = action_head.predict_action(
        tensors["actions_hidden_states"],
        proprio=tensors["proprio_input"],
        proprio_projector=proprio_projector,
        **_condition_kwargs(condition, tensors["incoming_warm_start_state"]),
    )
    debug = action_head.model.last_recurrence_debug
    if not isinstance(debug, Mapping):
        raise GPUMicrobenchmarkValidationError(
            "action head did not publish recurrence debug metadata"
        )
    if condition.kind == "legacy_fixed":
        schedule = {
            "K_t": int(returned_k),
            "recurrent_calls": int(returned_k),
            "coda_calls": int(returned_k),
            "canonical_recurrence_strategy": debug.get(
                "canonical_recurrence_strategy"
            ),
            "warm_start_state_used": bool(debug.get("warm_start_state_used")),
        }
    else:
        schedule = {
            "K_t": int(returned_k),
            "recurrent_calls": int(returned_k),
            "coda_calls": int(debug.get("coda_call_count", -1)),
            "canonical_recurrence_strategy": debug.get(
                "canonical_recurrence_strategy"
            ),
            "warm_start_state_used": bool(debug.get("warm_start_state_used")),
            "final_state_coda_executed": bool(
                debug.get("final_state_coda_executed")
            ),
        }
    mismatch = _schedule_mismatch(condition, tensors["actual_origin"], schedule)
    if mismatch is not None:
        raise GPUMicrobenchmarkValidationError(
            f"preflight schedule mismatch for {condition.condition_id}: {mismatch}"
        )
    if not bool(torch.isfinite(output).all().item()):
        raise GPUMicrobenchmarkValidationError(
            f"non-finite output from {condition.condition_id}"
        )
    return output.detach().clone()


def _compare_pair(legacy: torch.Tensor, terminal: torch.Tensor) -> dict[str, Any]:
    if tuple(legacy.shape) != tuple(terminal.shape):
        return {
            "shape_equal": False,
            "exact_equal": False,
            "max_abs_error": None,
            "max_rel_error": None,
        }
    exact = bool(torch.equal(legacy, terminal))
    difference = torch.abs(legacy.float() - terminal.float())
    denominator = torch.maximum(
        torch.abs(legacy.float()),
        torch.full_like(legacy.float(), 1e-12),
    )
    return {
        "shape_equal": True,
        "exact_equal": exact,
        "max_abs_error": float(torch.max(difference).item()),
        "max_rel_error": float(torch.max(difference / denominator).item()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--initial-state-manifest",
        type=Path,
        default=DEFAULT_INITIAL_STATE_MANIFEST,
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-seed", type=int, default=7)
    parser.add_argument("--max-workloads", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite fixed-Coda preflight report: {args.output}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("fixed-Coda preflight requires CUDA")

    protocol = load_json_object(args.protocol)
    _validate_protocol(protocol)
    all_conditions = _conditions(protocol)
    conditions_by_id = {
        condition.condition_id: condition for condition in all_conditions
    }
    paired_conditions = []
    for fixed_k in protocol["fixed_depths"]:
        paired_conditions.extend(
            [
                conditions_by_id[f"legacy_fixed_k{fixed_k}"],
                conditions_by_id[f"terminal_only_k{fixed_k}"],
            ]
        )

    calibration_validation = validate_calibration_run(
        str(args.run_root),
        str(args.initial_state_manifest),
        base_seed=args.base_seed,
    )
    if calibration_validation.get("complete_10_task_gate") is not True:
        raise GPUMicrobenchmarkValidationError(
            "fixed-Coda preflight requires complete ten-task calibration"
        )
    descriptors = load_workload_descriptors(args.run_root)
    formal_available = len(descriptors)
    if args.max_workloads is not None:
        if args.max_workloads < 1:
            raise ValueError("--max-workloads must be at least 1")
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

    comparisons: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with torch.inference_mode():
        for workload_index, descriptor in enumerate(descriptors):
            payload = load_action_head_workload(
                descriptor["path"],
                expected_sha256=descriptor["sha256"],
                expected_identity=descriptor["identity"],
                expected_origin=descriptor["actual_origin"],
            )
            tensors = _prepare_tensors(payload, device)
            tensors["actual_origin"] = descriptor["actual_origin"]
            _validate_projector(proprio_projector, tensors, device)
            outputs: dict[str, torch.Tensor] = {}
            with captured_cold_initial_state(
                action_head.model,
                tensors["selected_initial_state"],
                descriptor["actual_origin"],
            ):
                for condition in paired_conditions:
                    outputs[condition.condition_id] = _execute(
                        action_head,
                        proprio_projector,
                        tensors,
                        condition,
                    )
            torch.cuda.synchronize(device)
            for fixed_k in protocol["fixed_depths"]:
                legacy_id = f"legacy_fixed_k{fixed_k}"
                terminal_id = f"terminal_only_k{fixed_k}"
                comparison = _compare_pair(
                    outputs[legacy_id], outputs[terminal_id]
                )
                record = {
                    **descriptor["identity"],
                    "actual_origin": descriptor["actual_origin"],
                    "fixed_k": int(fixed_k),
                    "legacy_condition_id": legacy_id,
                    "terminal_condition_id": terminal_id,
                    **comparison,
                }
                comparisons.append(record)
                if comparison["exact_equal"] is not True:
                    failures.append(record)
            del outputs
            del tensors
            del payload
            if (workload_index + 1) % 20 == 0 or workload_index + 1 == len(
                descriptors
            ):
                print(
                    f"Validated workloads: {workload_index + 1}/{len(descriptors)}",
                    flush=True,
                )

    formal_run = (
        args.max_workloads is None
        and formal_available == int(protocol["expected_formal_workload_count"])
        and len(descriptors) == int(protocol["expected_formal_workload_count"])
    )
    report = {
        "schema_version": 1,
        "formal_run": formal_run,
        "code_git_commit": _git_commit(),
        "protocol": protocol,
        "inputs": {
            "run_root": str(args.run_root.resolve()),
            "protocol_manifest": {
                "path": str(args.protocol.resolve()),
                "sha256": sha256_file(args.protocol),
            },
            "initial_state_manifest": {
                "path": str(args.initial_state_manifest.resolve()),
                "sha256": sha256_file(args.initial_state_manifest),
            },
            "checkpoint": checkpoint_inputs,
        },
        "calibration_validation": calibration_validation,
        "workloads": {
            "formal_available": formal_available,
            "measured": len(descriptors),
            "cold": sum(
                descriptor["actual_origin"] == "COLD"
                for descriptor in descriptors
            ),
            "actual_warm": sum(
                descriptor["actual_origin"] == "ACTUAL_WARM"
                for descriptor in descriptors
            ),
        },
        "comparison_count": len(comparisons),
        "exact_match_count": sum(
            comparison["exact_equal"] is True for comparison in comparisons
        ),
        "failure_count": len(failures),
        "failures": failures,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Exact output mismatches: {len(failures)}")
    print(f"Formal run: {formal_run}")
    print(f"Wrote: {args.output}")
    if failures:
        raise GPUMicrobenchmarkValidationError(
            f"fixed-Coda output equivalence failed for {len(failures)} pairs"
        )


if __name__ == "__main__":
    main()
