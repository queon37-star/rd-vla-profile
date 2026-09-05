#!/usr/bin/env python3
"""Measure Coda-evaluation cost versus the frozen LDCE predictor cost.

Purpose
-------
This is a small component microprofile, not a policy-quality or end-to-end
benchmark.  It runs only the frozen LDCE arm and enables the existing inner
Coda profiler so that Coda and predictor costs are measured in the same live
execution environment.

Formal workload:
    10 LIBERO-Spatial tasks x 1 official final-partition state/task
    = 10 measured episodes total, plus one unmeasured warm-up rollout.

The selected manifest position is 22.  In the frozen 50-state ordering
(calibration 0..9 / screening 10..19 / final 20..49), this is a final-partition
state for every task.

Primary comparison
------------------
``coda_evaluation`` is the full ``_get_output`` evaluation that LDCE can avoid:
Coda processing plus output norm/projection.  ``predictor`` is the actual
production eager LDCE scorer invocation.  Both are reported as synchronized
per-call wall-clock costs from the runtime's existing profiling paths.

The report also exposes ``coda_internal`` (the Coda sub-block alone) when the
model has non-empty Coda layers.  The primary paper comparison should use
``coda_evaluation`` because that is the full computation eliminated when an
LDCE decision defers a Coda evaluation.

Do not report Action-head or policy-query latency from this run: enabling inner
Coda profiling intentionally inserts synchronization points inside
``predict_action`` and therefore perturbs those outer timings.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

import scripts.profile_spatial_paper_action_head as profiler
import scripts.run_spatial_paper_4arm as frozen


TASK_IDS = tuple(range(10))
STATE_POSITIONS = (22,)
PROTOCOL = "libero-spatial-coda-vs-ldce-predictor-microprofile-v1"
DEFAULT_OUTPUT_ROOT = "benchmark_results/paper_coda_vs_ldce_predictor_microprofile"


# ---------------------------------------------------------------------------
# Reuse the frozen paper configuration, changing only the diagnostic timing
# flag.  LDCE-only already has the final apply_to_cold=True semantics.
# ---------------------------------------------------------------------------
_original_build_arm_config = frozen._build_arm_config


def _build_arm_config_profiled(*, arm, **kwargs):
    if arm != "ldce":
        raise ValueError("Coda-vs-predictor microprofile is LDCE-only")
    cfg = _original_build_arm_config(arm=arm, **kwargs)
    cfg.profile_coda_cost = True
    if not cfg.action_delta_deferred_apply_to_cold:
        raise ValueError("Final LDCE semantics require apply_to_cold=True")
    return cfg


frozen._build_arm_config = _build_arm_config_profiled

_original_validate_cfg = profiler._validate_cfg


def _validate_cfg_profiled(cfg, *, arm: str, smoke: bool):
    if arm != "ldce":
        raise ValueError("Coda-vs-predictor microprofile is LDCE-only")
    # Reuse every frozen paper validation condition while bypassing only the
    # parent profiler's deliberate 'profile_coda_cost must stay False' rule.
    unprofiled = copy.copy(cfg)
    unprofiled.profile_coda_cost = False
    _original_validate_cfg(unprofiled, arm=arm, smoke=smoke)
    if not cfg.profile_coda_cost:
        raise ValueError("Microprofile requires profile_coda_cost=True")
    if not cfg.action_delta_deferred_apply_to_cold:
        raise ValueError("Microprofile requires final LDCE apply_to_cold=True")


profiler._validate_cfg = _validate_cfg_profiled
profiler.PROFILE_TASK_IDS = TASK_IDS
profiler.PROFILE_STATE_POSITIONS = STATE_POSITIONS
profiler.PROFILE_EPISODES_PER_TASK = len(STATE_POSITIONS)
profiler.PROFILE_PROTOCOL = PROTOCOL


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _stats(values: Iterable[float]) -> dict:
    a = np.asarray(list(values), dtype=np.float64)
    if a.size == 0:
        raise ValueError("Cannot summarize zero timing samples")
    if not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError("Timing samples must be finite and non-negative")
    return {
        "count": int(a.size),
        "mean_ms": float(a.mean()),
        "median_ms": float(np.median(a)),
        "std_ms": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "p90_ms": float(np.percentile(a, 90)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="outputs/12_24-24_24_Spatial_40k")
    p.add_argument(
        "--initial-state-manifest",
        default="experiments/robot/libero/manifests/libero_spatial_official_50_v1.json",
    )
    p.add_argument(
        "--action-delta-artifact",
        default="benchmark_results/coda_anchor_feasibility/action_delta_gate_fold4",
    )
    p.add_argument("--action-delta-sha256", default="")
    p.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _formal_records(step_path: Path) -> list[dict]:
    if not step_path.is_file():
        raise FileNotFoundError(f"Missing prediction log: {step_path}")
    rows = []
    for line in step_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("evaluation_protocol_phase") != profiler.PROFILE_PHASE:
            continue
        rows.append(row)
    if not rows:
        raise RuntimeError("No formal microprofile prediction records found")
    return rows


def _aggregate(root: Path) -> dict:
    rows = _formal_records(root / "ldce" / "debug_predictions.jsonl")
    predictor_ms: list[float] = []
    get_output_ms: list[float] = []
    coda_internal_ms: list[float] = []
    current_calls = 0
    backfill_calls = 0
    total_score_calls = 0

    for row in rows:
        if row.get("action_delta_deferred_backfill_filter_applied") is not True:
            raise RuntimeError("Formal record did not apply the LDCE filter")
        if row.get("action_delta_deferred_backfill_filter_apply_to_cold") is not True:
            raise RuntimeError("Formal record did not use final apply_to_cold=True semantics")

        pred = list(
            row.get("action_delta_deferred_backfill_filter_predictor_ms_list") or []
        )
        cur_get = list(
            row.get("action_delta_deferred_backfill_filter_current_get_output_ms_list")
            or []
        )
        back_get = list(
            row.get("action_delta_deferred_backfill_filter_backfill_get_output_ms_list")
            or []
        )
        cur_coda = list(
            row.get("action_delta_deferred_backfill_filter_current_coda_ms_list")
            or []
        )
        back_coda = list(
            row.get("action_delta_deferred_backfill_filter_backfill_coda_ms_list")
            or []
        )

        expected_scores = int(
            row.get("action_delta_deferred_backfill_filter_score_call_count", -1)
        )
        expected_current = int(
            row.get("action_delta_deferred_backfill_filter_current_state_coda_call_count", -1)
        )
        expected_backfill = int(
            row.get("action_delta_deferred_backfill_filter_backfill_coda_call_count", -1)
        )
        expected_total_coda = int(
            row.get("action_delta_deferred_backfill_filter_total_exact_coda_call_count", -1)
        )

        if expected_scores != len(pred):
            raise RuntimeError(
                f"Predictor timing count mismatch: expected={expected_scores}, actual={len(pred)}"
            )
        if expected_current != len(cur_get) or len(cur_get) != len(cur_coda):
            raise RuntimeError("Current-state Coda timing count mismatch")
        if expected_backfill != len(back_get) or len(back_get) != len(back_coda):
            raise RuntimeError("Backfill Coda timing count mismatch")
        if expected_total_coda != len(cur_get) + len(back_get):
            raise RuntimeError("Total exact Coda timing count mismatch")

        predictor_ms.extend(float(v) for v in pred)
        get_output_ms.extend(float(v) for v in cur_get)
        get_output_ms.extend(float(v) for v in back_get)
        coda_internal_ms.extend(float(v) for v in cur_coda)
        coda_internal_ms.extend(float(v) for v in back_coda)
        current_calls += len(cur_get)
        backfill_calls += len(back_get)
        total_score_calls += len(pred)

    predictor = _stats(predictor_ms)
    coda_eval = _stats(get_output_ms)
    coda_internal = _stats(coda_internal_ms)
    if predictor["mean_ms"] <= 0:
        raise RuntimeError("Predictor mean must be positive")

    ratio_mean = coda_eval["mean_ms"] / predictor["mean_ms"]
    ratio_median = coda_eval["median_ms"] / predictor["median_ms"]
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "purpose": "Coda evaluation versus frozen LDCE predictor per-call cost",
        "reportable_scope": "component microprofile only",
        "action_head_latency_reportable": False,
        "policy_query_latency_reportable": False,
        "environment_e2e_latency_reportable": False,
        "method": "ldce",
        "final_semantics": {"apply_to_cold": True},
        "tasks": list(TASK_IDS),
        "official_state_positions": list(STATE_POSITIONS),
        "measured_episodes": len(TASK_IDS) * len(STATE_POSITIONS),
        "formal_prediction_records": len(rows),
        "timing_definition": {
            "predictor": (
                "runtime perf_counter timing around the production eager LDCE scorer; "
                "the scalar decision is materialized before return"
            ),
            "coda_evaluation": (
                "synchronized _get_output wall-clock: Coda processing plus output norm/projection"
            ),
            "coda_internal": (
                "synchronized Coda sub-block only; may be zero when the configured model has no separate Coda layers"
            ),
        },
        "call_accounting": {
            "predictor_calls": total_score_calls,
            "coda_evaluation_calls": len(get_output_ms),
            "current_state_coda_calls": current_calls,
            "backfill_coda_calls": backfill_calls,
        },
        "predictor": predictor,
        "coda_evaluation": coda_eval,
        "coda_internal": coda_internal,
        "coda_eval_over_predictor": {
            "mean_ratio": float(ratio_mean),
            "median_ratio": float(ratio_median),
            "mean_difference_ms": float(coda_eval["mean_ms"] - predictor["mean_ms"]),
            "median_difference_ms": float(
                coda_eval["median_ms"] - predictor["median_ms"]
            ),
        },
    }
    _write_json(root / "component_report.json", report)
    return report


def main() -> int:
    args = _parse_args()
    root = Path(args.output_root)

    plan = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "method": "ldce",
        "task_ids": list(TASK_IDS),
        "official_state_positions": list(STATE_POSITIONS),
        "measured_episodes": len(TASK_IDS) * len(STATE_POSITIONS),
        "unmeasured_warmup_episodes": 1,
        "profile_coda_cost": True,
        "action_delta_deferred_apply_to_cold": True,
        "primary_comparison": "full Coda evaluation (_get_output) vs production eager LDCE predictor",
        "outer_latency_from_this_run": "do not report",
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    if root.exists() and not args.overwrite:
        raise FileExistsError(f"Output root exists: {root}; use --overwrite")

    parent_argv = [
        "profile_spatial_paper_action_head.py",
        "--checkpoint",
        args.checkpoint,
        "--initial-state-manifest",
        args.initial_state_manifest,
        "--action-delta-artifact",
        args.action_delta_artifact,
        "--output-root",
        str(root),
        "--seed",
        str(args.seed),
        "--arms",
        "ldce",
        "--overwrite",
    ]
    if args.action_delta_sha256:
        parent_argv.extend(["--action-delta-sha256", args.action_delta_sha256])

    old_argv = sys.argv
    try:
        sys.argv = parent_argv
        rc = profiler.main()
    finally:
        sys.argv = old_argv
    if rc != 0:
        raise RuntimeError(f"Parent profiler returned {rc}")

    # The parent profiler's outer Action-head report is intentionally not used
    # because inner Coda profiling inserts synchronization points.  Emit a
    # dedicated authoritative component report instead.
    report = _aggregate(root)
    _write_json(root / "microprofile_plan.json", plan)
    print("Completed Coda vs LDCE predictor microprofile")
    print(
        json.dumps(
            {
                "predictor_mean_ms": report["predictor"]["mean_ms"],
                "coda_evaluation_mean_ms": report["coda_evaluation"]["mean_ms"],
                "coda_over_predictor_mean_ratio": report[
                    "coda_eval_over_predictor"
                ]["mean_ratio"],
                "predictor_calls": report["call_accounting"]["predictor_calls"],
                "coda_calls": report["call_accounting"]["coda_evaluation_calls"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
