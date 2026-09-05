#!/usr/bin/env python3
"""Interleaved 4-arm Action-head latency validation for the paper.

Measured workload:
    10 LIBERO-Spatial tasks x 5 official states/task x 4 methods = 200 episodes.

Why this runner exists
----------------------
The long 4-arm closed-loop run executed whole methods sequentially, so method
identity was confounded with time-of-run.  This runner keeps the already
validated live Action-head profiler, but executes short task/method blocks in a
deterministic rotating order.  Each measured block is only five episodes.

The five official-state positions are 0, 10, 20, 30, 40 from the frozen
50-state manifest.  Method order rotates by task:

    task 0: baseline -> warm_start -> ldce -> combined
    task 1: warm_start -> ldce -> combined -> baseline
    task 2: ldce -> combined -> baseline -> warm_start
    task 3: combined -> baseline -> warm_start -> ldce
    ...

The underlying profiler measures synchronized wall-clock latency around exactly
``action_head.predict_action``.  VLM and environment time are excluded.  The
policy-query latency emitted incidentally by the live evaluator MUST NOT be
reported from this benchmark because the component timer intentionally inserts
CUDA synchronization around the Action-head call.

Final method semantics are enforced here: both LDCE and Combined apply LDCE to
cold-origin predictions.  Thus the first prediction of every Combined episode
is Cold+LDCE, and later predictions use Warm+LDCE when the midpoint cache is
available.

Implementation note
-------------------
Each task/method block invokes the existing validated profiler in-process with a
fresh model initialization.  Consequently each measured five-episode block has
one unmeasured warm-up rollout.  There are 40 such warm-ups in the full run;
none enter the reported latency statistics.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np

import scripts.profile_spatial_paper_action_head as profiler
import scripts.run_spatial_paper_4arm as frozen


ARMS = ("baseline", "warm_start", "ldce", "combined")
TASK_IDS = tuple(range(10))
STATE_POSITIONS = (0, 10, 20, 30, 40)
PROTOCOL = "libero-spatial-action-head-live-10x5-4arm-interleaved-v1"
SMOKE_PROTOCOL = "libero-spatial-action-head-live-1x1-4arm-interleaved-smoke-v1"
DEFAULT_OUTPUT_ROOT = "benchmark_results/paper_spatial_action_head_latency_interleaved_10x5"


# ---------------------------------------------------------------------------
# Final method semantics: LDCE is active on cold predictions in both LDCE and
# Combined.  Reuse the parent profiler's full validation contract, bypassing
# only its historical Combined=False restriction exactly as in the final
# one-pass runner.
# ---------------------------------------------------------------------------
_original_build_arm_config = frozen._build_arm_config


def _build_arm_config_final(*, arm, **kwargs):
    cfg = _original_build_arm_config(arm=arm, **kwargs)
    if arm in {"ldce", "combined"}:
        cfg.action_delta_deferred_apply_to_cold = True
    return cfg


frozen._build_arm_config = _build_arm_config_final

_original_validate_cfg = profiler._validate_cfg


def _validate_cfg_final(cfg, *, arm: str, smoke: bool):
    if arm == "combined":
        historical = copy.copy(cfg)
        historical.action_delta_deferred_apply_to_cold = False
        _original_validate_cfg(historical, arm=arm, smoke=smoke)
        if not cfg.action_delta_deferred_apply_to_cold:
            raise ValueError("Final Combined requires LDCE on cold predictions")
        return

    _original_validate_cfg(cfg, arm=arm, smoke=smoke)
    if arm == "ldce" and not cfg.action_delta_deferred_apply_to_cold:
        raise ValueError("Final LDCE requires LDCE on cold predictions")


profiler._validate_cfg = _validate_cfg_final


# Keep the validated synchronized wall-clock Action-head timer from the parent
# profiler.  Only the sampled state set and task subset are changed per block.
profiler.PROFILE_STATE_POSITIONS = STATE_POSITIONS
profiler.PROFILE_EPISODES_PER_TASK = len(STATE_POSITIONS)
profiler.PROFILE_PROTOCOL = PROTOCOL


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _stats(values: Iterable[float]) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot summarize zero latency samples")
    return {
        "count": int(array.size),
        "mean_ms": float(array.mean()),
        "median_ms": float(np.median(array)),
        "std_ms": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "p90_ms": float(np.percentile(array, 90)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
    }


def _rotating_order(task_id: int) -> tuple[str, ...]:
    shift = int(task_id) % len(ARMS)
    return ARMS[shift:] + ARMS[:shift]


def _block_root(root: Path, task_id: int, order_position: int, arm: str) -> Path:
    return root / "blocks" / f"task_{task_id:02d}" / f"pos_{order_position}_{arm}"


def _parse_arms(values: Iterable[str]) -> tuple[str, ...]:
    requested = []
    for value in values:
        for raw in value.split(","):
            arm = raw.strip()
            if not arm:
                continue
            if arm not in ARMS:
                raise ValueError(f"Unknown arm {arm!r}; choose from {ARMS}")
            if arm not in requested:
                requested.append(arm)
    if tuple(requested) != ARMS:
        raise ValueError(
            "This benchmark is frozen to all four arms in order: "
            "baseline warm_start ldce combined"
        )
    return tuple(requested)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interleaved 10x5 live Action-head latency validation."
    )
    parser.add_argument("--checkpoint", default="outputs/12_24-24_24_Spatial_40k")
    parser.add_argument(
        "--initial-state-manifest",
        default="experiments/robot/libero/manifests/libero_spatial_official_50_v1.json",
    )
    parser.add_argument(
        "--action-delta-artifact",
        default="benchmark_results/coda_anchor_feasibility/action_delta_gate_fold4",
    )
    parser.add_argument("--action-delta-sha256", default="")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _configure_parent_for_block(*, task_id: int, smoke: bool) -> None:
    profiler.PROFILE_TASK_IDS = (int(task_id),)
    if smoke:
        profiler.PROFILE_STATE_POSITIONS = (STATE_POSITIONS[0],)
        profiler.PROFILE_EPISODES_PER_TASK = 1
        profiler.PROFILE_PROTOCOL = SMOKE_PROTOCOL
    else:
        profiler.PROFILE_STATE_POSITIONS = STATE_POSITIONS
        profiler.PROFILE_EPISODES_PER_TASK = len(STATE_POSITIONS)
        profiler.PROFILE_PROTOCOL = PROTOCOL


def _invoke_parent(
    *,
    args: argparse.Namespace,
    arm: str,
    task_id: int,
    block_root: Path,
    smoke: bool,
) -> dict:
    _configure_parent_for_block(task_id=task_id, smoke=smoke)

    argv = [
        "profile_spatial_paper_action_head.py",
        "--checkpoint",
        args.checkpoint,
        "--initial-state-manifest",
        args.initial_state_manifest,
        "--action-delta-artifact",
        args.action_delta_artifact,
        "--output-root",
        str(block_root),
        "--seed",
        str(args.seed),
        "--arms",
        arm,
        "--overwrite",
    ]
    if args.action_delta_sha256:
        argv.extend(["--action-delta-sha256", args.action_delta_sha256])

    old_argv = sys.argv
    try:
        sys.argv = argv
        rc = profiler.main()
    finally:
        sys.argv = old_argv
    if rc != 0:
        raise RuntimeError(f"Parent profiler returned {rc} for task={task_id}, arm={arm}")

    summary_path = block_root / arm / "summary.json"
    task_path = block_root / arm / f"task_{task_id:02d}.json"
    if not summary_path.is_file() or not task_path.is_file():
        raise RuntimeError(f"Missing completed block outputs for task={task_id}, arm={arm}")
    summary = _load_json(summary_path)
    task = _load_json(task_path)
    if not summary.get("completed") or not task.get("completed"):
        raise RuntimeError(f"Incomplete block for task={task_id}, arm={arm}")
    expected_episodes = 1 if smoke else len(STATE_POSITIONS)
    if int(task.get("episodes", -1)) != expected_episodes:
        raise RuntimeError(
            f"task={task_id}, arm={arm}: expected {expected_episodes} episodes, "
            f"got {task.get('episodes')}"
        )
    return {"summary": summary, "task": task}


def _aggregate(root: Path, records: list[dict], *, protocol: str) -> dict:
    per_arm = {}
    for arm in ARMS:
        arm_records = [record for record in records if record["arm"] == arm]
        all_values = []
        task_means = {}
        total_predictions = 0
        total_episodes = 0
        total_successes = 0
        weighted_k = 0.0
        weighted_k_n = 0

        for record in arm_records:
            task = record["task_payload"]
            values = [float(sample["action_head_latency_ms"]) for sample in task["samples"]]
            all_values.extend(values)
            task_means[str(record["task_id"])] = float(np.mean(values))
            n = int(task["num_predictions"])
            total_predictions += n
            total_episodes += int(task["episodes"])
            total_successes += int(task["successes"])
            if task.get("prediction_weighted_avg_iters") is not None:
                weighted_k += float(task["prediction_weighted_avg_iters"]) * n
                weighted_k_n += n

        prediction_stats = _stats(all_values)
        macro_values = list(task_means.values())
        per_arm[arm] = {
            "episodes": total_episodes,
            "successes": total_successes,
            "num_predictions": total_predictions,
            "prediction_weighted_avg_K": (
                weighted_k / weighted_k_n if weighted_k_n else None
            ),
            "action_head_latency_prediction_weighted": prediction_stats,
            "action_head_latency_task_macro_mean_ms": (
                float(np.mean(macro_values)) if macro_values else None
            ),
            "task_mean_ms": task_means,
        }

    baseline = float(
        per_arm["baseline"]["action_head_latency_prediction_weighted"]["mean_ms"]
    )
    for arm in ARMS:
        mean_ms = float(per_arm[arm]["action_head_latency_prediction_weighted"]["mean_ms"])
        per_arm[arm]["speedup_vs_baseline_percent"] = 100.0 * (baseline - mean_ms) / baseline

    report = {
        "schema_version": 1,
        "protocol": protocol,
        "measurement_scope": "synchronized wall-clock around action_head.predict_action only",
        "policy_query_latency_reportable": False,
        "primary_aggregation": "prediction-weighted mean Action-head latency",
        "task_macro_latency_also_reported": True,
        "arms": per_arm,
        "execution_records": [
            {
                key: record[key]
                for key in (
                    "measurement_index",
                    "task_id",
                    "order_position",
                    "arm",
                    "started_unix_s",
                    "ended_unix_s",
                    "duration_s",
                )
            }
            for record in records
        ],
    }
    _write_json(root / "report.json", report)
    return report


def main() -> int:
    args = parse_args()
    _parse_arms(args.arms)

    checkpoint = Path(args.checkpoint)
    manifest = Path(args.initial_state_manifest)
    artifact = Path(args.action_delta_artifact)
    root = Path(args.output_root)

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest}")
    if not artifact.exists():
        raise FileNotFoundError(f"LDCE artifact does not exist: {artifact}")

    task_ids = (0,) if args.smoke else TASK_IDS
    positions = (STATE_POSITIONS[0],) if args.smoke else STATE_POSITIONS
    protocol = SMOKE_PROTOCOL if args.smoke else PROTOCOL
    orders = {str(task_id): list(_rotating_order(task_id)) for task_id in task_ids}
    measured_episodes = len(task_ids) * len(positions) * len(ARMS)
    measured_blocks = len(task_ids) * len(ARMS)

    plan = {
        "schema_version": 1,
        "protocol": protocol,
        "arms": list(ARMS),
        "task_ids": list(task_ids),
        "official_state_positions": list(positions),
        "episodes_per_task_per_arm": len(positions),
        "measured_episodes": measured_episodes,
        "measured_blocks": measured_blocks,
        "unmeasured_warmup_episodes": measured_blocks,
        "block_definition": "one task x one method; 5 episodes in full mode",
        "method_order_by_task": orders,
        "timer": "CUDA synchronize + perf_counter around action_head.predict_action",
        "combined_semantics": "Cold+LDCE on first prediction; Warm+LDCE when cache is available",
        "policy_query_latency_from_this_run": "do not report",
    }

    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    if root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output root already exists: {root}. Use --overwrite for a fresh benchmark."
            )
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "plan.json", plan)

    records = []
    state_path = root / "launcher_state.json"
    launcher = {
        **plan,
        "completed_blocks": 0,
        "active": None,
        "completed": False,
    }
    _write_json(state_path, launcher)

    measurement_index = 0
    for task_id in task_ids:
        order = _rotating_order(task_id)
        for order_position, arm in enumerate(order):
            block_root = _block_root(root, task_id, order_position, arm)
            launcher["active"] = {
                "measurement_index": measurement_index,
                "task_id": task_id,
                "order_position": order_position,
                "arm": arm,
            }
            _write_json(state_path, launcher)

            print(
                f"[block {measurement_index + 1}/{measured_blocks}] "
                f"task={task_id} pos={order_position} arm={arm}",
                flush=True,
            )
            started = time.time()
            result = _invoke_parent(
                args=args,
                arm=arm,
                task_id=task_id,
                block_root=block_root,
                smoke=args.smoke,
            )
            ended = time.time()
            records.append(
                {
                    "measurement_index": measurement_index,
                    "task_id": int(task_id),
                    "order_position": int(order_position),
                    "arm": arm,
                    "started_unix_s": float(started),
                    "ended_unix_s": float(ended),
                    "duration_s": float(ended - started),
                    "task_payload": result["task"],
                }
            )
            measurement_index += 1
            launcher["completed_blocks"] = measurement_index
            launcher["active"] = None
            _write_json(state_path, launcher)

    report = _aggregate(root, records, protocol=protocol)
    launcher["completed"] = True
    launcher["active"] = None
    _write_json(state_path, launcher)

    print("\nCompleted interleaved Action-head latency benchmark")
    print(json.dumps({arm: {
        "mean_ms": report["arms"][arm]["action_head_latency_prediction_weighted"]["mean_ms"],
        "speedup_vs_baseline_percent": report["arms"][arm]["speedup_vs_baseline_percent"],
        "avg_K": report["arms"][arm]["prediction_weighted_avg_K"],
    } for arm in ARMS}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
