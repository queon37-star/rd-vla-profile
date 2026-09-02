#!/usr/bin/env python3
"""One-pass final LIBERO-Spatial paper measurement.

Runs the four frozen paper arms over all 50 official states for each of the 10
LIBERO-Spatial tasks (2,000 formal episodes total) and collects, from the same
closed-loop rollouts:

* success / paired episode outcomes,
* recurrent depth K,
* exact Coda call accounting and LDCE elimination accounting,
* warm/cold origin accounting,
* CUDA-event Action-head latency, and
* the existing CUDA-synchronized get_action (policy-query) wall-clock latency.

The Action-head timer records CUDA events immediately before and after
``action_head.predict_action`` but performs no synchronization inside the
policy query.  The existing outer timer in run_libero_eval.py remains:

    cuda.synchronize -> perf_counter -> get_action -> cuda.synchronize

Thus the component timer does not insert a synchronization boundary between
VLM and Action-head execution.  CUDA events are resolved only after a task has
finished, outside all per-query policy latency intervals.

One unmeasured warm-up rollout is retained per arm.  The formal workload is
exactly 10 tasks x 50 episodes x 4 arms = 2,000 episodes.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import scripts.profile_spatial_paper_action_head as profiler
import scripts.run_spatial_paper_4arm as frozen


FINAL_PROTOCOL = "libero-spatial-final-onepass-50x10-4arm-v2"
DEFAULT_OUTPUT_ROOT = "benchmark_results/paper_spatial_final_onepass_50x10"

# Full official-state distribution.
profiler.PROFILE_STATE_POSITIONS = tuple(range(50))
profiler.PROFILE_EPISODES_PER_TASK = 50
profiler.PROFILE_TASK_IDS = tuple(range(10))
profiler.PROFILE_PROTOCOL = FINAL_PROTOCOL


# ---------------------------------------------------------------------------
# Method freeze: LDCE means LDCE is enabled whenever that method is selected.
# In Combined, the first policy query of each episode is Cold+LDCE and later
# queries are Warm+LDCE whenever the midpoint cache is available.
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
    # Reuse every existing validation condition.  The old paper harness had a
    # deliberate Combined-only restriction that disabled LDCE on cold-origin
    # predictions; validate a copy against that historical contract, then
    # enforce the final method definition on the real config.
    if arm == "combined":
        historical = copy.copy(cfg)
        historical.action_delta_deferred_apply_to_cold = False
        _original_validate_cfg(historical, arm=arm, smoke=smoke)
        if not cfg.action_delta_deferred_apply_to_cold:
            raise ValueError("Final Combined requires LDCE on cold predictions too")
        return

    _original_validate_cfg(cfg, arm=arm, smoke=smoke)
    if arm == "ldce" and not cfg.action_delta_deferred_apply_to_cold:
        raise ValueError("Final LDCE requires LDCE on cold predictions")


profiler._validate_cfg = _validate_cfg_final


# ---------------------------------------------------------------------------
# Lightweight component timer.
# Only two CUDA events are inserted per Action-head call.  No inner CUDA sync.
# ---------------------------------------------------------------------------
class _LazyCudaSampleList:
    def __init__(self):
        self._raw = []

    def __len__(self):
        return len(self._raw)

    def append(self, value):
        self._raw.append(value)

    @staticmethod
    def _resolve(sample):
        out = {
            "sequence_id": int(sample["sequence_id"]),
            "task_id": sample["task_id"],
            "failed_call": bool(sample["failed_call"]),
        }
        if sample.get("cuda"):
            out["action_head_latency_ms"] = float(
                sample["start_event"].elapsed_time(sample["end_event"])
            )
        else:
            out["action_head_latency_ms"] = float(
                (sample["host_end"] - sample["host_start"]) * 1000.0
            )
        return out

    def __getitem__(self, key):
        # The parent profiler slices samples only after run_task() returns.
        # Every policy query inside run_episode already ends with the outer CUDA
        # synchronization used for policy-query latency, so all events should be
        # complete.  One task-end synchronization is retained as a safety guard;
        # it is outside every measured policy-query interval.
        if isinstance(key, slice):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return [self._resolve(v) for v in self._raw[key]]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return self._resolve(self._raw[key])


class ActionHeadCudaEventTimer:
    """Wrap predict_action with CUDA events and no inner synchronization."""

    def __init__(self, action_head):
        self.action_head = action_head
        self.original = action_head.predict_action
        self.samples = _LazyCudaSampleList()
        self.current_task_id = None

    def __enter__(self):
        original = self.original
        samples = self.samples
        owner = self

        def timed_predict_action(*args, **kwargs):
            failed = False
            if torch.cuda.is_available():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                try:
                    return original(*args, **kwargs)
                except Exception:
                    failed = True
                    raise
                finally:
                    end_event.record()
                    samples.append(
                        {
                            "sequence_id": len(samples),
                            "task_id": owner.current_task_id,
                            "failed_call": failed,
                            "cuda": True,
                            "start_event": start_event,
                            "end_event": end_event,
                        }
                    )
            else:
                start = time.perf_counter()
                try:
                    return original(*args, **kwargs)
                except Exception:
                    failed = True
                    raise
                finally:
                    end = time.perf_counter()
                    samples.append(
                        {
                            "sequence_id": len(samples),
                            "task_id": owner.current_task_id,
                            "failed_call": failed,
                            "cuda": False,
                            "host_start": start,
                            "host_end": end,
                        }
                    )

        self.action_head.predict_action = timed_predict_action
        return self

    def __exit__(self, exc_type, exc, tb):
        self.action_head.predict_action = self.original


# _profile_arm resolves the timer class through this module global.
profiler.ActionHeadWallClockTimer = ActionHeadCudaEventTimer


# ---------------------------------------------------------------------------
# Enrich each completed task JSON from the existing prediction log.  This
# makes task JSON authoritative for both component and policy-query metrics,
# including after task-level resume.  Incomplete earlier attempts are deduped
# by (episode_id, prediction_step), keeping the latest record.
# ---------------------------------------------------------------------------
def _read_formal_records(step_path: Path, task_id: int):
    if not step_path.is_file():
        raise RuntimeError(f"Missing step log for completed task: {step_path}")

    latest = {}
    order = 0
    with step_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("evaluation_protocol_phase") != profiler.PROFILE_PHASE:
                continue
            if int(record.get("task_id", -1)) != int(task_id):
                continue
            episode = int(record.get("episode_id", -1))
            prediction = int(
                record.get("prediction_step", record.get("action_prediction_index", -1))
            )
            latest[(episode, prediction)] = (order, record)
            order += 1

    return [value[1] for value in sorted(latest.values(), key=lambda item: item[0])]


def _as_int(value, default=0):
    if value is None:
        return int(default)
    return int(value)


def _k_stats(values, max_iter=32):
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "min": None,
            "max": None,
            "max_iter_hit_count": 0,
            "max_iter_hit_rate": None,
        }
    hits = int(np.sum(arr >= float(max_iter)))
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "max_iter_hit_count": hits,
        "max_iter_hit_rate": float(hits / arr.size),
    }


def _accounting(samples):
    ks = [_as_int(s.get("recurrent_iteration_count")) for s in samples]
    actual_coda = sum(_as_int(s.get("coda_call_count")) for s in samples)
    potential_coda = sum(ks)
    computed_eliminated = potential_coda - actual_coda
    logged_eliminated = sum(
        _as_int(s.get("truly_eliminated_coda_call_count")) for s in samples
    )
    score_calls = sum(_as_int(s.get("ldce_score_call_count")) for s in samples)
    warm_count = sum(bool(s.get("warm_start_used")) for s in samples)
    ldce_applied = sum(bool(s.get("ldce_applied")) for s in samples)
    return {
        "potential_coda_calls": int(potential_coda),
        "actual_coda_calls": int(actual_coda),
        "coda_calls_per_prediction": (
            float(actual_coda / len(samples)) if samples else None
        ),
        "computed_eliminated_coda_calls": int(computed_eliminated),
        "logged_truly_eliminated_coda_calls": int(logged_eliminated),
        "coda_elimination_rate": (
            float(computed_eliminated / potential_coda) if potential_coda else 0.0
        ),
        "ldce_score_calls": int(score_calls),
        "warm_start_used_predictions": int(warm_count),
        "cold_predictions": int(len(samples) - warm_count),
        "ldce_applied_predictions": int(ldce_applied),
    }


def _enrich_task_payload(path: Path, payload: dict):
    if not payload.get("completed") or "samples" not in payload:
        return
    task_id = int(payload["task_id"])
    step_path = path.parent / "debug_predictions.jsonl"
    records = _read_formal_records(step_path, task_id)
    samples = payload["samples"]

    record_map = {
        (
            int(record.get("episode_id", -1)),
            int(record.get("prediction_step", record.get("action_prediction_index", -1))),
        ): record
        for record in records
    }
    if len(record_map) != len(samples):
        raise RuntimeError(
            f"task {task_id}: formal step records={len(record_map)} but "
            f"Action-head samples={len(samples)}"
        )

    per_episode_prediction = defaultdict(int)
    for sample in samples:
        episode = int(sample.get("episode", -1))
        prediction = per_episode_prediction[episode]
        per_episode_prediction[episode] += 1
        key = (episode, prediction)
        if key not in record_map:
            raise RuntimeError(f"task {task_id}: missing prediction record {key}")
        record = record_map[key]

        sample["prediction_step"] = prediction
        sample["policy_query_latency_ms"] = float(record["latency_ms"])
        sample["recurrent_iteration_count"] = _as_int(
            record.get("recurrent_iteration_count")
        )
        sample["max_recurrent_iteration"] = _as_int(
            record.get("max_recurrent_iteration"), 32
        )
        sample["coda_call_count"] = _as_int(record.get("coda_call_count"))
        sample["actual_origin"] = record.get("actual_origin")
        sample["warm_start_used"] = bool(record.get("warm_start_used", False))
        sample["ldce_applied"] = bool(
            record.get("action_delta_deferred_backfill_filter_applied", False)
        )
        sample["ldce_score_call_count"] = _as_int(
            record.get(
                "action_delta_deferred_backfill_filter_score_call_count",
                record.get("probe_score_call_count"),
            )
        )
        sample["truly_eliminated_coda_call_count"] = _as_int(
            record.get("truly_eliminated_coda_call_count")
        )
        sample["total_exact_coda_call_count"] = _as_int(
            record.get("total_exact_coda_call_count", record.get("coda_call_count"))
        )

    policy_values = [float(s["policy_query_latency_ms"]) for s in samples]
    k_values = [int(s["recurrent_iteration_count"]) for s in samples]
    payload["policy_query_latency"] = profiler._stats(policy_values)
    payload["recurrent_k"] = _k_stats(k_values)
    payload["compute_accounting"] = _accounting(samples)
    payload["measurement_contract"] = {
        "action_head": "CUDA-event elapsed around action_head.predict_action; no inner CUDA sync",
        "policy_query": "existing outer CUDA-sync + perf_counter get_action wall-clock",
        "same_closed_loop_predictions": True,
    }


def _augment_arm_summary(path: Path, payload: dict):
    if not payload.get("completed") or "arm" not in payload:
        return
    task_files = sorted(path.parent.glob("task_*.json"))
    if not task_files:
        return
    tasks = [profiler._load_json(task_path) for task_path in task_files]
    if any(not task.get("completed") for task in tasks):
        return

    samples = [sample for task in tasks for sample in task.get("samples", [])]
    if not samples:
        return
    policy = [float(s["policy_query_latency_ms"]) for s in samples]
    ks = [int(s["recurrent_iteration_count"]) for s in samples]

    by_episode = defaultdict(list)
    for sample in samples:
        key = (int(sample.get("task_id", -1)), int(sample.get("episode", -1)))
        by_episode[key].append(float(sample["policy_query_latency_ms"]))
    episode_policy_means = [float(np.mean(v)) for v in by_episode.values() if v]

    payload["success_rate"] = (
        float(payload["successes"] / payload["episodes"])
        if payload.get("episodes")
        else 0.0
    )
    payload["policy_query_latency_prediction_weighted"] = profiler._stats(policy)
    payload["policy_query_latency_episode_balanced"] = profiler._stats(
        episode_policy_means
    )
    payload["recurrent_k_prediction_weighted"] = _k_stats(ks)
    payload["compute_accounting"] = _accounting(samples)
    payload["measurement_contract"] = {
        "formal_episodes": int(payload.get("episodes", 0)),
        "action_head": "CUDA-event elapsed; no inner synchronization",
        "policy_query": "CUDA-synchronized get_action wall-clock",
        "profiling_coda_cost": False,
        "pytorch_profiler": False,
    }


def _episode_outcomes(root: Path, arm: str):
    outcomes = {}
    arm_dir = root / arm
    for task_path in sorted(arm_dir.glob("task_*.json")):
        task = profiler._load_json(task_path)
        task_id = int(task["task_id"])
        for episode in task.get("episode_stats", []):
            key = (
                task_id,
                int(episode.get("initial_state_id", -1)),
                int(episode.get("paired_trial_id", -1)),
            )
            outcomes[key] = bool(episode.get("success"))
    return outcomes


def _paired_counts(left, right):
    common = sorted(set(left) & set(right))
    result = {
        "paired_episodes": len(common),
        "same_success": 0,
        "same_failure": 0,
        "left_success_right_failure": 0,
        "left_failure_right_success": 0,
    }
    for key in common:
        a, b = bool(left[key]), bool(right[key])
        if a and b:
            result["same_success"] += 1
        elif not a and not b:
            result["same_failure"] += 1
        elif a:
            result["left_success_right_failure"] += 1
        else:
            result["left_failure_right_success"] += 1
    return result


# ---------------------------------------------------------------------------
# Metadata + task/arm summary interception.
# ---------------------------------------------------------------------------
_original_write_json = profiler._write_json


def _write_json_final(path: Path, payload):
    if isinstance(payload, dict):
        # plan.json / launcher_state.json
        if "measured_total_episodes" in payload and "timer" in payload:
            payload["timer"] = (
                "CUDA events around action_head.predict_action with no inner sync; "
                "existing outer CUDA-sync + perf_counter get_action timer retained"
            )
            payload["e2e_latency_from_this_run"] = (
                "reportable policy-query latency from the same formal run"
            )
            payload["metrics_collected"] = [
                "success",
                "recurrent_K",
                "Coda_calls",
                "LDCE_eliminated_Coda",
                "warm_cold_origin",
                "Action-head_CUDA_event_latency",
                "policy-query_wall-clock_latency",
            ]

        # protocol.json
        scope = payload.get("measurement_scope")
        if isinstance(scope, dict):
            scope["name"] = "action_head.predict_action CUDA-event elapsed"
            scope["start"] = "CUDA event recorded immediately before action_head.predict_action"
            scope["end"] = "CUDA event recorded immediately after action_head.predict_action returns"
            scope["includes"] = [
                "GPU work issued by Prelude",
                "GPU work issued by recurrent core",
                "GPU work issued by LDCE scorer/scheduler tensor operations",
                "GPU work issued by Coda/backfill and output projection",
            ]
            scope["excludes"] = [
                "VLM forward before Action-head",
                "image preprocessing",
                "environment stepping",
                "Python-only host overhead inside predict_action",
            ]
            scope["inner_cuda_synchronization"] = False
            scope["inner_coda_profiling_enabled"] = False
            scope["e2e_latency_from_this_profile_is_reportable"] = True
            scope["policy_query_timer"] = (
                "existing outer CUDA-sync + perf_counter around get_action"
            )

        if path.name.startswith("task_") and path.suffix == ".json":
            _enrich_task_payload(path, payload)

        if path.name == "summary.json" and "action_head_latency_prediction_weighted" in payload:
            _augment_arm_summary(path, payload)

        if path.name == "comparison.json" and "comparison" in payload:
            root = path.parent
            summaries = {}
            for arm in ("baseline", "ldce", "warm_start", "combined"):
                summary_path = root / arm / "summary.json"
                if summary_path.is_file():
                    summaries[arm] = profiler._load_json(summary_path)

            baseline = summaries.get("baseline")
            if baseline is not None:
                baseline_policy = float(
                    baseline["policy_query_latency_prediction_weighted"]["mean_ms"]
                )
                for arm, summary in summaries.items():
                    entry = payload["comparison"].setdefault(arm, {})
                    policy = float(
                        summary["policy_query_latency_prediction_weighted"]["mean_ms"]
                    )
                    entry["mean_action_head_cuda_event_ms"] = float(
                        summary["action_head_latency_prediction_weighted"]["mean_ms"]
                    )
                    entry["mean_policy_query_latency_ms"] = policy
                    entry["policy_query_speedup_vs_baseline_percent"] = (
                        100.0 * (baseline_policy - policy) / baseline_policy
                    )
                    entry["success_rate"] = summary.get("success_rate")
                    entry["avg_K"] = summary[
                        "recurrent_k_prediction_weighted"
                    ]["mean"]
                    entry["compute_accounting"] = summary["compute_accounting"]
                payload["primary_latency_aggregation"] = (
                    "same-run policy-query wall-clock + Action-head CUDA-event elapsed"
                )

            paired = {}
            if (root / "baseline" / "summary.json").is_file() and (
                root / "ldce" / "summary.json"
            ).is_file():
                paired["baseline_vs_ldce"] = _paired_counts(
                    _episode_outcomes(root, "baseline"),
                    _episode_outcomes(root, "ldce"),
                )
            if (root / "warm_start" / "summary.json").is_file() and (
                root / "combined" / "summary.json"
            ).is_file():
                paired["warm_start_vs_combined"] = _paired_counts(
                    _episode_outcomes(root, "warm_start"),
                    _episode_outcomes(root, "combined"),
                )
            payload["paired_success"] = paired

    return _original_write_json(path, payload)


profiler._write_json = _write_json_final


# Give this final wrapper its own default output root without adding a new CLI
# option.  All existing --resume/--overwrite/--smoke/--dry-run behavior remains.
_original_parse_args = profiler.parse_args


def _parse_args_final():
    args = _original_parse_args()
    if args.output_root == "benchmark_results/paper_spatial_action_head_latency_10x10":
        args.output_root = DEFAULT_OUTPUT_ROOT
    return args


profiler.parse_args = _parse_args_final


if __name__ == "__main__":
    sys.exit(profiler.main())
