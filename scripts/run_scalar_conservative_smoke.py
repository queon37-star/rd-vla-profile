#!/usr/bin/env python3
"""Run the frozen task-0/task-5 paired smoke for conservative scalar offsets."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO_ROOT / "outputs/12_24-24_24_Spatial_40k"
DEFAULT_INITIAL_STATE_MANIFEST = (
    REPO_ROOT
    / "experiments/robot/libero/manifests/libero_spatial_official_50_v1.json"
)
DEFAULT_OFFSET05 = (
    REPO_ROOT
    / "benchmark_results/preconvergence_trigger/seed7/scalar_runtime_policy_offset05"
)
DEFAULT_OFFSET10 = (
    REPO_ROOT
    / "benchmark_results/preconvergence_trigger/seed7/scalar_runtime_policy_offset10"
)
TASK_IDS = (0, 5)
ARMS = ("action_mse", "offset05", "offset10")


class ConservativeSmokeError(ValueError):
    """Raised when a candidate artifact or smoke output violates the plan."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ConservativeSmokeError(message)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def candidate_identity(path: Path, expected_limit: float) -> dict[str, Any]:
    manifest = load_json(path / "manifest.json")
    require(manifest.get("runtime_screening_only") is True, "candidate is not screening-only")
    require(manifest.get("promotion_allowed") is False, "candidate unexpectedly allows promotion")
    selection = manifest.get("threshold_selection")
    require(isinstance(selection, dict), "candidate has no threshold provenance")
    observed_source = str(selection.get("source_operating_point"))
    expected_source = f"severe_FPR<={expected_limit:.2f}"
    require(observed_source == expected_source, f"candidate source mismatch: {path}")
    artifact_sha256 = str(manifest.get("artifact_sha256", ""))
    require(len(artifact_sha256) == 64, f"candidate artifact SHA-256 is invalid: {path}")
    require((path / str(manifest.get("artifact_file", ""))).is_file(), f"candidate artifact is missing: {path}")
    return {
        "path": str(path.resolve()),
        "artifact_sha256": artifact_sha256,
        "policy_name": manifest.get("policy_name"),
        "uniform_threshold_offset": float(selection["uniform_threshold_offset"]),
        "source_operating_point": observed_source,
    }


def command_for(
    *,
    task_id: int,
    arm: str,
    output_dir: Path,
    checkpoint: Path,
    initial_state_manifest: Path,
    candidate: dict[str, Any] | None,
) -> list[str]:
    result_path = output_dir / "result.json"
    step_path = output_dir / "steps.jsonl"
    command = [
        sys.executable,
        "experiments/robot/libero/run_libero_eval.py",
        "--pretrained_checkpoint",
        str(checkpoint),
        "--sync_checkpoint_source_config",
        "False",
        "--task_suite_name",
        "libero_spatial",
        "--task_id",
        str(task_id),
        "--num_trials_per_task",
        "3",
        "--evaluation_protocol_phase",
        "smoke",
        "--initial_state_manifest_path",
        str(initial_state_manifest),
        "--initial_states_path",
        "DEFAULT",
        "--reset_rng_each_episode",
        "True",
        "--seed",
        "7",
        "--use_recurrent",
        "True",
        "--recurrence_max_iter",
        "32",
        "--use_warm_start",
        "True",
        "--warm_start_source",
        "midpoint",
        "--warm_start_min_iter",
        "2",
        "--use_latent_precheck",
        "False",
        "--latent_precheck_mode",
        "off",
        "--latent_precheck_trace_level",
        "off",
        "--shadow_full_depth",
        "False",
        "--num_exec_actions",
        "5",
        "--run_id_note",
        f"conservative-smoke-{arm}-task{task_id}",
        "--save_version",
        "scalar-conservative-smoke",
        "--step_log_file",
        str(step_path),
        "--json_log_file",
        str(result_path),
    ]
    if arm == "action_mse":
        command.extend(
            [
                "--recurrence_strategy",
                "adjacent_action_mse",
                "--recurrence_kl_thresh",
                "0.001",
                "--use_cached_final_output",
                "True",
            ]
        )
    else:
        require(candidate is not None, f"missing candidate for arm {arm}")
        command.extend(
            [
                "--recurrence_strategy",
                "scalar_policy",
                "--scalar_policy_artifact_path",
                candidate["path"],
                "--scalar_policy_expected_sha256",
                candidate["artifact_sha256"],
                "--scalar_policy_execution_mode",
                "confirm_next",
                "--use_cached_final_output",
                "False",
            ]
        )
    return command


def result_summary(result_path: Path) -> dict[str, Any]:
    result = load_json(result_path)
    total_episodes = int(result.get("total_episodes", -1))
    total_successes = int(result.get("total_successes", -1))
    require(total_episodes == 3, f"smoke result has {total_episodes} episodes: {result_path}")
    require(0 <= total_successes <= total_episodes, f"invalid success count: {result_path}")
    return {
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "success_rate": total_successes / total_episodes,
        "result_path": str(result_path.resolve()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--initial-state-manifest",
        type=Path,
        default=DEFAULT_INITIAL_STATE_MANIFEST,
    )
    parser.add_argument("--offset05-policy", type=Path, default=DEFAULT_OFFSET05)
    parser.add_argument("--offset10-policy", type=Path, default=DEFAULT_OFFSET10)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require(args.checkpoint.exists(), f"checkpoint does not exist: {args.checkpoint}")
    require(
        args.initial_state_manifest.is_file(),
        f"initial-state manifest does not exist: {args.initial_state_manifest}",
    )
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty smoke root: {args.output_root}")

    candidates = {
        "offset05": candidate_identity(args.offset05_policy, 0.05),
        "offset10": candidate_identity(args.offset10_policy, 0.10),
    }
    runs = []
    for task_id in TASK_IDS:
        for arm in ARMS:
            output_dir = args.output_root / f"task{task_id}" / arm
            command = command_for(
                task_id=task_id,
                arm=arm,
                output_dir=output_dir,
                checkpoint=args.checkpoint,
                initial_state_manifest=args.initial_state_manifest,
                candidate=candidates.get(arm),
            )
            runs.append(
                {
                    "task_id": task_id,
                    "arm": arm,
                    "output_dir": str(output_dir.resolve()),
                    "command": command,
                }
            )

    plan = {
        "schema_version": 1,
        "descriptive_candidate_screening": True,
        "promotion_allowed": False,
        "latency_scope": "post-VLM action-policy path; VLM backbone excluded",
        "tasks": list(TASK_IDS),
        "episodes_per_task": 3,
        "arms": list(ARMS),
        "candidates": candidates,
        "runs": runs,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    args.output_root.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_root / "run_plan.json"
    plan_path.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    environment = dict(os.environ)
    environment["NUMBA_DISABLE_JIT"] = "1"
    summaries = []
    for index, run in enumerate(runs, start=1):
        output_dir = Path(run["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=False)
        command = list(run["command"])
        print(
            f"[{index}/{len(runs)}] task={run['task_id']} arm={run['arm']}"
        )
        print(shlex.join(command))
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            check=True,
        )
        summary = {
            "task_id": run["task_id"],
            "arm": run["arm"],
            **result_summary(output_dir / "result.json"),
        }
        summaries.append(summary)
        print(
            f"Completed: success={summary['total_successes']}/"
            f"{summary['total_episodes']}"
        )

    execution_report = {
        **plan,
        "completed": True,
        "run_summaries": summaries,
    }
    report_path = args.output_root / "execution_report.json"
    report_path.write_text(
        json.dumps(execution_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("Smoke completed")
    for task_id in TASK_IDS:
        row = [summary for summary in summaries if summary["task_id"] == task_id]
        print(
            f"Task {task_id}: "
            + ", ".join(
                f"{item['arm']}={item['total_successes']}/3" for item in row
            )
        )
    print(f"Wrote: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
