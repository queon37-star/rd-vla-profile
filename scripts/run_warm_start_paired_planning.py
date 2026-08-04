#!/usr/bin/env python3
"""Run paired midpoint warm-start planning checks or a planning-only pilot.

Modes:
- preflight: tasks 0 and 5, frozen smoke phase, 3 pairs/task.
- extended_preflight: all ten tasks, frozen smoke phase, 3 pairs/task.
- pilot: all ten tasks, frozen calibration phase, 10 pairs/task.

No mode produces confirmatory evidence. The 10-pair/task pilot remains blocked
until the frozen task-0/task-5 preflight validation report has passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO_ROOT / "outputs/12_24-24_24_Spatial_40k"
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "experiments/robot/libero/manifests/libero_spatial_official_50_v1.json"
)
DEFAULT_SEED = 17007
ARMS = ("cold_initialized_adaptive", "midpoint_warm_start_adaptive")
MODE_CONFIG = {
    "preflight": {
        "phase": "smoke",
        "episodes_per_task": 3,
        "default_tasks": (0, 5),
        "role": "planning_preflight_not_evidence",
    },
    "extended_preflight": {
        "phase": "smoke",
        "episodes_per_task": 3,
        "default_tasks": tuple(range(10)),
        "role": "extended_planning_preflight_not_evidence",
    },
    "pilot": {
        "phase": "calibration",
        "episodes_per_task": 10,
        "default_tasks": tuple(range(10)),
        "role": "planning_only_pilot_not_confirmatory_evidence",
    },
}


class PlanningRunError(ValueError):
    """Raised when a planning run violates the frozen launch contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PlanningRunError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    require(path.is_dir(), f"checkpoint directory does not exist: {path}")
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    require(files, f"checkpoint directory is empty: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def source_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    require(len(value) == 40, "unable to resolve source commit")
    return value


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def draccus_string(value: str) -> str:
    require(isinstance(value, str) and value, "Draccus string must be non-empty")
    return json.dumps(value)


def parse_tasks(text: str | None, defaults: tuple[int, ...]) -> tuple[int, ...]:
    if text is None:
        return defaults
    tasks = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    require(tasks, "at least one task is required")
    require(len(tasks) == len(set(tasks)), "task IDs must be unique")
    require(all(0 <= task < 10 for task in tasks), "task IDs must lie in 0..9")
    return tasks


def validate_preflight_report(path: Path) -> dict[str, Any]:
    report = load_json(path)
    require(report.get("contract_passed") is True, "preflight contract did not pass")
    require(report.get("role") == "planning_preflight_not_evidence", "wrong preflight report role")
    require(report.get("tasks") == [0, 5], "preflight must cover tasks 0 and 5")
    require(int(report.get("episodes_per_task", -1)) == 3, "preflight must use 3 pairs/task")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
    }


def command_for(
    *,
    mode: str,
    phase: str,
    episodes_per_task: int,
    seed: int,
    task_id: int,
    arm: str,
    output_dir: Path,
    checkpoint: Path,
    manifest: Path,
) -> list[str]:
    use_warm = arm == "midpoint_warm_start_adaptive"
    return [
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
        str(episodes_per_task),
        "--evaluation_protocol_phase",
        phase,
        "--initial_state_manifest_path",
        str(manifest),
        "--initial_states_path",
        "DEFAULT",
        "--reset_rng_each_episode",
        "True",
        "--seed",
        str(seed),
        "--use_recurrent",
        "True",
        "--recurrence_strategy",
        "adjacent_action_mse",
        "--recurrence_kl_thresh",
        "0.001",
        "--recurrence_max_iter",
        "32",
        "--use_warm_start",
        str(use_warm),
        "--warm_start_source",
        "midpoint",
        "--warm_start_min_iter",
        "2",
        "--validate_warm_start_finite",
        "True",
        "--use_cached_final_output",
        "True",
        "--use_latent_precheck",
        "False",
        "--latent_precheck_mode",
        draccus_string("off"),
        "--latent_precheck_trace_level",
        draccus_string("off"),
        "--shadow_full_depth",
        "False",
        "--collect_preconvergence_raw_shadow",
        "False",
        "--profile_coda_cost",
        "False",
        "--profile_pytorch",
        "False",
        "--profile_timing_summary",
        "False",
        "--num_exec_actions",
        "5",
        "--adaptive_exec",
        "False",
        "--dynamic_exec",
        "False",
        "--use_linear_decay_horizon",
        "False",
        "--use_wandb",
        "False",
        "--run_id_note",
        f"warm-start-{mode}-{arm}-task{task_id}-seed{seed}",
        "--save_version",
        f"warm-start-{mode}",
        "--step_log_file",
        str(output_dir / "steps.jsonl"),
        "--json_log_file",
        str(output_dir / "result.json"),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(MODE_CONFIG), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--initial-state-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--tasks",
        help=(
            "Comma-separated task IDs. Frozen defaults are 0,5 for preflight "
            "and all 0..9 for extended_preflight/pilot."
        ),
    )
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = MODE_CONFIG[args.mode]
    tasks = parse_tasks(args.tasks, config["default_tasks"])

    if args.mode == "pilot":
        require(tasks == tuple(range(10)), "planning pilot must cover exactly tasks 0..9")
        require(args.preflight_report is not None, "pilot requires --preflight-report")
        preflight = validate_preflight_report(args.preflight_report)
    elif args.mode == "extended_preflight":
        require(
            tasks == tuple(range(10)),
            "extended preflight must cover exactly tasks 0..9",
        )
        require(
            args.preflight_report is None,
            "extended preflight must not receive --preflight-report",
        )
        preflight = None
    else:
        require(tasks == (0, 5), "preflight is frozen to tasks 0 and 5")
        require(args.preflight_report is None, "preflight must not receive --preflight-report")
        preflight = None

    require(args.seed >= 0, "seed must be non-negative")
    require(args.checkpoint.is_dir(), f"checkpoint does not exist: {args.checkpoint}")
    require(args.initial_state_manifest.is_file(), f"manifest does not exist: {args.initial_state_manifest}")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty run root: {args.output_root}")

    commit = source_commit()
    checkpoint_digest_before = sha256_tree(args.checkpoint)
    manifest_digest = sha256_file(args.initial_state_manifest)
    runs = []
    for task_id in tasks:
        for arm in ARMS:
            output_dir = (args.output_root / f"task{task_id}" / arm).resolve()
            runs.append(
                {
                    "task_id": task_id,
                    "arm": arm,
                    "output_dir": str(output_dir),
                    "command": command_for(
                        mode=args.mode,
                        phase=config["phase"],
                        episodes_per_task=config["episodes_per_task"],
                        seed=args.seed,
                        task_id=task_id,
                        arm=arm,
                        output_dir=output_dir,
                        checkpoint=args.checkpoint,
                        manifest=args.initial_state_manifest,
                    ),
                }
            )

    plan = {
        "schema_version": 1,
        "study": "midpoint_warm_start_primary_evaluation",
        "mode": args.mode,
        "role": config["role"],
        "confirmatory_evidence_allowed": False,
        "source_commit": commit,
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "tree_sha256_before": checkpoint_digest_before,
        },
        "initial_state_manifest": {
            "path": str(args.initial_state_manifest.resolve()),
            "sha256": manifest_digest,
        },
        "preflight_report": preflight,
        "evaluation_protocol_phase": config["phase"],
        "tasks": list(tasks),
        "episodes_per_task": config["episodes_per_task"],
        "seed": args.seed,
        "arms": list(ARMS),
        "latency_scope": (
            "synchronized online policy-query latency around get_action; "
            "includes processor, VLM prediction, action policy, and postprocessing"
        ),
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
        print(f"[{index}/{len(runs)}] task={run['task_id']} arm={run['arm']}")
        print(shlex.join(command))
        subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)
        result = load_json(output_dir / "result.json")
        summaries.append(
            {
                "task_id": run["task_id"],
                "arm": run["arm"],
                "total_episodes": int(result.get("total_episodes", -1)),
                "total_successes": int(result.get("total_successes", -1)),
            }
        )

    checkpoint_digest_after = sha256_tree(args.checkpoint)
    require(
        checkpoint_digest_after == checkpoint_digest_before,
        "checkpoint tree changed during the run",
    )
    execution_report = {
        **plan,
        "completed": True,
        "checkpoint_tree_sha256_after": checkpoint_digest_after,
        "run_summaries": summaries,
    }
    report_path = args.output_root / "execution_report.json"
    report_path.write_text(
        json.dumps(execution_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    validation_path = args.output_root / "validation_report.json"
    validation_command = [
        sys.executable,
        "scripts/validate_warm_start_paired_runs.py",
        "--run-root",
        str(args.output_root),
        "--output",
        str(validation_path),
    ]
    print(shlex.join(validation_command))
    subprocess.run(validation_command, cwd=REPO_ROOT, env=environment, check=True)
    print(f"{args.mode.capitalize()} completed")
    print(f"Execution report: {report_path}")
    print(f"Validation report: {validation_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
