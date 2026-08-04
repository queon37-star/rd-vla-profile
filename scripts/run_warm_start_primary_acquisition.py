#!/usr/bin/env python3
"""Run the authorized 50-state/task warm-start primary acquisition.

The launcher executes calibration, screening, and final protocol phases for the
cold adaptive and midpoint warm-start arms. It is resumable only at completed
run-directory boundaries. All phases must be completed regardless of interim
outcomes. Statistical analysis is performed separately after acquisition.
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

from run_warm_start_paired_planning import command_for


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "experiment/warm-start-primary-evaluation"
PHASE_COUNTS = {"calibration": 10, "screening": 10, "final": 30}
PHASES = tuple(PHASE_COUNTS)
TASKS = tuple(range(10))
ARMS = ("cold_initialized_adaptive", "midpoint_warm_start_adaptive")
BASE_SEED = 47007


class AcquisitionError(ValueError):
    """Raised when acquisition cannot proceed under the frozen contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AcquisitionError(message)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    require(path.is_file(), f"missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    require(path.is_dir(), f"missing directory: {path}")
    files = sorted(item for item in path.rglob("*") if item.is_file())
    require(files, f"empty directory: {path}")
    digest = hashlib.sha256()
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def git_text(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def validate_authorization(path: Path) -> dict[str, Any]:
    authorization = load_json(path)
    require(
        authorization.get("role") == "warm_start_runtime_authorization",
        "wrong authorization role",
    )
    require(authorization.get("authorized") is True, "acquisition is not authorized")
    contract = authorization.get("acquisition_contract")
    require(isinstance(contract, dict), "authorization lacks acquisition contract")
    require(contract.get("base_seed") == BASE_SEED, "base seed mismatch")
    require(contract.get("phases") == list(PHASES), "phase contract mismatch")
    require(contract.get("tasks") == list(TASKS), "task contract mismatch")
    require(contract.get("arms") == list(ARMS), "arm contract mismatch")
    require(
        contract.get("complete_all_phases_regardless_of_interim_results") is True,
        "authorization does not require complete acquisition",
    )
    return authorization


def current_runtime_matches(authorization: dict[str, Any]) -> None:
    branch = git_text("branch", "--show-current")
    commit = git_text("rev-parse", "HEAD")
    tracked_status = git_text("status", "--porcelain", "--untracked-files=no")
    require(branch == EXPECTED_BRANCH, f"expected branch {EXPECTED_BRANCH}, got {branch}")
    require(not tracked_status, f"tracked source tree is dirty:\n{tracked_status}")
    require(commit == authorization["source"]["commit"], "source commit differs from authorization")
    require(sys.executable == authorization["python_executable"], "Python executable differs from authorization")

    for relative, expected in authorization["source"]["required_source_file_sha256"].items():
        actual = sha256_file(REPO_ROOT / relative)
        require(actual == expected, f"authorized source file changed: {relative}")

    checkpoint = Path(authorization["checkpoint"]["path"])
    manifest = Path(authorization["protocol_manifest"]["path"])
    require(
        sha256_tree(checkpoint) == authorization["checkpoint"]["tree_sha256"],
        "checkpoint tree differs from authorization",
    )
    require(
        sha256_file(manifest) == authorization["protocol_manifest"]["sha256"],
        "protocol manifest differs from authorization",
    )


def build_plan(
    *,
    output_root: Path,
    authorization_path: Path,
    authorization: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = Path(authorization["checkpoint"]["path"])
    manifest = Path(authorization["protocol_manifest"]["path"])
    phase_plans = {}
    for phase, episodes in PHASE_COUNTS.items():
        phase_root = (output_root / phase).resolve()
        runs = []
        for task_id in TASKS:
            for arm in ARMS:
                output_dir = phase_root / f"task{task_id}" / arm
                runs.append(
                    {
                        "phase": phase,
                        "task_id": task_id,
                        "arm": arm,
                        "output_dir": str(output_dir),
                        "command": command_for(
                            mode="primary",
                            phase=phase,
                            episodes_per_task=episodes,
                            seed=BASE_SEED,
                            task_id=task_id,
                            arm=arm,
                            output_dir=output_dir,
                            checkpoint=checkpoint,
                            manifest=manifest,
                        ),
                    }
                )
        phase_plans[phase] = {
            "schema_version": 1,
            "study": "midpoint_warm_start_primary_evaluation",
            "mode": "primary_acquisition",
            "role": "primary_acquisition_phase_runtime_validation",
            "confirmatory_evidence_allowed": False,
            "source_commit": authorization["source"]["commit"],
            "checkpoint": {
                "path": str(checkpoint.resolve()),
                "tree_sha256_before": authorization["checkpoint"]["tree_sha256"],
            },
            "initial_state_manifest": {
                "path": str(manifest.resolve()),
                "sha256": authorization["protocol_manifest"]["sha256"],
            },
            "evaluation_protocol_phase": phase,
            "tasks": list(TASKS),
            "episodes_per_task": episodes,
            "seed": BASE_SEED,
            "arms": list(ARMS),
            "latency_scope": (
                "synchronized online policy-query latency around get_action; "
                "includes processor, VLM prediction, action policy, and postprocessing"
            ),
            "runs": runs,
        }
    return {
        "schema_version": 1,
        "study": "midpoint_warm_start_primary_evaluation",
        "role": "authorized_primary_acquisition_plan",
        "authorization_path": str(authorization_path.resolve()),
        "authorization_sha256": sha256_file(authorization_path),
        "analysis_freeze_path": authorization["analysis_freeze"]["path"],
        "analysis_freeze_sha256": authorization["analysis_freeze"]["sha256"],
        "source_commit": authorization["source"]["commit"],
        "checkpoint_tree_sha256": authorization["checkpoint"]["tree_sha256"],
        "protocol_manifest_sha256": authorization["protocol_manifest"]["sha256"],
        "base_seed": BASE_SEED,
        "phase_order": list(PHASES),
        "complete_all_phases_required": True,
        "phase_plans": phase_plans,
    }


def write_or_validate_plan(output_root: Path, plan: dict[str, Any]) -> None:
    path = output_root / "acquisition_plan.json"
    if path.exists():
        existing = load_json(path)
        require(existing == plan, "existing acquisition plan differs from frozen plan")
        return
    output_root.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for phase in PHASES:
        phase_root = output_root / phase
        phase_root.mkdir(parents=True, exist_ok=True)
        phase_plan_path = phase_root / "run_plan.json"
        phase_plan_path.write_text(
            json.dumps(plan["phase_plans"][phase], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def completed_run(output_dir: Path) -> bool:
    return (output_dir / "result.json").is_file() and (output_dir / "steps.jsonl").is_file()


def execute_runs(plan: dict[str, Any], output_root: Path) -> None:
    environment = dict(os.environ)
    environment["NUMBA_DISABLE_JIT"] = "1"
    total_runs = sum(len(plan["phase_plans"][phase]["runs"]) for phase in PHASES)
    run_index = 0
    for phase in PHASES:
        for run in plan["phase_plans"][phase]["runs"]:
            run_index += 1
            output_dir = Path(run["output_dir"])
            if completed_run(output_dir):
                print(
                    f"[{run_index}/{total_runs}] SKIP complete "
                    f"phase={phase} task={run['task_id']} arm={run['arm']}"
                )
                continue
            if output_dir.exists():
                raise AcquisitionError(
                    "incomplete run directory found; inspect it and remove only that "
                    f"directory before resuming: {output_dir}"
                )
            output_dir.mkdir(parents=True, exist_ok=False)
            command = list(run["command"])
            print(
                f"[{run_index}/{total_runs}] phase={phase} "
                f"task={run['task_id']} arm={run['arm']}"
            )
            print(shlex.join(command))
            subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)
            require(completed_run(output_dir), f"run did not produce complete outputs: {output_dir}")

    # Validate only after all 60 runs are complete; no statistical decision is made here.
    for phase in PHASES:
        phase_root = output_root / phase
        validation_path = phase_root / "validation_report.json"
        if validation_path.exists():
            validation = load_json(validation_path)
            require(validation.get("contract_passed") is True, f"{phase}: existing validation failed")
            continue
        command = [
            sys.executable,
            "scripts/validate_warm_start_paired_runs.py",
            "--run-root",
            str(phase_root),
            "--output",
            str(validation_path),
        ]
        print(shlex.join(command))
        subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)
        validation = load_json(validation_path)
        require(validation.get("contract_passed") is True, f"{phase}: runtime validation failed")


def finalize_report(
    *,
    output_root: Path,
    plan: dict[str, Any],
    authorization: dict[str, Any],
) -> None:
    current_runtime_matches(authorization)
    phase_reports = {}
    for phase in PHASES:
        validation_path = output_root / phase / "validation_report.json"
        validation = load_json(validation_path)
        require(validation.get("contract_passed") is True, f"{phase}: contract not passed")
        phase_reports[phase] = {
            "validation_report": str(validation_path.resolve()),
            "validation_report_sha256": sha256_file(validation_path),
            "pairs": validation.get("total_pairs"),
        }
    report_path = output_root / "acquisition_report.json"
    payload = {
        "schema_version": 1,
        "role": "completed_warm_start_primary_acquisition",
        "study": "midpoint_warm_start_primary_evaluation",
        "completed": True,
        "authorization_sha256": plan["authorization_sha256"],
        "analysis_freeze_sha256": plan["analysis_freeze_sha256"],
        "source_commit": plan["source_commit"],
        "checkpoint_tree_sha256": plan["checkpoint_tree_sha256"],
        "protocol_manifest_sha256": plan["protocol_manifest_sha256"],
        "base_seed": BASE_SEED,
        "all_phases_completed": True,
        "phase_reports": phase_reports,
        "statistical_analysis_performed": False,
    }
    if report_path.exists():
        require(load_json(report_path) == payload, "existing acquisition report differs")
    else:
        report_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"Acquisition completed: {report_path}")
    print("All 50 states/task were acquired for both arms.")
    print("Primary statistical analysis has not been performed by this launcher.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    authorization = validate_authorization(args.authorization)
    current_runtime_matches(authorization)
    plan = build_plan(
        output_root=args.output_root.resolve(),
        authorization_path=args.authorization,
        authorization=authorization,
    )
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    write_or_validate_plan(args.output_root, plan)
    execute_runs(plan, args.output_root)
    finalize_report(
        output_root=args.output_root,
        plan=plan,
        authorization=authorization,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
