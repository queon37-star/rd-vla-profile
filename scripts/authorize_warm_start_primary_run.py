#!/usr/bin/env python3
"""Authorize the frozen warm-start primary acquisition.

Authorization succeeds only when the analysis freeze, runtime snapshot, source
commit, checkpoint, protocol manifest, required scripts, and analyzer self-test
all match. The resulting JSON is required by the acquisition launcher.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "experiment/warm-start-primary-evaluation"
REQUIRED_SOURCE_FILES = (
    "scripts/run_warm_start_primary_acquisition.py",
    "scripts/analyze_warm_start_primary_final.py",
    "scripts/validate_warm_start_paired_runs.py",
    "scripts/authorize_warm_start_primary_run.py",
    "experiments/robot/libero/run_libero_eval.py",
    "experiments/robot/libero/evaluation_protocol.py",
    "experiments/robot/robot_utils.py",
    "prismatic/models/action_heads.py",
)


class AuthorizationError(ValueError):
    """Raised when final runtime authorization cannot be granted."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuthorizationError(message)


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
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-freeze", type=Path, required=True)
    parser.add_argument("--runtime-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require(not args.output.exists(), f"refusing to overwrite output: {args.output}")

    freeze = load_json(args.analysis_freeze)
    snapshot = load_json(args.runtime_snapshot)
    require(
        freeze.get("status") == "primary_analysis_frozen_runtime_not_authorized",
        "analysis freeze has wrong status",
    )
    require(freeze.get("final_run_authorized") is False, "analysis freeze unexpectedly authorizes rollout")
    require(
        snapshot.get("role") == "warm_start_runtime_snapshot_not_authorization",
        "runtime snapshot has wrong role",
    )
    require(snapshot.get("final_run_authorized") is False, "runtime snapshot unexpectedly authorizes rollout")

    freeze_sha = sha256_file(args.analysis_freeze)
    require(
        snapshot.get("analysis_freeze", {}).get("sha256") == freeze_sha,
        "runtime snapshot does not reference the supplied analysis freeze",
    )

    branch = git_text("branch", "--show-current")
    commit = git_text("rev-parse", "HEAD")
    tracked_status = git_text("status", "--porcelain", "--untracked-files=no")
    require(branch == EXPECTED_BRANCH, f"expected branch {EXPECTED_BRANCH}, got {branch}")
    require(not tracked_status, f"tracked source tree is dirty:\n{tracked_status}")
    require(snapshot.get("source", {}).get("commit") == commit, "source commit differs from runtime snapshot")
    require(snapshot.get("source", {}).get("branch") == branch, "source branch differs from runtime snapshot")
    require(snapshot.get("python", {}).get("executable") == sys.executable, "Python executable differs from runtime snapshot")

    checkpoint_path = Path(snapshot["checkpoint"]["path"])
    protocol_manifest_path = Path(snapshot["protocol_manifest"]["path"])
    checkpoint_digest = sha256_tree(checkpoint_path)
    protocol_digest = sha256_file(protocol_manifest_path)
    require(checkpoint_digest == snapshot["checkpoint"]["tree_sha256"], "checkpoint tree changed after snapshot")
    require(protocol_digest == snapshot["protocol_manifest"]["sha256"], "protocol manifest changed after snapshot")

    source_hashes = {}
    for relative in REQUIRED_SOURCE_FILES:
        path = REPO_ROOT / relative
        source_hashes[relative] = sha256_file(path)

    self_test = subprocess.run(
        [sys.executable, "scripts/analyze_warm_start_primary_final.py", "--self-test"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    require(
        self_test.returncode == 0,
        "final analyzer self-test failed:\n"
        + self_test.stdout
        + "\n"
        + self_test.stderr,
    )
    require(
        "Warm-start final analyzer self-test: PASS" in self_test.stdout,
        "final analyzer self-test did not emit PASS",
    )

    payload = {
        "schema_version": 1,
        "role": "warm_start_runtime_authorization",
        "study": "midpoint_warm_start_primary_evaluation",
        "authorized": True,
        "source": {
            "branch": branch,
            "commit": commit,
            "tracked_tree_clean": True,
            "required_source_file_sha256": source_hashes,
        },
        "analysis_freeze": {
            "path": str(args.analysis_freeze.resolve()),
            "sha256": freeze_sha,
        },
        "runtime_snapshot": {
            "path": str(args.runtime_snapshot.resolve()),
            "sha256": sha256_file(args.runtime_snapshot),
        },
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "tree_sha256": checkpoint_digest,
        },
        "protocol_manifest": {
            "path": str(protocol_manifest_path.resolve()),
            "sha256": protocol_digest,
        },
        "python_executable": sys.executable,
        "analyzer_self_test": {
            "passed": True,
            "stdout": self_test.stdout.strip(),
        },
        "acquisition_contract": {
            "base_seed": 47007,
            "phases": ["calibration", "screening", "final"],
            "phase_pairs_per_task": {
                "calibration": 10,
                "screening": 10,
                "final": 30,
            },
            "tasks": list(range(10)),
            "arms": [
                "cold_initialized_adaptive",
                "midpoint_warm_start_adaptive",
            ],
            "complete_all_phases_regardless_of_interim_results": True,
            "primary_pairs_per_task_after_exclusion": 47,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote: {args.output}")
    print(f"Authorized source commit: {commit}")
    print(f"Authorized checkpoint tree SHA-256: {checkpoint_digest}")
    print("Warm-start primary acquisition authorized: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
