#!/usr/bin/env python3
"""Capture the immutable runtime identity for the warm-start primary study.

The snapshot records source, checkpoint, protocol-manifest, Python, PyTorch,
CUDA, GPU, and package identities. It does not authorize the final rollout.
Tracked source changes are rejected; untracked benchmark outputs are ignored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "experiment/warm-start-primary-evaluation"
DEFAULT_CHECKPOINT = REPO_ROOT / "outputs/12_24-24_24_Spatial_40k"
DEFAULT_PROTOCOL_MANIFEST = (
    REPO_ROOT
    / "experiments/robot/libero/manifests/libero_spatial_official_50_v1.json"
)
DEFAULT_ANALYSIS_FREEZE = (
    REPO_ROOT / "benchmark_results/warm_start_primary/analysis_freeze_v2.json"
)


class SnapshotError(ValueError):
    """Raised when the runtime cannot be frozen safely."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SnapshotError(message)


def run_text(command: list[str], *, check: bool = True) -> str:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=check,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


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


def command_snapshot(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def gpu_snapshot() -> dict[str, Any]:
    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": int(properties.total_memory),
                    "compute_capability": [int(properties.major), int(properties.minor)],
                    "multi_processor_count": int(properties.multi_processor_count),
                }
            )
    nvidia_smi = command_snapshot(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    return {
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_device_count": int(torch.cuda.device_count()),
        "devices": devices,
        "nvidia_smi": nvidia_smi,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-freeze", type=Path, default=DEFAULT_ANALYSIS_FREEZE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--protocol-manifest", type=Path, default=DEFAULT_PROTOCOL_MANIFEST
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require(not args.output.exists(), f"refusing to overwrite output: {args.output}")

    branch = run_text(["git", "branch", "--show-current"])
    commit = run_text(["git", "rev-parse", "HEAD"])
    require(branch == EXPECTED_BRANCH, f"expected branch {EXPECTED_BRANCH}, got {branch}")
    require(len(commit) == 40, "invalid source commit")

    tracked_status = run_text(
        ["git", "status", "--porcelain", "--untracked-files=no"]
    )
    require(not tracked_status, f"tracked source tree is dirty:\n{tracked_status}")

    analysis_freeze = json.loads(args.analysis_freeze.read_text(encoding="utf-8"))
    require(
        analysis_freeze.get("status")
        == "primary_analysis_frozen_runtime_not_authorized",
        "analysis freeze has the wrong status",
    )
    require(
        analysis_freeze.get("final_run_authorized") is False,
        "analysis freeze unexpectedly authorizes rollout",
    )

    pip_freeze = command_snapshot([sys.executable, "-m", "pip", "freeze"])
    require(pip_freeze["returncode"] == 0, "pip freeze failed")
    pip_payload = pip_freeze["stdout"].encode("utf-8")

    conda_list = command_snapshot(["conda", "list", "--json"])
    cudnn_version = torch.backends.cudnn.version()
    payload = {
        "schema_version": 1,
        "role": "warm_start_runtime_snapshot_not_authorization",
        "study": "midpoint_warm_start_primary_evaluation",
        "source": {
            "repository_root": str(REPO_ROOT),
            "branch": branch,
            "commit": commit,
            "tracked_tree_clean": True,
            "untracked_files_ignored": True,
        },
        "analysis_freeze": {
            "path": str(args.analysis_freeze.resolve()),
            "sha256": sha256_file(args.analysis_freeze),
        },
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "tree_sha256": sha256_tree(args.checkpoint),
        },
        "protocol_manifest": {
            "path": str(args.protocol_manifest.resolve()),
            "sha256": sha256_file(args.protocol_manifest),
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "pytorch": {
            "version": torch.__version__,
            "cuda_build_version": torch.version.cuda,
            "cudnn_version": int(cudnn_version) if cudnn_version is not None else None,
            "deterministic_algorithms_enabled": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
        },
        "gpu": gpu_snapshot(),
        "packages": {
            "pip_freeze_sha256": hashlib.sha256(pip_payload).hexdigest(),
            "pip_freeze": pip_freeze["stdout"].splitlines(),
            "conda_list": conda_list,
        },
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CONDA_DEFAULT_ENV",
                "CONDA_PREFIX",
                "CUDA_VISIBLE_DEVICES",
                "CUBLAS_WORKSPACE_CONFIG",
                "PYTHONHASHSEED",
            )
        },
        "final_run_authorized": False,
    }

    require(payload["gpu"]["torch_cuda_available"], "CUDA is unavailable")
    require(payload["gpu"]["torch_cuda_device_count"] >= 1, "no CUDA GPU detected")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote: {args.output}")
    print(f"Frozen source commit: {commit}")
    print(f"Checkpoint tree SHA-256: {payload['checkpoint']['tree_sha256']}")
    print(
        "Runtime: "
        f"python={sys.version.split()[0]}, torch={torch.__version__}, "
        f"cuda={torch.version.cuda}, GPUs={payload['gpu']['torch_cuda_device_count']}"
    )
    print("Final run authorized: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
