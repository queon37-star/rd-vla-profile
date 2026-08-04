#!/usr/bin/env python3
"""Run the authorized warm-start acquisition with immutable identity checks.

This wrapper reuses the resumable base launcher, verifies the frozen runtime
identity, and propagates the canonical authorization identity expected by the
frozen final analyzer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

import run_warm_start_primary_acquisition as base


REPO_ROOT = Path(__file__).resolve().parents[1]


class AcquisitionV2Error(ValueError):
    """Raised when immutable runtime identity verification fails."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AcquisitionV2Error(message)


def pip_freeze_sha256() -> str:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return hashlib.sha256(completed.stdout.strip().encode("utf-8")).hexdigest()


def current_gpu_names() -> list[str]:
    if not torch.cuda.is_available():
        return []
    return [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]


def validate_runtime_identity(authorization: dict[str, Any]) -> None:
    identity = authorization.get("runtime_identity")
    require(isinstance(identity, dict), "authorization lacks runtime identity")
    require(identity.get("python_executable") == sys.executable, "Python executable changed")
    require(identity.get("torch_version") == torch.__version__, "PyTorch version changed")
    require(
        identity.get("torch_cuda_build_version") == torch.version.cuda,
        "PyTorch CUDA build changed",
    )
    cudnn = torch.backends.cudnn.version()
    cudnn = int(cudnn) if cudnn is not None else None
    require(identity.get("cudnn_version") == cudnn, "cuDNN version changed")
    require(identity.get("gpu_names") == current_gpu_names(), "GPU identity changed")
    require(
        identity.get("pip_freeze_sha256") == pip_freeze_sha256(),
        "installed Python package set changed",
    )
    token = authorization.get("self_sha256")
    require(isinstance(token, str) and len(token) == 64, "authorization identity is invalid")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    authorization = base.validate_authorization(args.authorization)
    validate_runtime_identity(authorization)
    base.current_runtime_matches(authorization)
    plan = base.build_plan(
        output_root=args.output_root.resolve(),
        authorization_path=args.authorization,
        authorization=authorization,
    )
    plan["schema_version"] = 2
    plan["role"] = "authorized_primary_acquisition_plan_v2"
    plan["authorization_file_sha256"] = plan["authorization_sha256"]
    plan["authorization_sha256"] = authorization["self_sha256"]
    plan["authorization_identity_definition"] = authorization.get(
        "authorization_identity_definition"
    )

    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    base.write_or_validate_plan(args.output_root, plan)
    base.execute_runs(plan, args.output_root)
    validate_runtime_identity(authorization)
    base.finalize_report(
        output_root=args.output_root,
        plan=plan,
        authorization=authorization,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
