#!/usr/bin/env python3
"""Create the final warm-start runtime authorization with immutable identity.

The script delegates all base checks to authorize_warm_start_primary_run.py,
then adds the runtime identities and canonical authorization identity required
by the resumable v2 launcher and frozen analyzer.
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


REPO_ROOT = Path(__file__).resolve().parents[1]
V2_SOURCE_FILES = (
    "scripts/authorize_warm_start_primary_run_v2.py",
    "scripts/run_warm_start_primary_acquisition_v2.py",
)


class AuthorizationV2Error(ValueError):
    """Raised when the v2 immutable authorization cannot be created."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuthorizationV2Error(message)


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


def canonical_identity(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-freeze", type=Path, required=True)
    parser.add_argument("--runtime-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require(not args.output.exists(), f"refusing to overwrite output: {args.output}")
    snapshot = load_json(args.runtime_snapshot)
    freeze_sha = sha256_file(args.analysis_freeze)
    require(
        snapshot.get("analysis_freeze", {}).get("sha256") == freeze_sha,
        "runtime snapshot does not match the supplied analysis freeze",
    )

    temporary = args.output.with_suffix(args.output.suffix + ".base.tmp")
    require(not temporary.exists(), f"temporary authorization already exists: {temporary}")
    command = [
        sys.executable,
        "scripts/authorize_warm_start_primary_run.py",
        "--analysis-freeze",
        str(args.analysis_freeze),
        "--runtime-snapshot",
        str(args.runtime_snapshot),
        "--output",
        str(temporary),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    require(
        completed.returncode == 0,
        "base authorization failed:\n" + completed.stdout + "\n" + completed.stderr,
    )
    base = load_json(temporary)
    temporary.unlink()
    require(base.get("authorized") is True, "base authorization did not authorize")

    source_hashes = base["source"]["required_source_file_sha256"]
    for relative in V2_SOURCE_FILES:
        source_hashes[relative] = sha256_file(REPO_ROOT / relative)

    gpu_names = [item["name"] for item in snapshot.get("gpu", {}).get("devices", [])]
    require(gpu_names, "runtime snapshot contains no GPU devices")
    payload = {
        **base,
        "schema_version": 2,
        "role": "warm_start_runtime_authorization",
        "frozen_inputs": {
            "analysis_freeze_sha256": freeze_sha,
            "runtime_snapshot_sha256": sha256_file(args.runtime_snapshot),
        },
        "runtime_identity": {
            "python_executable": snapshot["python"]["executable"],
            "python_version": snapshot["python"]["version"],
            "torch_version": snapshot["pytorch"]["version"],
            "torch_cuda_build_version": snapshot["pytorch"]["cuda_build_version"],
            "cudnn_version": snapshot["pytorch"]["cudnn_version"],
            "gpu_names": gpu_names,
            "pip_freeze_sha256": snapshot["packages"]["pip_freeze_sha256"],
        },
        "authorization_identity_definition": (
            "SHA-256 of canonical authorization JSON before adding self_sha256; "
            "used as an immutable cross-artifact identifier, not as the file-byte hash"
        ),
    }
    payload["self_sha256"] = canonical_identity(payload)

    require(payload["runtime_identity"]["python_executable"] == sys.executable, "Python executable changed")
    require(payload["runtime_identity"]["torch_version"] == torch.__version__, "PyTorch version changed")
    require(payload["runtime_identity"]["torch_cuda_build_version"] == torch.version.cuda, "PyTorch CUDA build changed")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote: {args.output}")
    print(f"Authorization identity: {payload['self_sha256']}")
    print(f"Authorized source commit: {payload['source']['commit']}")
    print("Warm-start primary acquisition authorized: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
