#!/usr/bin/env python3
"""Build the scalar-only learned-convergence dataset from validated traces."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.learned_convergence_probe_lib import (  # noqa: E402
    build_dataset_records,
    canonical_json_bytes,
    dependency_versions,
    sha256_bytes,
    sha256_file,
    source_git_commit,
    write_dataset,
)
from scripts.origin_aware_calibration_lib import validate_calibration_run  # noqa: E402
from scripts.origin_aware_replay_lib import parse_shadow_predictions  # noqa: E402


DEFAULT_INITIAL_STATE_MANIFEST = (
    REPO_ROOT / "experiments/robot/libero/manifests/libero_spatial_official_50_v1.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--initial-state-manifest", type=Path, default=DEFAULT_INITIAL_STATE_MANIFEST
    )
    parser.add_argument("--base-seed", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _load_records(paths: list[Path]) -> list[dict]:
    records = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record must be an object: {path}:{line_number}")
            records.append(value)
    return records


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite dataset: {manifest_path}")

    # This is intentionally mandatory rather than an optional fast path.  It
    # rechecks the complete formal artifact contract before reading trace rows.
    validation = validate_calibration_run(
        str(args.run_root), str(args.initial_state_manifest), base_seed=args.base_seed
    )
    if validation.get("complete_10_task_gate") is not True:
        raise ValueError("learned probe requires the complete ten-task calibration gate")
    if validation["totals"] != {
        "tasks": 10,
        "episodes": 100,
        "successes": 92,
        "predictions": 2398,
        "actual_warm_predictions": 2298,
        "workload_shards": 200,
        "cold_workloads": 100,
        "actual_warm_workloads": 100,
    }:
        raise ValueError(f"frozen calibration totals changed: {validation['totals']}")

    trace_paths = [args.run_root / f"task{task_id}" / "steps.jsonl" for task_id in range(10)]
    trace_files = [
        {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in trace_paths
    ]
    trace_set_sha256 = sha256_bytes(
        canonical_json_bytes(
            [{"task_id": task_id, "sha256": item["sha256"]} for task_id, item in enumerate(trace_files)]
        )
    )
    predictions = parse_shadow_predictions(_load_records(trace_paths))
    if len(predictions) != validation["totals"]["predictions"]:
        raise ValueError("validation/trace prediction count mismatch")
    records = build_dataset_records(predictions)
    metadata = {
        "source_git_commit": source_git_commit(REPO_ROOT),
        "calibration_run_root": str(args.run_root.resolve()),
        "calibration_validation": validation,
        "calibration_validation_sha256": sha256_bytes(canonical_json_bytes(validation)),
        "initial_state_manifest": {
            "path": str(args.initial_state_manifest.resolve()),
            "sha256": sha256_file(args.initial_state_manifest),
        },
        "trace_files": trace_files,
        "trace_set_sha256": trace_set_sha256,
        "random_seed": args.base_seed,
        "dependency_versions": dependency_versions(),
        "excluded_sources": ["smoke", "screening", "final"],
    }
    manifest = write_dataset(args.output_dir, records, metadata)
    print(f"Validated episodes: {validation['totals']['episodes']}")
    print(f"Validated predictions: {validation['totals']['predictions']}")
    print(f"Scalar transitions: {manifest['transition_count']}")
    print(f"Dataset SHA-256: {manifest['dataset_sha256']}")
    print(f"Wrote: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
