#!/usr/bin/env python3
"""Train task-level OOF scalar convergence probes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.learned_convergence_probe_lib import (  # noqa: E402
    canonical_json_bytes,
    load_dataset,
    load_fold_manifest,
    sha256_file,
    train_oof_bundle,
)


DEFAULT_FOLD_MANIFEST = (
    REPO_ROOT / "experiments/robot/libero/manifests/libero_spatial_task_oof_5fold_v1.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--fold-manifest", type=Path, default=DEFAULT_FOLD_MANIFEST)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite training bundle: {args.output}")
    dataset_manifest, records = load_dataset(args.dataset_dir)
    _, assignment = load_fold_manifest(
        args.fold_manifest, {str(record["task_id"]) for record in records}
    )
    bundle = train_oof_bundle(records, assignment, seed=args.seed)
    bundle["inputs"] = {
        "dataset_manifest": str((args.dataset_dir / "manifest.json").resolve()),
        "dataset_sha256": dataset_manifest["dataset_sha256"],
        "fold_manifest": str(args.fold_manifest.resolve()),
        "fold_manifest_sha256": sha256_file(args.fold_manifest),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(bundle))
    print(f"Leakage audit passed: {bundle['leakage_audit']['passed']}")
    print(f"Models trained: {', '.join(bundle['model_order'])}")
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
