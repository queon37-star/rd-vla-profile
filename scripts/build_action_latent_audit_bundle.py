#!/usr/bin/env python3
"""Build the frozen compact trajectory bundle for offline audits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.action_latent_audit_lib import (  # noqa: E402
    DEFAULT_EPSILON,
    build_trajectory_bundle,
)


DEFAULT_DATASET_DIR = (
    REPO_ROOT / "benchmark_results/preconvergence_trigger/seed7/dataset"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "benchmark_results/preconvergence_trigger/seed7/action_latent_audit"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_trajectory_bundle(
        args.dataset_dir,
        args.output_dir,
        epsilon=args.epsilon,
        repo_root=REPO_ROOT,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "prediction_count": manifest["prediction_count"],
                "output_bundle_sha256": manifest["output_bundle_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
