#!/usr/bin/env python3
"""Measure descriptive in-sample latent/action trajectory associations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.action_latent_audit_lib import (  # noqa: E402
    ACTION_THRESHOLDS,
    build_latent_signal_audit,
    load_trajectory_bundle,
    write_csv,
    write_json,
)


DEFAULT_BUNDLE_DIR = (
    REPO_ROOT / "benchmark_results/preconvergence_trigger/seed7/action_latent_audit"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.bundle_dir
    bundle_manifest, records = load_trajectory_bundle(args.bundle_dir)
    report, rows = build_latent_signal_audit(records, thresholds=ACTION_THRESHOLDS)
    report["trajectory_bundle_sha256"] = bundle_manifest["output_bundle_sha256"]
    write_json(output_dir / "latent_signal_audit.json", report)
    write_csv(output_dir / "latent_signal_by_task.csv", rows)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "prediction_count": report["prediction_count"],
                "row_count": report["row_count"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
