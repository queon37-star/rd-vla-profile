#!/usr/bin/env python3
"""Audit authoritative adjacent-action stopping rules offline."""

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
    build_action_stopping_audit,
    load_trajectory_bundle,
    write_csv,
    write_json,
    write_jsonl,
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
    report, predictions, task_rows = build_action_stopping_audit(
        records, thresholds=ACTION_THRESHOLDS
    )
    report["trajectory_bundle_sha256"] = bundle_manifest["output_bundle_sha256"]
    write_json(output_dir / "action_stopping_audit.json", report)
    write_jsonl(output_dir / "action_stopping_predictions.jsonl", predictions)
    write_csv(output_dir / "action_stopping_by_task.csv", task_rows)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "actual_warm_predictions": report["primary_actual_warm"]["prediction_count"],
                "cold_predictions": report["cold_reported_separately"]["prediction_count"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
