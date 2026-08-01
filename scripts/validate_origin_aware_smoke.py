#!/usr/bin/env python3
"""Validate origin-aware calibration smoke artifacts and emit a JSON report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.origin_aware_smoke_lib import (  # noqa: E402
    SMOKE_TASK_IDS,
    validate_smoke_run,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base-seed", type=int, default=7)
    parser.add_argument("--task-ids", nargs="+", type=int, default=list(SMOKE_TASK_IDS))
    parser.add_argument("--output", help="Optional JSON validation-report path")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = validate_smoke_run(
            args.run_root,
            args.manifest,
            base_seed=args.base_seed,
            task_ids=args.task_ids,
        )
    except (OSError, ValueError) as exc:
        print(f"Smoke validation failed: {exc}", file=sys.stderr)
        return 2

    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
