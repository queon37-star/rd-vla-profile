#!/usr/bin/env python3
"""Build the frozen official-state manifest for paired LIBERO evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.evaluation_protocol import build_protocol_manifest, write_protocol_manifest  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-suite-name", default="libero_spatial", choices=("libero_spatial",))
    parser.add_argument("--output", required=True, help="Output JSON manifest path")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    from libero.libero import benchmark, get_libero_path

    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task_initial_states = {}
    task_initial_state_files = {}
    task_names = {}
    for task_id in range(task_suite.n_tasks):
        task_initial_states[task_id] = task_suite.get_task_init_states(task_id)
        task = task_suite.get_task(task_id)
        task_initial_state_files[task_id] = str(
            Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
        )
        task_names[task_id] = getattr(task, "name", getattr(task, "language", str(task_id)))

    manifest = build_protocol_manifest(
        args.task_suite_name,
        task_initial_states,
        task_names=task_names,
        task_initial_state_files=task_initial_state_files,
    )
    manifest_sha256 = write_protocol_manifest(args.output, manifest, overwrite=args.overwrite)
    print(f"Wrote {args.output}")
    print(f"manifest_sha256={manifest_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
