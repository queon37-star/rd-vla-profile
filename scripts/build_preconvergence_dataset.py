#!/usr/bin/env python3
"""Build the raw-latent preconvergence dataset from optional shadow shards."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.preconvergence_trigger_lib import (  # noqa: E402
    load_raw_manifest_sequences,
    save_dataset_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-manifest", type=Path, action="append", required=True)
    parser.add_argument("--authoritative-dataset-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset: {manifest_path}")
    metadata, sequences = load_raw_manifest_sequences(
        args.raw_manifest, args.authoritative_dataset_dir
    )
    manifest = save_dataset_bundle(args.output_dir, metadata, sequences)
    print(
        "Built preconvergence dataset: "
        f"{manifest['prediction_count']} predictions, "
        f"sha256={manifest['dataset_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
