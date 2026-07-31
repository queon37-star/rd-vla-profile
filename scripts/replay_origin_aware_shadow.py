#!/usr/bin/env python
"""Validate shadow traces and rank origin-aware scheduler families by task OOF replay."""

import argparse
import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.origin_aware_replay_lib import (  # noqa: E402
    CONFIRMATION_MODES,
    FIXED_WARM_THRESHOLDS,
    MAX_SKIP_VALUES,
    WARM_QUANTILES,
    CostModel,
    SelectionConstraints,
    parse_shadow_predictions,
    run_task_level_oof_selection,
)


def load_json_records(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        records = json.loads(text)
        if not isinstance(records, list):
            raise ValueError(f"Expected a JSON list in {path}")
        return records
    records = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Replay origin-aware Coda schedules on clean midpoint shadow traces. "
            "Results are baseline-conditioned pruning estimates, not closed-loop estimates."
        )
    )
    parser.add_argument("trace_paths", nargs="+", type=Path)
    parser.add_argument("--fold-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--recurrent-ms", required=True, type=float)
    parser.add_argument("--decode-ms", required=True, type=float)
    parser.add_argument("--latent-gate-ms", required=True, type=float)
    parser.add_argument("--action-compare-ms", required=True, type=float)
    parser.add_argument("--finite-check-ms", required=True, type=float)
    parser.add_argument(
        "--fixed-thresholds", nargs="+", type=float, default=list(FIXED_WARM_THRESHOLDS)
    )
    parser.add_argument("--quantiles", nargs="+", type=float, default=list(WARM_QUANTILES))
    parser.add_argument(
        "--max-skip-values", nargs="+", type=int, default=list(MAX_SKIP_VALUES)
    )
    parser.add_argument(
        "--confirmation-modes", nargs="+", choices=CONFIRMATION_MODES,
        default=list(CONFIRMATION_MODES),
    )
    parser.add_argument("--top-n", type=int, default=6)
    parser.add_argument("--min-convergence-capture", type=float, default=0.995)
    parser.add_argument("--max-mean-delta-k", type=float, default=0.25)
    parser.add_argument("--max-p95-delta-k", type=float, default=1.0)
    parser.add_argument("--max-max-iter-rate-delta", type=float, default=0.0)
    parser.add_argument("--max-false-convergence-rate", type=float, default=0.0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    records = []
    for path in args.trace_paths:
        records.extend(load_json_records(path))
    predictions = parse_shadow_predictions(records)
    fold_manifest = json.loads(args.fold_manifest.read_text(encoding="utf-8"))
    costs = CostModel(
        recurrent_ms=args.recurrent_ms,
        decode_ms=args.decode_ms,
        latent_gate_ms=args.latent_gate_ms,
        action_compare_ms=args.action_compare_ms,
        finite_check_ms=args.finite_check_ms,
    )
    constraints = SelectionConstraints(
        min_convergence_capture=args.min_convergence_capture,
        max_mean_delta_k=args.max_mean_delta_k,
        max_p95_delta_k=args.max_p95_delta_k,
        max_max_iter_rate_delta=args.max_max_iter_rate_delta,
        max_false_convergence_rate=args.max_false_convergence_rate,
    )
    report = run_task_level_oof_selection(
        predictions,
        fold_manifest,
        costs,
        constraints=constraints,
        fixed_thresholds=args.fixed_thresholds,
        quantiles=args.quantiles,
        max_skip_values=args.max_skip_values,
        confirmation_modes=args.confirmation_modes,
        top_n=args.top_n,
    )
    report["inputs"] = [
        {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for path in args.trace_paths
    ]
    report["fold_manifest"] = {
        "path": str(args.fold_manifest.resolve()),
        "sha256": sha256_file(args.fold_manifest),
    }
    report["prediction_count"] = len(predictions)
    report["task_ids"] = sorted({prediction.task_id for prediction in predictions})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Validated predictions: {len(predictions)}")
    print(f"Candidate families: {report['family_grid_size']}")
    print(f"Safety-passing families: {report['passing_family_count']}")
    print(f"Distinct refit configs: {report['selected_distinct_config_count']}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
