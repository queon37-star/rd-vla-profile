#!/usr/bin/env python3
"""Run the frozen adaptive-Coda failure audit and safety-wrapper replay."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.adaptive_coda_gate_oof import parse_gate_predictions  # noqa: E402
from scripts.analyze_latent_dynamics_features import (  # noqa: E402
    DEFAULT_INPUT_ROOT,
    EXPECTED_PREDICTION_COUNT,
    EXPECTED_TRANSITION_COUNT,
    sha256_file,
    validate_input_records,
)
from scripts.audit_adaptive_coda_gate import (  # noqa: E402
    EXPECTED_PREDICTION_COUNT as EXPECTED_WARM_PREDICTION_COUNT,
    EXPECTED_WORKLOAD_IDENTITY,
    WRAPPER_COMPONENTS,
    build_failure_audit,
    index_replays,
    load_frozen_oof_artifacts,
    reconstruct_frozen_scores,
    replay_predeclared_wrappers,
    score_margins,
    selected_thresholds_by_fold,
    write_safety_outputs,
)
from scripts.check_latent_dynamics_trace import load_jsonl  # noqa: E402


DEFAULT_OOF_INPUT = DEFAULT_INPUT_ROOT / "adaptive_coda_gate_oof_975da90"


def source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def default_output_dir(commit: str) -> Path:
    return DEFAULT_INPUT_ROOT / f"adaptive_coda_gate_safety_{commit[:7]}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_OOF_INPUT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _trace_paths(report: Mapping[str, object]) -> list[Path]:
    inputs = report["inputs"]
    assert isinstance(inputs, Mapping)
    values = inputs["trace_files"]
    assert isinstance(values, list)
    return [Path(str(value)) for value in values]


def _print_summary(report: Mapping[str, object]) -> None:
    print("Predeclared calibration-development safety wrappers:")
    print("wrapper                                      reduction  exact-K   mean dK  p95  pass")
    print("-------------------------------------------  ---------  --------  --------  ---  ----")
    results = report["wrapper_results"]
    assert isinstance(results, Mapping)
    for policy in WRAPPER_COMPONENTS:
        result = results[policy]
        metrics = result["metrics"]
        print(
            f"{policy:<43}  {metrics['coda_call_reduction']:>8.3%}  "
            f"{metrics['exact_K_preservation_rate']:>7.3%}  "
            f"{metrics['mean_delta_K']:>8.4f}  {metrics['p95_delta_K']:>3.0f}  "
            f"{str(result['passes_full_promotion_gate']):>4}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    commit = source_commit()
    output_dir = args.output_dir or default_output_dir(commit)
    artifacts = load_frozen_oof_artifacts(args.input_dir)
    source_report = artifacts["metric_report"]
    trace_paths = _trace_paths(source_report)
    expected_trace_hashes = source_report["inputs"]["trace_sha256"]
    for path in trace_paths:
        actual = sha256_file(path)
        expected = expected_trace_hashes[str(path.resolve())]
        if actual != expected:
            raise ValueError(f"raw trace hash mismatch: {path}")
    records = load_jsonl(trace_paths)
    contract = validate_input_records(
        records,
        expected_identity_sha256=EXPECTED_WORKLOAD_IDENTITY,
        expected_prediction_count=EXPECTED_PREDICTION_COUNT,
        expected_transition_count=EXPECTED_TRANSITION_COUNT,
    )
    predictions = parse_gate_predictions(records)
    if len(predictions) != EXPECTED_WARM_PREDICTION_COUNT:
        raise ValueError(
            "ACTUAL_WARM prediction count mismatch: "
            f"expected={EXPECTED_WARM_PREDICTION_COUNT} actual={len(predictions)}"
        )
    print(
        f"Validated frozen inputs: {contract['prediction_count']} trace predictions, "
        f"{contract['transition_count']} transitions, {len(predictions)} ACTUAL_WARM"
    )

    thresholds = selected_thresholds_by_fold(artifacts["model_summary"])
    scores = reconstruct_frozen_scores(predictions, artifacts["model_summary"])
    replay_index = index_replays(artifacts["replays"])
    margins = score_margins(predictions, replay_index, scores, thresholds)
    failure_audit, failure_rows = build_failure_audit(
        artifacts["replays"], artifacts["model_summary"], margins=margins
    )
    wrappers = replay_predeclared_wrappers(predictions, artifacts["replays"])
    inputs = {
        "safety_evaluator_git_commit": commit,
        "adaptive_oof_input_dir": str(args.input_dir.resolve()),
        "adaptive_oof_output_hashes_sha256": sha256_file(
            args.input_dir / "output_hashes.json"
        ),
        "adaptive_oof_source_git_commit": source_report["inputs"]["source_git_commit"],
        "workload_identity_sha256": contract["workload_identity_sha256"],
        "trace_files": [str(path.resolve()) for path in trace_paths],
        "trace_sha256": {str(path.resolve()): sha256_file(path) for path in trace_paths},
    }
    write_safety_outputs(
        output_dir,
        failure_audit=failure_audit,
        failure_rows=failure_rows,
        wrapper_evaluation=wrappers,
        inputs=inputs,
        overwrite=args.overwrite,
    )
    _print_summary(wrappers)
    audit_requirement = failure_audit["exact_95_percent_requirement"]
    print(
        "Combined exact-K requirement: "
        f"required={audit_requirement['required_exact_prediction_count']} "
        f"actual={audit_requirement['actual_combined_exact_prediction_count']} "
        f"shortfall={audit_requirement['shortfall_count']}"
    )
    print(f"Wrote safety audit outputs to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
