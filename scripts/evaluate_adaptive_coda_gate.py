#!/usr/bin/env python3
"""Run nested task-level OOF evaluation for learned adaptive Coda activation."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.adaptive_coda_gate_oof import (  # noqa: E402
    LEARNED_POLICIES,
    evaluate_nested_oof,
    parse_gate_predictions,
    write_evaluation_outputs,
)
from scripts.analyze_latent_dynamics_features import (  # noqa: E402
    DEFAULT_FOLD_MANIFEST,
    DEFAULT_INPUT_ROOT,
    EXPECTED_IDENTITY_SHA256,
    EXPECTED_PREDICTION_COUNT,
    EXPECTED_TRANSITION_COUNT,
    default_trace_paths,
    load_fold_assignment,
    sha256_file,
    validate_input_records,
)
from scripts.check_latent_dynamics_trace import load_jsonl  # noqa: E402


DEFAULT_PRIOR_ANALYSIS = DEFAULT_INPUT_ROOT / "analysis/analysis_report.json"


def source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def default_output_dir(commit: str) -> Path:
    return DEFAULT_INPUT_ROOT / f"adaptive_coda_gate_oof_{commit[:7]}"


def _classification_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    micro = result["oof_metrics"]["micro"]
    task_macro = result["oof_metrics"]["task_macro"]
    return {
        "micro": {
            key: micro[key]
            for key in ("auroc", "auprc", "balanced_accuracy", "positive_prevalence")
        },
        "task_macro": {
            key: task_macro[key]
            for key in ("auroc", "auprc", "balanced_accuracy", "positive_prevalence")
        },
    }


def load_prior_classification_context(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    groups = report["feature_group_oof"]
    univariate = report["univariate_oof"]
    return {
        "source": str(path.resolve()),
        "source_sha256": sha256_file(path),
        "scope": "prior transition-level activation_due classification; not scheduler replay",
        "metrics": {
            "iteration_only": _classification_metrics(
                groups["iteration_index"]["logistic_regression"]
            ),
            "raw_mse_logistic": _classification_metrics(
                groups["raw_mse"]["logistic_regression"]
            ),
            "iteration_raw_mse": {
                "available": False,
                "reason": "this exact two-feature group was not in the prior analysis",
            },
            "token_update_p95": _classification_metrics(univariate["token_update_p95"]),
            "update_dynamics": _classification_metrics(
                groups["update_dynamics"]["logistic_regression"]
            ),
            "combined": _classification_metrics(
                groups["combined"]["logistic_regression"]
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--trace", action="append", type=Path)
    parser.add_argument("--fold-manifest", type=Path, default=DEFAULT_FOLD_MANIFEST)
    parser.add_argument("--prior-analysis", type=Path, default=DEFAULT_PRIOR_ANALYSIS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _print_summary(report: Mapping[str, Any]) -> None:
    print("Sequence-level outer OOF results:")
    print("policy                         reduction   mean dK   p95 dK   exact-K   forced")
    print("-----------------------------  ----------  --------  --------  --------  --------")
    sections = {
        **report["reference_policies"],
        **report["learned_candidates"],
    }
    for policy in (
        "coda_every_iteration",
        "fixed_raw_mse_beta_0_05",
        *LEARNED_POLICIES,
    ):
        metrics = sections[policy]["oof_sequence_metrics"]
        print(
            f"{policy:<29}  {metrics['coda_call_reduction']:>9.4%}  "
            f"{metrics['mean_delta_K']:>8.4f}  {metrics['p95_delta_K']:>8.4f}  "
            f"{metrics['exact_K_preservation_rate']:>8.4%}  "
            f"{metrics['forced_trigger_rate']:>8.4%}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    commit = source_commit()
    output_dir = args.output_dir or default_output_dir(commit)
    trace_paths = args.trace or default_trace_paths(args.input_root)
    records = load_jsonl(trace_paths)
    contract = validate_input_records(
        records,
        expected_identity_sha256=EXPECTED_IDENTITY_SHA256,
        expected_prediction_count=EXPECTED_PREDICTION_COUNT,
        expected_transition_count=EXPECTED_TRANSITION_COUNT,
    )
    predictions = parse_gate_predictions(records)
    if len(predictions) != 2298:
        raise ValueError(
            f"ACTUAL_WARM prediction count mismatch: expected=2298 actual={len(predictions)}"
        )
    fold_manifest, assignment = load_fold_assignment(
        args.fold_manifest, {int(item["task_id"]) for item in predictions}
    )
    prior_context = load_prior_classification_context(args.prior_analysis)
    print(
        f"Validated frozen traces: {contract['prediction_count']} predictions, "
        f"{contract['transition_count']} transitions, {len(predictions)} ACTUAL_WARM"
    )
    evaluation = evaluate_nested_oof(
        predictions,
        assignment,
        prior_classification_context=prior_context,
    )
    inputs = {
        "source_git_commit": commit,
        "trace_files": [str(path.resolve()) for path in trace_paths],
        "trace_sha256": {str(path.resolve()): sha256_file(path) for path in trace_paths},
        "workload_identity_sha256": contract["workload_identity_sha256"],
        "fold_manifest": str(args.fold_manifest.resolve()),
        "fold_manifest_name": fold_manifest.get("name"),
        "fold_manifest_sha256": sha256_file(args.fold_manifest),
        "prior_analysis": str(args.prior_analysis.resolve()),
        "prior_analysis_sha256": sha256_file(args.prior_analysis),
    }
    write_evaluation_outputs(
        output_dir,
        evaluation,
        inputs=inputs,
        overwrite=args.overwrite,
    )
    _print_summary(evaluation["metric_report"])
    print(f"Wrote nested OOF outputs to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
