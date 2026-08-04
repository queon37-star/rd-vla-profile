#!/usr/bin/env python3
"""Build a hash-verified scalar runtime artifact with one uniform threshold offset.

The source task-OOF model parameters are left unchanged. Only each task's
selected threshold is shifted by the same descriptive offset read from the
formal k=3 separability audit. The resulting artifact is screening-only and is
not a new task-OOF threshold result.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prismatic.models.scalar_stopping_policy import (  # noqa: E402
    load_scalar_policy_artifact,
    sha256_file,
    validate_scalar_policy_artifact,
)


DEFAULT_SOURCE_POLICY = (
    REPO_ROOT
    / "benchmark_results/preconvergence_trigger/seed7/scalar_runtime_policy_kfirst_v1"
)
DEFAULT_K3_REPORT = (
    REPO_ROOT
    / "benchmark_results/preconvergence_trigger/seed7/k3_separability_audit/report.json"
)
SUPPORTED_FPR_LIMITS = (0.01, 0.05, 0.10, 0.20)


class ScalarOffsetArtifactError(ValueError):
    """Raised when the source artifact or offset provenance is invalid."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ScalarOffsetArtifactError(message)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON input: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def resolve_operating_point(
    report: Mapping[str, Any],
    fpr_limit: float,
) -> tuple[str, dict[str, Any]]:
    require(report.get("formal_run") is True, "k=3 separability report is not formal")
    require(
        report.get("latency_reporting_scope")
        == "post-VLM action-policy path; VLM backbone excluded",
        "k=3 report latency scope mismatch",
    )
    selected = report.get("uniform_margin_offset_sweep", {}).get(
        "selected_descriptive_operating_points"
    )
    require(isinstance(selected, Mapping), "k=3 report has no offset sweep")
    key = f"severe_FPR<={fpr_limit:.2f}"
    point = selected.get(key)
    require(isinstance(point, Mapping), f"missing operating point: {key}")
    offset = float(point.get("margin_offset", float("nan")))
    observed_fpr = float(point.get("severe_false_trigger_rate", float("nan")))
    safe_recall = float(point.get("safe_trigger_recall", float("nan")))
    require(math.isfinite(offset) and offset >= 0.0, "invalid threshold offset")
    require(
        math.isfinite(observed_fpr) and observed_fpr <= fpr_limit + 1e-12,
        "operating point violates its severe-FPR limit",
    )
    require(
        math.isfinite(safe_recall) and 0.0 <= safe_recall <= 1.0,
        "invalid safe-trigger recall",
    )
    return key, dict(point)


def build_offset_payload(
    source_payload: Mapping[str, Any],
    *,
    offset: float,
    source_report_sha256: str,
    operating_point_key: str,
    operating_point: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    payload = copy.deepcopy(dict(source_payload))
    policies = payload.get("policies_by_task")
    require(isinstance(policies, Mapping), "source payload has no task policies")

    thresholds: dict[str, dict[str, float]] = {}
    for task_id in range(10):
        key = str(task_id)
        require(key in policies, f"source policy is missing task {task_id}")
        policy = policies[key]
        require(isinstance(policy, dict), f"task {task_id} policy must be mutable")
        base = float(policy.get("selected_threshold", float("nan")))
        effective = base + float(offset)
        require(math.isfinite(base), f"task {task_id} base threshold is invalid")
        require(
            0.0 < effective < 1.0,
            f"task {task_id} effective threshold is outside (0, 1): {effective}",
        )
        policy["base_selected_threshold"] = base
        policy["selected_threshold"] = effective
        policy["uniform_threshold_offset"] = float(offset)
        thresholds[key] = {
            "base_threshold": base,
            "effective_threshold": effective,
        }

    payload["runtime_threshold_provenance"] = {
        "selection_type": "uniform_calibration_margin_offset",
        "screening_only": True,
        "promotion_allowed": False,
        "task_oof_model_parameters_preserved": True,
        "task_oof_threshold_claim": False,
        "uniform_threshold_offset": float(offset),
        "source_k3_report_sha256": source_report_sha256,
        "source_operating_point": operating_point_key,
        "observed_k3_severe_false_trigger_rate": float(
            operating_point["severe_false_trigger_rate"]
        ),
        "observed_k3_safe_trigger_recall": float(
            operating_point["safe_trigger_recall"]
        ),
        "selection_population": "seen formal calibration ACTUAL_WARM predictions",
    }
    validate_scalar_policy_artifact(payload)
    return payload, thresholds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-policy", type=Path, default=DEFAULT_SOURCE_POLICY)
    parser.add_argument("--k3-report", type=Path, default=DEFAULT_K3_REPORT)
    parser.add_argument(
        "--severe-fpr-limit",
        type=float,
        required=True,
        choices=SUPPORTED_FPR_LIMITS,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to reuse non-empty output directory: {args.output_dir}"
        )

    k3_report = load_json(args.k3_report)
    operating_point_key, operating_point = resolve_operating_point(
        k3_report, float(args.severe_fpr_limit)
    )
    source_manifest, source_payload = load_scalar_policy_artifact(args.source_policy)
    source_artifact_sha256 = str(source_manifest.get("artifact_sha256", ""))
    require(len(source_artifact_sha256) == 64, "source manifest has no artifact SHA-256")

    report_sha256 = sha256_file(args.k3_report)
    offset = float(operating_point["margin_offset"])
    payload, thresholds = build_offset_payload(
        source_payload,
        offset=offset,
        source_report_sha256=report_sha256,
        operating_point_key=operating_point_key,
        operating_point=operating_point,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = args.output_dir / "scalar_policy.pt"
    manifest_path = args.output_dir / "manifest.json"
    torch.save(payload, artifact_path)
    artifact_sha256 = sha256_file(artifact_path)

    policy_name = (
        "scalar_combo_oof_model_uniform_offset_"
        f"severe_fpr_{int(round(float(args.severe_fpr_limit) * 100)):02d}pct"
    )
    manifest = {
        "schema_version": int(payload["schema_version"]),
        "artifact_type": payload["artifact_type"],
        "artifact_file": artifact_path.name,
        "artifact_sha256": artifact_sha256,
        "policy_name": policy_name,
        "target_reference": payload["target_reference"],
        "model_configuration": payload["model_configuration"],
        "feature_names": list(payload["feature_names"]),
        "runtime_screening_only": True,
        "promotion_allowed": False,
        "task_oof_model_parameters_preserved": True,
        "task_oof_threshold_claim": False,
        "threshold_selection": payload["runtime_threshold_provenance"],
        "thresholds_by_task": thresholds,
        "source_artifact": {
            "path": str(args.source_policy.resolve()),
            "artifact_sha256": source_artifact_sha256,
            "policy_name": source_manifest.get("policy_name"),
        },
        "source_k3_separability_report": {
            "path": str(args.k3_report.resolve()),
            "sha256": report_sha256,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    # Re-open through the production loader so hash and schema failures are caught now.
    load_scalar_policy_artifact(
        args.output_dir,
        expected_sha256=artifact_sha256,
    )

    print("Built scalar offset runtime artifact")
    print(f"Severe-FPR operating point: {operating_point_key}")
    print(f"Uniform threshold offset: {offset:.9f}")
    print(f"Artifact SHA256: {artifact_sha256}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
