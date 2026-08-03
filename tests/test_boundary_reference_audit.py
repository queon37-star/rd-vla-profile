from __future__ import annotations

import math

import pytest
import torch

from scripts.action_latent_audit_lib import build_trajectory_record
from scripts.boundary_latent_oof_lib import (
    build_boundary_reference_audit,
    compute_boundary_references,
)
from scripts.preconvergence_trigger_lib import RawPreconvergenceSequence, SequenceIdentity


def make_record(
    prediction_id: int = 0,
    *,
    task_id: int = 0,
    origin: str = "ACTUAL_WARM",
    authoritative: list[float | None] | None = None,
    recomputed: list[float | None] | None = None,
    latent_dim: int = 4,
) -> dict:
    authoritative = authoritative or [None, None, 0.002, 0.0005, 0.003, 0.0004, 0.0003]
    recomputed = recomputed or [None, None, 0.002, 0.002, 0.0004, 0.0003, 0.0002]
    max_iter = len(authoritative) - 1
    values = [0.0]
    for k in range(2, max_iter + 1):
        values.append(values[-1] + math.sqrt(float(recomputed[k])))
    actions = torch.tensor(values, dtype=torch.float32).reshape(max_iter, 1, 1, 1).repeat(1, 1, 2, 2)
    base = torch.arange(max_iter, dtype=torch.float32).reshape(max_iter, 1, 1)
    states = torch.cat([base * (index + 1) for index in range(latent_dim)], dim=2)
    phases = [None, None] + ["production"] * (max_iter - 1)
    first = next((k for k in range(2, max_iter + 1) if float(authoritative[k]) < 0.001), max_iter)
    sequence = RawPreconvergenceSequence(
        identity=SequenceIdentity(task_id, 0, prediction_id),
        actual_origin=origin,
        states=states,
        actions=actions,
        action_mse=tuple(authoritative),
        action_mse_phase=tuple(phases),
        baseline_k=first,
        max_iter=max_iter,
    )
    return build_trajectory_record(sequence)


def test_all_four_reference_definitions_and_rebound() -> None:
    result = compute_boundary_references(make_record())
    assert result["references"] == {
        "K_first_authoritative": 3,
        "K_stable_suffix_authoritative": 5,
        "K_first_recomputed_fp32": 4,
        "K_stable_suffix_recomputed_fp32": 4,
    }
    assert result["authoritative_first_hit_rebound"] is True


def test_no_hit_remains_null() -> None:
    no_hit = [None, None] + [0.01] * 5
    result = compute_boundary_references(
        make_record(authoritative=no_hit, recomputed=no_hit)
    )
    assert all(value is None for value in result["references"].values())


def test_first_hit_masking_is_evaluated_only_at_authoritative_first_hit() -> None:
    result = compute_boundary_references(make_record())
    assert result["first_hit_masking"]["evaluated_at_k"] == 3
    assert result["first_hit_masking"]["eligible"] is True
    # The audit carries a single evaluation, not a stable-tail transition list.
    assert "transitions" not in result["first_hit_masking"]


def test_reference_report_separates_origins_and_reports_pairwise_deltas() -> None:
    warm = make_record()
    cold = make_record(1, task_id=1, origin="COLD")
    report, task_rows = build_boundary_reference_audit(
        [warm, cold],
        source_bundle_sha256="source",
        fold_manifest_sha256="fold",
        git_commit="commit",
    )
    assert report["primary_actual_warm"]["prediction_count"] == 1
    assert report["cold_excluded_from_oof"]["prediction_count"] == 1
    comparison = report["primary_actual_warm"]["pairwise_K_comparisons"][
        "K_first_authoritative__vs__K_first_recomputed_fp32"
    ]
    assert comparison["K_disagreement_count"] == 1
    assert comparison["delta_right_minus_left"]["mean"] == 1.0
    assert {row["actual_origin"] for row in task_rows} == {"ACTUAL_WARM", "COLD"}

