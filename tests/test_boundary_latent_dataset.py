from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.boundary_latent_oof_lib import (
    BoundaryLatentOOFError,
    build_boundary_dataset_payload,
    load_boundary_dataset,
    save_boundary_dataset,
)
from test_boundary_reference_audit import make_record


def record_with_k(prediction_id: int, task_id: int, k: int, *, origin: str = "ACTUAL_WARM") -> dict:
    max_iter = 7
    high = 0.01
    low = 0.0001
    authoritative = [None, None] + [high if iteration < k else low for iteration in range(2, max_iter + 1)]
    recomputed = list(authoritative)
    return make_record(
        prediction_id,
        task_id=task_id,
        origin=origin,
        authoritative=authoritative,
        recomputed=recomputed,
    )


def test_boundary_excludes_k_at_or_after_reference_and_k4_is_positive_only() -> None:
    payload, coverage = build_boundary_dataset_payload([record_with_k(0, 0, 4)])
    for target in payload["target_references"]:
        rows = [row for row in payload["rows"] if row["target_reference"] == target]
        assert [(row["k"], row["label"], row["boundary_offset"]) for row in rows] == [(3, 1, 0)]
        assert coverage["targets"][target]["positive_only_prediction_count"] == 1


def test_history_unavailable_and_cold_are_explicitly_excluded() -> None:
    payload, coverage = build_boundary_dataset_payload(
        [record_with_k(0, 0, 3), record_with_k(1, 1, 5, origin="COLD")]
    )
    assert payload["rows"] == []
    for target in payload["target_references"]:
        assert coverage["targets"][target]["history_unavailable_prediction_count"] == 1
        assert coverage["targets"][target]["history_unavailable_K_3_count"] == 1
    assert coverage["cold_excluded_from_oof_prediction_count"] == 1


def test_features_use_no_future_state_and_prediction_weights_sum_to_one() -> None:
    first = record_with_k(0, 0, 6)
    second = {**first, "state_mean": first["state_mean"].clone(), "latent_metrics": {name: values.clone() for name, values in first["latent_metrics"].items()}}
    second["state_mean"][5:] += 10000
    before, _ = build_boundary_dataset_payload([first])
    after, _ = build_boundary_dataset_payload([second])
    before_k4 = next(row for row in before["rows"] if row["k"] == 4)
    after_k4 = next(row for row in after["rows"] if row["k"] == 4)
    assert torch.equal(before_k4["scalar_features"], after_k4["scalar_features"])
    assert torch.equal(before_k4["current_mean_pooled_delta"], after_k4["current_mean_pooled_delta"])
    grouped = {}
    for row in before["rows"]:
        key = (row["target_reference"], row["prediction_id"])
        grouped.setdefault(key, 0.0)
        grouped[key] += row["weight"]
    assert all(value == pytest.approx(1.0) for value in grouped.values())


def test_nonfinite_features_fail_closed() -> None:
    record = record_with_k(0, 0, 5)
    record["state_mean"][3, 0] = float("inf")
    with pytest.raises(BoundaryLatentOOFError, match="non-finite"):
        build_boundary_dataset_payload([record])


def test_manifest_hash_load_source_immutability_and_overwrite_refusal(tmp_path: Path) -> None:
    record = record_with_k(0, 0, 5)
    state_before = record["state_mean"].clone()
    action_before = record["action_flat"].clone()
    payload, coverage = build_boundary_dataset_payload([record])
    output = tmp_path / "dataset"
    manifest = save_boundary_dataset(
        output,
        payload,
        coverage,
        source_bundle_path=tmp_path / "trajectory_bundle.pt",
        source_bundle_sha256="source-hash",
        fold_manifest_sha256="fold-hash",
        git_commit="commit",
    )
    loaded_manifest, loaded = load_boundary_dataset(output)
    assert loaded_manifest["boundary_dataset_sha256"] == manifest["boundary_dataset_sha256"]
    assert loaded["boundary_dataset_sha256"] == manifest["boundary_dataset_sha256"]
    assert torch.equal(record["state_mean"], state_before)
    assert torch.equal(record["action_flat"], action_before)
    with pytest.raises(BoundaryLatentOOFError, match="refusing to overwrite"):
        save_boundary_dataset(
            output,
            payload,
            coverage,
            source_bundle_path=tmp_path / "trajectory_bundle.pt",
            source_bundle_sha256="source-hash",
            fold_manifest_sha256="fold-hash",
            git_commit="commit",
        )
