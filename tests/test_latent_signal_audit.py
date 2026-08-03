from __future__ import annotations

import pytest
import torch

from scripts.action_latent_audit_lib import (
    build_latent_rows,
    build_latent_signal_audit,
    build_trajectory_record,
    exact_rank_auc,
)
from scripts.preconvergence_trigger_lib import RawPreconvergenceSequence, SequenceIdentity


def make_record(
    prediction_id: int,
    *,
    task_id: int,
    origin: str = "ACTUAL_WARM",
    max_iter: int = 6,
    state_scale: float = 1.0,
) -> dict:
    increments = torch.arange(1, max_iter + 1, dtype=torch.float32) * state_scale
    states = torch.cumsum(increments, dim=0).reshape(max_iter, 1, 1).repeat(1, 2, 2)
    actions = torch.tensor(
        [[1.0 / (k + 1), 0.0] for k in range(max_iter)], dtype=torch.float32
    )
    mse = [None, None] + [float(torch.mean((actions[k - 1] - actions[k - 2]) ** 2)) for k in range(2, max_iter + 1)]
    phases = [None, None] + ["production"] * (max_iter - 1)
    sequence = RawPreconvergenceSequence(
        identity=SequenceIdentity(task_id, 0, prediction_id),
        actual_origin=origin,
        states=states,
        actions=actions,
        action_mse=tuple(mse),
        action_mse_phase=tuple(phases),
        baseline_k=max_iter,
        max_iter=max_iter,
    )
    return build_trajectory_record(sequence)


def test_exact_rank_auc_supports_ties_and_missing_class() -> None:
    assert exact_rank_auc([0, 1, 0, 1], [1.0, 1.0, 2.0, 3.0]) == pytest.approx(0.625)
    assert exact_rank_auc([1, 1], [1.0, 2.0]) is None


def test_latent_rows_use_only_current_and_previous_state() -> None:
    original = make_record(0, task_id=0)
    rows, _ = build_latent_rows([original], thresholds=[0.001])
    modified = {**original, "state_mean": original["state_mean"].clone(), "latent_metrics": {name: values.clone() for name, values in original["latent_metrics"].items()}}
    modified["state_mean"][4:] += 10000
    # Rebuild scalar metrics from a source sequence to ensure only future values differ.
    future_source = make_record(0, task_id=0)
    future_source["state_mean"] = modified["state_mean"]
    future_source["latent_metrics"] = modified["latent_metrics"]
    changed_rows, _ = build_latent_rows([future_source], thresholds=[0.001])
    assert rows[0]["k"] == 3
    assert rows[0]["features"] == changed_rows[0]["features"]


def test_second_difference_cosine_and_target_coverage() -> None:
    record = make_record(0, task_id=0)
    rows, coverage = build_latent_rows([record], thresholds=[0.1])
    first = rows[0]
    assert first["features"]["delta_cosine"] == pytest.approx(1.0)
    assert first["features"]["second_difference_rms"] == pytest.approx(1.0)
    assert coverage["K_first@0.1"]["prediction_with_label"] == 1
    assert first["targets"]["log10_next_recomputed_action_mse"] is not None
    assert rows[-1]["targets"]["log10_next_recomputed_action_mse"] is None


def test_task_macro_is_unweighted_and_differs_from_row_micro() -> None:
    records = [
        make_record(0, task_id=0, max_iter=8, state_scale=1.0),
        make_record(1, task_id=1, max_iter=4, state_scale=-1.0),
    ]
    report, csv_rows = build_latent_signal_audit(records, thresholds=[0.001])
    association = report["associations"]["log10_current_recomputed_action_mse"]["iteration_k"]
    assert association["global_row_micro"]["sample_count"] == 8
    assert association["task_macro"]["spearman"] == pytest.approx(
        sum(item["spearman"] for item in association["per_task"].values()) / 2
    )
    assert association["task_macro"]["pearson"] != pytest.approx(
        association["global_row_micro"]["pearson"]
    )
    assert len([row for row in csv_rows if row["scope"] == "per_task"]) > 0


def test_cold_rows_are_excluded_from_descriptive_audit() -> None:
    warm = make_record(0, task_id=0)
    cold = make_record(1, task_id=1, origin="COLD")
    report, _ = build_latent_signal_audit([warm, cold], thresholds=[0.001])
    assert report["prediction_count"] == 1
    assert report["row_count"] == warm["max_iter"] - 2
    assert report["not_oof"] is True
