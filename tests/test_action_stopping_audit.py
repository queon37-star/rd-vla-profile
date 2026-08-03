from __future__ import annotations

import pytest
import torch

from scripts.action_latent_audit_lib import (
    action_prediction_audit,
    aggregate_action_predictions,
    build_action_stopping_audit,
    build_trajectory_record,
    consecutive_hit,
    first_hit,
    rebound_diagnostics,
    stable_suffix_hit,
    tail_action_diagnostics,
)
from scripts.preconvergence_trigger_lib import RawPreconvergenceSequence, SequenceIdentity


def make_record(
    prediction_id: int = 0,
    *,
    task_id: int = 0,
    origin: str = "ACTUAL_WARM",
    mse: list[float | None] | None = None,
    actions: torch.Tensor | None = None,
) -> dict:
    actions = actions if actions is not None else torch.tensor(
        [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.21, 0.0], [0.22, 0.0], [0.23, 0.0]],
        dtype=torch.float32,
    )
    max_iter = len(actions)
    states = torch.arange(max_iter * 4, dtype=torch.float32).reshape(max_iter, 2, 2)
    mse = mse or [None, None, 0.0004, 0.002, 0.0003, 0.0002, 0.0001]
    phases = [None, None] + ["production"] * 3 + ["shadow_tail"] * (max_iter - 4)
    sequence = RawPreconvergenceSequence(
        identity=SequenceIdentity(task_id, 0, prediction_id),
        actual_origin=origin,
        states=states,
        actions=actions,
        action_mse=tuple(mse),
        action_mse_phase=tuple(phases),
        baseline_k=4,
        max_iter=max_iter,
    )
    return build_trajectory_record(sequence)


def test_first_consecutive_rebound_stable_suffix_and_no_hit() -> None:
    values = [None, None, 0.0004, 0.002, 0.0003, 0.0002, 0.0001]
    assert first_hit(values, 0.001) == 2
    assert consecutive_hit(values, 0.001, 2) == 5
    assert consecutive_hit(values, 0.001, 3) == 6
    assert stable_suffix_hit(values, 0.001) == 4
    rebound = rebound_diagnostics(values, 0.001, 2)
    assert rebound == {
        "exists": True,
        "count": 1,
        "maximum_post_hit_mse": 0.002,
        "maximum_post_hit_mse_over_threshold": 2.0,
        "first_rebound_iteration": 3,
    }
    assert first_hit(values, 0.00001) is None
    assert stable_suffix_hit(values, 0.00001) is None


def test_tail_mean_and_interpolated_median_diagnostics() -> None:
    record = make_record(actions=torch.tensor([[0.0], [1.0], [2.0], [10.0], [20.0], [30.0]]))
    result = tail_action_diagnostics(record, 4, 100.0)
    # Last four values have mean 15.5 and interpolated median 15.0.
    assert result["mean"]["center_action"] == [15.5]
    assert result["median"]["center_action"] == [15.0]
    assert result["mean"]["mse_to_center"][4] == pytest.approx((10.0 - 15.5) ** 2)
    assert result["median"]["mse_to_center"][4] == pytest.approx((10.0 - 15.0) ** 2)
    assert result["offline_only_future_action_diagnostic"] is True


def test_stored_recomputed_disagreement_is_reported_by_phase() -> None:
    record = make_record()
    prediction = action_prediction_audit(record, thresholds=[0.001], tail_windows=[4])
    aggregate = aggregate_action_predictions([prediction], [0.001])
    phase = aggregate["thresholds"]["0.001"]["stored_vs_recomputed_by_phase"]
    assert phase["production"]["threshold_hit_disagreement_count"] > 0
    assert phase["production"]["K_disagreement_count"] == 1
    assert phase["shadow_tail"]["transition_count"] == 2


def test_dimension_and_timestep_masking_detect_hidden_large_element() -> None:
    actions = torch.zeros(4, 20, 20)
    actions[1:, 0, 0] = 0.5
    mse = [None, None, 0.000625, 0.0005, 0.0005]
    record = make_record(actions=actions, mse=mse)
    prediction = action_prediction_audit(record, thresholds=[0.001], tail_windows=[4])
    transition = prediction["thresholds"]["0.001"]["transitions"][0]
    assert transition["recomputed_mse"] == pytest.approx(0.000625)
    assert transition["dimension_masking"]["max_dimension_ge_10x"] is True
    assert transition["dimension_masking"]["max_timestep_ge_10x"] is True


def test_actual_warm_and_cold_are_never_mixed_in_primary_rates() -> None:
    warm = make_record(0, origin="ACTUAL_WARM")
    cold = make_record(1, origin="COLD", mse=[None, None, 0.01, 0.01, 0.01, 0.01, 0.01])
    report, predictions, task_rows = build_action_stopping_audit([warm, cold], thresholds=[0.001])
    assert len(predictions) == 2
    assert report["primary_actual_warm"]["prediction_count"] == 1
    assert report["cold_reported_separately"]["prediction_count"] == 1
    assert report["primary_actual_warm"]["thresholds"]["0.001"]["candidate_coverage"]["K_first"]["hit_prediction_count"] == 1
    assert report["cold_reported_separately"]["thresholds"]["0.001"]["candidate_coverage"]["K_first"]["no_hit_prediction_count"] == 1
    assert {row["actual_origin"] for row in task_rows} == {"ACTUAL_WARM", "COLD"}
