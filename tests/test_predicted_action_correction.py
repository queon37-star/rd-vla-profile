import numpy as np
import pytest
import torch

from prismatic.models.action_delta_gate import (
    build_action_delta_gate_corrected_output,
)
from scripts.coda_anchor_feasibility.evaluate_predicted_action_correction import (
    compute_error_metrics,
    replay_first_hits,
    summarize_population,
)


def test_runtime_dtype_correction_preserves_float32_correction_benefit():
    anchor = torch.tensor(
        [[[0.25, -0.5], [0.75, -1.0]]],
        dtype=torch.bfloat16,
    )
    target_delta = torch.tensor(
        [[[0.125, -0.25], [0.5, -0.125]]],
        dtype=torch.bfloat16,
    ).float()
    pred_delta = torch.tensor(
        [[[0.12, -0.24], [0.48, -0.12]]],
        dtype=torch.float32,
    )
    exact_terminal = anchor.float() + target_delta
    float32_corrected = anchor.float() + pred_delta
    runtime_corrected = build_action_delta_gate_corrected_output(
        anchor,
        pred_delta,
    )

    assert runtime_corrected.shape == anchor.shape
    assert runtime_corrected.device == anchor.device
    assert runtime_corrected.dtype == anchor.dtype
    reuse_mse = target_delta.square().mean()
    float32_mse = (exact_terminal - float32_corrected).square().mean()
    runtime_mse = (exact_terminal - runtime_corrected.float()).square().mean()
    benefit_retention = (reuse_mse - runtime_mse) / (reuse_mse - float32_mse)

    assert float32_mse < reuse_mse
    assert runtime_mse < reuse_mse
    assert benefit_retention >= 0.95


def test_error_metrics_match_direct_action_chunk_reductions():
    error = torch.tensor(
        [
            [
                [1.0, -2.0],
                [3.0, -4.0],
                [5.0, -6.0],
                [7.0, -8.0],
                [9.0, -10.0],
                [11.0, -12.0],
            ],
            [
                [0.5, -0.25],
                [0.0, 0.0],
                [-1.0, 2.0],
                [0.25, -0.5],
                [1.5, 1.0],
                [-2.0, 0.5],
            ],
        ],
        dtype=torch.float32,
    )

    observed = compute_error_metrics(error, prefix_steps=5)
    squared = error.square()
    absolute = error.abs()
    expected = {
        "full_mse": squared.mean(dim=(1, 2)),
        "prefix5_mse": squared[:, :5].mean(dim=(1, 2)),
        "full_max_abs": absolute.amax(dim=(1, 2)),
        "prefix5_max_abs": absolute[:, :5].amax(dim=(1, 2)),
        "max_per_step_mse": squared.mean(dim=2).amax(dim=1),
        "max_per_dim_mse": squared.mean(dim=1).amax(dim=1),
    }

    assert set(observed) == set(expected)
    for metric_name, expected_values in expected.items():
        np.testing.assert_array_equal(observed[metric_name], expected_values.numpy())


def test_replay_counts_only_scores_before_each_sequential_first_hit():
    # Deliberately interleave trajectories and leave their rows out of k order.
    trajectory_ids = np.array([10, 20, 30, 10, 30, 20, 10, 30, 20])
    ks = np.array([5, 5, 5, 1, 2, 4, 4, 4, 6])
    scores = np.array([0.01, 0.01, 0.30, 0.01, 0.01, 0.01, 0.20, 0.20, 0.00])
    target_safe = np.array([True, True, True, True, True, False, True, True, True])

    replay, activated, correct_safe, false_early = replay_first_hits(
        scores=scores,
        target_safe=target_safe,
        trajectory_ids=trajectory_ids,
        ks=ks,
        threshold=0.05,
        min_gate_k=4,
    )

    assert replay == {
        "runtime_min_terminal_iteration": 5,
        "offline_min_gate_k": 4,
        "trajectory_count": 3,
        "score_call_count": 5,
        "activated": 2,
        "correct_safe": 1,
        "false_early": 1,
        "no_skip": 1,
    }
    np.testing.assert_array_equal(activated, np.array([0, 5]))
    np.testing.assert_array_equal(correct_safe, np.array([0]))
    np.testing.assert_array_equal(false_early, np.array([5]))


def test_population_summary_reports_correction_reduction_ratios():
    reuse = {
        metric: np.array([4.0, 4.0, 4.0], dtype=np.float32)
        for metric in (
            "full_mse",
            "prefix5_mse",
            "full_max_abs",
            "prefix5_max_abs",
            "max_per_step_mse",
            "max_per_dim_mse",
        )
    }
    correction = {
        metric: np.array([1.0, 2.0, 8.0], dtype=np.float32)
        for metric in reuse
    }

    summary = summarize_population(
        "synthetic", np.arange(3), reuse, correction
    )

    comparison = summary["full_mse_comparison"]
    assert summary["N"] == 3
    assert comparison["mean_reduction_ratio"] == (0.25 + 0.5 + 2.0) / 3
    assert comparison["median_reduction_ratio"] == 0.5
    assert comparison["correction_better_percent"] == pytest.approx(200.0 / 3)
    assert comparison["correction_worse_percent"] == pytest.approx(100.0 / 3)
    assert comparison[
        "correction_reduces_mse_at_least_50_percent"
    ] == pytest.approx(200.0 / 3)
    assert comparison["worst_correction_degradation_ratio"] == 2.0
