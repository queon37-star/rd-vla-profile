import inspect

import torch

import prismatic.models.action_delta_gate as production_scorer
from prismatic.models.action_delta_gate import (
    ACTION_DELTA_GATE_ARTIFACT_TYPE,
    ACTION_DELTA_GATE_CALIBRATION_METHOD,
    ACTION_DELTA_GATE_MODEL_TYPE,
    PreparedActionDeltaGate,
)
from scripts.coda_anchor_feasibility import profile_action_delta_scorer as profiler


def make_gate():
    return PreparedActionDeltaGate(
        schema_version=1,
        artifact_type=ACTION_DELTA_GATE_ARTIFACT_TYPE,
        model_type=ACTION_DELTA_GATE_MODEL_TYPE,
        hidden_dim=4,
        action_dim=2,
        action_chunk_len=2,
        held_out_task_ids=(4, 5),
        outer_fold=4,
        threshold=0.0007,
        x_mean=torch.tensor([0.25, -0.5, 1.0, 0.125]),
        x_std=torch.tensor([0.5, 2.0, 0.25, 4.0]),
        y_mean=torch.tensor([0.01, -0.02]),
        y_std=torch.tensor([0.5, 2.0]),
        linear_weight=torch.tensor(
            [[0.1, -0.2, 0.3, 0.4], [-0.5, 0.25, 0.125, -0.1]]
        ),
        linear_bias=torch.tensor([0.03, -0.04]),
        delta_quantization_dtype="bfloat16",
        training_seed=1011,
        calibration_method=ACTION_DELTA_GATE_CALIBRATION_METHOD,
    )


def inputs():
    anchor = torch.tensor(
        [[[0.11, -0.23, 0.37, 0.41], [0.53, -0.67, 0.79, -0.83]]],
        dtype=torch.bfloat16,
    )
    current = torch.tensor(
        [[[0.19, -0.17, 0.49, 0.28], [0.61, -0.52, 0.71, -0.69]]],
        dtype=torch.bfloat16,
    )
    return anchor, current


def test_decomposition_reproduces_production_score_and_high_side_decision():
    gate = make_gate()
    anchor, current = inputs()

    decomposed = profiler.decomposed_action_delta_score(gate, anchor, current)
    production = production_scorer.score_action_delta_gate(
        gate, anchor, current
    )
    evaluated, _low_side_decision = production_scorer.evaluate_action_delta_gate(
        gate, anchor, current
    )

    assert float(decomposed.item()) == float(production.item()) == evaluated
    assert (float(decomposed.item()) >= profiler.HIGH_SIDE_THRESHOLD) == (
        evaluated >= profiler.HIGH_SIDE_THRESHOLD
    )


def test_pure_tensor_scorer_reproduces_production_score():
    gate = make_gate()
    anchor, current = inputs()

    pure = profiler.call_tensor_scorer(
        profiler.tensor_action_delta_score, gate, anchor, current
    )
    production = production_scorer.score_action_delta_gate(gate, anchor, current)

    assert torch.equal(pure, production)
    source = inspect.getsource(profiler.tensor_action_delta_score)
    assert "current_state.float() - anchor_state.float()" in source
    assert "to(torch.bfloat16).float()" in source


def test_decomposition_preserves_exact_bf16_quantization_order():
    gate = make_gate()
    anchor, current = inputs()

    _, intermediates = profiler.decomposed_action_delta_score(
        gate,
        anchor,
        current,
        return_intermediates=True,
    )
    expected = (current.float() - anchor.float()).to(torch.bfloat16).float()

    torch.testing.assert_close(
        intermediates["delta_bfloat16_restored_fp32"],
        expected,
        rtol=0,
        atol=0,
    )
    assert intermediates["delta_bfloat16"].dtype == torch.bfloat16


def test_profiler_math_does_not_mutate_inputs_or_gate():
    gate = make_gate()
    anchor, current = inputs()
    anchor_before = anchor.clone()
    current_before = current.clone()
    weight_before = gate.linear_weight.clone()

    profiler.decomposed_action_delta_score(gate, anchor, current)

    assert torch.equal(anchor, anchor_before)
    assert torch.equal(current, current_before)
    assert torch.equal(gate.linear_weight, weight_before)


def test_profiler_is_diagnostic_and_uses_unmodified_production_functions():
    assert profiler.DIAGNOSTIC_ONLY is True
    assert profiler.score_action_delta_gate is production_scorer.score_action_delta_gate
    assert (
        profiler.evaluate_action_delta_gate
        is production_scorer.evaluate_action_delta_gate
    )
    assert profiler.HIGH_SIDE_THRESHOLD == 0.0015
    assert "action_heads" not in profiler.__dict__
    assert profiler.tensor_action_delta_score.__module__ == profiler.__name__


def test_compiled_eager_high_side_decision_comparison_reports_exact_counts():
    identities = [{"row": index} for index in range(4)]
    eager = torch.tensor([0.0014, 0.0015, 0.0016, 0.0020])
    same_decisions = torch.tensor([0.00145, 0.00151, 0.00155, 0.0021])

    result = profiler.compare_score_vectors(eager, same_decisions, identities)

    assert result["transition_count"] == 4
    assert result["exact_score_match_count"] == 0
    assert result["scores_bitwise_identical"] is False
    assert result["high_side_decision_match_count"] == 4
    assert result["high_side_decision_mismatch_count"] == 0
    assert result["largest_score_difference_transition"]["identity"] == {"row": 3}


def test_compiled_scalar_snapshots_remain_independent_across_reused_outputs():
    reused_output = torch.empty((), dtype=torch.float32)
    snapshots = []

    for value in (1.0, 2.0, 3.0):
        reused_output.fill_(value)
        snapshots.append(profiler.snapshot_compiled_scalar(reused_output))

    reused_output.fill_(99.0)
    assert [float(snapshot.item()) for snapshot in snapshots] == [1.0, 2.0, 3.0]
    assert all(snapshot.device.type == "cpu" for snapshot in snapshots)


def test_parity_snapshot_is_not_added_to_runtime_like_timing():
    parity_source = inspect.getsource(profiler.evaluate_candidate_parity)
    timing_source = inspect.getsource(profiler.benchmark_tensor_runtime_wall)
    module_source = inspect.getsource(profiler)

    assert "snapshot_compiled_scalar(compiled_score)" in parity_source
    assert "snapshot_compiled_scalar" not in timing_source
    assert "set_float32_matmul_precision" not in module_source


def test_candidate_acceptance_requires_full_parity_and_twenty_percent_speedup():
    parity = {
        "transition_count": profiler.EXPECTED_DEV8_TRANSITIONS,
        "high_side_decision_match_count": profiler.EXPECTED_DEV8_TRANSITIONS,
        "high_side_decision_mismatch_count": 0,
        "input_and_artifact_tensor_mutation_free": True,
    }

    strong = profiler.classify_compiled_candidate(
        parity, candidate_wall_ms=0.7, eager_tensor_wall_ms=1.0
    )
    too_slow = profiler.classify_compiled_candidate(
        parity, candidate_wall_ms=0.81, eager_tensor_wall_ms=1.0
    )

    assert strong["classification"] == "ACCEPT_FOR_RUNTIME_TRIAL_STRONG_CANDIDATE"
    assert too_slow["classification"] == "REJECT_FOR_RUNTIME_TRIAL"


def test_decision_mismatch_prevents_runtime_trial_acceptance():
    parity = {
        "transition_count": profiler.EXPECTED_DEV8_TRANSITIONS,
        "high_side_decision_match_count": profiler.EXPECTED_DEV8_TRANSITIONS - 1,
        "high_side_decision_mismatch_count": 1,
        "input_and_artifact_tensor_mutation_free": True,
    }

    result = profiler.classify_compiled_candidate(
        parity, candidate_wall_ms=0.5, eager_tensor_wall_ms=1.0
    )

    assert result["classification"] == "REJECT_FOR_RUNTIME_TRIAL"
    assert result["decision_parity_7139_of_7139"] is False


def test_latency_statistics_reports_required_quantiles():
    summary = profiler.latency_statistics([1.0, 2.0, 3.0, 4.0], warmup_count=100)

    assert summary["count"] == 4
    assert summary["warmup_count"] == 100
    assert summary["mean_ms"] == 2.5
    assert summary["median_ms"] == 2.5
    assert set(summary) == {
        "count",
        "warmup_count",
        "mean_ms",
        "median_ms",
        "p90_ms",
        "p95_ms",
        "p99_ms",
        "std_ms",
        "min_ms",
        "max_ms",
    }
