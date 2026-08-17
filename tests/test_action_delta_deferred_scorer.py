import inspect

import pytest
import torch

import prismatic.models.action_delta_deferred_scorer as deferred_scorer
from prismatic.models.action_delta_gate import (
    ACTION_DELTA_GATE_ARTIFACT_TYPE,
    ACTION_DELTA_GATE_CALIBRATION_METHOD,
    ACTION_DELTA_GATE_MODEL_TYPE,
    ACTION_DELTA_NONCONVERGENCE_THRESHOLD,
    NonFiniteActionDeltaGateError,
    PreparedActionDeltaGate,
    score_action_delta_gate,
)


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


def test_pure_compiled_candidate_math_matches_eager_and_preserves_bf16_order():
    gate = make_gate()
    anchor, current = inputs()

    score = deferred_scorer.action_delta_deferred_tensor_score(
        anchor,
        current,
        gate.x_mean,
        gate.x_std,
        gate.linear_weight,
        gate.linear_bias,
        gate.y_std,
        gate.y_mean,
    )
    eager = score_action_delta_gate(gate, anchor, current)
    quantized = (current.float() - anchor.float()).to(torch.bfloat16).float()
    normalized = (quantized - gate.x_mean) / gate.x_std
    predicted = (
        torch.nn.functional.linear(
            normalized, gate.linear_weight, gate.linear_bias
        )
        * gate.y_std
        + gate.y_mean
    )

    assert torch.equal(score, eager)
    assert torch.equal(score, predicted.square().mean())
    source = inspect.getsource(deferred_scorer)
    assert "set_float32_matmul_precision" not in source


def test_compile_default_prepares_once_with_only_accepted_compile_options(monkeypatch):
    compile_calls = []
    scorer_calls = 0

    def fake_compile(function, **options):
        compile_calls.append((function, options))

        def compiled(*args):
            nonlocal scorer_calls
            scorer_calls += 1
            return function(*args)

        return compiled

    monkeypatch.setattr(deferred_scorer.torch, "compile", fake_compile)
    gate = make_gate()
    prepared = deferred_scorer.prepare_action_delta_deferred_scorer(
        gate, "compile_default"
    )

    assert len(compile_calls) == 1
    assert compile_calls[0][1] == {"fullgraph": True, "dynamic": False}
    assert scorer_calls == 1
    assert prepared.backend == "compile_default"
    assert prepared.compile_setup_ms >= 0.0
    assert prepared.development_decision_parity_count == 7139
    assert "not_bitwise_equal" in prepared.numerical_equivalence


def test_compiled_backend_returns_scalar_and_uses_fixed_high_side_rule():
    gate = make_gate()
    anchor, current = inputs()
    calls = 0

    def fixed_score(*_args):
        nonlocal calls
        calls += 1
        return torch.tensor(ACTION_DELTA_NONCONVERGENCE_THRESHOLD)

    prepared = deferred_scorer.PreparedActionDeltaDeferredScorer(
        backend="compile_default",
        tensor_scorer=fixed_score,
        compile_setup_ms=123.0,
        compile_fullgraph=True,
        compile_dynamic=False,
        numerical_equivalence="not_bitwise_equal",
        development_decision_parity_count=7139,
        development_decision_parity_total=7139,
    )

    score, high = deferred_scorer.evaluate_action_delta_deferred_scorer(
        gate,
        anchor,
        current,
        backend="compile_default",
        prepared_scorer=prepared,
    )

    assert isinstance(score, float)
    assert score == pytest.approx(ACTION_DELTA_NONCONVERGENCE_THRESHOLD)
    assert high is True
    assert calls == 1
    assert ACTION_DELTA_NONCONVERGENCE_THRESHOLD == 0.0015


def test_compiled_nonfinite_score_fails_closed():
    gate = make_gate()
    anchor, current = inputs()
    prepared = deferred_scorer.PreparedActionDeltaDeferredScorer(
        backend="compile_default",
        tensor_scorer=lambda *_args: torch.tensor(float("nan")),
        compile_setup_ms=0.0,
        compile_fullgraph=True,
        compile_dynamic=False,
        numerical_equivalence="not_bitwise_equal",
        development_decision_parity_count=7139,
        development_decision_parity_total=7139,
    )

    with pytest.raises(NonFiniteActionDeltaGateError, match="non-finite"):
        deferred_scorer.evaluate_action_delta_deferred_scorer(
            gate,
            anchor,
            current,
            backend="compile_default",
            prepared_scorer=prepared,
        )


def test_compile_backend_is_deferred_only_and_eager_requires_no_preparation():
    assert (
        deferred_scorer.prepare_action_delta_deferred_scorer(make_gate(), "eager")
        is None
    )
    deferred_scorer.validate_action_delta_deferred_scorer_configuration(
        deferred_filter_enabled=False,
        backend="eager",
        prepared_scorer=None,
    )
    with pytest.raises(ValueError, match="deferred/backfill-only"):
        deferred_scorer.validate_action_delta_deferred_scorer_configuration(
            deferred_filter_enabled=False,
            backend="compile_default",
            prepared_scorer=None,
        )
