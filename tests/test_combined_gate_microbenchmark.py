import inspect
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.combined_gate_microbenchmark_lib import (
    COMBINED_FEATURE_NAMES,
    EXPECTED_MODEL_SOURCE_COMMIT,
    EXPECTED_TRACE_IDENTITY,
    CombinedGateMicrobenchmarkError,
    SerializedCombinedModel,
    StateCase,
    assert_case_parity,
    combined_current_diagnostic,
    combined_optimized_decision,
    combined_optimized_tensor,
    deterministic_operation_order,
    gate_evaluations_per_prediction,
    load_serialized_combined_models,
    load_actual_warm_workload_descriptors,
    model_to_device,
    optimized_correctness_values,
    optimized_host_transfer_audit,
    project_policy_latency,
    raw_mse_decision,
    raw_mse_tensor,
    select_stratified_workloads,
    sha256_file,
    summarize_latency_samples,
)


def _expanded_names():
    history = {
        "contraction_ratio",
        "update_turning_cosine",
        "acceleration_rms",
        "acceleration_ratio",
    }
    values = []
    for name in COMBINED_FEATURE_NAMES:
        values.append(name)
        if name in history:
            values.append(f"{name}__available")
    return tuple(values)


def _model(fold=0, tasks=(0, 9), threshold=0.5):
    return SerializedCombinedModel(
        outer_fold=fold,
        held_out_task_ids=tuple(tasks),
        feature_names=COMBINED_FEATURE_NAMES,
        expanded_feature_names=_expanded_names(),
        imputation_medians=tuple(0.1 + index * 0.01 for index in range(18)),
        scaling_mean=tuple(0.01 * index for index in range(22)),
        scaling_scale=tuple(1.0 + 0.01 * index for index in range(22)),
        weights=tuple(-0.2 + 0.02 * index for index in range(22)),
        bias=-0.03,
        threshold=threshold,
        threshold_hex=float(threshold).hex(),
    )


def _case(previous_update=True):
    torch.manual_seed(4)
    previous = torch.randn(1, 4, 3, dtype=torch.float32)
    update = torch.randn_like(previous) * 0.1
    current = previous + update
    prior = torch.randn_like(previous) * 0.1 if previous_update else None
    anchor = previous - 0.05
    return StateCase(
        "k_ge_3" if previous_update else "k2",
        3 if previous_update else 2,
        current,
        previous,
        prior,
        anchor,
    )


def test_raw_mse_mathematical_parity():
    current = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
    previous = torch.tensor([[0.0, 1.0], [2.0, 2.0]])
    expected = torch.mean((current.float() - previous.float()).square())
    actual = raw_mse_tensor(current, previous)
    assert torch.equal(actual, expected)
    assert raw_mse_decision(current, previous, float(expected)) is True


@pytest.mark.parametrize("history", [False, True])
def test_combined_feature_preprocessing_logit_probability_and_decision_parity(history):
    case = _case(history)
    serialized = _model(threshold=0.45)
    tensors = model_to_device(serialized, "cpu")
    result = assert_case_parity(case, serialized, tensors)
    reference = combined_current_diagnostic(
        case.current_state,
        case.previous_state,
        case.previous_update,
        case.warm_anchor,
        serialized,
    )
    optimized = optimized_correctness_values(
        case.current_state,
        case.previous_state,
        case.previous_update,
        case.warm_anchor,
        tensors,
    )
    assert result["parity_passed"] is True
    assert np.allclose(
        reference["normalized_features"], optimized["normalized_features"], rtol=1e-5, atol=1e-6
    )
    assert math.isclose(reference["logit"], optimized["logit"], rel_tol=1e-5, abs_tol=1e-6)
    assert math.isclose(
        reference["probability"], optimized["probability"], rel_tol=1e-5, abs_tol=1e-6
    )
    assert combined_optimized_decision(
        case.current_state,
        case.previous_state,
        case.previous_update,
        case.warm_anchor,
        tensors,
    ) == reference["decision"]


def test_k2_null_history_has_explicit_zero_availability_indicators():
    case = _case(False)
    serialized = _model()
    reference = combined_current_diagnostic(
        case.current_state,
        case.previous_state,
        None,
        case.warm_anchor,
        serialized,
    )
    assert reference["features"][4:8] == [None, None, None, None]
    for name in (
        "contraction_ratio__available",
        "update_turning_cosine__available",
        "acceleration_rms__available",
        "acceleration_ratio__available",
    ):
        assert reference["expanded_features"][_expanded_names().index(name)] == 0.0


def test_k_ge_3_history_fields_are_available_and_finite():
    case = _case(True)
    reference = combined_current_diagnostic(
        case.current_state,
        case.previous_state,
        case.previous_update,
        case.warm_anchor,
        _model(),
    )
    assert all(math.isfinite(value) for value in reference["features"][4:8])
    for name in (
        "contraction_ratio__available",
        "update_turning_cosine__available",
        "acceleration_rms__available",
        "acceleration_ratio__available",
    ):
        assert reference["expanded_features"][_expanded_names().index(name)] == 1.0


def test_optimized_tensor_has_no_item_and_decision_has_exactly_one_host_transfer():
    audit = optimized_host_transfer_audit()
    assert audit == {
        "optimized_tensor_item_calls": 0,
        "optimized_decision_item_calls": 1,
    }
    source = inspect.getsource(combined_optimized_tensor)
    assert ".item(" not in source


def test_input_tensors_remain_bitwise_unchanged():
    case = _case(True)
    originals = [
        case.current_state.clone(),
        case.previous_state.clone(),
        case.previous_update.clone(),
        case.warm_anchor.clone(),
    ]
    combined_optimized_tensor(
        case.current_state,
        case.previous_state,
        case.previous_update,
        case.warm_anchor,
        model_to_device(_model(), "cpu"),
    )
    assert all(
        torch.equal(original, actual)
        for original, actual in zip(
            originals,
            (
                case.current_state,
                case.previous_state,
                case.previous_update,
                case.warm_anchor,
            ),
        )
    )


def test_workload_selection_preserves_unique_warm_identity_and_task_stratification():
    descriptors = [
        {
            "workload_id": f"{task}:{index}",
            "identity": {"task_id": task},
            "actual_origin": "ACTUAL_WARM",
        }
        for task in range(10)
        for index in range(2)
    ]
    selected = select_stratified_workloads(descriptors, 10)
    assert [item["identity"]["task_id"] for item in selected] == list(range(10))
    assert len({item["workload_id"] for item in selected}) == 10
    with pytest.raises(CombinedGateMicrobenchmarkError, match="only 20 are available"):
        select_stratified_workloads(descriptors, 21)


def test_frozen_manifest_exposes_only_100_distinct_actual_warm_shards():
    manifest = Path(
        "benchmark_results/learned_convergence_probe/20260801_seed7/dataset/manifest.json"
    )
    descriptors, provenance = load_actual_warm_workload_descriptors(manifest)
    assert len(descriptors) == 100
    assert provenance["counts_by_task"] == {str(task): 10 for task in range(10)}
    assert all(item["actual_origin"] == "ACTUAL_WARM" for item in descriptors)
    assert all(item["path"].is_file() for item in descriptors)
    with pytest.raises(
        CombinedGateMicrobenchmarkError,
        match="requested 200 distinct ACTUAL_WARM workloads, but only 100 are available",
    ):
        select_stratified_workloads(descriptors, 200)


def _write_artifacts(path: Path):
    folds = []
    pairs = ((0, 9), (1, 8), (2, 7), (3, 6), (4, 5))
    for fold_id, tasks in enumerate(pairs):
        model = _model(fold_id, tasks, threshold=0.4 + fold_id * 0.01)
        folds.append(
            {
                "outer_fold": fold_id,
                "outer_held_out_task_ids": list(tasks),
                "fixed_raw_mse_reference": {"threshold": 0.1},
                "learned_models": {
                    "combined": {
                        "outer_training_refit": {
                            "feature_names": list(model.feature_names),
                            "preprocessor": {
                                "expanded_feature_names": list(model.expanded_feature_names),
                                "imputation_medians": list(model.imputation_medians),
                                "scaling_mean": list(model.scaling_mean),
                                "scaling_scale": list(model.scaling_scale),
                            },
                            "model": {"weights": list(model.weights), "bias": model.bias},
                        },
                        "threshold_selection": {
                            "selected_threshold": model.threshold,
                            "selected_threshold_hex": model.threshold_hex,
                        },
                    }
                },
            }
        )
    report = {
        "inputs": {
            "workload_identity_sha256": EXPECTED_TRACE_IDENTITY,
            "source_git_commit": EXPECTED_MODEL_SOURCE_COMMIT,
        }
    }
    summary = {
        "global_model_fitted": False,
        "global_threshold_fitted": False,
        "outer_folds": folds,
    }
    (path / "metric_report.json").write_text(json.dumps(report), encoding="utf-8")
    (path / "model_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    hashes = {
        name: sha256_file(path / name)
        for name in ("metric_report.json", "model_summary.json")
    }
    (path / "output_hashes.json").write_text(
        json.dumps({"files": hashes}), encoding="utf-8"
    )


def test_task_to_outer_fold_mapping_uses_serialized_artifacts_without_refitting(tmp_path):
    _write_artifacts(tmp_path)
    models, task_to_fold, provenance = load_serialized_combined_models(tmp_path)
    assert task_to_fold == {0: 0, 9: 0, 1: 1, 8: 1, 2: 2, 7: 2, 3: 3, 6: 3, 4: 4, 5: 4}
    assert models[3].threshold == pytest.approx(0.43)
    assert provenance["models_refit"] is False
    assert provenance["thresholds_reselected"] is False


def test_timing_summary_keeps_cuda_event_and_wall_fields_distinct():
    samples = [
        {
            "workload_id": "0:0",
            "operation": "raw_mse_tensor",
            "case_id": "k2",
            "cuda_event_ms": value,
            "wall_time_ms": value + 1.0,
        }
        for value in (1.0, 2.0, 3.0)
    ]
    summary = summarize_latency_samples(samples)
    row = summary["aggregates"]["raw_mse_tensor:k2"]
    assert row["cuda_event_device"]["p50_ms"] == 2.0
    assert row["synchronized_wall"]["p50_ms"] == 3.0
    assert row["cuda_event_is_end_to_end_decision_latency"] is False


def test_break_even_formula_and_trigger_derived_gate_evaluations():
    replays = [
        {
            "baseline_coda_calls": 6,
            "scheduled_coda_calls": 4,
            "delta_k": 0,
            "trigger_k": 3,
        },
        {
            "baseline_coda_calls": 6,
            "scheduled_coda_calls": 5,
            "delta_k": 1,
            "trigger_k": 5,
        },
    ]
    assert gate_evaluations_per_prediction(replays) == 3.0
    result = project_policy_latency(
        replays,
        coda_latency_ms=2.0,
        recurrent_iteration_latency_ms=1.0,
        gate_decision_latency_ms=0.1,
        baseline_action_head_latency_ms=5.0,
    )
    assert result["saved_coda_calls_per_prediction"] == 1.5
    assert result["gross_coda_saving_ms"] == 3.0
    assert result["added_recurrent_cost_ms"] == 0.5
    assert result["gate_overhead_ms"] == pytest.approx(0.3)
    assert result["projected_net_saving_ms"] == pytest.approx(2.2)
    assert result["projected_net_latency_change_ms"] == pytest.approx(-2.2)


def test_non_timing_order_is_deterministic_and_timing_boundary_excludes_io():
    first = deterministic_operation_order(workload_index=2, trial_index=1, case_index=0, seed=9)
    second = deterministic_operation_order(workload_index=2, trial_index=1, case_index=0, seed=9)
    assert first == second
    runner = Path("scripts/benchmark_combined_gate_overhead.py").read_text(encoding="utf-8")
    measured = runner[runner.index("def _measure_call"):runner.index("def _measure_peak_memory")]
    assert ".to(" not in measured
    assert "open(" not in measured
    assert "write" not in measured
