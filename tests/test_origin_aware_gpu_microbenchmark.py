import copy
from collections import Counter
from pathlib import Path

import pytest

from scripts.origin_aware_gpu_microbenchmark_lib import (
    BASELINE_CONDITION_ID,
    GPUMicrobenchmarkValidationError,
    balanced_condition_order,
    build_benchmark_summary,
    conditions_from_shortlist,
    load_json_object,
    validate_protocol_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SHORTLIST_PATH = (
    REPO_ROOT
    / "experiments/robot/libero/manifests/origin_aware_oof_seed7_shortlist_v1.json"
)
PROTOCOL_PATH = (
    REPO_ROOT
    / "experiments/robot/libero/manifests/origin_aware_gpu_microbenchmark_v1.json"
)


def _protocol():
    protocol = load_json_object(PROTOCOL_PATH)
    protocol = copy.deepcopy(protocol)
    protocol["measurement_repeats"] = 3
    protocol["bootstrap"]["draws"] = 1000
    return protocol


def _measurements(conditions):
    records = []
    for task_id in (0, 1):
        for episode_id in (0, 1):
            for prediction_step, origin in ((0, "COLD"), (1, "ACTUAL_WARM")):
                for condition in conditions:
                    if origin == "ACTUAL_WARM":
                        if condition.condition_id == BASELINE_CONDITION_ID:
                            latency = 10.0
                        elif condition.rank == 1:
                            latency = 9.0
                        else:
                            latency = 9.8
                    else:
                        latency = 12.0 if condition.kind == "baseline" else 11.8
                    for repeat_index in range(3):
                        records.append(
                            {
                                "task_id": task_id,
                                "episode_id": episode_id,
                                "prediction_step": prediction_step,
                                "repeat_index": repeat_index,
                                "actual_origin": origin,
                                "condition_id": condition.condition_id,
                                "latency_ms": latency,
                            }
                        )
    return records


def test_committed_protocol_and_shortlist_define_seven_distinct_conditions():
    protocol = load_json_object(PROTOCOL_PATH)
    validate_protocol_manifest(protocol)
    conditions = conditions_from_shortlist(load_json_object(SHORTLIST_PATH))

    assert len(conditions) == 7
    assert conditions[0].condition_id == BASELINE_CONDITION_ID
    assert len({condition.exact_key for condition in conditions}) == 7


def test_balanced_order_is_deterministic_complete_and_position_balanced():
    condition_ids = [f"condition_{index}" for index in range(7)]
    observed = Counter()
    first = balanced_condition_order(condition_ids, block_index=0, repeat_index=0, seed=7)
    assert first == balanced_condition_order(
        condition_ids, block_index=0, repeat_index=0, seed=7
    )
    for block_index in range(700):
        order = balanced_condition_order(
            condition_ids, block_index=block_index, repeat_index=0, seed=7
        )
        assert set(order) == set(condition_ids)
        for position, condition_id in enumerate(order):
            observed[(condition_id, position)] += 1
    assert set(observed.values()) == {100}


def test_primary_simultaneous_bound_promotes_only_measured_five_percent_winner():
    conditions = conditions_from_shortlist(load_json_object(SHORTLIST_PATH))
    summary = build_benchmark_summary(
        _measurements(conditions),
        conditions,
        _protocol(),
        schedule_mismatch_count=0,
        required_task_ids=(0, 1),
        episodes_per_task=2,
    )

    assert summary["primary_scope"] == "actual_warm"
    assert summary["screening_candidates"] == ["candidate_rank_1"]
    assert summary["online_screening_allowed"] is True
    reports = summary["scopes"]["actual_warm"]["conditions"]
    assert reports[1]["improvement_vs_baseline"] == pytest.approx(0.1)
    assert reports[1]["simultaneous_one_sided_lower_bound"] == pytest.approx(0.1)
    assert all(not report["promotion_gate_passed"] for report in reports[2:])


def test_schedule_mismatch_vetoes_otherwise_passing_candidate():
    conditions = conditions_from_shortlist(load_json_object(SHORTLIST_PATH))
    summary = build_benchmark_summary(
        _measurements(conditions),
        conditions,
        _protocol(),
        schedule_mismatch_count=1,
        required_task_ids=(0, 1),
        episodes_per_task=2,
    )
    assert summary["screening_candidates"] == []
    assert summary["online_screening_allowed"] is False


def test_missing_repeat_is_rejected():
    conditions = conditions_from_shortlist(load_json_object(SHORTLIST_PATH))
    records = _measurements(conditions)
    records.pop()
    with pytest.raises(GPUMicrobenchmarkValidationError, match="repeats"):
        build_benchmark_summary(
            records,
            conditions,
            _protocol(),
            schedule_mismatch_count=0,
            required_task_ids=(0, 1),
            episodes_per_task=2,
        )
