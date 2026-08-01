from scripts.learned_convergence_probe_lib import (
    FEATURE_NAMES,
    FINAL_ONLY_POLICY,
    LEGACY_POLICY,
    aggregate_scheduler_metrics,
    replay_scored_records,
)


def _record(task_id, labels, scores):
    transitions = [
        {
            "k": k,
            "phase": "production",
            "action_mse": 0.0005 if label else 0.01,
            "label": int(label),
            "features": [float(k)] * len(FEATURE_NAMES),
        }
        for k, label in enumerate(labels, start=2)
    ]
    baseline_k = next(
        (index + 2 for index, label in enumerate(labels) if label),
        len(labels) + 1,
    )
    return {
        "key": [str(task_id), 0, 0],
        "task_id": str(task_id),
        "episode_id": 0,
        "prediction_index": 0,
        "actual_origin": "ACTUAL_WARM",
        "baseline_k": baseline_k,
        "baseline_decode_calls": baseline_k,
        "max_iter": len(labels) + 1,
        "transitions": transitions,
        "scores": scores,
    }


def test_positive_at_k3_uses_one_final_only_decode_and_two_legacy_decodes():
    record = _record(0, [0, 1, 1], [0.1, 0.9, 0.9])
    final_only = replay_scored_records(
        [record], 0.5, policy=FINAL_ONLY_POLICY
    )[0]
    legacy = replay_scored_records([record], 0.5, policy=LEGACY_POLICY)[0]
    assert final_only["terminal_k"] == legacy["terminal_k"] == 3
    assert final_only["recurrent_calls"] == 3
    assert final_only["probe_calls"] == 2
    assert final_only["coda_decode_calls"] == 1
    assert final_only["final_action_source"] == "Coda(S_3)"
    assert legacy["coda_decode_calls"] == 2


def test_final_only_uses_one_decode_when_probe_never_fires():
    record = _record(0, [0, 0, 0], [0.1, 0.2, 0.3])
    replay = replay_scored_records([record], 0.5, policy=FINAL_ONLY_POLICY)[0]
    assert replay["terminal_k"] == record["max_iter"]
    assert replay["recurrent_calls"] == record["max_iter"]
    assert replay["probe_calls"] == record["max_iter"] - 1
    assert replay["coda_decode_calls"] == 1


def test_false_positive_keeps_safety_failure_with_one_final_decode():
    record = _record(0, [0, 1, 1], [0.8, 0.9, 0.9])
    replay = replay_scored_records([record], 0.5, policy=FINAL_ONLY_POLICY)[0]
    assert replay["terminal_k"] == 2
    assert replay["false_convergence"] is True
    assert replay["coda_decode_calls"] == 1


def test_every_final_only_prediction_has_exactly_one_coda_decode():
    records = [
        _record(0, [0, 1, 1], [0.1, 0.9, 0.9]),
        _record(1, [0, 0, 0], [0.1, 0.2, 0.3]),
        _record(2, [0, 1, 1], [0.8, 0.9, 0.9]),
    ]
    replays = replay_scored_records(records, 0.5, policy=FINAL_ONLY_POLICY)
    assert all(item["coda_decode_calls"] == 1 for item in replays)


def test_final_only_total_decode_reduction_uses_prediction_count():
    records = [
        _record(0, [0, 1, 1], [0.1, 0.9, 0.9]),
        _record(1, [0, 0, 0], [0.1, 0.2, 0.3]),
    ]
    metrics = aggregate_scheduler_metrics(
        replay_scored_records(records, 0.5, policy=FINAL_ONLY_POLICY)
    )
    baseline_calls = sum(record["baseline_decode_calls"] for record in records)
    assert metrics["candidate_coda_decode_calls"] == len(records)
    assert metrics["relative_coda_decode_call_reduction"] == (
        baseline_calls - len(records)
    ) / baseline_calls


def test_policies_share_terminal_and_safety_metrics():
    records = [
        _record(0, [0, 1, 1], [0.1, 0.9, 0.9]),
        _record(1, [0, 1, 1], [0.8, 0.9, 0.9]),
        _record(2, [0, 0, 0], [0.1, 0.2, 0.3]),
    ]
    legacy = aggregate_scheduler_metrics(
        replay_scored_records(records, 0.5, policy=LEGACY_POLICY)
    )
    final_only = aggregate_scheduler_metrics(
        replay_scored_records(records, 0.5, policy=FINAL_ONLY_POLICY)
    )
    for field in (
        "false_convergence_count",
        "convergence_capture",
        "precision",
        "recall",
        "mean_terminal_k",
        "mean_delta_k",
        "p95_delta_k",
        "max_iter_rate_delta",
    ):
        assert final_only[field] == legacy[field]
