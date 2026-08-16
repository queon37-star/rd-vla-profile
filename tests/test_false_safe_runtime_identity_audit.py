import json

from scripts.coda_anchor_feasibility.audit_false_safe_runtime_identity import (
    extract_runtime_rejected_events,
    offline_identity,
    runtime_identity,
)


def test_runtime_and_offline_counter_identity_mapping_is_explicit():
    runtime = {
        "task_id": 4,
        "episode_id": 2,
        "action_prediction_index": 15,
        "timestep": 85,
    }
    offline = {
        "task_id": 4,
        "episode_id": 2,
        "prediction_id": 15,
        "timestep": 85,
    }

    assert runtime_identity(runtime) == (4, 2, 15, 85)
    assert offline_identity(offline) == runtime_identity(runtime)


def test_rejected_event_extraction_joins_confirmation_to_exact_score_row(
    tmp_path,
):
    record = {
        "task_id": 4,
        "episode_id": 2,
        "action_prediction_index": 15,
        "prediction_step": 15,
        "timestep": 85,
        "paired_trial_id": 2,
        "initial_state_id": 2,
        "episode_seed": 123,
        "initial_states_sha256": "states",
        "initial_states_file_sha256": "file",
        "evaluation_protocol_phase": "screening",
        "initial_state_partition": "screening",
        "actual_origin": "ACTUAL_WARM",
        "warm_start_source": "midpoint",
        "warm_start_source_iteration": 3,
        "warm_start_source_K": 6,
        "warm_start_cache_age": 15,
        "K_t": 6,
        "action_delta_gate_score_trace": [
            {
                "anchor_iteration": 4,
                "terminal_iteration": 5,
                "score": 0.0007,
                "triggered": True,
            },
            {
                "anchor_iteration": 5,
                "terminal_iteration": 6,
                "score": 0.0004,
                "triggered": True,
            },
        ],
        "action_delta_gate_exact_confirmation_trace": [
            {
                "mode": "oracle_confirm",
                "anchor_iteration": 4,
                "terminal_iteration": 5,
                "exact_adjacent_mse": 0.0011,
                "exact_safe": False,
                "accepted": False,
            },
            {
                "mode": "oracle_confirm",
                "anchor_iteration": 5,
                "terminal_iteration": 6,
                "exact_adjacent_mse": 0.0004,
                "exact_safe": True,
                "accepted": True,
            },
        ],
    }
    path = tmp_path / "runtime.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    events = extract_runtime_rejected_events(path)

    assert len(events) == 1
    event = events[0]
    assert runtime_identity(event) == (4, 2, 15, 85)
    assert event["anchor_iteration"] == 4
    assert event["terminal_iteration"] == 5
    assert event["predicted_score"] == 0.0007
    assert event["exact_adjacent_action_mse"] == 0.0011
