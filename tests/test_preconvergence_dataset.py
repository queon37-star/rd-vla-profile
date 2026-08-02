from __future__ import annotations

import json
from pathlib import Path

import torch

from scripts.preconvergence_trigger_lib import (
    RawPreconvergenceSequence,
    SequenceIdentity,
    build_training_batch,
    derive_k_action,
    leakage_audit,
    load_dataset_bundle,
    load_raw_manifest_sequences,
    pooled_trigger_input,
    preconvergence_label,
    save_dataset_bundle,
    sha256_file,
    tensor_metadata,
)


def make_sequence(
    prediction_id: int,
    *,
    task_id: int = 0,
    episode_id: int = 0,
    origin: str = "ACTUAL_WARM",
    k_action: int = 5,
    max_iter: int = 8,
) -> RawPreconvergenceSequence:
    states = torch.arange(max_iter * 2 * 3, dtype=torch.float32).reshape(
        max_iter, 2, 3
    )
    states = states + prediction_id * 0.01
    actions = torch.arange(max_iter * 4, dtype=torch.float32).reshape(max_iter, 4)
    actions = actions + prediction_id * 0.02
    mse = [None, None] + [0.01] * (max_iter - 1)
    mse[k_action] = 0.0005
    phases = [None, None] + [
        "production" if k <= k_action else "shadow_tail"
        for k in range(2, max_iter + 1)
    ]
    sequence = RawPreconvergenceSequence(
        identity=SequenceIdentity(task_id, episode_id, prediction_id),
        actual_origin=origin,
        states=states,
        actions=actions,
        action_mse=tuple(mse),
        action_mse_phase=tuple(phases),
        baseline_k=k_action,
        max_iter=max_iter,
    )
    sequence.validate()
    return sequence


def test_k_action_and_first_hit_labels() -> None:
    sequence = make_sequence(0, k_action=6)
    assert derive_k_action(sequence.action_mse) == 6
    labels = [preconvergence_label(k, sequence.k_action) for k in range(3, 6)]
    assert labels == [0, 0, 1]


def test_post_convergence_transitions_are_excluded() -> None:
    sequence = make_sequence(0, k_action=6)
    batch = build_training_batch([sequence])
    assert [row.k for row in batch.rows] == [3, 4, 5]
    assert all(row.k < row.k_action for row in batch.rows)
    assert sum(row.label for row in batch.rows) == 1


def test_future_state_changes_do_not_change_current_input() -> None:
    sequence = make_sequence(0, k_action=6)
    before = pooled_trigger_input(sequence.states, 4)
    changed = sequence.states.clone()
    changed[4:] = changed[4:] + 100000.0
    after = pooled_trigger_input(changed, 4)
    assert torch.equal(before, after)


def test_task_level_oof_has_zero_overlap() -> None:
    sequences = [
        make_sequence(0, task_id=0),
        make_sequence(1, task_id=1),
        make_sequence(2, task_id=2),
        make_sequence(3, task_id=3),
    ]
    audit = leakage_audit(sequences, {"0": 0, "1": 0, "2": 1, "3": 1})
    assert audit["passed"] is True
    assert all(fold["task_overlap_count"] == 0 for fold in audit["folds"])
    assert all(fold["prediction_overlap_count"] == 0 for fold in audit["folds"])


def test_prediction_level_weighting_gives_each_prediction_unit_mass() -> None:
    batch = build_training_batch(
        [make_sequence(0, k_action=5), make_sequence(1, k_action=8)]
    )
    by_prediction: dict[int, float] = {}
    for row, weight in zip(batch.rows, batch.weights):
        by_prediction.setdefault(row.identity.prediction_id, 0.0)
        by_prediction[row.identity.prediction_id] += float(weight)
    assert by_prediction == {0: 1.0, 1: 1.0}
    for prediction_id in by_prediction:
        indices = [
            index
            for index, row in enumerate(batch.rows)
            if row.identity.prediction_id == prediction_id
        ]
        positive_mass = sum(
            float(batch.weights[index])
            for index in indices
            if batch.rows[index].label == 1
        )
        negative_mass = sum(
            float(batch.weights[index])
            for index in indices
            if batch.rows[index].label == 0
        )
        assert positive_mass == 0.5
        assert negative_mass == 0.5


def test_building_features_does_not_modify_raw_inputs() -> None:
    sequence = make_sequence(0, k_action=6)
    states_before = sequence.states.clone()
    actions_before = sequence.actions.clone()
    build_training_batch([sequence])
    assert torch.equal(sequence.states, states_before)
    assert torch.equal(sequence.actions, actions_before)


def test_raw_shadow_manifest_joins_only_to_authoritative_label_source(
    tmp_path: Path,
) -> None:
    sequence = make_sequence(0, k_action=5)
    authoritative_dir = tmp_path / "authoritative"
    authoritative_dir.mkdir()
    transitions = [
        {
            "k": k,
            "phase": sequence.action_mse_phase[k],
            "action_mse": sequence.action_mse[k],
            "label": int(sequence.action_mse[k] < 0.001),
            "features": [0.0] * 18,
        }
        for k in range(2, sequence.max_iter + 1)
    ]
    record = {
        "key": list(sequence.identity.key),
        "task_id": 0,
        "episode_id": 0,
        "prediction_index": 0,
        "actual_origin": "ACTUAL_WARM",
        "baseline_k": 5,
        "baseline_decode_calls": 5,
        "max_iter": sequence.max_iter,
        "transitions": transitions,
    }
    data_path = authoritative_dir / "dataset.jsonl"
    data_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    trace_identity = "frozen-trace-identity"
    (authoritative_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_file": data_path.name,
                "dataset_sha256": sha256_file(data_path),
                "trace_set_sha256": trace_identity,
            }
        ),
        encoding="utf-8",
    )

    shard_path = tmp_path / "prediction.pt"
    torch.save(
        {
            "schema_version": 1,
            "identity": {"task_id": 0, "episode_id": 0, "prediction_id": 0},
            "actual_origin": "ACTUAL_WARM",
            "states": sequence.states,
            "actions": sequence.actions,
        },
        shard_path,
    )
    raw_manifest_path = tmp_path / "raw_manifest.json"
    raw_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "collection_mode": "optional_post_production_shadow",
                "source_trace_set_sha256": trace_identity,
                "sequences": [
                    {
                        "task_id": 0,
                        "episode_id": 0,
                        "prediction_id": 0,
                        "shard_path": shard_path.name,
                        "sha256": sha256_file(shard_path),
                        "tensor_metadata": {
                            "states": tensor_metadata(sequence.states),
                            "actions": tensor_metadata(sequence.actions),
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    metadata, loaded = load_raw_manifest_sequences(
        raw_manifest_path, authoritative_dir
    )
    assert metadata["authoritative_label_sources"]["production"].startswith(
        "native BF16"
    )
    assert loaded[0].k_action == 5
    output_dir = tmp_path / "derived"
    manifest = save_dataset_bundle(output_dir, metadata, loaded)
    reloaded_manifest, reloaded = load_dataset_bundle(output_dir)
    assert reloaded_manifest["dataset_sha256"] == manifest["dataset_sha256"]
    assert reloaded[0].identity == sequence.identity
    assert torch.equal(reloaded[0].states, sequence.states)
