import json
from pathlib import Path

import pytest

import scripts.audit_adaptive_coda_gate as safety
from scripts.adaptive_coda_gate_oof import recorded_action_mse_by_iteration, replay_trigger
from scripts.analyze_latent_dynamics_features import canonical_json
from scripts.audit_adaptive_coda_gate import (
    AUDIT_POLICIES,
    SOURCE_FIXED_POLICY,
    WRAPPER_COMPONENTS,
    combine_trigger_k,
    exact_required_count,
    failure_sets,
    replay_predeclared_wrappers,
    selected_thresholds_by_fold,
    write_safety_outputs,
)


def _prediction(
    prediction_id,
    *,
    task_id="0",
    baseline_k=4,
    true_iterations=None,
):
    true_iterations = {baseline_k} if true_iterations is None else set(true_iterations)
    transitions = [
        {
            "k": iteration,
            "iteration_index": iteration,
            "label": iteration in true_iterations,
            "action_mse": 0.0005 if iteration in true_iterations else 0.01,
            "action_mse_source": (
                "production_iteration_mse"
                if iteration <= baseline_k
                else "shadow_fp32_adjacent_action_mse"
            ),
        }
        for iteration in range(2, 33)
    ]
    return {
        "key": (task_id, 0, prediction_id),
        "task_id": task_id,
        "episode_id": 0,
        "prediction_id": prediction_id,
        "actual_origin": "ACTUAL_WARM",
        "baseline_k": baseline_k,
        "activation_target": max(2, baseline_k - 1),
        "difficulty": "easy" if baseline_k <= 4 else "medium",
        "max_iter": 32,
        "transitions": transitions,
    }


def _enriched_replay(prediction, trigger, policy, *, forced=False, fold=0):
    return {
        **replay_trigger(prediction, trigger, forced_trigger=forced),
        "policy": policy,
        "outer_fold": fold,
    }


def _source_replays(predictions, trigger_by_policy):
    policies = set(AUDIT_POLICIES) | {
        component for components in WRAPPER_COMPONENTS.values() for component in components
    }
    result = {}
    for policy in policies:
        trigger = trigger_by_policy.get(policy, 2)
        result[policy] = [
            _enriched_replay(prediction, trigger, policy) for prediction in predictions
        ]
    return result


def test_exact_95_percent_required_count_uses_ceiling():
    assert exact_required_count(2298) == 2184
    assert exact_required_count(20) == 19
    assert exact_required_count(21) == 20


def test_fixed_combined_failure_intersections_are_exact():
    predictions = [_prediction(index) for index in range(4)]
    fixed = [_enriched_replay(item, 2, SOURCE_FIXED_POLICY) for item in predictions]
    combined = [_enriched_replay(item, 2, "combined") for item in predictions]
    for index, delta in enumerate((1, 1, 0, 0)):
        fixed[index]["delta_k"] = delta
    for index, delta in enumerate((0, 1, 1, 0)):
        combined[index]["delta_k"] = delta
    sets = failure_sets(fixed, combined)
    assert {key[2] for key in sets["shared"]} == {1}
    assert {key[2] for key in sets["fixed_only"]} == {0}
    assert {key[2] for key in sets["combined_only"]} == {2}
    assert {key[2] for key in sets["union"]} == {0, 1, 2}


def test_min_trigger_combination_uses_earliest_component():
    prediction = _prediction(0)
    components = [
        _enriched_replay(prediction, 7, "combined"),
        _enriched_replay(prediction, 3, SOURCE_FIXED_POLICY),
        _enriched_replay(prediction, 5, "iteration_raw_mse"),
    ]
    assert combine_trigger_k(components) == 3


def test_trigger_combination_is_followed_by_exact_replay():
    prediction = _prediction(0, baseline_k=6, true_iterations={6})
    source = _source_replays(
        [prediction],
        {
            "combined": 7,
            SOURCE_FIXED_POLICY: 3,
            "iteration_raw_mse": 5,
            "raw_mse_logistic": 6,
        },
    )
    result = replay_predeclared_wrappers([prediction], source)
    hybrid = result["all_replays"]["combined_or_fixed"][0]
    assert hybrid["trigger_k"] == 3
    assert hybrid["first_action_mse_check_k"] == 4
    assert hybrid["terminal_k"] == 6
    assert hybrid["executed_action_mse_checks"] == [4, 5, 6]


def test_nonmonotonic_action_labels_are_replayed_in_sequence():
    prediction = _prediction(0, baseline_k=4, true_iterations={4, 6})
    source = _source_replays(
        [prediction],
        {
            "combined": 4,
            SOURCE_FIXED_POLICY: 2,
            "iteration_raw_mse": 4,
            "raw_mse_logistic": 4,
        },
    )
    result = replay_predeclared_wrappers([prediction], source)
    assert result["all_replays"]["combined"][0]["terminal_k"] == 6
    assert result["all_replays"]["combined_or_fixed"][0]["terminal_k"] == 4


def test_hybrid_terminal_k_never_exceeds_component_terminal_k():
    predictions = [
        _prediction(0, baseline_k=4, true_iterations={4, 7}),
        _prediction(1, baseline_k=6, true_iterations={6, 9}),
    ]
    source = _source_replays(
        predictions,
        {
            "combined": 5,
            SOURCE_FIXED_POLICY: 2,
            "iteration_raw_mse": 4,
            "raw_mse_logistic": 3,
        },
    )
    result = replay_predeclared_wrappers(predictions, source)
    for wrapper, components in WRAPPER_COMPONENTS.items():
        for row in result["all_replays"][wrapper]:
            assert all(
                row["terminal_k"] <= row["component_terminal_k"][component]
                for component in components
            )


def test_hybrid_exact_k_rate_is_not_lower_than_components():
    predictions = [
        _prediction(0, baseline_k=4, true_iterations={4, 7}),
        _prediction(1, baseline_k=6, true_iterations={6, 9}),
    ]
    source = _source_replays(
        predictions,
        {
            "combined": 5,
            SOURCE_FIXED_POLICY: 2,
            "iteration_raw_mse": 4,
            "raw_mse_logistic": 3,
        },
    )
    result = replay_predeclared_wrappers(predictions, source)
    for wrapper, components in WRAPPER_COMPONENTS.items():
        wrapper_rate = result["wrapper_results"][wrapper]["metrics"][
            "exact_K_preservation_rate"
        ]
        for component in components:
            component_rate = sum(
                bool(row["exact_k_preserved"]) for row in source[component]
            ) / len(source[component])
            assert wrapper_rate >= component_rate


def test_coda_calls_are_recomputed_from_hybrid_replay():
    prediction = _prediction(0, baseline_k=6, true_iterations={6})
    source = _source_replays(
        [prediction],
        {
            "combined": 5,
            SOURCE_FIXED_POLICY: 3,
            "iteration_raw_mse": 4,
            "raw_mse_logistic": 4,
        },
    )
    result = replay_predeclared_wrappers([prediction], source)
    row = result["all_replays"]["combined_or_fixed"][0]
    assert row["scheduled_coda_calls"] == 1 + row["terminal_k"] - row["trigger_k"] + 1
    assert row["scheduled_coda_calls"] == len(row["executed_coda_iterations"])
    assert row["scheduled_coda_calls"] != (
        source["combined"][0]["scheduled_coda_calls"]
        + source[SOURCE_FIXED_POLICY][0]["scheduled_coda_calls"]
    )


def test_production_and_shadow_action_label_phase_semantics():
    record = {
        "K_t": 4,
        "iteration_mse": [0.1, 0.01, 0.0009],
        "latent_metric_trace": [
            {
                "iteration_index": iteration,
                "adjacent_action_mse": 0.0011 if iteration == 4 else 0.0008,
            }
            for iteration in range(2, 33)
        ],
    }
    values = recorded_action_mse_by_iteration(record)
    assert values[4]["label"] is True
    assert values[4]["source"] == "production_iteration_mse"
    assert values[5]["label"] is True
    assert values[5]["source"] == "shadow_fp32_adjacent_action_mse"


def _minimal_model_summary(threshold=0.25):
    models = {}
    for policy in ("raw_mse_logistic", "iteration_raw_mse", "combined"):
        models[policy] = {
            "threshold_selection": {
                "selected_threshold": threshold,
                "selected_threshold_hex": float(threshold).hex(),
            },
            "outer_training_refit": {"serialized": policy},
        }
    return {
        "outer_folds": [
            {
                "outer_fold": 0,
                "outer_held_out_task_ids": ["0"],
                "fixed_raw_mse_reference": {
                    "threshold": 0.1,
                    "threshold_hex": float(0.1).hex(),
                },
                "learned_models": models,
            }
        ]
    }


def test_frozen_score_reconstruction_does_not_refit(monkeypatch):
    prediction = _prediction(0)
    calls = []

    def fake_score(predictions, fitted):
        calls.append(fitted["serialized"])
        return [
            {
                "prediction": predictions[0],
                "scores_by_k": {iteration: 0.5 for iteration in range(2, 32)},
            }
        ]

    monkeypatch.setattr(safety, "score_gate_predictions", fake_score)
    monkeypatch.setattr(
        "scripts.adaptive_coda_gate_oof.fit_gate_model",
        lambda *_args, **_kwargs: pytest.fail("model refitting is forbidden"),
    )
    scores = safety.reconstruct_frozen_scores([prediction], _minimal_model_summary())
    assert calls == ["raw_mse_logistic", "iteration_raw_mse", "combined"]
    assert scores["combined"][("0", 0, 0)][2] == 0.5


def test_thresholds_are_read_from_frozen_outer_artifacts_only():
    summary = _minimal_model_summary(threshold=0.375)
    first = selected_thresholds_by_fold(summary)
    second = selected_thresholds_by_fold(json.loads(canonical_json(summary)))
    assert first == second
    assert first["0"]["combined"]["threshold"] == 0.375
    assert first["0"]["combined"]["selection_scope"].endswith("_frozen")


def test_deterministic_output_bytes(tmp_path: Path):
    predictions = [_prediction(0), _prediction(1, task_id="1", baseline_k=5)]
    source = _source_replays(
        predictions,
        {
            "combined": 2,
            SOURCE_FIXED_POLICY: 2,
            "iteration_raw_mse": 2,
            "raw_mse_logistic": 2,
        },
    )
    wrappers = replay_predeclared_wrappers(predictions, source)
    audit = {"prediction_count": 2, "exact_95_percent_requirement": {}}
    failures = []
    hashes = []
    for name in ("first", "second"):
        output = tmp_path / name
        hashes.append(
            write_safety_outputs(
                output,
                failure_audit=audit,
                failure_rows=failures,
                wrapper_evaluation=wrappers,
                inputs={"frozen": True},
            )
        )
    assert hashes[0] == hashes[1]
    for filename in (*hashes[0], "output_hashes.json"):
        assert (tmp_path / "first" / filename).read_bytes() == (
            tmp_path / "second" / filename
        ).read_bytes()
