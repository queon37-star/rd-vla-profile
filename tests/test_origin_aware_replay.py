import json
import types

import pytest
import torch

from scripts.origin_aware_replay_lib import (
    CostModel,
    SchedulerConfig,
    SelectionConstraints,
    ShadowTraceValidationError,
    eligible_warm_latent_values,
    parse_shadow_prediction,
    parse_shadow_predictions,
    replay_prediction,
    run_task_level_oof_selection,
)
from prismatic.models.action_heads import RecurrentConfigInternal, VLARecurrent
from scripts.replay_origin_aware_shadow import main as replay_cli_main


def _shadow_record(
    *,
    task_id=0,
    episode_id=0,
    prediction_step=0,
    baseline_k=5,
    max_iter=6,
    warm=True,
    latent_mse=None,
    action_mse=None,
):
    latent_mse = latent_mse or [1.0] * max_iter
    action_mse = action_mse or [None, 1.0, 1.0, 0.2, 0.05, 0.01][:max_iter]
    trace = []
    for k in range(1, max_iter + 1):
        action_value = action_mse[k - 1]
        trace.append(
            {
                "k": k,
                "phase": "production" if k <= baseline_k else "shadow_tail",
                "state_finite": True,
                "output_finite": True,
                "latent_mse": latent_mse[k - 1],
                "latent_l2": latent_mse[k - 1] ** 0.5,
                "action_mse": action_value,
                "action_l2": None if action_value is None else action_value ** 0.5,
            }
        )
    return {
        "task_id": task_id,
        "episode_id": episode_id,
        "prediction_step": prediction_step,
        "K_t": baseline_k,
        "max_recurrent_iteration": max_iter,
        "action_mse_threshold": 0.1,
        "effective_min_iter": 2,
        "latent_precheck_min_iter": 3,
        "warm_start_used": warm,
        "numerical_retry_attempted": False,
        "shadow_full_depth_enabled": True,
        "shadow_trace_complete": True,
        "shadow_error": None,
        "shadow_production_snapshot": {
            "K_t": baseline_k,
            "terminal_iteration": baseline_k,
            "stop_reason": "kl_divergence",
            "midpoint_source_iteration": max(1, baseline_k // 2),
            "cached_final_output_reused": True,
        },
        "shadow_trace": trace,
    }


def test_next_iter_replay_uses_refresh_then_forced_confirmation():
    prediction = parse_shadow_prediction(_shadow_record())
    result = replay_prediction(
        prediction,
        SchedulerConfig(
            warm_threshold=0.1,
            max_skip_iters=1,
            confirmation_mode="next_iter",
        ),
    )

    assert result.terminal_k == 5
    assert result.adaptive_stop is True
    assert result.decode_calls == 4
    assert result.backfill_decode_calls == 0
    assert result.latent_gate_calls == 2
    assert result.decoded_calls == (
        (1, "forced_initial"),
        (2, "forced_second"),
        (4, "max_skip_reached"),
        (5, "confirmation"),
    )
    assert result.comparison_pairs == ((1, 2), (4, 5))
    assert result.reference_first_convergence_k == 5
    assert result.captured_reference_convergence is True
    assert result.recovered_persistent_convergence is True
    assert result.stopped_before_persistent_tail is False


def test_backfill_replay_decodes_previous_and_current_states():
    record = _shadow_record(
        baseline_k=4,
        action_mse=[None, 1.0, 1.0, 0.05, 0.01, 0.01],
    )
    result = replay_prediction(
        parse_shadow_prediction(record),
        SchedulerConfig(
            warm_threshold=0.1,
            max_skip_iters=1,
            confirmation_mode="backfill_pair",
        ),
    )

    assert result.terminal_k == 4
    assert result.decode_calls == 4
    assert result.backfill_decode_calls == 1
    assert result.decoded_calls[-2:] == (
        (3, "backfill_previous"),
        (4, "max_skip_reached"),
    )
    assert result.comparison_pairs == ((1, 2), (3, 4))
    assert result.captured_reference_convergence is True


def test_max_iter_has_priority_over_backfill_confirmation():
    record = _shadow_record(
        baseline_k=4,
        max_iter=4,
        action_mse=[None, 1.0, 1.0, 1.0],
    )
    result = replay_prediction(
        parse_shadow_prediction(record),
        SchedulerConfig(
            warm_threshold=0.1,
            max_skip_iters=1,
            confirmation_mode="backfill_pair",
        ),
    )

    assert result.terminal_k == 4
    assert result.adaptive_stop is False
    assert result.stop_reason == "max_iter"
    assert result.decode_calls == 3
    assert result.backfill_decode_calls == 0
    assert result.decoded_calls[-1] == (4, "max_iter")
    assert result.comparison_pairs == ((1, 2),)
    assert result.max_iteration_convergence_evaluable is False


def test_actual_origin_selects_cold_or_warm_threshold():
    common = {
        "baseline_k": 3,
        "latent_mse": [1.0, 1.0, 0.1, 1.0, 1.0, 1.0],
        "action_mse": [None, 1.0, 0.05, 1.0, 0.05, 0.01],
    }
    cold = parse_shadow_prediction(_shadow_record(warm=False, **common))
    warm = parse_shadow_prediction(_shadow_record(warm=True, **common))
    config = SchedulerConfig(
        warm_threshold=0.05,
        max_skip_iters=1,
        confirmation_mode="next_iter",
    )

    assert replay_prediction(cold, config).terminal_k == 3
    assert replay_prediction(warm, config).terminal_k == 5


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda record: record.update(shadow_trace_complete=False), "complete"),
        (
            lambda record: record["shadow_trace"][2].update(state_finite=False),
            "non-finite",
        ),
        (
            lambda record: record["shadow_trace"][5].update(phase="production"),
            "phase",
        ),
        (lambda record: record["shadow_trace"].pop(), "cover every iteration"),
    ],
)
def test_parser_fails_closed_on_invalid_shadow_traces(mutation, message):
    record = _shadow_record()
    mutation(record)
    with pytest.raises(ShadowTraceValidationError, match=message):
        parse_shadow_prediction(record)


def test_quantile_pool_uses_only_eligible_actual_warm_transitions():
    warm = parse_shadow_prediction(
        _shadow_record(
            max_iter=5,
            baseline_k=4,
            latent_mse=[0.9, 0.8, 0.1, 0.2, 0.3],
            action_mse=[None, 1.0, 1.0, 0.05, 0.01],
        )
    )
    cold = parse_shadow_prediction(
        _shadow_record(
            task_id=1,
            max_iter=5,
            baseline_k=4,
            warm=False,
            latent_mse=[0.9, 0.8, 9.0, 9.0, 9.0],
            action_mse=[None, 1.0, 1.0, 0.05, 0.01],
        )
    )

    assert eligible_warm_latent_values([warm, cold]) == [0.1, 0.2]


def _constant_threshold_records():
    records = []
    for task_id in (0, 1):
        records.append(
            _shadow_record(
                task_id=task_id,
                latent_mse=[1.0, 1.0, 0.1, 0.1, 0.1, 0.1],
            )
        )
    return records


def test_oof_quantile_refit_and_exact_config_dedup():
    predictions = parse_shadow_predictions(_constant_threshold_records())
    report = run_task_level_oof_selection(
        predictions,
        {
            "folds": [
                {"fold_id": 0, "validation_task_ids": [0]},
                {"fold_id": 1, "validation_task_ids": [1]},
            ]
        },
        CostModel(
            recurrent_ms=1.0,
            decode_ms=1.0,
            latent_gate_ms=0.0,
            action_compare_ms=0.0,
            finite_check_ms=0.0,
        ),
        fixed_thresholds=[0.1],
        quantiles=[0.5],
        max_skip_values=[1],
        confirmation_modes=["next_iter"],
        top_n=6,
    )

    assert report["family_grid_size"] == 2
    assert report["passing_family_count"] == 2
    assert report["selected_distinct_config_count"] == 1
    selected = report["selected_refit_configs"][0]
    assert selected["config"]["warm_threshold"] == 0.1
    quantile_report = next(
        item
        for item in report["family_reports"]
        if item["threshold_family"]["kind"] == "quantile"
    )
    assert [fold["warm_threshold"] for fold in quantile_report["fold_thresholds"]] == [
        0.1,
        0.1,
    ]


def test_oof_quantile_threshold_is_fit_on_other_tasks_only():
    records = _constant_threshold_records()
    for point in records[0]["shadow_trace"]:
        if 3 <= point["k"] < 6:
            point["latent_mse"] = 0.1
    for point in records[1]["shadow_trace"]:
        if 3 <= point["k"] < 6:
            point["latent_mse"] = 0.3
    report = run_task_level_oof_selection(
        parse_shadow_predictions(records),
        {
            "folds": [
                {"fold_id": 0, "validation_task_ids": [0]},
                {"fold_id": 1, "validation_task_ids": [1]},
            ]
        },
        CostModel(1.0, 1.0, 0.0, 0.0, 0.0),
        constraints=SelectionConstraints(
            min_convergence_capture=0.0,
            max_mean_delta_k=10.0,
            max_p95_delta_k=10.0,
            max_max_iter_rate_delta=1.0,
        ),
        fixed_thresholds=[],
        quantiles=[0.5],
        max_skip_values=[1],
        confirmation_modes=["next_iter"],
    )
    thresholds = report["family_reports"][0]["fold_thresholds"]
    assert thresholds[0]["warm_threshold"] == pytest.approx(0.3)
    assert thresholds[1]["warm_threshold"] == pytest.approx(0.1)


def test_cli_writes_strict_reproducible_report(tmp_path):
    trace_path = tmp_path / "shadow.jsonl"
    trace_path.write_text(
        "\n".join(json.dumps(record) for record in _constant_threshold_records()) + "\n",
        encoding="utf-8",
    )
    folds_path = tmp_path / "folds.json"
    folds_path.write_text(
        json.dumps(
            {
                "folds": [
                    {"fold_id": 0, "validation_task_ids": [0]},
                    {"fold_id": 1, "validation_task_ids": [1]},
                ]
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "report.json"

    replay_cli_main(
        [
            str(trace_path),
            "--fold-manifest",
            str(folds_path),
            "--output",
            str(output_path),
            "--recurrent-ms",
            "1",
            "--decode-ms",
            "1",
            "--latent-gate-ms",
            "0",
            "--action-compare-ms",
            "0",
            "--finite-check-ms",
            "0",
            "--fixed-thresholds",
            "0.1",
            "--quantiles",
            "0.5",
            "--max-skip-values",
            "1",
            "--confirmation-modes",
            "next_iter",
        ]
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["prediction_count"] == 2
    assert report["selected_distinct_config_count"] == 1
    assert len(report["inputs"][0]["sha256"]) == 64
    assert "baseline-conditioned" in report["scope"]


def _deterministic_scheduler_model():
    cfg = RecurrentConfigInternal(
        hidden_dim=4,
        num_heads=1,
        recurrent_vlm_layers=(0,),
        coda_vlm_layers=(),
        action_chunk_len=2,
        action_dim=2,
        mean_recurrence=3,
        backprop_depth=1,
        random_iterations=False,
    )
    model = VLARecurrent(cfg).eval()
    action_deltas = [None, None, 1.0, 1.0, 0.2 ** 0.5, 0.05 ** 0.5, 0.01 ** 0.5]
    action_values = [0.0]
    for k in range(1, 7):
        delta = 0.0 if k == 1 else action_deltas[k]
        action_values.append(action_values[-1] + delta)

    def init_state(self, batch_size, device, dtype):
        return torch.zeros(batch_size, 2, 4, device=device, dtype=dtype)

    def run_one_iteration(self, state, *args):
        return state + 1

    def get_output(self, state, *args, profile=False):
        iteration = int(round(float(state[0, 0, 0].item())))
        return torch.full_like(state[..., :2], action_values[iteration])

    model.init_state = types.MethodType(init_state, model)
    model._run_one_iteration = types.MethodType(run_one_iteration, model)
    model._get_output = types.MethodType(get_output, model)
    return model


def _scheduler_inputs():
    return (
        torch.zeros(1, 1, 1, 4),
        torch.zeros(1, 1, 1, 4),
        torch.zeros(1, 1, 4),
    )


@pytest.mark.parametrize("warm_threshold", [0.5, 1.0])
@pytest.mark.parametrize("max_skip", [1, 2])
@pytest.mark.parametrize("confirmation", ["next_iter", "backfill_pair"])
def test_offline_replay_matches_online_scheduler(warm_threshold, max_skip, confirmation):
    shadow_model = _deterministic_scheduler_model()
    shadow_result = shadow_model(
        *_scheduler_inputs(),
        convergence_strategy="kl_divergence",
        kl_thresh=0.1,
        max_iter=6,
        warm_start_state=torch.zeros(1, 2, 4),
        enable_warm_start=True,
        warm_start_source="midpoint",
        use_cached_final_output=True,
        use_latent_precheck=False,
        latent_precheck_mode="off",
        latent_precheck_trace_level="off",
        latent_precheck_min_iter=3,
        shadow_full_depth=True,
    )
    shadow_debug = shadow_model.last_recurrence_debug
    shadow_metadata = shadow_model.last_inference_metadata
    record = {
        "task_id": 0,
        "episode_id": 0,
        "prediction_step": 0,
        "K_t": shadow_result[1],
        "max_recurrent_iteration": 6,
        "action_mse_threshold": 0.1,
        "effective_min_iter": 2,
        "latent_precheck_min_iter": 3,
        "warm_start_used": shadow_metadata["warm_start"]["state_used"],
        "numerical_retry_attempted": False,
        **{
            key: shadow_debug[key]
            for key in (
                "shadow_full_depth_enabled",
                "shadow_trace_complete",
                "shadow_error",
                "shadow_production_snapshot",
                "shadow_trace",
            )
        },
    }
    prediction = parse_shadow_prediction(record)
    offline = replay_prediction(
        prediction,
        SchedulerConfig(
            warm_threshold=warm_threshold,
            max_skip_iters=max_skip,
            confirmation_mode=confirmation,
        ),
    )

    online_model = _deterministic_scheduler_model()
    online_result = online_model(
        *_scheduler_inputs(),
        convergence_strategy="kl_divergence",
        kl_thresh=0.1,
        max_iter=6,
        warm_start_state=torch.zeros(1, 2, 4),
        enable_warm_start=True,
        warm_start_source="midpoint",
        use_cached_final_output=True,
        use_latent_precheck=True,
        latent_precheck_mode="origin_aware",
        latent_precheck_trace_level="full",
        latent_precheck_warm_thresh=warm_threshold,
        latent_precheck_min_iter=3,
        latent_precheck_max_skip_iters=max_skip,
        latent_precheck_confirmation_mode=confirmation,
        nonfinite_policy="cold_retry_once",
    )
    online = online_model.last_recurrence_debug

    assert offline.terminal_k == online_result[1] == online["K_t"]
    assert offline.decode_calls == online["latent_precheck_call_count"]
    assert offline.backfill_decode_calls == online["coda_reason_counts"].get(
        "backfill_previous", 0
    )
    assert offline.decoded_calls == tuple(
        (record["iteration"], record["reason"])
        for record in online["coda_call_records"]
    )
    assert offline.comparison_pairs == tuple(
        tuple(pair) for pair in online["adjacent_comparison_pairs"]
    )
    assert offline.adaptive_stop == online["adaptive_stop"]
    assert offline.final_convergence_evaluable == online["final_convergence_evaluable"]
    assert (
        offline.max_iteration_convergence_evaluable
        == online["max_iteration_convergence_evaluable"]
    )
