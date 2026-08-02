import pytest

from configs.rdvla_precheck import (
    canonicalize_recurrence_strategy,
    validate_latent_only_configuration,
    validate_latent_precheck_configuration,
)


@pytest.mark.parametrize(
    ("requested", "canonical"),
    [
        (None, None),
        ("fixed", "fixed"),
        ("kl_divergence", "adjacent_action_mse"),
        ("adjacent_action_mse", "adjacent_action_mse"),
        ("cosine_similarity", "cosine_similarity"),
        ("latent_only", "latent_only"),
    ],
)
def test_recurrence_strategy_canonicalization(requested, canonical):
    assert canonicalize_recurrence_strategy(requested) == canonical


def test_unknown_recurrence_strategy_is_rejected():
    with pytest.raises(ValueError, match="Unsupported recurrence strategy"):
        canonicalize_recurrence_strategy("typo")


def _validate_latent_only(**overrides):
    values = {
        "recurrence_strategy": "latent_only",
        "metric": "raw_mse",
        "cold_threshold": 0.1,
        "warm_threshold": 0.2,
        "min_iter": 2,
        "eps": 1e-8,
    }
    values.update(overrides)
    return validate_latent_only_configuration(**values)


def test_latent_only_configuration_accepts_independent_defaults():
    assert _validate_latent_only() is None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"metric": "unknown"}, "Unsupported latent_only_metric"),
        ({"cold_threshold": -1.0}, "finite non-negative"),
        ({"warm_threshold": float("nan")}, "finite non-negative"),
        ({"min_iter": 1}, "integer >= 2"),
        ({"eps": 0.0}, "finite positive"),
        ({"use_latent_precheck": True}, "cannot use a latent pre-check"),
        ({"latent_precheck_mode": "origin_aware"}, "cannot use a latent pre-check"),
        ({"shadow_full_depth": True}, "cannot enable shadow_full_depth"),
        ({"use_cached_final_output": True}, "cannot reuse a cached action"),
    ],
)
def test_latent_only_configuration_rejects_invalid_or_scheduler_settings(
    overrides, message
):
    with pytest.raises(ValueError, match=message):
        _validate_latent_only(**overrides)


@pytest.mark.parametrize("trace_level", ["off", "summary", "full"])
@pytest.mark.parametrize("use_latent_precheck", [False, True])
def test_legacy_mode_ignores_new_trace_policy(trace_level, use_latent_precheck):
    assert (
        validate_latent_precheck_configuration(
            "legacy",
            trace_level,
            use_latent_precheck,
        )
        == "legacy"
    )


def test_off_mode_requires_precheck_and_trace_to_be_off():
    assert validate_latent_precheck_configuration("off", "off", False) == "off"

    with pytest.raises(ValueError, match="use_latent_precheck=False"):
        validate_latent_precheck_configuration("off", "off", True)

    with pytest.raises(ValueError, match="trace_level='off'"):
        validate_latent_precheck_configuration("off", "summary", False)


def test_origin_aware_mode_fails_closed_until_scheduler_is_implemented():
    with pytest.raises(NotImplementedError, match="not implemented yet"):
        validate_latent_precheck_configuration("origin_aware", "off", False)


@pytest.mark.parametrize(
    ("mode", "trace_level", "message"),
    [
        ("unknown", "off", "Unsupported latent_precheck_mode"),
        ("legacy", "verbose", "Unsupported latent_precheck_trace_level"),
    ],
)
def test_unknown_precheck_settings_are_rejected(mode, trace_level, message):
    with pytest.raises(ValueError, match=message):
        validate_latent_precheck_configuration(mode, trace_level, False)


def test_nonfinite_policy_rejects_unknown_and_non_origin_aware_use():
    with pytest.raises(ValueError, match="Unsupported nonfinite_policy"):
        validate_latent_precheck_configuration(
            "legacy", "off", False, nonfinite_policy="typo"
        )
    with pytest.raises(ValueError, match="requires latent_precheck_mode='origin_aware'"):
        validate_latent_precheck_configuration(
            "off", "off", False, nonfinite_policy="cold_retry_once"
        )


def _validate_origin_aware(*, use_latent_precheck=True, **overrides):
    scheduler = {
        "origin_aware_implemented": True,
        "warm_threshold": 0.1,
        "max_skip_iters": 2,
        "confirmation_mode": "next_iter",
        "warm_start_source": "midpoint",
        "recurrence_strategy": "kl_divergence",
        "use_warm_start": True,
        "min_iter": 2,
        "nonfinite_policy": "cold_retry_once",
    }
    scheduler.update(overrides)
    return validate_latent_precheck_configuration(
        "origin_aware",
        "full",
        use_latent_precheck,
        **scheduler,
    )


@pytest.mark.parametrize("strategy", ["kl_divergence", "adjacent_action_mse"])
@pytest.mark.parametrize("confirmation_mode", ["next_iter", "backfill_pair"])
def test_origin_aware_configuration_accepts_frozen_scheduler_contract(
    strategy, confirmation_mode
):
    assert (
        _validate_origin_aware(
            recurrence_strategy=strategy,
            confirmation_mode=confirmation_mode,
        )
        == "origin_aware"
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"use_latent_precheck": False}, "requires use_latent_precheck=True"),
        ({"use_warm_start": False}, "requires use_warm_start=True"),
        ({"warm_start_source": "final"}, "requires warm_start_source='midpoint'"),
        ({"warm_threshold": None}, "finite non-negative"),
        ({"warm_threshold": float("nan")}, "finite non-negative"),
        ({"warm_threshold": -0.1}, "finite non-negative"),
        ({"warm_threshold": True}, "finite non-negative"),
        ({"max_skip_iters": 0}, "integer >= 1"),
        ({"max_skip_iters": True}, "integer >= 1"),
        ({"confirmation_mode": "unknown"}, "Unsupported latent_precheck_confirmation_mode"),
        ({"recurrence_strategy": "cosine_similarity"}, "requires recurrence_strategy"),
        ({"min_iter": 1}, "integer >= 2"),
        ({"nonfinite_policy": "legacy"}, "requires nonfinite_policy='cold_retry_once'"),
    ],
)
def test_origin_aware_configuration_rejects_invalid_scheduler_contract(overrides, message):
    overrides = dict(overrides)
    use_latent_precheck = overrides.pop("use_latent_precheck", True)
    with pytest.raises(ValueError, match=message):
        _validate_origin_aware(
            use_latent_precheck=use_latent_precheck,
            **overrides,
        )
