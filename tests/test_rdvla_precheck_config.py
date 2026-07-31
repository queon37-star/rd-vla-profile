import pytest

from configs.rdvla_precheck import (
    canonicalize_recurrence_strategy,
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
    ],
)
def test_recurrence_strategy_canonicalization(requested, canonical):
    assert canonicalize_recurrence_strategy(requested) == canonical


def test_unknown_recurrence_strategy_is_rejected():
    with pytest.raises(ValueError, match="Unsupported recurrence strategy"):
        canonicalize_recurrence_strategy("typo")


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
