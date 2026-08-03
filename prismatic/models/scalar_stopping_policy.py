"""Hash-verified scalar stopping policy utilities for RD-VLA inference."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch


SCALAR_POLICY_SCHEMA_VERSION = 1
SCALAR_POLICY_ARTIFACT_TYPE = "rdvla_task_oof_scalar_stopping_policy"

SCALAR_FEATURE_NAMES = (
    "iteration_k",
    "delta_rms",
    "previous_delta_rms",
    "relative_delta_rms",
    "delta_ratio",
    "delta_cosine",
    "second_difference_rms",
)

SUPPORTED_SCALAR_EXECUTION_MODES = (
    "direct",
    "confirm_next",
)


class ScalarStoppingPolicyError(ValueError):
    """Raised when a scalar-policy artifact violates its frozen contract."""


class NonFiniteScalarPolicyError(RuntimeError):
    """Raised before a non-finite scalar score can affect recurrence stopping."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ScalarStoppingPolicyError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_positive(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ScalarStoppingPolicyError(f"{name} must be finite and positive")

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ScalarStoppingPolicyError(
            f"{name} must be finite and positive"
        ) from exc

    _require(
        math.isfinite(result) and result > 0.0,
        f"{name} must be finite and positive",
    )
    return result


def _finite_tensor(
    value: Any,
    *,
    shape: tuple[int, ...],
    name: str,
) -> torch.Tensor:
    _require(torch.is_tensor(value), f"{name} must be a tensor")

    tensor = (
        value.detach()
        .to(device="cpu", dtype=torch.float32)
        .contiguous()
        .clone()
    )

    _require(
        tuple(tensor.shape) == shape,
        f"{name} shape mismatch: expected={shape}, actual={tuple(tensor.shape)}",
    )
    _require(
        bool(torch.isfinite(tensor).all().item()),
        f"{name} is non-finite",
    )
    return tensor


def validate_scalar_policy_artifact(
    payload: Mapping[str, Any],
) -> None:
    _require(
        int(payload.get("schema_version", -1))
        == SCALAR_POLICY_SCHEMA_VERSION,
        "scalar-policy schema mismatch",
    )
    _require(
        payload.get("artifact_type")
        == SCALAR_POLICY_ARTIFACT_TYPE,
        "scalar-policy artifact type mismatch",
    )
    _require(
        payload.get("target_reference")
        == "K_first_authoritative",
        "runtime scalar policy must target K_first_authoritative",
    )
    _require(
        payload.get("model_configuration") == "scalar_combo",
        "runtime scalar policy must use scalar_combo",
    )
    _require(
        tuple(payload.get("feature_names", ()))
        == SCALAR_FEATURE_NAMES,
        "scalar feature order mismatch",
    )

    minimum_gate_iteration = payload.get(
        "minimum_gate_iteration"
    )
    _require(
        isinstance(minimum_gate_iteration, int)
        and not isinstance(minimum_gate_iteration, bool)
        and minimum_gate_iteration == 3,
        "schema v1 requires minimum_gate_iteration=3",
    )

    _finite_positive(payload.get("epsilon"), "epsilon")

    modes = tuple(payload.get("supported_execution_modes", ()))
    _require(
        modes == SUPPORTED_SCALAR_EXECUTION_MODES,
        "supported execution modes mismatch",
    )

    raw_task_to_fold = payload.get("task_to_fold")
    raw_policies = payload.get("policies_by_task")

    _require(
        isinstance(raw_task_to_fold, Mapping),
        "task_to_fold must be a mapping",
    )
    _require(
        isinstance(raw_policies, Mapping),
        "policies_by_task must be a mapping",
    )

    task_to_fold = {
        int(task_id): int(fold_id)
        for task_id, fold_id in raw_task_to_fold.items()
    }

    _require(
        set(task_to_fold) == set(range(10)),
        "scalar policy must cover LIBERO Spatial task IDs 0..9",
    )
    _require(
        set(raw_policies) == {
            str(task_id) for task_id in range(10)
        },
        "policies_by_task must exactly cover task IDs 0..9",
    )

    for task_id in range(10):
        policy = raw_policies[str(task_id)]
        _require(
            isinstance(policy, Mapping),
            f"task {task_id} policy must be a mapping",
        )
        _require(
            int(policy.get("task_id", -1)) == task_id,
            f"task {task_id} policy identity mismatch",
        )
        _require(
            int(policy.get("outer_fold", -1))
            == task_to_fold[task_id],
            f"task {task_id} outer-fold mismatch",
        )
        _require(
            task_id
            in {
                int(value)
                for value in policy.get(
                    "held_out_task_ids", []
                )
            },
            f"task {task_id} is not held out by its policy fold",
        )

        threshold = float(
            policy.get("selected_threshold", float("nan"))
        )
        _require(
            math.isfinite(threshold)
            and 0.0 < threshold < 1.0,
            f"task {task_id} threshold is invalid",
        )

        mean = _finite_tensor(
            policy.get("normalizer_mean"),
            shape=(7,),
            name=f"task {task_id} normalizer_mean",
        )
        scale = _finite_tensor(
            policy.get("normalizer_scale"),
            shape=(7,),
            name=f"task {task_id} normalizer_scale",
        )
        _finite_tensor(
            policy.get("linear_weight"),
            shape=(7,),
            name=f"task {task_id} linear_weight",
        )
        _finite_tensor(
            policy.get("linear_bias"),
            shape=(),
            name=f"task {task_id} linear_bias",
        )

        _require(
            bool(torch.all(scale > 0).item()),
            f"task {task_id} normalizer scale must be positive",
        )
        _require(
            mean.numel() == len(SCALAR_FEATURE_NAMES),
            f"task {task_id} feature dimension mismatch",
        )


def load_scalar_policy_artifact(
    source: str | Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a scalar policy only after verifying its SHA-256."""

    source_path = Path(source)
    manifest: dict[str, Any] = {}

    if source_path.is_dir():
        manifest_path = source_path / "manifest.json"
        _require(
            manifest_path.is_file(),
            f"scalar-policy manifest does not exist: {manifest_path}",
        )
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        artifact_path = source_path / str(
            manifest.get("artifact_file", "")
        )
    else:
        artifact_path = source_path
        manifest_path = artifact_path.parent / "manifest.json"

        if manifest_path.is_file():
            candidate = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            if candidate.get("artifact_file") == artifact_path.name:
                manifest = candidate

    _require(
        artifact_path.is_file(),
        f"scalar-policy artifact does not exist: {artifact_path}",
    )

    manifest_sha = manifest.get("artifact_sha256")

    if expected_sha256 is not None and manifest_sha is not None:
        _require(
            str(expected_sha256) == str(manifest_sha),
            "requested and manifest scalar-policy hashes differ",
        )

    required_sha = (
        str(expected_sha256)
        if expected_sha256 is not None
        else (
            str(manifest_sha)
            if manifest_sha is not None
            else None
        )
    )

    _require(
        required_sha is not None,
        "scalar-policy loading requires a manifest or expected SHA-256",
    )

    actual_sha = sha256_file(artifact_path)
    _require(
        actual_sha == required_sha,
        "scalar-policy artifact SHA-256 mismatch: "
        f"expected={required_sha}, actual={actual_sha}",
    )

    payload = torch.load(
        artifact_path,
        map_location="cpu",
        weights_only=True,
    )
    _require(
        isinstance(payload, Mapping),
        "scalar-policy payload must be a mapping",
    )

    validate_scalar_policy_artifact(payload)

    if manifest:
        _require(
            int(manifest.get("schema_version", -1))
            == SCALAR_POLICY_SCHEMA_VERSION,
            "scalar-policy manifest schema mismatch",
        )
        _require(
            manifest.get("artifact_type")
            == SCALAR_POLICY_ARTIFACT_TYPE,
            "scalar-policy manifest artifact type mismatch",
        )

    return manifest, dict(payload)



def validate_scalar_runtime_configuration(
    recurrence_strategy: str | None,
    *,
    task_policy: Any,
    execution_mode: str,
    use_warm_start: bool,
    warm_start_source: str,
    use_latent_precheck: bool,
    latent_precheck_mode: str,
    latent_precheck_trace_level: str,
    shadow_full_depth: bool,
    collect_preconvergence_raw_shadow: bool,
    use_cached_final_output: bool,
    max_iter: int,
) -> None:
    """Validate scalar-policy combinations without loading an artifact."""

    if recurrence_strategy != "scalar_policy":
        _require(
            task_policy is None,
            "scalar_task_policy may only be supplied with "
            "recurrence_strategy='scalar_policy'",
        )
        return

    _require(
        isinstance(
            task_policy,
            PreparedScalarTaskPolicy,
        ),
        "scalar_policy requires a prepared task policy",
    )
    _require(
        execution_mode
        in SUPPORTED_SCALAR_EXECUTION_MODES,
        "scalar_policy execution mode must be direct "
        "or confirm_next",
    )
    _require(
        use_warm_start,
        "scalar_policy requires warm-start inference",
    )
    _require(
        warm_start_source == "midpoint",
        "scalar_policy requires midpoint warm-start",
    )
    _require(
        not use_latent_precheck,
        "scalar_policy cannot use latent pre-check",
    )
    _require(
        latent_precheck_mode == "off",
        "scalar_policy requires latent_precheck_mode='off'",
    )
    _require(
        latent_precheck_trace_level == "off",
        "scalar_policy requires latent_precheck_trace_level='off'",
    )
    _require(
        not shadow_full_depth,
        "scalar_policy cannot enable shadow_full_depth",
    )
    _require(
        not collect_preconvergence_raw_shadow,
        "scalar_policy cannot collect raw shadow trajectories",
    )
    _require(
        not use_cached_final_output,
        "scalar_policy performs one terminal Coda call and "
        "cannot reuse a cached output",
    )
    _require(
        isinstance(max_iter, int)
        and not isinstance(max_iter, bool)
        and max_iter >= 3,
        "scalar_policy requires max_iter >= 3",
    )


@dataclass(frozen=True)
class PreparedScalarTaskPolicy:
    """One task's OOF scalar policy moved to the inference device."""

    task_id: int
    outer_fold: int
    threshold: float
    minimum_gate_iteration: int
    epsilon: float
    normalizer_mean: torch.Tensor
    normalizer_scale: torch.Tensor
    linear_weight: torch.Tensor
    linear_bias: torch.Tensor


def prepare_scalar_task_policy(
    payload: Mapping[str, Any],
    task_id: int,
    *,
    device: torch.device | str,
) -> PreparedScalarTaskPolicy:
    validate_scalar_policy_artifact(payload)

    if isinstance(task_id, bool) or not isinstance(task_id, int):
        raise ScalarStoppingPolicyError(
            "scalar-policy task_id must be an integer"
        )

    policies = payload["policies_by_task"]
    _require(
        str(task_id) in policies,
        f"scalar policy does not contain task ID {task_id}",
    )

    policy = policies[str(task_id)]
    target_device = torch.device(device)

    def move(name: str) -> torch.Tensor:
        return (
            policy[name]
            .detach()
            .to(
                device=target_device,
                dtype=torch.float32,
            )
            .contiguous()
            .clone()
        )

    prepared = PreparedScalarTaskPolicy(
        task_id=task_id,
        outer_fold=int(policy["outer_fold"]),
        threshold=float(policy["selected_threshold"]),
        minimum_gate_iteration=int(
            payload["minimum_gate_iteration"]
        ),
        epsilon=float(payload["epsilon"]),
        normalizer_mean=move("normalizer_mean"),
        normalizer_scale=move("normalizer_scale"),
        linear_weight=move("linear_weight"),
        linear_bias=move("linear_bias").reshape(()),
    )

    _require(
        bool(
            torch.isfinite(
                prepared.normalizer_mean
            ).all().item()
        ),
        "prepared scalar-policy mean is non-finite",
    )
    _require(
        bool(
            torch.isfinite(
                prepared.normalizer_scale
            ).all().item()
        )
        and bool(
            torch.all(
                prepared.normalizer_scale > 0
            ).item()
        ),
        "prepared scalar-policy scale is invalid",
    )

    return prepared


def compute_scalar_stopping_features(
    current_state: torch.Tensor,
    previous_state: torch.Tensor,
    previous_previous_state: torch.Tensor,
    *,
    iteration: int,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Compute the exact seven online features used by scalar_combo."""

    if (
        isinstance(iteration, bool)
        or not isinstance(iteration, int)
        or iteration < 3
    ):
        raise ScalarStoppingPolicyError(
            "scalar stopping features require iteration >= 3"
        )

    epsilon = _finite_positive(epsilon, "epsilon")

    for name, state in (
        ("current_state", current_state),
        ("previous_state", previous_state),
        ("previous_previous_state", previous_previous_state),
    ):
        if not torch.is_tensor(state):
            raise ScalarStoppingPolicyError(
                f"{name} must be a tensor"
            )
        if state.numel() == 0:
            raise ScalarStoppingPolicyError(
                f"{name} must be non-empty"
            )

    if not (
        current_state.shape
        == previous_state.shape
        == previous_previous_state.shape
    ):
        raise ScalarStoppingPolicyError(
            "scalar stopping states must have identical shapes"
        )

    current = current_state.to(dtype=torch.float32)
    previous = previous_state.to(dtype=torch.float32)
    previous_previous = previous_previous_state.to(
        dtype=torch.float32
    )

    if not (
        bool(torch.isfinite(current).all().item())
        and bool(torch.isfinite(previous).all().item())
        and bool(
            torch.isfinite(
                previous_previous
            ).all().item()
        )
    ):
        raise NonFiniteScalarPolicyError(
            "scalar stopping state is non-finite"
        )

    current_delta = current - previous
    previous_delta = previous - previous_previous

    delta_rms = torch.sqrt(
        torch.mean(current_delta.square())
    )
    previous_delta_rms = torch.sqrt(
        torch.mean(previous_delta.square())
    )
    state_rms = torch.sqrt(
        torch.mean(current.square())
    )

    relative_delta_rms = delta_rms / torch.clamp(
        state_rms,
        min=epsilon,
    )
    delta_ratio = delta_rms / torch.clamp(
        previous_delta_rms,
        min=epsilon,
    )

    reduce_dimensions = tuple(
        range(current.ndim - 1)
    )
    if reduce_dimensions:
        current_mean = current.mean(
            dim=reduce_dimensions
        )
        previous_mean = previous.mean(
            dim=reduce_dimensions
        )
        previous_previous_mean = (
            previous_previous.mean(
                dim=reduce_dimensions
            )
        )
    else:
        current_mean = current
        previous_mean = previous
        previous_previous_mean = previous_previous

    current_mean_delta = (
        current_mean - previous_mean
    ).reshape(-1)
    previous_mean_delta = (
        previous_mean - previous_previous_mean
    ).reshape(-1)

    cosine_denominator = (
        torch.linalg.vector_norm(
            current_mean_delta
        )
        * torch.linalg.vector_norm(
            previous_mean_delta
        )
    )
    delta_cosine = torch.dot(
        current_mean_delta,
        previous_mean_delta,
    ) / torch.clamp(
        cosine_denominator,
        min=epsilon,
    )

    second_difference_rms = torch.sqrt(
        torch.mean(
            (
                current_delta
                - previous_delta
            ).square()
        )
    )

    iteration_value = torch.tensor(
        float(iteration),
        device=current.device,
        dtype=torch.float32,
    )

    features = torch.stack(
        (
            iteration_value,
            delta_rms,
            previous_delta_rms,
            relative_delta_rms,
            delta_ratio,
            delta_cosine,
            second_difference_rms,
        )
    ).contiguous()

    if not bool(torch.isfinite(features).all().item()):
        raise NonFiniteScalarPolicyError(
            "scalar stopping features are non-finite"
        )

    return features


def score_scalar_stopping_policy(
    policy: PreparedScalarTaskPolicy,
    features: torch.Tensor,
) -> torch.Tensor:
    """Return one finite sigmoid score without applying the threshold."""

    if not torch.is_tensor(features):
        raise ScalarStoppingPolicyError(
            "scalar stopping features must be a tensor"
        )

    values = features.to(dtype=torch.float32)

    _require(
        tuple(values.shape) == (7,),
        "scalar stopping feature shape must be (7,)",
    )
    _require(
        values.device == policy.normalizer_mean.device,
        "scalar features and prepared policy must share a device",
    )

    if not bool(torch.isfinite(values).all().item()):
        raise NonFiniteScalarPolicyError(
            "scalar stopping features are non-finite"
        )

    normalized = (
        values - policy.normalizer_mean
    ) / policy.normalizer_scale

    logit = (
        torch.dot(
            normalized,
            policy.linear_weight,
        )
        + policy.linear_bias
    )
    score = torch.sigmoid(logit)

    if not bool(torch.isfinite(score).item()):
        raise NonFiniteScalarPolicyError(
            "scalar stopping score is non-finite"
        )

    return score


def evaluate_scalar_stopping_policy(
    policy: PreparedScalarTaskPolicy,
    features: torch.Tensor,
) -> tuple[float, bool]:
    """Synchronize one score and return score plus threshold decision."""

    score = float(
        score_scalar_stopping_policy(
            policy,
            features,
        ).item()
    )
    return score, score >= policy.threshold


def resolve_scalar_terminal_iteration(
    gate_iteration: int | None,
    *,
    maximum_iteration: int,
    execution_mode: str,
) -> int:
    if execution_mode not in SUPPORTED_SCALAR_EXECUTION_MODES:
        raise ScalarStoppingPolicyError(
            f"unsupported scalar execution mode: {execution_mode}"
        )
    if (
        isinstance(maximum_iteration, bool)
        or not isinstance(maximum_iteration, int)
        or maximum_iteration < 1
    ):
        raise ScalarStoppingPolicyError(
            "maximum_iteration must be an integer >= 1"
        )

    if gate_iteration is None:
        return maximum_iteration

    if (
        isinstance(gate_iteration, bool)
        or not isinstance(gate_iteration, int)
        or not 1 <= gate_iteration <= maximum_iteration
    ):
        raise ScalarStoppingPolicyError(
            "gate_iteration is outside the recurrence range"
        )

    if execution_mode == "direct":
        return gate_iteration

    return min(
        gate_iteration + 1,
        maximum_iteration,
    )
