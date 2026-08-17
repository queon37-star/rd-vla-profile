"""Hash-verified Action-Delta Gate artifacts and runtime scoring."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from prismatic.utils.rdvla_profiler import rdvla_range


ACTION_DELTA_GATE_SCHEMA_VERSION = 1
ACTION_DELTA_GATE_ARTIFACT_TYPE = "rdvla_action_delta_gate"
ACTION_DELTA_GATE_MODEL_TYPE = "delta_only_shared_token_linear"
ACTION_DELTA_GATE_CALIBRATION_METHOD = "cp95_false_safe_risk_0.01"
ACTION_DELTA_GATE_DELTA_DTYPE = "bfloat16"

ACTION_DELTA_GATE_HIDDEN_DIM = 896
ACTION_DELTA_GATE_ACTION_DIM = 7
ACTION_DELTA_GATE_CHUNK_LEN = 8
ACTION_DELTA_GATE_OUTER_FOLD = 4
ACTION_DELTA_GATE_HELD_OUT_TASK_IDS = (4, 5)
ACTION_DELTA_GATE_SHADOW_CALIBRATION_TASK_IDS = (0, 1, 2, 3, 6, 7, 8, 9)
ACTION_DELTA_NONCONVERGENCE_THRESHOLD = 0.0015
ACTION_DELTA_NONCONVERGENCE_MIN_TERMINAL_ITER = 5
# Fixed measurement anchors used only for diagnostic latency estimates. Runtime
# measurements remain separately reported and never replace these provenance
# values.
ACTION_DELTA_NONCONVERGENCE_SCORER_COST_MS = 0.36627289134678936
ACTION_DELTA_NONCONVERGENCE_CODA_COST_MS = 1.864207385085098
ACTION_DELTA_GATE_PRODUCTION_RETURN_MODES = (
    "anchor",
    "predicted_correction",
)
ACTION_DELTA_GATE_DIAGNOSTIC_RETURN_MODES = (
    "exact_terminal",
    "oracle_confirm",
)
ACTION_DELTA_GATE_RETURN_MODES = (
    ACTION_DELTA_GATE_PRODUCTION_RETURN_MODES
    + ACTION_DELTA_GATE_DIAGNOSTIC_RETURN_MODES
)


class ActionDeltaGateError(ValueError):
    """Raised when an Action-Delta Gate artifact violates its contract."""


class NonFiniteActionDeltaGateError(RuntimeError):
    """Raised before a non-finite value can cause a Coda skip."""


class ActionDeltaGateCorrectionError(RuntimeError):
    """Raised before an invalid predicted correction can cause a Coda skip."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ActionDeltaGateError(message)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_float(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ActionDeltaGateError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ActionDeltaGateError(f"{name} must be finite") from exc
    _require(math.isfinite(result), f"{name} must be finite")
    if positive:
        _require(result > 0.0, f"{name} must be positive")
    return result


def _integer(value: Any, name: str, expected: int) -> None:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value == expected,
        f"{name} must equal {expected}",
    )


def _finite_tensor(value: Any, *, shape: tuple[int, ...], name: str) -> torch.Tensor:
    _require(torch.is_tensor(value), f"{name} must be a tensor")
    tensor = value.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
    _require(
        tuple(tensor.shape) == shape,
        f"{name} shape mismatch: expected={shape}, actual={tuple(tensor.shape)}",
    )
    _require(bool(torch.isfinite(tensor).all().item()), f"{name} is non-finite")
    return tensor


def validate_action_delta_gate_artifact(payload: Mapping[str, Any]) -> None:
    _require(isinstance(payload, Mapping), "Action-Delta Gate payload must be a mapping")
    _integer(payload.get("schema_version"), "schema_version", ACTION_DELTA_GATE_SCHEMA_VERSION)
    _require(
        payload.get("artifact_type") == ACTION_DELTA_GATE_ARTIFACT_TYPE,
        "Action-Delta Gate artifact type mismatch",
    )
    _require(
        payload.get("model_type") == ACTION_DELTA_GATE_MODEL_TYPE,
        "Action-Delta Gate model type mismatch",
    )
    _integer(payload.get("hidden_dim"), "hidden_dim", ACTION_DELTA_GATE_HIDDEN_DIM)
    _integer(payload.get("action_dim"), "action_dim", ACTION_DELTA_GATE_ACTION_DIM)
    _integer(payload.get("action_chunk_len"), "action_chunk_len", ACTION_DELTA_GATE_CHUNK_LEN)
    _integer(payload.get("outer_fold"), "outer_fold", ACTION_DELTA_GATE_OUTER_FOLD)
    _require(
        tuple(payload.get("held_out_task_ids", ())) == ACTION_DELTA_GATE_HELD_OUT_TASK_IDS,
        "held-out task identity mismatch",
    )
    _require(
        payload.get("delta_quantization_dtype") == ACTION_DELTA_GATE_DELTA_DTYPE,
        "delta quantization dtype mismatch",
    )
    _require(
        payload.get("calibration_method") == ACTION_DELTA_GATE_CALIBRATION_METHOD,
        "calibration method mismatch",
    )
    _integer(payload.get("training_seed"), "training_seed", 1011)
    _integer(payload.get("epochs"), "epochs", 60)
    _integer(payload.get("batch_size"), "batch_size", 512)
    lr = _finite_float(payload.get("lr"), "lr", positive=True)
    weight_decay = _finite_float(
        payload.get("weight_decay"), "weight_decay", positive=True
    )
    _require(math.isclose(lr, 1e-3, rel_tol=0.0, abs_tol=1e-12), "lr must equal 1e-3")
    _require(
        math.isclose(weight_decay, 1e-4, rel_tol=0.0, abs_tol=1e-12),
        "weight_decay must equal 1e-4",
    )
    _finite_float(payload.get("threshold"), "threshold", positive=True)

    x_mean = _finite_tensor(payload.get("x_mean"), shape=(896,), name="x_mean")
    x_std = _finite_tensor(payload.get("x_std"), shape=(896,), name="x_std")
    y_mean = _finite_tensor(payload.get("y_mean"), shape=(7,), name="y_mean")
    y_std = _finite_tensor(payload.get("y_std"), shape=(7,), name="y_std")
    _finite_tensor(payload.get("linear_weight"), shape=(7, 896), name="linear_weight")
    _finite_tensor(payload.get("linear_bias"), shape=(7,), name="linear_bias")
    _require(bool(torch.all(x_std > 0).item()), "x_std must be positive")
    _require(bool(torch.all(y_std > 0).item()), "y_std must be positive")
    _require(x_mean.numel() == 896 and y_mean.numel() == 7, "artifact dimensions are inconsistent")


def load_action_delta_gate_artifact(
    source: str | Path,
    *,
    expected_sha256: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify SHA-256 before loading a weights-only-compatible artifact."""

    source_path = Path(source)
    manifest: dict[str, Any] = {}
    if source_path.is_dir():
        manifest_path = source_path / "manifest.json"
        _require(manifest_path.is_file(), f"Action-Delta Gate manifest does not exist: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifact_path = source_path / str(manifest.get("artifact_file", ""))
    else:
        artifact_path = source_path
        manifest_path = artifact_path.parent / "manifest.json"
        if manifest_path.is_file():
            candidate = json.loads(manifest_path.read_text(encoding="utf-8"))
            if candidate.get("artifact_file") == artifact_path.name:
                manifest = candidate

    _require(artifact_path.is_file(), f"Action-Delta Gate artifact does not exist: {artifact_path}")
    manifest_sha = manifest.get("artifact_sha256")
    if expected_sha256 is not None:
        _require(
            len(expected_sha256) == 64
            and all(character in "0123456789abcdef" for character in expected_sha256.lower()),
            "expected Action-Delta Gate SHA-256 is invalid",
        )
        if manifest_sha is not None:
            _require(
                expected_sha256.lower() == str(manifest_sha).lower(),
                "requested and manifest Action-Delta Gate hashes differ",
            )
    required_sha = expected_sha256.lower() if expected_sha256 is not None else manifest_sha
    _require(required_sha is not None, "Action-Delta Gate loading requires an expected SHA-256 or manifest")
    actual_sha = sha256_file(artifact_path)
    _require(
        actual_sha == str(required_sha).lower(),
        "Action-Delta Gate artifact SHA-256 mismatch: "
        f"expected={required_sha}, actual={actual_sha}",
    )

    payload = torch.load(artifact_path, map_location="cpu", weights_only=True)
    validate_action_delta_gate_artifact(payload)
    if manifest:
        _integer(manifest.get("schema_version"), "manifest schema_version", ACTION_DELTA_GATE_SCHEMA_VERSION)
        _require(
            manifest.get("artifact_type") == ACTION_DELTA_GATE_ARTIFACT_TYPE,
            "Action-Delta Gate manifest artifact type mismatch",
        )
        _require(
            int(manifest.get("outer_fold", -1)) == ACTION_DELTA_GATE_OUTER_FOLD,
            "Action-Delta Gate manifest outer-fold mismatch",
        )
        _require(
            tuple(manifest.get("held_out_task_ids", ())) == ACTION_DELTA_GATE_HELD_OUT_TASK_IDS,
            "Action-Delta Gate manifest held-out tasks mismatch",
        )
    return manifest, dict(payload)


@dataclass(frozen=True)
class PreparedActionDeltaGate:
    schema_version: int
    artifact_type: str
    model_type: str
    hidden_dim: int
    action_dim: int
    action_chunk_len: int
    held_out_task_ids: tuple[int, ...]
    outer_fold: int
    threshold: float
    x_mean: torch.Tensor
    x_std: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor
    linear_weight: torch.Tensor
    linear_bias: torch.Tensor
    delta_quantization_dtype: str
    training_seed: int
    calibration_method: str


def prepare_action_delta_gate(
    payload: Mapping[str, Any],
    *,
    device: torch.device | str,
    task_id: int,
) -> PreparedActionDeltaGate:
    validate_action_delta_gate_artifact(payload)
    _require(
        isinstance(task_id, int) and not isinstance(task_id, bool),
        "Action-Delta Gate task_id must be an integer",
    )
    held_out = tuple(int(value) for value in payload["held_out_task_ids"])
    _require(task_id in held_out, f"task {task_id} is not held out by Action-Delta Gate fold 4")
    target_device = torch.device(device)

    def move(name: str) -> torch.Tensor:
        return payload[name].detach().to(device=target_device, dtype=torch.float32).contiguous().clone()

    return PreparedActionDeltaGate(
        schema_version=int(payload["schema_version"]),
        artifact_type=str(payload["artifact_type"]),
        model_type=str(payload["model_type"]),
        hidden_dim=int(payload["hidden_dim"]),
        action_dim=int(payload["action_dim"]),
        action_chunk_len=int(payload["action_chunk_len"]),
        held_out_task_ids=held_out,
        outer_fold=int(payload["outer_fold"]),
        threshold=float(payload["threshold"]),
        x_mean=move("x_mean"),
        x_std=move("x_std"),
        y_mean=move("y_mean"),
        y_std=move("y_std"),
        linear_weight=move("linear_weight"),
        linear_bias=move("linear_bias"),
        delta_quantization_dtype=str(payload["delta_quantization_dtype"]),
        training_seed=int(payload["training_seed"]),
        calibration_method=str(payload["calibration_method"]),
    )


def prepare_action_delta_gate_shadow(
    payload: Mapping[str, Any],
    *,
    device: torch.device | str,
    task_id: int,
) -> PreparedActionDeltaGate:
    """Prepare the frozen fold-4 predictor for development-task shadow data.

    Production preparation remains restricted to the artifact's held-out tasks
    4 and 5.  This separate diagnostic entry point does the inverse: it accepts
    only the eight development tasks and can never authorize production gate
    inference.
    """

    _require(
        isinstance(task_id, int) and not isinstance(task_id, bool),
        "Action-Delta Gate shadow task_id must be an integer",
    )
    _require(
        task_id in ACTION_DELTA_GATE_SHADOW_CALIBRATION_TASK_IDS,
        "Action-Delta Gate shadow collection permits only development tasks "
        f"{ACTION_DELTA_GATE_SHADOW_CALIBRATION_TASK_IDS}",
    )
    # Reuse the exact artifact validation and tensor materialization contract.
    # The task identity check above is the only intentional difference from
    # prepare_action_delta_gate().
    validate_action_delta_gate_artifact(payload)
    target_device = torch.device(device)

    def move(name: str) -> torch.Tensor:
        return payload[name].detach().to(
            device=target_device, dtype=torch.float32
        ).contiguous().clone()

    return PreparedActionDeltaGate(
        schema_version=int(payload["schema_version"]),
        artifact_type=str(payload["artifact_type"]),
        model_type=str(payload["model_type"]),
        hidden_dim=int(payload["hidden_dim"]),
        action_dim=int(payload["action_dim"]),
        action_chunk_len=int(payload["action_chunk_len"]),
        held_out_task_ids=tuple(int(value) for value in payload["held_out_task_ids"]),
        outer_fold=int(payload["outer_fold"]),
        threshold=float(payload["threshold"]),
        x_mean=move("x_mean"),
        x_std=move("x_std"),
        y_mean=move("y_mean"),
        y_std=move("y_std"),
        linear_weight=move("linear_weight"),
        linear_bias=move("linear_bias"),
        delta_quantization_dtype=str(payload["delta_quantization_dtype"]),
        training_seed=int(payload["training_seed"]),
        calibration_method=str(payload["calibration_method"]),
    )


def score_action_delta_gate(
    gate: PreparedActionDeltaGate,
    anchor_state: torch.Tensor,
    current_state: torch.Tensor,
    *,
    return_pred_delta: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Return the on-device predicted next action-delta MSE without synchronizing."""

    if not isinstance(gate, PreparedActionDeltaGate):
        raise ActionDeltaGateError("a prepared Action-Delta Gate is required")
    for name, state in (("anchor_state", anchor_state), ("current_state", current_state)):
        _require(torch.is_tensor(state), f"{name} must be a tensor")
        _require(
            tuple(state.shape) == (1, gate.action_chunk_len, gate.hidden_dim),
            f"{name} shape mismatch: expected={(1, gate.action_chunk_len, gate.hidden_dim)}, "
            f"actual={tuple(state.shape)}",
        )
        _require(
            state.device == gate.x_mean.device,
            f"{name} and prepared Action-Delta Gate must share a device",
        )
    with rdvla_range("RDVLA/action_head/action_delta_gate/delta_compute"):
        delta = (
            current_state.float() - anchor_state.float()
        ).to(torch.bfloat16).float()
        x = (delta - gate.x_mean) / gate.x_std
    with rdvla_range("RDVLA/action_head/action_delta_gate/predict"):
        pred_norm = F.linear(x, gate.linear_weight, gate.linear_bias)
        pred_delta = pred_norm * gate.y_std + gate.y_mean
        score = pred_delta.square().mean()

    if return_pred_delta:
        return score, pred_delta
    return score


def evaluate_action_delta_gate(
    gate: PreparedActionDeltaGate,
    anchor_state: torch.Tensor,
    current_state: torch.Tensor,
    *,
    return_pred_delta: bool = False,
) -> tuple[float, bool] | tuple[float, bool, torch.Tensor]:
    """Return score and decision with exactly one device-to-host synchronization.

    Artifact tensors are finite, and both normalization scales are finite and
    positive. A non-finite state therefore remains non-finite through the BF16
    transition, normalization, finite linear projection, positive finite output
    scale, and final squared mean. Likewise, overflow in any intermediate makes
    the final score non-finite. Checking the final score on the host is thus
    sufficient for fail-closed behavior under the frozen artifact contract.
    """

    pred_delta = None
    if return_pred_delta:
        score_tensor, pred_delta = score_action_delta_gate(
            gate,
            anchor_state,
            current_state,
            return_pred_delta=True,
        )
    else:
        score_tensor = score_action_delta_gate(
            gate,
            anchor_state,
            current_state,
        )
    # One scalar D2H transfer is the runtime decision's only sync point.
    score = float(score_tensor.item())
    if not math.isfinite(score):
        raise NonFiniteActionDeltaGateError(
            "Action-Delta Gate state or score is non-finite"
        )
    triggered = score <= gate.threshold
    if return_pred_delta:
        return score, triggered, pred_delta
    return score, triggered


def build_action_delta_gate_corrected_output(
    anchor_output: torch.Tensor,
    pred_delta: torch.Tensor,
) -> torch.Tensor:
    """Add a float32 predicted delta and restore the exact anchor output contract."""

    if not torch.is_tensor(anchor_output) or not torch.is_tensor(pred_delta):
        raise ActionDeltaGateCorrectionError(
            "Action-Delta Gate correction inputs must be tensors"
        )
    if tuple(pred_delta.shape) != tuple(anchor_output.shape):
        raise ActionDeltaGateCorrectionError(
            "Action-Delta Gate correction shape mismatch: "
            f"anchor={tuple(anchor_output.shape)}, "
            f"predicted_delta={tuple(pred_delta.shape)}"
        )
    if pred_delta.device != anchor_output.device:
        raise ActionDeltaGateCorrectionError(
            "Action-Delta Gate correction tensors must share a device"
        )
    if not anchor_output.is_floating_point() or not pred_delta.is_floating_point():
        raise ActionDeltaGateCorrectionError(
            "Action-Delta Gate correction tensors must be floating point"
        )

    corrected_float = anchor_output.float() + pred_delta.float()
    corrected_output = corrected_float.to(dtype=anchor_output.dtype)
    valid = torch.isfinite(corrected_float).all() & torch.isfinite(
        corrected_output
    ).all()
    if not bool(valid.item()):
        raise ActionDeltaGateCorrectionError(
            "Action-Delta Gate predicted correction is non-finite"
        )
    if (
        tuple(corrected_output.shape) != tuple(anchor_output.shape)
        or corrected_output.device != anchor_output.device
        or corrected_output.dtype != anchor_output.dtype
    ):
        raise ActionDeltaGateCorrectionError(
            "Action-Delta Gate predicted correction violated the output contract"
        )
    return corrected_output


def validate_action_delta_gate_runtime_configuration(
    *,
    enabled: bool,
    canonical_recurrence_strategy: str | None,
    prepared_gate: Any,
    batch_size: int,
    use_warm_start: bool,
    warm_start_source: str,
    warm_start_min_iter: int,
    use_latent_precheck: bool,
    latent_precheck_mode: str,
    latent_precheck_trace_level: str,
    shadow_full_depth: bool,
    collect_preconvergence_raw_shadow: bool,
    use_cached_final_output: bool,
    max_skip: int,
    min_terminal_iter: int,
    exact_coda_audit: bool,
    return_mode: str,
) -> None:
    _require(
        return_mode in ACTION_DELTA_GATE_RETURN_MODES,
        "Action-Delta Gate return mode must be one of "
        f"{ACTION_DELTA_GATE_RETURN_MODES}",
    )
    if not enabled:
        return
    _require(isinstance(prepared_gate, PreparedActionDeltaGate), "Action-Delta Gate requires a prepared artifact")
    _require(batch_size == 1, "Action-Delta Gate Phase B requires batch size 1")
    _require(canonical_recurrence_strategy == "adjacent_action_mse", "Action-Delta Gate requires adjacent action-MSE recurrence")
    _require(use_warm_start, "Action-Delta Gate requires warm-start inference")
    _require(warm_start_source == "midpoint", "Action-Delta Gate requires midpoint warm-start")
    _require(warm_start_min_iter == 2, "Action-Delta Gate requires warm_start_min_iter=2")
    _require(not use_latent_precheck, "Action-Delta Gate cannot use latent pre-check")
    _require(latent_precheck_mode == "off", "Action-Delta Gate requires latent_precheck_mode='off'")
    _require(latent_precheck_trace_level == "off", "Action-Delta Gate requires latent_precheck_trace_level='off'")
    _require(not shadow_full_depth, "Action-Delta Gate cannot enable shadow_full_depth")
    _require(not collect_preconvergence_raw_shadow, "Action-Delta Gate cannot collect raw shadow trajectories")
    _require(use_cached_final_output, "Action-Delta Gate Phase B requires use_cached_final_output=True")
    _require(max_skip == 1, "Action-Delta Gate Phase B requires max_skip=1")
    _require(
        isinstance(min_terminal_iter, int)
        and not isinstance(min_terminal_iter, bool)
        and min_terminal_iter >= 2,
        "Action-Delta Gate minimum terminal iteration must be an integer >= 2",
    )
    _require(
        isinstance(exact_coda_audit, bool),
        "Action-Delta Gate exact Coda audit must be boolean",
    )


def validate_action_delta_nonconvergence_filter_configuration(
    *,
    enabled: bool,
    production_gate_enabled: bool,
    shadow_collection_enabled: bool,
    canonical_recurrence_strategy: str | None,
    prepared_gate: Any,
    batch_size: int,
    use_warm_start: bool,
    warm_start_source: str,
    warm_start_min_iter: int,
    use_latent_precheck: bool,
    latent_precheck_mode: str,
    latent_precheck_trace_level: str,
    shadow_full_depth: bool,
    collect_preconvergence_raw_shadow: bool,
    use_cached_final_output: bool,
    profile_coda_cost: bool,
) -> None:
    """Validate the development-only high-side non-convergence filter."""

    if not enabled:
        return
    _require(
        not production_gate_enabled,
        "non-convergence filter cannot run with the production Action-Delta Gate",
    )
    _require(
        not shadow_collection_enabled,
        "non-convergence filter cannot run with Action-Delta shadow collection",
    )
    _require(
        isinstance(prepared_gate, PreparedActionDeltaGate),
        "non-convergence filter requires a prepared frozen Action-Delta artifact",
    )
    _require(batch_size == 1, "non-convergence filter requires batch size 1")
    _require(
        canonical_recurrence_strategy == "adjacent_action_mse",
        "non-convergence filter requires adjacent action-MSE recurrence",
    )
    _require(use_warm_start, "non-convergence filter requires warm-start inference")
    _require(
        warm_start_source == "midpoint",
        "non-convergence filter requires midpoint warm-start",
    )
    _require(
        warm_start_min_iter == 2,
        "non-convergence filter requires warm_start_min_iter=2",
    )
    _require(not use_latent_precheck, "non-convergence filter cannot use latent pre-check")
    _require(
        latent_precheck_mode == "off",
        "non-convergence filter requires latent_precheck_mode='off'",
    )
    _require(
        latent_precheck_trace_level == "off",
        "non-convergence filter requires latent_precheck_trace_level='off'",
    )
    _require(not shadow_full_depth, "non-convergence filter cannot enable shadow_full_depth")
    _require(
        not collect_preconvergence_raw_shadow,
        "non-convergence filter cannot collect raw shadow trajectories",
    )
    _require(
        use_cached_final_output,
        "non-convergence filter requires use_cached_final_output=True",
    )
    _require(
        profile_coda_cost,
        "non-convergence filter requires profile_coda_cost=True for diagnostic accounting",
    )


def validate_action_delta_deferred_backfill_configuration(
    *,
    enabled: bool,
    max_skip_filter_enabled: bool,
    production_gate_enabled: bool,
    shadow_collection_enabled: bool,
    canonical_recurrence_strategy: str | None,
    prepared_gate: Any,
    batch_size: int,
    use_warm_start: bool,
    warm_start_source: str,
    warm_start_min_iter: int,
    use_latent_precheck: bool,
    latent_precheck_mode: str,
    latent_precheck_trace_level: str,
    shadow_full_depth: bool,
    collect_preconvergence_raw_shadow: bool,
    use_cached_final_output: bool,
    profile_coda_cost: bool,
) -> None:
    """Validate the development-only adjacent-history deferred policy."""

    if not enabled:
        return
    _require(
        not max_skip_filter_enabled,
        "deferred/backfill filter cannot run with the max-skip=1 non-convergence filter",
    )
    validate_action_delta_nonconvergence_filter_configuration(
        enabled=True,
        production_gate_enabled=production_gate_enabled,
        shadow_collection_enabled=shadow_collection_enabled,
        canonical_recurrence_strategy=canonical_recurrence_strategy,
        prepared_gate=prepared_gate,
        batch_size=batch_size,
        use_warm_start=use_warm_start,
        warm_start_source=warm_start_source,
        warm_start_min_iter=warm_start_min_iter,
        use_latent_precheck=use_latent_precheck,
        latent_precheck_mode=latent_precheck_mode,
        latent_precheck_trace_level=latent_precheck_trace_level,
        shadow_full_depth=shadow_full_depth,
        collect_preconvergence_raw_shadow=collect_preconvergence_raw_shadow,
        use_cached_final_output=use_cached_final_output,
        profile_coda_cost=profile_coda_cost,
    )
