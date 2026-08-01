"""Serializable, layout-checked action-head workloads for GPU schedule replay."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch


ACTION_HEAD_WORKLOAD_SCHEMA_VERSION = 1
ACTION_HEAD_WORKLOAD_TENSORS = (
    "actions_hidden_states",
    "proprio_input",
    "proprio_features",
    "incoming_warm_start_state",
    "selected_initial_state",
)


class ActionHeadWorkloadError(ValueError):
    """Raised when a captured workload is incomplete or layout-incompatible."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _capture_tensor(name: str, tensor: Optional[torch.Tensor]):
    if tensor is None:
        return None, None
    if not torch.is_tensor(tensor):
        raise ActionHeadWorkloadError(f"{name} must be a tensor or null")
    if not tensor.is_contiguous():
        raise ActionHeadWorkloadError(
            f"{name} must be contiguous at the production action-head boundary"
        )
    if not bool(torch.isfinite(tensor).all().item()):
        raise ActionHeadWorkloadError(f"{name} contains non-finite values")
    metadata = {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "contiguous": True,
    }
    return tensor.detach().to(device="cpu", copy=True), metadata


def build_action_head_workload(
    *,
    actions_hidden_states: torch.Tensor,
    proprio_input: torch.Tensor,
    proprio_features: torch.Tensor,
    incoming_warm_start_state: Optional[torch.Tensor],
    selected_initial_state: torch.Tensor,
    actual_origin: str,
) -> Dict[str, Any]:
    """Copy one exact production action-head input workload to CPU."""

    if actual_origin not in {"COLD", "ACTUAL_WARM"}:
        raise ActionHeadWorkloadError(f"unsupported actual_origin: {actual_origin!r}")
    tensors = {}
    tensor_metadata = {}
    values = {
        "actions_hidden_states": actions_hidden_states,
        "proprio_input": proprio_input,
        "proprio_features": proprio_features,
        "incoming_warm_start_state": incoming_warm_start_state,
        "selected_initial_state": selected_initial_state,
    }
    for name in ACTION_HEAD_WORKLOAD_TENSORS:
        tensors[name], tensor_metadata[name] = _capture_tensor(name, values[name])

    if actual_origin == "COLD" and tensors["incoming_warm_start_state"] is not None:
        raise ActionHeadWorkloadError("COLD workload cannot contain an incoming warm cache")
    if actual_origin == "ACTUAL_WARM":
        incoming = tensors["incoming_warm_start_state"]
        if incoming is None:
            raise ActionHeadWorkloadError("ACTUAL_WARM workload requires an incoming warm cache")
        if not torch.equal(incoming, tensors["selected_initial_state"]):
            raise ActionHeadWorkloadError(
                "ACTUAL_WARM selected initial state must exactly match the accepted cache"
            )

    return {
        "schema_version": ACTION_HEAD_WORKLOAD_SCHEMA_VERSION,
        "actual_origin": actual_origin,
        "identity": None,
        "tensors": tensors,
        "tensor_metadata": tensor_metadata,
    }


def save_action_head_workload(
    path: Path,
    workload: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
) -> str:
    """Atomically save one CPU workload shard and return its SHA-256."""

    output_path = Path(path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite action-head workload: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(workload)
    payload["identity"] = dict(identity)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return sha256_file(output_path)


def load_action_head_workload(
    path: Path,
    *,
    expected_sha256: Optional[str] = None,
    expected_identity: Optional[Mapping[str, Any]] = None,
    expected_origin: Optional[str] = None,
) -> Dict[str, Any]:
    """Load and fail-closed validate one serialized action-head workload."""

    workload_path = Path(path)
    if not workload_path.is_file():
        raise ActionHeadWorkloadError(f"missing action-head workload: {workload_path}")
    actual_sha256 = sha256_file(workload_path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ActionHeadWorkloadError(
            f"action-head workload SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    try:
        payload = torch.load(workload_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ActionHeadWorkloadError(f"cannot load action-head workload: {workload_path}") from exc
    if not isinstance(payload, dict):
        raise ActionHeadWorkloadError("action-head workload root must be a mapping")
    if payload.get("schema_version") != ACTION_HEAD_WORKLOAD_SCHEMA_VERSION:
        raise ActionHeadWorkloadError("unsupported action-head workload schema version")
    if expected_identity is not None and payload.get("identity") != dict(expected_identity):
        raise ActionHeadWorkloadError("action-head workload identity mismatch")

    actual_origin = payload.get("actual_origin")
    if actual_origin not in {"COLD", "ACTUAL_WARM"}:
        raise ActionHeadWorkloadError("invalid action-head workload origin")
    if expected_origin is not None and actual_origin != expected_origin:
        raise ActionHeadWorkloadError(
            f"action-head workload origin mismatch: expected {expected_origin}, got {actual_origin}"
        )
    tensors = payload.get("tensors")
    tensor_metadata = payload.get("tensor_metadata")
    if not isinstance(tensors, dict) or not isinstance(tensor_metadata, dict):
        raise ActionHeadWorkloadError("workload tensors and metadata must be mappings")
    if set(tensors) != set(ACTION_HEAD_WORKLOAD_TENSORS):
        raise ActionHeadWorkloadError("action-head workload tensor fields are incomplete")

    for name in ACTION_HEAD_WORKLOAD_TENSORS:
        tensor = tensors[name]
        metadata = tensor_metadata.get(name)
        if tensor is None:
            if metadata is not None:
                raise ActionHeadWorkloadError(f"{name} metadata must be null with a null tensor")
            continue
        if not torch.is_tensor(tensor) or tensor.device.type != "cpu":
            raise ActionHeadWorkloadError(f"{name} must be a CPU tensor")
        if not tensor.is_contiguous() or not bool(torch.isfinite(tensor).all().item()):
            raise ActionHeadWorkloadError(f"{name} must be finite and contiguous")
        if not isinstance(metadata, dict):
            raise ActionHeadWorkloadError(f"{name} metadata must be a mapping")
        expected_layout = {
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype),
            "contiguous": True,
        }
        for field, expected in expected_layout.items():
            if metadata.get(field) != expected:
                raise ActionHeadWorkloadError(f"{name} metadata field {field!r} mismatch")

    required_dims = {
        "actions_hidden_states": 4,
        "proprio_input": 2,
        "proprio_features": 3,
        "selected_initial_state": 3,
    }
    for name, ndim in required_dims.items():
        tensor = tensors.get(name)
        if tensor is None or tensor.ndim != ndim:
            raise ActionHeadWorkloadError(f"{name} must be a rank-{ndim} tensor")

    incoming = tensors["incoming_warm_start_state"]
    selected = tensors["selected_initial_state"]
    if actual_origin == "COLD" and incoming is not None:
        raise ActionHeadWorkloadError("COLD workload cannot contain an incoming warm cache")
    if actual_origin == "ACTUAL_WARM":
        if incoming is None or incoming.ndim != 3:
            raise ActionHeadWorkloadError("ACTUAL_WARM workload requires a rank-3 incoming cache")
        if not torch.equal(incoming, selected):
            raise ActionHeadWorkloadError(
                "ACTUAL_WARM selected initial state does not match the incoming cache"
            )
    return payload
